"""Check causal label alignment and gradient paths in a tiny multimodal model.

Python 3.10+, PyTorch. CPU only, synthetic inputs, no InternVL weights.
The small attention layer illustrates autograd; it is not InternVL architecture.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Linear(4, 6)
        self.projector = nn.Linear(6, 8)
        self.text_embedding = nn.Embedding(16, 8)
        self.language = nn.ModuleDict({'q': nn.Linear(8, 8), 'k': nn.Linear(8, 8),
                                       'v': nn.Linear(8, 8), 'head': nn.Linear(8, 16)})

    def forward(self, image, ids, visible, cut_language_graph=False):
        visual = self.projector(torch.tanh(self.vision(image)))
        features = torch.cat((visual, self.text_embedding(ids[2:])), dim=0)

        def language_forward():
            q, k, v = (self.language[name](features) for name in ('q', 'k', 'v'))
            length = len(features)
            allowed = torch.ones(length, length, dtype=torch.bool).tril()
            allowed &= visible.unsqueeze(0)
            # Masked early rows can have no visible key. Supply one finite key
            # for these unused rows; they never enter the supervised positions.
            no_key = ~allowed.any(dim=-1)
            allowed[no_key, 0] = True
            scores = (q @ k.T / math.sqrt(8)).masked_fill(~allowed, -torch.inf)
            hidden = torch.tanh(features + scores.softmax(dim=-1) @ v)
            return self.language['head'](hidden)

        if cut_language_graph:
            with torch.no_grad():
                return language_forward()
        return language_forward()


def shifted_loss(logits, labels):
    return F.cross_entropy(logits[:-1], labels[1:], ignore_index=-100)


def module_grad_norm(module):
    return sum(float(p.grad.square().sum()) for p in module.parameters()
               if p.grad is not None) ** .5


def run_case(freeze_language=False, hide_image=False, cut_graph=False, extra_pad=0):
    torch.manual_seed(31)
    model = TinyVLM()
    if freeze_language:
        for parameter in model.language.parameters():
            parameter.requires_grad_(False)
        model.text_embedding.requires_grad_(False)
    image = torch.tensor([[.2, -.3, .5, .8], [-.4, .1, .7, -.2]])
    ids = torch.tensor([0, 0, 2, 3, 5, 6, 7, 0] + [0] * extra_pad)
    labels = torch.tensor([-100] * 4 + [5, 6, 7, -100] + [-100] * extra_pad)
    visible = torch.tensor([True] * 7 + [False] * (1 + extra_pad))
    if hide_image:
        visible[:2] = False
    logits = model(image, ids, visible, cut_language_graph=cut_graph)
    loss = shifted_loss(logits, labels)
    # Independent indexing of the three answer targets checks the shift.
    manual = -logits.log_softmax(-1)[torch.tensor([3,4,5]), torch.tensor([5,6,7])].mean()
    torch.testing.assert_close(loss, manual)
    if loss.requires_grad:
        loss.backward()
    return {'loss': float(loss.detach()), 'loss_requires_grad': loss.requires_grad,
            'vision_grad_norm': module_grad_norm(model.vision),
            'projector_grad_norm': module_grad_norm(model.projector),
            'language_grad_norm': module_grad_norm(model.language),
            'language_gradients_absent': all(p.grad is None for p in model.language.parameters())}


def check_square_average():
    theta = torch.tensor(1., requires_grad=True)
    lengths = torch.tensor([100., 400.])
    sample_mean = torch.stack((2 * theta, theta))
    global_loss = (lengths.sqrt() * sample_mean).sum() / lengths.sqrt().sum()
    local_means_then_average = sample_mean.mean()
    gradient = torch.autograd.grad(global_loss, theta, retain_graph=True)[0]
    wrong_gradient = torch.autograd.grad(local_means_then_average, theta)[0]
    torch.testing.assert_close(gradient, torch.tensor(4 / 3))
    torch.testing.assert_close(wrong_gradient, torch.tensor(1.5))
    return {'global_weighted_gradient': float(gradient),
            'average_of_rank_local_means_gradient': float(wrong_gradient)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = {'train_all': run_case(), 'frozen_language': run_case(freeze_language=True),
             'no_grad_language': run_case(freeze_language=True, cut_graph=True),
             'hidden_visual_prefix': run_case(hide_image=True),
             'extra_padding': run_case(extra_pad=9)}
    for name in ('train_all', 'frozen_language'):
        assert cases[name]['vision_grad_norm'] > 1e-8
        assert cases[name]['projector_grad_norm'] > 1e-8
    assert cases['train_all']['language_grad_norm'] > 1e-8
    assert cases['frozen_language']['language_gradients_absent']
    assert not cases['no_grad_language']['loss_requires_grad']
    assert cases['no_grad_language']['projector_grad_norm'] == 0
    assert cases['hidden_visual_prefix']['projector_grad_norm'] == 0
    assert math.isclose(cases['train_all']['loss'], cases['extra_padding']['loss'], abs_tol=1e-6)
    report = {'scope': 'Synthetic tiny causal attention model; no InternVL weights or training reproduction.',
              'torch': torch.__version__, 'device': 'cpu', 'cases': cases,
              'supervised_logit_positions': [3, 4, 5], 'target_token_ids': [5, 6, 7],
              'square_averaging': check_square_average(),
              'checks': ['causal one-position shift', 'visual gradients without visual targets',
                         'frozen language preserves input gradients', 'no_grad cuts the path',
                         'hidden visual keys cut answer dependence', 'padding invariance',
                         'global weighting differs from local means']}
    result = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result, end='')


if __name__ == '__main__':
    main()
