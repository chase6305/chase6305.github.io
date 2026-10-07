"""CPU tensor-to-objective demo: causal shift, EOS/PAD, DPO and GRPO.

A trainable 5x5 bigram table predicts the next token from only the last token.
Two fixed unequal-length answers are a loss/gradient fixture, NOT GRPO rollouts
or a language-model training benchmark. DPO and GRPO each get one independent
SGD step from identical initialization. There is no generation or tokenizer.
"""
import argparse
import json
from pathlib import Path

import torch

from rl_lab import (DTYPE, categorical_kl, clipped_policy_loss, dpo_loss,
                    group_advantages, response_token_logps)


def fixture(extra_padding=0):
    # Vocabulary: BOS=0, prompt=1, good=2, bad=3, EOS=PAD=4.
    ids = torch.tensor([[0, 1, 2, 4, 4], [0, 1, 3, 3, 4]])
    mask = torch.tensor([[False, False, True, True, False],
                         [False, False, True, True, True]])
    if type(extra_padding) is not int or extra_padding < 0:
        raise ValueError("extra_padding must be a nonnegative integer")
    ids = torch.cat((ids, torch.full((2, extra_padding), 4, dtype=torch.long)), dim=1)
    mask = torch.cat((mask, torch.zeros(2, extra_padding, dtype=torch.bool)), dim=1)
    return ids, mask


def objectives(actor, old, reference, ids, mask):
    """Exactly two responses to one prompt: chosen then rejected, rewards [1,0]."""
    logits = actor[ids]
    current, valid = response_token_logps(logits, ids, mask)
    with torch.no_grad():
        old_logps, _ = response_token_logps(old[ids], ids, mask)
        ref_logps, _ = response_token_logps(reference[ids], ids, mask)
        advantages = group_advantages(torch.tensor([[1., 0.]], dtype=DTYPE)).flatten()
    lengths = valid.sum(-1)
    sequence = current.sum(-1)
    ref_sequence = ref_logps.sum(-1)
    dpo = dpo_loss(sequence[:1], sequence[1:], ref_sequence[:1], ref_sequence[1:]).mean()
    policy = clipped_policy_loss(current, old_logps,
                                 advantages[:, None].expand_as(current))
    # Tiny vocabulary allows exact current||reference KL at each observed prefix.
    kl = categorical_kl(logits[:, :-1], reference.detach()[ids][:, :-1])
    token_loss = (policy + .03 * kl).masked_fill(~valid, 0)
    return dict(dpo=dpo, grpo=token_loss.sum(-1).div(lengths).mean(),
                global_token_loss=token_loss.sum() / lengths.sum(),
                relative_margin=(sequence - ref_sequence)[0] - (sequence - ref_sequence)[1],
                sequence_logps=sequence, mean_logps=sequence / lengths,
                token_logps=current, lengths=lengths,
                advantages=advantages, valid=valid)


def run():
    torch.set_num_threads(1)
    ids, mask = fixture()
    report = dict(scope="fixed two-answer bigram gradient fixture; no online sampling",
                  device="cpu", dtype=str(DTYPE), torch=torch.__version__,
                  vocabulary={"BOS": 0, "prompt": 1, "good": 2, "bad": 3, "EOS_and_PAD": 4},
                  input_ids=ids.tolist(), response_mask=mask.tolist(),
                  dpo_beta=.5, grpo_kl_beta=.03, clip_epsilon=.2,
                  learning_rate=.1, reduction="GRPO: mean tokens per response, then mean responses")
    for name in ("dpo", "grpo"):
        actor = torch.nn.Parameter(torch.zeros(5, 5, dtype=DTYPE))
        old = actor.detach().clone()
        reference = actor.detach().clone()
        optimizer = torch.optim.SGD([actor], lr=.1)
        before = objectives(actor, old, reference, ids, mask)
        optimizer.zero_grad(set_to_none=True)
        before[name].backward()
        gradient_norm = actor.grad.norm().item()
        optimizer.step()
        with torch.no_grad():
            after = objectives(actor, old, reference, ids, mask)
        report[name] = dict(initial_loss=before[name].item(), final_loss=after[name].item(),
                            gradient_norm=gradient_norm,
                            relative_margin_before=before["relative_margin"].item(),
                            relative_margin_after=after["relative_margin"].item())
        if name == "grpo":
            report["alignment"] = {key: before[key].tolist() for key in
                                    ("lengths", "sequence_logps", "mean_logps", "token_logps", "advantages", "valid")}
            report["initial_global_token_loss"] = before["global_token_loss"].item()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results-tokens.json"))
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
