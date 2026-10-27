#!/usr/bin/env python3
"""Check pinned MorphMimic tensor interfaces on CPU, without its task registry.

Requires an existing PyTorch environment and a read-only checkout supplied with
--source. Namespace loading skips package __init__ (which registers mjlab tasks).
This checks isolated tensor modules, not simulation, checkpoints or training.
"""
import argparse
import ast
import importlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

COMMIT = 'aa40fe46a7fa995ba7a7354ad065a2c8ad64a302'
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == COMMIT, 'Review source changes before updating the pinned revision'
    root = source / 'src/morphmimic'
    for suffix in ['', '.robots', '.model', '.train']:
        module = types.ModuleType('morphmimic' + suffix)
        module.__path__ = [str(root.joinpath(*suffix.strip('.').split('.'))) if suffix else str(root)]
        sys.modules[module.__name__] = module
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(19)
    constants = importlib.import_module('morphmimic.constants')
    spec = importlib.import_module('morphmimic.robots.spec')
    policy_module = importlib.import_module('morphmimic.model.policy')
    distributed = importlib.import_module('morphmimic.train.distributed')
    assert (constants.PROPRIO_DIM, constants.GOAL_DIM, constants.KP_GOAL_DIM,
            constants.LOCAL_DIM, constants.CTX_DIM) == (93, 670, 465, 268, 33)
    translator = spec.Translator(perm=tuple(reversed(range(29))),
        sign=tuple(-1. if i % 2 else 1. for i in range(29)),
        offset=tuple(i * .01 for i in range(29)), num_native=31)
    native = torch.linspace(-.7, .8, 31, dtype=torch.float64, device='cpu')
    canonical = translator.to_canonical(native)
    restored = translator.to_native(canonical, native)
    torch.testing.assert_close(restored, native, atol=1e-12, rtol=0)
    assert translator.untracked == (29, 30)
    limits = torch.stack([native - .2, native + .3], dim=-1)
    mapped_limits = translator.limits_to_canonical(limits)
    assert torch.all(mapped_limits[:, 0] <= mapped_limits[:, 1])
    for fraction in [0., .2, 1.]:
        sample = limits[:, 0] + fraction * (limits[:, 1] - limits[:, 0])
        mapped = translator.to_canonical(sample)
        assert torch.all(mapped >= mapped_limits[:, 0] - 1e-12)
        assert torch.all(mapped <= mapped_limits[:, 1] + 1e-12)
    cfg = policy_module.PolicyCfg(enc_hidden=(12, 8), dec_hidden=(13, 9),
        latent=32, keypoint=True, morph_dim=125, film=True)
    policy = policy_module.Policy(cfg).cpu()
    assert policy.encoder.net[0].in_features == 670
    assert policy.kp_encoder.net[0].in_features == 465
    assert policy.decoder.net[0].in_features == 1087
    observation = torch.randn(2, 930, device='cpu')
    goal = torch.randn(2, 670, device='cpu')
    morphology = torch.linspace(-1, 1, 125, device='cpu')
    command = policy.encode(goal)
    actions = policy(observation, goal, morphology)
    assert command.shape == (2, 32) and actions.shape == (2, 29)
    assert torch.all(command.abs() < 1)
    # At zero FiLM initialization the same decoder is concatenation-only.
    direct = policy.decoder.net(torch.cat([command, observation, morphology.expand(2, -1)], dim=-1))
    torch.testing.assert_close(actions, direct, atol=0, rtol=0)
    motion_ids = {id(p) for p in policy.motion_parameters()}
    kp_ids = {id(p) for p in policy.kp_parameters()}
    assert motion_ids.isdisjoint(kp_ids)
    for parameter in list(policy.encoder.parameters()) + list(policy.decoder.parameters()):
        parameter.requires_grad_(False)
    keypoints = torch.randn(2, 465, device='cpu', requires_grad=True)
    policy.forward_kp(observation, keypoints, morphology).square().mean().backward()
    assert keypoints.grad is not None and torch.isfinite(keypoints.grad).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.kp_parameters())
    assert all(p.grad is None for p in policy.motion_parameters())
    points = [torch.tensor([value], dtype=torch.float64, device='cpu')
              for value in [[.2, .1], [.6, -.1], [.4, 0.]]]
    z = points[0].clone().requires_grad_()
    def gather(outputs, _input):
        for output, point in zip(outputs, points):
            output.copy_(point)
    with patch.object(distributed, 'rank_and_world_size', return_value=(0, 3)), \
         patch.object(distributed.dist, 'all_gather', side_effect=gather):
        loss, distance = distributed.latent_alignment_loss(z)
    loss.backward()
    torch.testing.assert_close(z.grad, torch.tensor([[-.4, .2]], dtype=torch.float64), atol=1e-12, rtol=0)
    assert math.isclose(loss.item(), .05, abs_tol=1e-12)
    tree = ast.parse((root / 'train/trainer.py').read_text())
    config = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TrainCfg')
    pooled = next(node for node in config.body if isinstance(node, ast.AnnAssign)
                  and isinstance(node.target, ast.Name) and node.target.id == 'pool_norms')
    assert ast.literal_eval(pooled.value) is True
    added = next(node for node in tree.body if isinstance(node, ast.AnnAssign)
                 and isinstance(node.target, ast.Name) and node.target.id == 'ADDED_CFG_DEFAULTS')
    assert ast.literal_eval(added.value)['pool_norms'] is False
    print(json.dumps(dict(commit=revision, torch=torch.__version__, device='cpu',
        translator_roundtrip_and_limits=True, tensor_dimensions=True,
        zero_film_equals_concatenation=True, frozen_decoder_passes_keypoint_gradients=True,
        local_alignment_loss=loss.item(), local_alignment_gradient=z.grad.tolist(),
        simulated_gather_pair_distance=distance.item(),
        current_pool_norms=True, legacy_pool_norms=False,
        limits='Tiny random networks and mocked gather; no task registration, checkpoint, simulator, GPU or robot.'), indent=2))


if __name__ == '__main__':
    main()
