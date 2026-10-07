#!/usr/bin/env python3
"""Read-only CPU probes for the fixed lucidrains/light-loco-parkour implementation.

Requires that repository's dependencies, including PyTorch. These probes exercise
array/gradient contracts; they do not train a humanoid or reproduce paper scores.
Run: python source_probe.py --source-root /path/to/light-loco-parkour --output result.json
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

EXPECTED_COMMIT = '963a6ec3b8b42eb29dd6b9dfed34ededd3c64c7b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.source_root.resolve()
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != EXPECTED_COMMIT:
        parser.error(f'expected source commit {EXPECTED_COMMIT}, got {commit}')
    if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip():
        parser.error('source checkout must be clean so the recorded commit identifies tested code')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    import torch
    from torch import nn
    from light_loco_parkour import (
        Actor, Agent, Critic, StateEncoder, Gaussian, Beta, DistillationWrapper,
        RewardHyperParams, FootAccelerationPenalty, RewardShapingWrapper,
        MotionPrior, Discriminator, PhaseConditionalMotionPrior,
    )
    from light_loco_parkour.light_loco_parkour import default_stateful_reward_fns
    from types import SimpleNamespace
    torch.set_num_threads(1)
    torch.manual_seed(42)
    rows = {}

    # A nonnegative acceleration excess is added with the repository's positive default.
    hparams = RewardHyperParams()
    penalty = FootAccelerationPenalty()
    impulse = SimpleNamespace(foot_acceleration=torch.tensor([[[50., 0., 0.], [0., 0., 0.]]]))
    rest = SimpleNamespace(foot_acceleration=torch.zeros(1, 2, 3))
    filtered = [float(penalty(impulse, hparams))]
    filtered += [float(penalty(rest, hparams)) for _ in range(2)]
    weight = default_stateful_reward_fns()[0][1]
    assert weight == 0.01 and filtered[0] == 20.
    wrapper = RewardShapingWrapper(reward_fns=(), reward_hparams=hparams)
    actual_contribution = float(wrapper(impulse))
    assert abs(actual_contribution - 0.2) < 1e-6
    rows['foot_acceleration'] = dict(filtered=filtered, default_weight=weight,
        default_reward_contribution=actual_contribution, paper_weight=-0.01)

    # A global reset cannot isolate an episode boundary in one vectorized environment.
    batched = FootAccelerationPenalty()
    two_impulses = SimpleNamespace(foot_acceleration=torch.tensor(
        [[[50., 0., 0.]], [[70., 0., 0.]]]))
    two_rest = SimpleNamespace(foot_acceleration=torch.zeros(2, 1, 3))
    first = batched(two_impulses, hparams)
    unchanged = batched(two_rest, hparams)
    batched.reset_()
    global_reset = batched(two_rest, hparams)
    expected_selective = unchanged.clone()
    expected_selective[0] = 0.
    torch.testing.assert_close(first, torch.tensor([20., 40.]))
    torch.testing.assert_close(global_reset, torch.zeros(2))
    assert float(unchanged[0]) > 0 and float(expected_selective[1]) > 0
    rows['vectorized_reset'] = {
        'initial_memory': first.tolist(), 'without_reset_next': unchanged.tolist(),
        'global_reset_next': global_reset.tolist(),
        'desired_next_if_only_environment_zero_resets': expected_selective.tolist(),
        'selective_reset_is_upstream_feature': False,
    }

    def actor(frames=1, recurrent=False, latent=None, actions=1):
        return Actor(16, state_encoder=StateEncoder(16, dim_state=2,
            num_stacked_frames=frames, use_rnn=recurrent), num_actions=actions,
            depth=2, action_distr=Gaussian(), next_latent_prediction=latent)

    # Same recurrent weights: a whole sequence versus streaming individual frames.
    states = torch.randn(1, 6, 2)
    rows['streaming'] = {}
    for frames in (1, 5):
        model = actor(frames=frames, recurrent=True, latent=False).eval()
        with torch.no_grad():
            full, _ = model((states,), deterministic=True)
            hidden = None
            chunks = []
            for t in range(states.shape[1]):
                out, hidden = model((states[:, t:t+1],), time_hiddens=hidden, deterministic=True)
                chunks.append(out)
            streaming = torch.cat(chunks, dim=1)
            error = float((full-streaming).abs().max())
        if frames == 1:
            assert error < 1e-6
        else:
            assert error > 1e-4
        rows['streaming'][str(frames)] = {'max_absolute_error':error}

    # None is the Actor's auto-enabled default; literal False really disables it.
    rows['latent_default'] = {}
    for label, setting in [('None', None), ('False', False), ('True', True)]:
        model = actor(latent=setting).train()
        model((states,), deterministic=True)
        active = model.to_actions.next_latent_prediction_loss is not None
        assert active == (setting is not False)
        rows['latent_default'][label] = {'active':active, 'loss':float(model.next_latent_prediction_loss.detach())}

    # Distillation weights scale a valid-token mean; they are not normalized by their sum.
    class FixedActor(nn.Module):
        aux_decoder = None
        def __init__(self, values):
            super().__init__()
            self.register_buffer('values', torch.tensor(values).reshape(1, 2, 1))
        def forward(self, states, **kwargs):
            return self.values, None
    student = FixedActor([1., 3.])
    teacher = FixedActor([0., 0.])
    distiller = DistillationWrapper(student, teacher, student_state_keys=('s',), teacher_state_keys=('s',))
    loss = float(distiller({'s':torch.zeros(1,2,2)}, weights=torch.tensor([[1.,3.]])))
    assert loss == 14.
    rows['weighted_distillation'] = {'actual':loss,'sum_weight_normalized_alternative':7.}

    # calc_gae returns value targets, using continuation masks at terminals.
    critic = Critic(16,state_encoder=StateEncoder(16,dim_state=2,num_stacked_frames=1),depth=2)
    agent = Agent(actor(latent=False),critic)
    returns=agent.calc_gae(torch.tensor([[1.,2.]]),torch.tensor([[.5,.4,.3]]),
        torch.tensor([[1.,0.]]),gamma=.9,lam=.8,use_accelerated=False)
    torch.testing.assert_close(returns,torch.tensor([[2.512,2.]]))
    rows['gae'] = {'returns':returns.tolist(),'advantages':(returns-torch.tensor([[.5,.4]])).tolist()}

    # Equal-to-threshold is still the previous phase. Smooth mode needs ordered triggers.
    priors=[MotionPrior(Discriminator(8,dim_in=2,depth=2)) for _ in range(3)]
    phase=PhaseConditionalMotionPrior(priors,prior_transition_positions=(1.,2.))
    positions=torch.tensor([[.9,1.,1.1,2.,2.1]])
    phases=phase.resolve_phases(positions)
    assert phases.tolist()==[[0,0,1,1,2]]
    smooth=PhaseConditionalMotionPrior(priors,prior_transition_positions=(1.,2.),
        smooth_handoff=True,handoff_temperature=4.)
    weights=smooth.phase_weights(positions)
    torch.testing.assert_close(weights.sum(-1),torch.ones_like(positions))
    unsorted=smooth.phase_weights(positions,prior_transition_positions=torch.tensor([2.,1.]))
    assert float(unsorted.min()) < 0.
    shared=PhaseConditionalMotionPrior(priors[0],prior_transition_positions=(1.,2.))
    assert shared.motion_priors[0] is shared.motion_priors[1]
    rows['phase_prior']={'positions':positions.tolist(),'hard_phases':phases.tolist(),
        'smooth_weights':weights.tolist(),'unsorted_min_weight':float(unsorted.min()),
        'single_prior_shares_weights':True}

    # Identical parameters, but losing the rollout's GRU state changes PPO log probabilities.
    model=actor(recurrent=True,latent=False).eval()
    with torch.no_grad():
        distribution, _=model((states,),return_action_distr=True)
        actions=distribution.mean
        original=model.action_distr.log_prob(distribution,actions)
        replay, _=model((states[:,3:],),return_action_distr=True)
        replay_log=model.action_distr.log_prob(replay,actions[:,3:])
        ratios=(replay_log-original[:,3:]).exp()
    assert float((ratios-1.).abs().max()) > 1e-4
    rows['recurrent_replay']={'ratio_with_unchanged_weights_but_missing_initial_hidden':ratios.tolist()}

    # AMP reward subtracted after detach contributes no gradient to a parameter.
    parameter=torch.tensor(2.,requires_grad=True)
    amp=(parameter*3.).detach()
    (parameter.square()-.01*amp).backward()
    assert float(parameter.grad)==4.
    rows['detached_amp']={'gradient_with_detached_reward':float(parameter.grad),
        'gradient_without_reward':4.}

    # The exported model's stochastic output and deterministic action have different types/shapes.
    gaussian_actor=actor(actions=3,latent=False).eval()
    with torch.no_grad():
        distribution,_=gaussian_actor((states,))
        action_mean,_=gaussian_actor((states,),deterministic=True)
    rows['action_contract']={'default_type':type(distribution).__name__,
        'distribution_mean_shape':list(distribution.mean.shape),'deterministic_shape':list(action_mean.shape)}
    beta=Beta()
    params=torch.zeros(1,1,2)
    beta_dist=beta(params)
    rows['beta_default']={'raw_zero_mean':float(beta.mean(params)),
        'base_alpha':float(beta_dist.base_dist.concentration1),
        'base_beta':float(beta_dist.base_dist.concentration0),
        'concentration_before_unimodality_floor':float(beta.concentration(params))}

    report={'source_commit':commit,'source_module':str(Path(sys.modules['light_loco_parkour'].__file__).relative_to(root)),
        'scope':'CPU component probes; no simulator, humanoid training, checkpoint or hardware',
        'versions':{name:importlib.metadata.version(name) for name in (
            'torch','numpy','light-loco-parkour','assoc-scan','mean-conc-beta',
            'torch-einops-utils','x-mlps-pytorch','hl-gauss-pytorch')},'results':rows,'passed':True}
    rendered=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.write_text(rendered,encoding='utf-8')
    print(rendered,end='')


if __name__=='__main__':
    main()
