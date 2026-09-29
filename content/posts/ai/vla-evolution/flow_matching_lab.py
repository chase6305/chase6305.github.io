"""Learn a conditional 1D flow on a known two-mode distribution.

Python 3.10+, PyTorch; Matplotlib is optional for --figure. CPU only.
This is a small neural-network experiment, not a VLA, action controller, or
reproduction of pi0. Training uses uniform flow times and synthetic samples.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

DATA_STD = 0.15


class Velocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(5, 64), nn.SiLU(),
                                    nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x, s, condition):
        features = torch.cat((x, s, condition, torch.sin(math.pi*s),
                              torch.cos(math.pi*s)), dim=-1)
        return self.layers(features)


def sample_batch(size, generator):
    condition = torch.randint(2, (size, 1), generator=generator).float() - 0.5
    mode = 2*torch.randint(2, (size, 1), generator=generator).float() - 1
    action = condition + mode + DATA_STD*torch.randn(size, 1, generator=generator)
    noise = torch.randn(size, 1, generator=generator)
    s = torch.rand(size, 1, generator=generator)
    return condition, action, noise, s


def exact_velocity(x, s, condition):
    """Gaussian conditioning, independently derived from the data mixture."""
    means = condition + torch.tensor([-1., 1.], dtype=x.dtype)
    variance = (1-s).square()*DATA_STD**2 + s.square()
    residual = x - (1-s)*means
    weights = torch.softmax(-0.5*residual.square()/variance, dim=-1)
    covariance = s - (1-s)*DATA_STD**2
    conditional_velocity = -means + covariance/variance*residual
    return (weights*conditional_velocity).sum(dim=-1, keepdim=True)


def mixture_cdf(x, condition, s=0.):
    variance = (1-s)**2*DATA_STD**2+s**2
    means = (1-s)*(condition+torch.tensor([-1., 1.], dtype=x.dtype))
    component_cdf = 0.5*(1+torch.erf((x-means)/math.sqrt(2*variance)))
    return component_cdf.mean(dim=-1, keepdim=True)


def target_quantiles(probabilities, condition):
    low, high = condition-4., condition+4.
    low, high = low.expand_as(probabilities).clone(), high.expand_as(probabilities).clone()
    for _ in range(60):
        mid = (low+high)/2
        smaller = mixture_cdf(mid, condition) < probabilities
        low = torch.where(smaller, mid, low)
        high = torch.where(smaller, high, mid)
    return (low+high)/2


@torch.no_grad()
def integrate(field, initial, condition, steps, method='euler', return_path=False):
    if type(steps) is not int or steps < 1 or method not in ('euler', 'heun'):
        raise ValueError('Use positive integer steps and euler/heun')
    x = initial.clone()
    dt = -1./steps
    path = [x.clone()]
    for i in range(steps):
        s = torch.full_like(x, 1-i/steps)
        velocity = field(x, s, condition)
        candidate = x + dt*velocity
        if method == 'heun':
            next_s = torch.full_like(s, 1-(i+1)/steps)
            candidate = x + dt*0.5*(velocity+field(candidate, next_s, condition))
        x = candidate
        if return_path:
            path.append(x.clone())
    return (x, path) if return_path else x


def numerical_checks():
    # One normal endpoint: E[A|X] and hence the exact velocity has a closed form.
    x = torch.tensor([[-1.2], [0.1], [2.3]], dtype=torch.float64)
    c = torch.full_like(x, 0.5)
    torch.testing.assert_close(exact_velocity(x, torch.ones_like(x), c), x-c)
    # Independent CDF invariant: analytic transport must preserve probability.
    p = torch.tensor([[.05], [.2], [.8], [.95]], dtype=torch.float64)
    initial = math.sqrt(2)*torch.erfinv(2*p-1)
    for value in (-.5, .5):
        condition = torch.full_like(initial, value)
        result = integrate(exact_velocity, initial, condition, 512, 'heun')
        torch.testing.assert_close(mixture_cdf(result, condition), p, atol=2e-5, rtol=0)
    # Oracle path checks the sign and all entries of a 2 x 3 action chunk.
    action = torch.tensor([[1., 2., 3.], [-1., 0., 2.]])
    noise = torch.tensor([[.1, .4, -.2], [1., 2., -3.]])
    generated = integrate(lambda x, s, c: noise-action, noise, noise*0, 10)
    torch.testing.assert_close(generated, action, atol=1e-6, rtol=0)
    return ['noise-endpoint conditional mean', 'CDF conservation for both conditions',
            'whole-chunk oracle direction']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 29, 43])
    parser.add_argument('--output', type=Path, default=Path('flow-learning-results.json'))
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checks = numerical_checks()
    # Fixed probability grid avoids random sample variance in the solver comparison.
    p = (torch.arange(2048, dtype=torch.float64).reshape(-1, 1)+.5)/2048
    initial64 = math.sqrt(2)*torch.erfinv(2*p-1)
    rows, selected = [], None
    for seed in args.seeds:
        torch.manual_seed(seed)
        training = torch.Generator().manual_seed(seed+1000)
        model = Velocity()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(args.steps):
            condition, action, noise, s = sample_batch(256, training)
            mixed = (1-s)*action+s*noise
            target = noise-action
            loss = (model(mixed, s, condition)-target).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        heldout = torch.Generator().manual_seed(20260929)
        c, a, noise, s = sample_batch(8192, heldout)
        mixed = (1-s)*a+s*noise
        with torch.no_grad():
            prediction = model(mixed, s, c)
            oracle = exact_velocity(mixed, s, c)
            validation = {'sample_target_mse': float((prediction-(noise-a)).square().mean()),
                          'exact_field_mse': float((prediction-oracle).square().mean()),
                          'oracle_sample_target_mse': float((oracle-(noise-a)).square().mean())}
        for value in (-.5, .5):
            c64 = torch.full_like(initial64, value)
            truth = target_quantiles(p, c64)
            for method, steps in [('euler', 1), ('euler', 4), ('euler', 16),
                                  ('euler', 64), ('heun', 8)]:
                result = integrate(model, initial64.float(), c64.float(), steps, method).double()
                exact = integrate(exact_velocity, initial64, c64, steps, method)
                assert torch.isfinite(result).all()
                rows.append({'seed': seed, 'condition': value, 'method': method,
                             'steps': steps, 'nfe': steps*(2 if method == 'heun' else 1),
                             'learned_quantile_W1': float((result.sort(dim=0).values-truth).abs().mean()),
                             'exact_field_quantile_W1': float((exact.sort(dim=0).values-truth).abs().mean()),
                             'right_mode_fraction': float((result > value).double().mean()),
                             'gap_fraction_abs_center_lt_0_4': float(((result-value).abs()<.4).double().mean()),
                             'mean': float(result.mean()), 'field_validation': validation})
        if selected is None:
            selected = model
    parameter_count = sum(v.numel() for v in selected.parameters())
    report = {'torch': torch.__version__, 'device': 'cpu', 'dtype': 'float32 training; float64 oracle',
              'optimizer': 'Adam, lr=0.001', 'batch_size': 256, 'training_steps_per_seed': args.steps,
              'seeds': args.seeds, 'parameters': parameter_count,
              'target': 'A = c + sign + 0.15 * Normal(0,1); c in {-0.5,0.5}; sign equiprobable +/-1',
              'time_sampling': 'Uniform[0,1)', 'reference_quantiles': 2048,
              'direct_L2_optimum': 'E[A|c]=c; population MSE=1.0225; in-gap fraction=1',
              'checks': checks, 'rows': rows,
              'scope': 'Synthetic scalar conditional density, online independent training samples. No robot, no pretrained VLM, no pi0 reproduction. Quantile-grid W1 is a finite-grid approximation; solver error and learned field error both contribute.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if args.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
        c = torch.full_like(initial64, .5)
        generated = integrate(selected, initial64.float(), c.float(), 64).reshape(-1)
        grid = torch.linspace(-2.5, 3., 800, dtype=torch.float64)
        density = sum(torch.exp(-.5*((grid-(.5+sign))/DATA_STD)**2)
                      for sign in (-1, 1))/(2*DATA_STD*math.sqrt(2*math.pi))
        axes[0].plot(grid.tolist(), density.tolist(), color='#4c8c78', label='Known target density')
        axes[0].hist(generated.tolist(), bins=60, density=True, alpha=.4, color='#699bc3', label='Learned flow, Euler 64')
        axes[0].axvline(.5, color='#b87a40', ls='--', label='Direct L2 optimum')
        axes[0].set(title=f'Condition c = 0.5; seed {args.seeds[0]}', xlabel='Synthetic action', ylabel='Density')
        axes[0].legend(fontsize=8)
        for seed in args.seeds:
            group=[r for r in rows if r['seed']==seed and r['condition']==.5 and r['method']=='euler']
            axes[1].plot([r['nfe'] for r in group], [r['learned_quantile_W1'] for r in group], 'o-', label=f'Learned: seed {seed}')
        group=[r for r in rows if r['seed']==args.seeds[0] and r['condition']==.5 and r['method']=='euler']
        axes[1].plot([r['nfe'] for r in group], [r['exact_field_quantile_W1'] for r in group], 's--', color='#777777', label='Exact field')
        axes[1].set(xscale='log', yscale='log', xlabel='Velocity evaluations (NFE)', ylabel='Quantile-grid W1', title='More steps do not fix the field')
        axes[1].legend(fontsize=8)
        start=torch.tensor([[-1.6],[-.9],[-.4],[.4],[.9],[1.6]])
        _, path=integrate(selected,start,torch.full_like(start,.5),64,return_path=True)
        for index in range(len(start)):
            axes[2].plot([i/64 for i in range(65)],[float(x[index,0]) for x in path],color='#8c77b5' if index<3 else '#679abd')
        axes[2].set(xlabel='Sampling progress 1 - s', ylabel='Synthetic action', title='Different initial noise values')
        for ax in axes:ax.grid(alpha=.15)
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure,dpi=160);plt.close(fig)
    print(json.dumps({'parameters':parameter_count,'rows':len(rows),'checks':checks,'output':str(args.output)}))


if __name__ == '__main__':
    main()
