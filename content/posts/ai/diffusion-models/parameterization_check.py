"""Check VP prediction conversions and DDIM steps without training a model.

NumPy, with optional Matplotlib for --figure. Clean samples and noise are known
for algebra checks; this oracle is unavailable when generating real samples.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def ddim_moments(xt, epsilon, alpha_bar_t, alpha_bar_s, eta):
    """One backward VP step; s is cleaner than t. Returns mean and variance."""
    if not (0 < alpha_bar_t < alpha_bar_s <= 1 and 0 <= eta <= 1):
        raise ValueError('Need 0 < alpha_bar_t < alpha_bar_s <= 1 and eta in [0,1]')
    a_t, sigma_t = np.sqrt(alpha_bar_t), np.sqrt(1-alpha_bar_t)
    predicted_x0 = (xt - sigma_t * epsilon) / a_t
    variance = eta**2 * (1-alpha_bar_s)/(1-alpha_bar_t) * (1-alpha_bar_t/alpha_bar_s)
    direction_variance = 1-alpha_bar_s-variance
    assert direction_variance >= -1e-14
    mean = np.sqrt(alpha_bar_s)*predicted_x0 + np.sqrt(max(0.,direction_variance))*epsilon
    return mean, variance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('parameterization-results.json'))
    parser.add_argument('--figure',type=Path)
    args=parser.parse_args()
    rng=np.random.default_rng(20260928)
    x0=rng.normal(size=(128,4));epsilon=rng.normal(size=x0.shape)
    conversion=[]
    for alpha_bar in [1.,.999,.8,.1,1e-4,1e-8,0.]:
        a,sigma=np.sqrt(alpha_bar),np.sqrt(1-alpha_bar)
        xt=a*x0+sigma*epsilon
        velocity=a*epsilon-sigma*x0
        restored_x0=a*xt-sigma*velocity
        restored_epsilon=sigma*xt+a*velocity
        np.testing.assert_allclose(restored_x0,x0,atol=1e-12)
        np.testing.assert_allclose(restored_epsilon,epsilon,atol=1e-12)
        delta=rng.normal(scale=.01,size=x0.shape)
        predicted_v=velocity+delta
        predicted_epsilon=sigma*xt+a*predicted_v
        predicted_x0=a*xt-sigma*predicted_v
        v_mse=float(np.mean(delta**2))
        eps_mse=float(np.mean((predicted_epsilon-epsilon)**2))
        x0_mse=float(np.mean((predicted_x0-x0)**2))
        np.testing.assert_allclose(eps_mse,alpha_bar*v_mse,rtol=1e-9,atol=1e-20)
        np.testing.assert_allclose(x0_mse,(1-alpha_bar)*v_mse,rtol=1e-9,atol=1e-20)
        conversion.append({'alpha_bar':alpha_bar,'x0_roundtrip_max_error':float(abs(restored_x0-x0).max()),
                           'epsilon_roundtrip_max_error':float(abs(restored_epsilon-epsilon).max()),
                           'v_mse':v_mse,'converted_epsilon_mse':eps_mse,'converted_x0_mse':x0_mse})

    # A skipped DDIM step must use alpha_bar at BOTH actual endpoints.
    alpha_t,alpha_s=.2,.7
    xt=np.sqrt(alpha_t)*x0+np.sqrt(1-alpha_t)*epsilon
    deterministic,_=ddim_moments(xt,epsilon,alpha_t,alpha_s,0.)
    desired=np.sqrt(alpha_s)*x0+np.sqrt(1-alpha_s)*epsilon
    np.testing.assert_allclose(deterministic,desired,atol=1e-12)
    stochastic_mean,stochastic_variance=ddim_moments(xt,epsilon,alpha_t,alpha_s,1.)
    # Independent conditioning of the forward Gaussian x_t = sqrt(a_t/a_s)*x_s + noise.
    ratio=alpha_t/alpha_s
    covariance=np.sqrt(ratio)*(1-alpha_s)
    conditional_mean=np.sqrt(alpha_s)*x0+covariance/(1-alpha_t)*(xt-np.sqrt(alpha_t)*x0)
    conditional_variance=1-alpha_s-covariance**2/(1-alpha_t)
    np.testing.assert_allclose(stochastic_mean,conditional_mean,atol=1e-12)
    np.testing.assert_allclose(stochastic_variance,conditional_variance,atol=1e-12)
    clean,clean_variance=ddim_moments(xt,epsilon,alpha_t,1.,1.)
    np.testing.assert_allclose(clean,x0,atol=1e-12);assert clean_variance==0.

    beta=np.linspace(1e-4,.02,1000);alpha=1-beta;bar=np.cumprod(alpha)
    adjacent_errors=[]
    for t in [1,20,499,999]:
        noisy=np.sqrt(bar[t])*x0+np.sqrt(1-bar[t])*epsilon
        mean,variance=ddim_moments(noisy,epsilon,bar[t],bar[t-1],1.)
        ddpm_mean=(noisy-beta[t]/np.sqrt(1-bar[t])*epsilon)/np.sqrt(alpha[t])
        posterior_var=beta[t]*(1-bar[t-1])/(1-bar[t])
        np.testing.assert_allclose(mean,ddpm_mean,atol=1e-12)
        np.testing.assert_allclose(variance,posterior_var,atol=1e-12)
        adjacent_errors.append(float(abs(mean-ddpm_mean).max()))
    snr=np.logspace(-6,4,300);a2=snr/(1+snr);sigma=np.sqrt(1-a2)
    prediction_error=.01
    error_from_epsilon=prediction_error/np.sqrt(snr)
    error_from_v=prediction_error*sigma
    if args.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9.3,4.8),constrained_layout=True)
        ax.loglog(snr,error_from_epsilon,label='ε error of 0.01 → x₀ error',color='#c17c39',linewidth=2)
        ax.loglog(snr,error_from_v,label='v error of 0.01 → x₀ error',color='#547fab',linewidth=2,linestyle='--')
        ax.set(xlabel='SNR = ᾱ / (1 − ᾱ)',ylabel='Absolute error in reconstructed x₀',
               title='Same numerical output error, different conversion sensitivity')
        ax.legend();ax.grid(alpha=.2,which='both');ax.spines[['top','right']].set_visible(False)
        args.figure.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.figure,dpi=160);plt.close(fig)
    example_snr = 1e-4
    example_a2 = example_snr / (1 + example_snr)
    example_a, example_sigma = np.sqrt(example_a2), np.sqrt(1-example_a2)
    example_xt = example_a*x0 + example_sigma*epsilon
    example_v = example_a*epsilon - example_sigma*x0
    epsilon_reconstruction = (example_xt-example_sigma*(epsilon+.01))/example_a
    v_reconstruction = example_a*example_xt-example_sigma*(example_v+.01)
    report={'numpy':np.__version__,'seed':20260928,'conversion_cases':conversion,
            'skipped_step':{'alpha_bar_t':alpha_t,'alpha_bar_s':alpha_s,
                           'deterministic_max_error':float(abs(deterministic-desired).max()),
                           'eta_one_conditional_mean_max_error':float(abs(stochastic_mean-conditional_mean).max()),
                           'eta_one_conditional_variance':float(stochastic_variance)},
            'adjacent_ddpm_max_mean_error':max(adjacent_errors),'clean_endpoint_variance':clean_variance,
            'sensitivity':{'output_error':prediction_error,'snr':example_snr,
                           'x0_error_from_epsilon':float(np.sqrt(np.mean((epsilon_reconstruction-x0)**2))),
                           'x0_error_from_v':float(np.sqrt(np.mean((v_reconstruction-x0)**2)))},
            'scope':'Oracle algebra and error propagation only; no trained denoiser, FID or model-quality comparison.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
