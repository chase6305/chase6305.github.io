"""Optional CPU-only autograd verification of the new mathematical constructs."""
import argparse,hashlib,json,math,sys
from itertools import product
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[3]
BUNDLE=ROOT/'content/posts/ai/pion-muon-vla-rlvr'
sys.path.insert(0,str(BUNDLE))
from grpo_group_signal import group_moments,one_group
from vla_gradient_rank import hidden_example


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    report={'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu',
            'optimizer_steps':0,'hidden_networks':[],'group_distributions':[],
            'autograd_groups':0,'max_group_gradient_absolute_error':0.,
            'source_sha256':{name:hashlib.sha256((BUNDLE/name).read_bytes()).hexdigest()
                             for name in ['grpo_group_signal.py','vla_gradient_rank.py']},
            'scope':'Tiny constructed forward/backward evaluations only. No VLA/LLM checkpoint, rollout engine, GPU or model training.'}
    for size in (2,3,8,32):
        example=hidden_example(size)
        inputs=torch.tensor(example['inputs'],dtype=torch.float64,device='cpu')
        hidden=torch.eye(size,dtype=torch.float64,device='cpu',requires_grad=True)
        head=torch.ones(1,size,dtype=torch.float64,device='cpu',requires_grad=True)
        prediction=torch.relu(inputs@hidden.T)@head.T
        loss=prediction.abs().mean()
        gradient,head_gradient=torch.autograd.grad(loss,(hidden,head))
        expected=torch.tensor(example['hidden_gradient'],dtype=torch.float64)
        expected_head=torch.tensor(example['output_head_gradient'],dtype=torch.float64)
        torch.testing.assert_close(gradient,expected,rtol=2e-14,atol=1e-15)
        torch.testing.assert_close(head_gradient,expected_head,rtol=2e-14,atol=1e-15)
        hidden_rank=int(torch.linalg.matrix_rank(gradient).item())
        head_rank=int(torch.linalg.matrix_rank(head_gradient).item())
        if hidden_rank!=size or head_rank!=1 or loss.detach().item()!=1.:
            raise AssertionError('Constructed scalar-output network rank/loss mismatch')
        report['hidden_networks'].append({'width':size,'scalar_output':1,'positions':size,
            'hidden_gradient_rank':hidden_rank,'head_gradient_rank':head_rank,
            'max_gradient_absolute_error':(gradient-expected).abs().max().item()})
    for size in (2,3,8):
        for p in (.01,.1,.5,.9):
            gradients=[];weights=[];mixed=[]
            for outcomes in product((0,1),repeat=size):
                rewards=torch.tensor(outcomes,dtype=torch.float64,device='cpu')
                std=rewards.std(correction=0)
                advantages=((rewards-rewards.mean())/std if std.item() else torch.zeros_like(rewards)).detach()
                theta=torch.tensor(math.log(p/(1-p)),dtype=torch.float64,device='cpu',requires_grad=True)
                current=torch.distributions.Bernoulli(logits=theta)
                old_logs=torch.distributions.Bernoulli(logits=theta.detach()).log_prob(rewards)
                objective=(advantages*(current.log_prob(rewards)-old_logs).exp()).mean()
                gradient=torch.autograd.grad(objective,theta)[0].item()
                expected=one_group(outcomes,p)['ascent_gradient']
                error=abs(gradient-expected)
                report['max_group_gradient_absolute_error']=max(report['max_group_gradient_absolute_error'],error)
                if error>2e-14:raise AssertionError('On-policy ratio-surrogate autograd mismatch')
                successes=sum(outcomes);weight=p**successes*(1-p)**(size-successes)
                gradients.append(gradient);weights.append(weight)
                if 0<successes<size:mixed.append((gradient,weight))
                report['autograd_groups']+=1
            predicted=group_moments(size,str(p))
            mean=math.fsum(g*w for g,w in zip(gradients,weights))
            variance=math.fsum(w*(g-mean)**2 for g,w in zip(gradients,weights))
            q=math.fsum(w for g,w in mixed)
            conditional_mean=math.fsum(g*w for g,w in mixed)/q
            conditional_variance=math.fsum(w*(g-conditional_mean)**2 for g,w in mixed)/q
            for observed,expected in [(mean,float(predicted['full_group']['mean'])),
                                      (variance,float(predicted['full_group']['variance'])),
                                      (conditional_mean,float(predicted['nondegenerate_only']['mean'])),
                                      (conditional_variance,float(predicted['nondegenerate_only']['variance']))]:
                if not math.isclose(observed,expected,rel_tol=2e-12,abs_tol=2e-14):
                    raise AssertionError('Autograd distribution moment mismatch')
            snr=mean*mean/variance
            if not math.isclose(snr,float(predicted['full_group']['snr']),rel_tol=2e-11):
                raise AssertionError('Full-group SNR mismatch')
            if predicted['nondegenerate_only']['variance_zero']:
                if max(g for g,w in mixed)-min(g for g,w in mixed)>2e-14:
                    raise AssertionError('A constant retained coefficient varied beyond Float64 rounding')
                conditional_snr=None
            else:
                conditional_snr=conditional_mean*conditional_mean/conditional_variance
                if not math.isclose(conditional_snr,float(predicted['nondegenerate_only']['snr']),rel_tol=2e-10):
                    raise AssertionError('Retained-group SNR mismatch')
            report['group_distributions'].append({'size':size,'p':p,'patterns':len(gradients),
                'full_snr':snr,'retained_snr':conditional_snr,'nondegenerate_probability':q})
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'hidden_networks':len(report['hidden_networks']),
                     'autograd_groups':report['autograd_groups'],
                     'group_distributions':len(report['group_distributions']),
                     'max_group_gradient_absolute_error':report['max_group_gradient_absolute_error'],
                     'device':'cpu','optimizer_steps':0}))


if __name__=='__main__':main()
