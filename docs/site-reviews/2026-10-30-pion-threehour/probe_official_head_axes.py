"""Pinned RL optimizer on small constructed CPU gradients; no model training."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--official-file',type=Path,required=True)
source=parser.parse_args().official_file
if hashlib.sha256(source.read_bytes()).hexdigest()!='15c1331453ef3db4b565a514d1e43f74692e2e910eb8cb939894d54371dd402d':
    parser.error('Official RL source does not match the pinned Pion commit')
import numpy as np
import torch
spec=importlib.util.spec_from_file_location('pinned_rl_muon',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_num_threads(1)
rows=[]
def check(shape,axis,heads,amplitudes,name,fallback=False):
    gradient=np.zeros(shape)
    width=shape[axis]//len(amplitudes)
    for head,value in enumerate(amplitudes):
        position=(head*width,(2*head+1)%shape[1])if axis==0 else((2*head+1)%shape[0],head*width)
        gradient[position]=value
    parameter=torch.nn.Parameter(torch.zeros(shape,dtype=torch.float64,device='cpu'))
    parameter.grad=torch.from_numpy(gradient.copy())
    optimizer=module.PerHeadPionAdamW([{'params':[parameter],'num_heads':heads,'head_split_dim':axis}],lr=1.,weight_decay=0.,momentum=0.,nesterov=False,promotion_steps=1,ns_steps=5)
    assert optimizer.param_groups[0]['scale_factor']==5.
    optimizer.step()
    block_shape=shape if fallback else(shape[0]//heads,shape[1])if axis==0 else(shape[0],shape[1]//heads)
    scale=5*math.sqrt(max(block_shape))
    expected=-np.sign(gradient)*scale
    actual=parameter.detach().numpy()
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
    assert np.isfinite(actual).all()
    if fallback:
        result=module._multihead_ortho_2d(torch.from_numpy(gradient),heads,axis,lambda x:module._batched_pion(x,1,4),scale_factor=5.)
        assert result is None
    rows.append({'case':name,'shape':shape,'axis':axis,'heads':heads,'amplitudes':amplitudes,'fallback':fallback,'scale':scale,'nonzero_count':int(np.count_nonzero(actual)),'maximum_absolute_error':float(np.max(np.abs(actual-expected)))})
for axis,shape in [(0,(12,8)),(1,(8,12))]:
    for name,values in [('amplitude_ratio',[8.,2.,.5]),('signed',[-8.,2.,-.5]),('zero_head',[8.,0.,.5]),('all_zero',[0.,0.,0.])]:check(shape,axis,3,values,name)
    check(shape,axis,5,[8.],'incompatible_head_fallback',True)
check((4,8),0,1,[2.],'KV_single_head')
result={'scope':'Small unsharded CPU gradients and a single optimizer step; not model/GPU/FSDP training','official_commit':'87965014b360bde604632cba0aa12e423c6c63c6','official_file_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','parameter_dtype':'float64','official_internal_dtype':'bfloat16','lr':1.,'weight_decay':0.,'momentum':0.,'nesterov':False,'promotion_steps':1,'suppression_steps':4,'cases':rows}
print(json.dumps(result,indent=2,allow_nan=False))
