#!/usr/bin/env python3
"""Published sinusoidal embedding against scalar math; isolated CPU PyTorch required."""
import math
from pathlib import Path
import re
import torch
article=Path(__file__).resolve().parents[1]/'content/posts/ai/diffusion-models/index.md'
blocks=re.findall(r'^```python\n(.*?)^```',article.read_text(),re.M|re.S)
source=next(block for block in blocks if 'def timestep_embedding(' in block)
namespace={}
exec(compile(source,str(article),'exec'),namespace)
embedding=namespace['timestep_embedding']
cases=0
for dim in [1,2,3,4,7,8,63,64,65]:
    for batch in [0,1,3]:
        for period in [100,10_000]:
            t=torch.arange(batch,device='cpu')
            actual=embedding(t,dim,period)
            assert actual.shape==(batch,dim),(dim,batch,actual.shape)
            assert actual.dtype==torch.float32 and actual.device.type=='cpu'
            half=dim//2
            for row in range(batch):
                angles=[row*math.exp(-math.log(period)*i/max(half-1,1)) for i in range(half)]
                expected=[math.cos(a) for a in angles]+[math.sin(a) for a in angles]+([0.] if dim%2 else [])
                assert all(abs(float(x)-y)<2e-6 for x,y in zip(actual[row],expected)),(dim,row)
            cases+=1
print(f'PASS {cases} actual CPU PyTorch embedding cases, including dim=1 and empty batches; no training validation')
