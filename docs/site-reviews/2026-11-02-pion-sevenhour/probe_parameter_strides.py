"""Pinned VLA four-dimensional view compatibility; adapted CPU calls only.

CUDA buffer allocations are redirected to CPU. Communication is a synchronous
one-rank copy. Eight one-parameter step attempts cover four constructed gradient
layouts and both Nesterov settings; no model, GPU or backend is loaded.
"""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import torch
PINNED='8e7ca8d6f40ebc4fb066acb15387e618d515896cca5d1eb0ad234e5310e9157e'
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--official-file',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
 def digest():return hashlib.sha256(a.official_file.read_bytes()).hexdigest()
 if digest()!=PINNED:raise ValueError('Pinned source mismatch')
 spec=importlib.util.spec_from_file_location('pinned_stride_optimizer',a.official_file);source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
 class Communication:
  def __init__(self):self.calls=0
  def all_gather_into_tensor(self,out,value,async_op):
   assert async_op;out.copy_(value.reshape_as(out));self.calls+=1
   class Handle:
    def wait(self):return True
   return Handle()
 comm=Communication();source.dist=comm;original_empty=torch.empty;allocations=[]
 def cpu_empty(*shape,**kwargs):
  if kwargs.get('device')=='cuda':kwargs={**kwargs,'device':'cpu'};allocations.append('CUDA allocation redirected to CPU')
  return original_empty(*shape,**kwargs)
 def make(layout,fill):
  if layout=='contiguous':return torch.full((2,3,5,7),fill,dtype=torch.float64)
  if layout=='channels_last':return torch.full((2,3,5,7),fill,dtype=torch.float64).to(memory_format=torch.channels_last)
  if layout=='spatial_transpose':return torch.full((2,3,7,5),fill,dtype=torch.float64).transpose(-1,-2)
  if layout=='regular_stride':return torch.full((2,3,5,14),fill,dtype=torch.float64)[...,::2]
  raise ValueError(layout)
 rows=[];torch.set_num_threads(1)
 for nesterov in(False,True):
  for layout in('contiguous','channels_last','spatial_transpose','regular_stride'):
   value=make(layout,1.);weight=torch.nn.Parameter(value);gradient=make(layout,0.);gradient[0,0,0,0]=1.;weight.grad=gradient
   before=weight.detach().clone();torch.empty=cpu_empty
   try:optimizer=source.DefaultPion([weight],lr=.01,weight_decay=0.,momentum=.5,nesterov=nesterov,promotion_steps=0,ns_steps=5,scale_factor=.2,rank=0,world_size=1)
   finally:torch.empty=original_empty
   calls=comm.calls;error=None
   try:optimizer.step()
   except RuntimeError as e:error=str(e)
   expected_rejection=layout in('channels_last','spatial_transpose')
   assert (error is not None)==expected_rejection,(layout,error)
   if error:assert 'view size is not compatible' in error and torch.equal(weight.detach(),before)
   else:assert comm.calls==calls+1 and float(weight.detach()[0,0,0,0])<1
   rows.append({'layout':layout,'nesterov':nesterov,'shape':list(weight.shape),'weight_stride':list(weight.stride()),'gradient_stride':list(gradient.stride()),'buffer_stride':list(optimizer.state[weight]['momentum_buffer'].stride()),'contiguous':gradient.is_contiguous(),'step_completed':error is None,'expected_rejection':expected_rejection,'error':error,'communication_calls':comm.calls-calls})
 assert digest()==PINNED
 result={'scope':__doc__,'torch':torch.__version__,'device':'cpu','official_file_sha256':PINNED,'official_source_modified':False,'step_attempts':len(rows),'completed_steps':sum(r['step_completed']for r in rows),'expected_view_rejections':sum(r['expected_rejection']for r in rows),'communication_calls':comm.calls,'buffer_allocation_adaptations':len(allocations),'cases':rows,'interpretation':'Shape alone does not establish view compatibility. The fixed4Dcaller uses view rather than reshape. Some noncontiguous views work; channels-last and spatially transposed examples here fail. No claim is made about every layout or a production training backend.'}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in('step_attempts','completed_steps','expected_view_rejections','official_source_modified')}))
if __name__=='__main__':main()
