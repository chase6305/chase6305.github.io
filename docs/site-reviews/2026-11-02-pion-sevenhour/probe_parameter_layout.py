"""Pinned VLA routing and 4D-layout audit; CPU allocation and communication fixture.

No model, GPU, distributed backend or training data is loaded. Two one-parameter
optimizer calls use the unchanged source class, with CUDA buffer allocations
redirected to CPU and all_gather replaced by a synchronous one-rank copy.
"""
import argparse,ast,hashlib,importlib.util,json,math
from pathlib import Path
from types import SimpleNamespace
import torch

VLA_SHA='8e7ca8d6f40ebc4fb066acb15387e618d515896cca5d1eb0ad234e5310e9157e'
ROUTING_SHA='0cb8c1453e8d9d60690d0cbc3bef99e48430f0f27bfafbdb5542737229d6c74b'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--optimizer',type=Path,required=True);parser.add_argument('--routing',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 if digest(args.optimizer)!=VLA_SHA or digest(args.routing)!=ROUTING_SHA:raise ValueError('Pinned source mismatch')
 tree=ast.parse(args.routing.read_text());functions=[]
 for name in('force_adamw_aux','get_modality'):
  matches=[node for node in ast.walk(tree)if isinstance(node,ast.FunctionDef)and node.name==name]
  if len(matches)!=1:raise ValueError('Ambiguous source function: '+name)
  functions.append(matches[0])
 scope={'torch':torch};exec(compile(ast.Module(body=functions,type_ignores=[]),str(args.routing),'exec'),scope)
 routes=[]
 for name,ndim in [('vla.vision_backbone.patch_embed.proj.weight',4),('vla.vision_backbone.linear.weight',2),('vla.language_model.embed_tokens.weight',2),('action_head.fc2.weight',2),('proprio_projector.weight',2),('vla.vision_backbone.linear.bias',1)]:
  aux=scope['force_adamw_aux'](name,SimpleNamespace(ndim=ndim));mod=scope['get_modality'](name)
  routes.append({'name':name,'ndim':ndim,'force_adamw_aux':aux,'modality':mod})
 assert not routes[0]['force_adamw_aux'] and routes[0]['modality']=='V'
 assert [r['force_adamw_aux']for r in routes]==[False,False,True,True,False,True]
 spec=importlib.util.spec_from_file_location('pinned_layout_optimizer',args.optimizer);official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
 class Communication:
  def __init__(self):self.calls=0;self.waits=0
  def all_gather_into_tensor(self,out,value,async_op):
   assert async_op and out.numel()==value.numel();out.copy_(value.reshape_as(out));self.calls+=1
   parent=self
   class Handle:
    def wait(self):parent.waits+=1;return True
   return Handle()
 communication=Communication();official.dist=communication;old_empty=torch.empty;allocations=[]
 def cpu_empty(*shape,**kwargs):
  if kwargs.get('device')=='cuda':kwargs={**kwargs,'device':'cpu'};allocations.append('CUDA buffer redirected to CPU')
  return old_empty(*shape,**kwargs)
 rows=[];torch.set_num_threads(1)
 for shape in ((2,3,5,7),(2,105)):
  weight=torch.nn.Parameter(torch.ones(shape,dtype=torch.float64));weight.grad=torch.zeros_like(weight);weight.grad.reshape(-1)[0]=1
  torch.empty=cpu_empty
  try:optimizer=official.DefaultPion([weight],lr=.01,weight_decay=0.,momentum=0.,nesterov=False,promotion_steps=0,ns_steps=5,scale_factor=.2,rank=0,world_size=1)
  finally:torch.empty=old_empty
  optimizer.step();delta=1-weight.detach();actual=float(delta.reshape(-1)[0]);last_two=max(shape[-2:]);expected=.01*.2*math.sqrt(last_two)
  assert math.isclose(actual,expected,rel_tol=1e-13,abs_tol=1e-15)
  assert torch.count_nonzero(delta)==1
  logical=(shape[0],math.prod(shape[1:]));transformed=official.high_pass_ns(weight.grad.reshape(logical),0,5)
  assert float(transformed.reshape(-1)[0])==1 and torch.count_nonzero(transformed)==1
  rows.append({'parameter_shape':shape,'spectral_matrix_shape':logical,'source_scale_dimension':last_two,'source_scale_factor':.2*math.sqrt(last_two),'first_coordinate_decrement':actual,'expected_decrement':expected,'flattened_matrix_scale_dimension':max(logical),'other_coordinates_changed':0})
 assert math.isclose(rows[1]['first_coordinate_decrement']/rows[0]['first_coordinate_decrement'],math.sqrt(15),rel_tol=2e-13)
 assert communication.calls==communication.waits==2
 assert digest(args.optimizer)==VLA_SHA and digest(args.routing)==ROUTING_SHA
 result={'scope':__doc__,'torch':torch.__version__,'device':'cpu','source_sha256':{'optimizer':VLA_SHA,'routing':ROUTING_SHA},'source_modified':False,'optimizer_steps':2,'buffer_allocation_adaptations':allocations,'communication_calls':communication.calls,'routes':routes,'cases':rows,'interpretation':'The2Dscaleformula is correct for2Dparameters. The fixed VLA class flattens4Dgradients but reads its update scale from the original parameter last two dimensions. This is a source behavior audit, not a recommendation to alter that scale and not distributed or hardware validation.'}
 args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'routes':len(routes),'optimizer_steps':2,'cases':len(rows),'device':'cpu','source_modified':False}))

if __name__=='__main__':main()
