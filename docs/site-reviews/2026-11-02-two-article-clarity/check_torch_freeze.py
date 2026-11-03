from pathlib import Path
from datetime import datetime,timezone
import contextlib,io,json,re,hashlib,math
import torch
w=Path(__file__).parent;p=Path('content/posts/ai/supermimic/index.md');source=p.read_text();code=next(c for c in re.findall(r'```python\n(.*?)```',source,re.S)if'import torch'in c);namespace={};out=io.StringIO()
with contextlib.redirect_stdout(out):exec(compile(code,str(p), 'exec'),namespace)
assert out.getvalue()=='actor gradient=8.0\nmodel gradient=None\n';assert namespace['gain'].grad is None;assert math.isclose(namespace['theta'].grad.item(),8.,abs_tol=1e-12)
def run(*,frozen=True,detach=False,no_grad=False,evaluation=False):
 model=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
 with torch.no_grad():model.weight.fill_(2.)
 model.requires_grad_(not frozen)
 if evaluation:model.eval()
 theta=torch.tensor([.1],dtype=torch.float64,requires_grad=True);state=torch.zeros(1,dtype=torch.float64);total=torch.zeros(1,dtype=torch.float64)
 with torch.no_grad()if no_grad else contextlib.nullcontext():
  for index in range(2):
   if detach and index:state=state.detach()
   state=state+model(theta);total=total-(state-1.).square()
 if total.requires_grad:total.backward()
 return {'frozen':frozen,'detach':detach,'no_grad':no_grad,'evaluation':evaluation,'return':total.item(),'actor_gradient':theta.grad.item()if theta.grad is not None else None,'model_gradient':model.weight.grad.item()if model.weight.grad is not None else None,'output_requires_grad':total.requires_grad}
rows=[run(),run(frozen=False,evaluation=True),run(detach=True),run(no_grad=True)];assert math.isclose(rows[0]['actor_gradient'],8.);assert rows[0]['model_gradient']is None;assert math.isclose(rows[1]['actor_gradient'],8.)and math.isclose(rows[1]['model_gradient'],.4);assert math.isclose(rows[2]['actor_gradient'],5.6);assert rows[3]['actor_gradient']is None and not rows[3]['output_requires_grad'];assert all(math.isclose(r['return'],-1.)for r in rows)
result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'torch':torch.__version__,'device':'cpu','source_article_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'published_snippet_sha256':hashlib.sha256(code.encode()).hexdigest(),'published_output':out.getvalue(),'cases':rows,'optimizer_steps':0,'scope':'Published scalar snippet and independent Linear-module variations; no learned world model, G1 asset, GPU, robot or training reproduction.'};(w/'torch-freeze.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
