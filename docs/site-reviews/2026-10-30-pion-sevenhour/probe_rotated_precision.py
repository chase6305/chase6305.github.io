"""Constructed rotated spectra: inspect finite precision, not model quality."""
import argparse,hashlib,importlib.util,json,pathlib,sys,time,math,platform
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--official-file',type=pathlib.Path,required=True)
parser.add_argument('--output',type=pathlib.Path,required=True)
parser.add_argument('--bundle',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[3]/'content/posts/ai/pion-muon-vla-rlvr')
args=parser.parse_args()
source=args.official_file
try:
 digest=hashlib.sha256(source.read_bytes()).hexdigest()
except OSError as error:
 parser.error(str(error))
if digest!='8e7ca8d6f40ebc4fb066acb15387e618d515896cca5d1eb0ad234e5310e9157e':
 parser.error('Official VLA source does not match the pinned Pion commit')
import numpy as np
import torch
spec=importlib.util.spec_from_file_location('official_pion',source);official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
sys.path.insert(0,str(args.bundle))
from pion_matrix import transform
from pion_spectral import pion
def svdvals(value):return np.linalg.svd(value,compute_uv=False)
np.seterr(all='raise',under='ignore');torch.set_num_threads(1)
def frob(value):return math.hypot(*value.flat)
rng=np.random.default_rng(402);rows=[];start=time.monotonic()
for shape in [(4,8),(8,4),(16,32),(32,16),(64,128),(128,64),(128,128)]:
 rank=min(shape)
 for seed in range(12):
  u,_=np.linalg.qr(rng.normal(size=(shape[0],rank)));v,_=np.linalg.qr(rng.normal(size=(shape[1],rank)))
  for profile in ['rank1','rank2','decay','flat']:
   singular=np.zeros(rank)
   if profile=='rank1':singular[0]=1
   elif profile=='rank2':singular[:2]=[1,.25]
   elif profile=='decay':singular=np.geomspace(1,1e-4,rank)
   else:singular[:]=1
   source_matrix=(u*singular)@v.T
   for kp in [0,1,2]:
    ideal=(u*np.array([pion(float(x),kp)for x in singular/np.linalg.norm(singular)]))@v.T
    numerical=transform(source_matrix,kp)
    with torch.no_grad():observed=official.high_pass_ns(torch.from_numpy(source_matrix.copy()),kp,5-kp).double().numpy()
    output_norm=frob(observed);expected_norm=frob(ideal)
    row={'shape':shape,'seed':seed,'profile':profile,'kp':kp,'expected_norm':expected_norm,'bf16_norm':output_norm,'f64_max_error':float(np.max(np.abs(numerical-ideal))),'bf16_max_error':float(np.max(np.abs(observed-ideal))),'bf16_max_singular':float(svdvals(observed)[0])}
    if profile=='rank1':
     direction=np.outer(u[:,0],v[:,0]);parallel=float(np.sum(observed*direction));residual=observed-parallel*direction
     row['rank1_subspace_residual_norm']=frob(residual)
     row['rank1_cosine']=parallel/output_norm if output_norm else None
    rows.append(row)
 print(json.dumps({'completed_shape':shape,'cases':len(rows),'elapsed_seconds':time.monotonic()-start}),flush=True)
out={'scope':'Rotated constructed spectra at ordinary scale; no training or statistical noise model','seed':402,'python':platform.python_version(),'architecture':platform.machine(),'official_source_sha256':digest,'bundle_source_sha256':{name:hashlib.sha256((args.bundle/name).read_bytes()).hexdigest()for name in ['pion_matrix.py','pion_spectral.py']},'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','rows':rows,'elapsed_seconds':time.monotonic()-start}
args.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'cases':len(rows),'max_f64_error':max(r['f64_max_error']for r in rows),'max_bf16_error':max(r['bf16_max_error']for r in rows),'max_bf16_singular':max(r['bf16_max_singular']for r in rows),'max_rank1_residual':max(r.get('rank1_subspace_residual_norm',0)for r in rows),'elapsed_seconds':out['elapsed_seconds']}),flush=True)
