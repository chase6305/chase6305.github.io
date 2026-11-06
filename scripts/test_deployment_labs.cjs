#!/usr/bin/env node
// Independent Python references for the three article-scoped experiments.
const assert=require('node:assert/strict');
const path=require('node:path');
const {execFileSync}=require('node:child_process');
const models=require('../assets/js/deployment-labs.js');
const cases=[];
for(const h of [8,16,32,64]) for(const s of [1,3,4,5,8,16,32,64].filter(s=>s<=h))
  for(const period of [10,20,33.3]) for(const obs of [0,40]) for(const work of [0,.1,60,80,159.9,160,160.1,240])
    for(const execution of [0,20]) cases.push({kind:'timing',args:[h,s,period,obs,work,execution]});
for(const gain of [.1,1,3]) for(const feedback of [-1,0,.99,1,1.2,4])
  for(const correction of [-.2,0,.2]) for(const bias of [0,.2]) for(const period of [.01,.1,.8,1])
    cases.push({kind:'residual',args:[gain,feedback,correction,bias,period]});
for(const sigma of [1,2,4,8]) for(const angle of [0,.001,1,15,30,45,89.999,90]) for(const second of [false,true])
  cases.push({kind:'posterior',args:[sigma,angle,second]});
const oracle=`import importlib.util,json,math,sys
from decimal import Decimal,localcontext
from pathlib import Path
root=Path(sys.argv[1])
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,root/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
timing=load('timing','content/posts/ai/rma-umi-rtc-deployment/deployment_contracts.py')
contact=load('contact','content/posts/calibration/interactive-contact-calibration/contact_calibration_sanity.py')
result=[]
for case in json.load(sys.stdin):
 kind=case['kind'];args=case['args']
 if kind=='timing':
  model=timing.timing_budget(*args)
  # Independently evaluate the printed Eq. 5 with high-precision Decimal.
  if model['schedulable']:
   with localcontext() as context:
    context.prec=60
    h,s,*_=args;d=model['delay'];end=h-s;one=Decimal(1)
    profile=[]
    for i in range(h):
     c=Decimal(end-i)/Decimal(end-d+1)
     profile.append(1. if i<d else float(c*(c.exp()-1)/(one.exp()-1)) if i<end else 0.)
    model['weights']=profile
  result.append(model);continue
 if kind=='posterior':
  result.append(contact.posterior_example(*args));continue
 k,alpha,a,b,dt=args;p=1+dt*(alpha-k);q=1-dt*k
 # High-precision closed forms avoid cancellation between two large terms
 # at an unstable fixed point (e.g. p=1.2, forcing=-0.2, e0=1).
 with localcontext() as context:
  context.prec=80
  kd,ad,cd,bd,td=map(lambda v:Decimal(str(v)),args)
  pd=1+td*(ad-kd);qd=1-td*kd
  power=lambda coefficient,n:Decimal(1) if n==0 else coefficient**n
  series=lambda coefficient,forcing:[float(power(coefficient,n)+td*forcing*sum((power(coefficient,j) for j in range(n)),Decimal(0))) for n in range(61)]
  actual=series(pd,cd+bd);baseline=series(qd,bd)
 stable=abs(p)<1
 result.append(dict(pole=p,stable=stable,forcing=a+b,finalError=actual[-1],duration=60*dt,
                    fixedPoint=(a+b)/(k-alpha) if stable else None,actual=actual,baseline=baseline,steps=60))
print(json.dumps(result))`;
const expected=JSON.parse(execFileSync('python3',['-B','-c',oracle,path.resolve(__dirname,'..')],
  {input:JSON.stringify(cases),encoding:'utf8',maxBuffer:32*1024*1024}));
let numbers=0,maxNormalizedError=0,maxErrorCase=null;
function compare(actual,expected,label) {
  if(typeof expected==='number') {
    assert(Number.isFinite(actual),label);
    const error=Math.abs(actual-expected)/Math.max(1,Math.abs(expected));
    if(error>maxNormalizedError) {maxNormalizedError=error;maxErrorCase={label,actual,expected}}
    numbers++;
    assert(error<2e-11,JSON.stringify({label,actual,expected,error}));return;
  }
  if(Array.isArray(expected)) {
    assert.equal(actual.length,expected.length,label);expected.forEach((v,i)=>compare(actual[i],v,label+'['+i+']'));return;
  }
  if(expected!==null&&typeof expected==='object') {
    assert.deepEqual(Object.keys(actual).sort(),Object.keys(expected).sort(),label);
    Object.entries(expected).forEach(([k,v])=>compare(actual[k],v,label+'.'+k));return;
  }
  assert.equal(actual,expected,label);
}
cases.forEach((c,i)=>compare(models[c.kind](...c.args),expected[i],c.kind+JSON.stringify(c.args)));
const invalid={timing:[[7,5,20,0,80,0],[16,17,20,0,80,0],[16,5,20,0,80.01,0],[16,5,0,0,80,0],[16,5,20,NaN,80,0]],
  residual:[[0,0,0,0,.1],[1,5,0,0,.1],[1,0,2,0,.1],[1,0,0,Infinity,.1],[1,0,0,0,0]],
  posterior:[[0,90,false],[2,91,true],[2,90,1],[NaN,0,false]]};
for(const [kind,items] of Object.entries(invalid)) items.forEach(args=>assert.throws(()=>models[kind](...args),RangeError));
assert.equal(models.timing(16,4,20,40,60,20).schedulable,true);
assert.equal(models.timing(16,3,20,0,80,0).schedulable,false);
const halfWeight=models.timing(8,3,20,0,40,0).weights[3];
assert(Math.abs(halfWeight-.5/(Math.sqrt(Math.E)+1))<1e-15,'RTC Eq. 5 includes the linear c_i factor');
assert(Math.abs(halfWeight-1/(Math.sqrt(Math.E)+1))>.18,'The exponential-only regression must be distinguishable');
assert.equal(models.residual(1,1.2,0,0,.1).stable,false);
assert.equal(models.residual(3,0,0,0,.8).stable,false);
assert(Math.abs(models.posterior(2,90,true).meanX-6)<1e-10);
assert(models.posterior(2,0,true).stdX>11);
console.log(JSON.stringify({cases:cases.length,byKind:Object.fromEntries(Object.keys(models).map(k=>[k,cases.filter(c=>c.kind===k).length])),
  numbers,invalidCases:Object.values(invalid).flat().length,maximumNormalizedError:maxNormalizedError,maxErrorCase}));
