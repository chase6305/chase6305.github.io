#!/usr/bin/env node
// Exact binomial enumeration in Python, independent of the JS closed form.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {spawnSync}=require('node:child_process');
const root=path.resolve(__dirname,'..'),lab=require('../assets/js/pear-sampling.js');
const cases=[];
for(const probability of [0,.4,2,10,50,90,99.9,100])for(const group of [2,3,8,16,32,64])for(const groups of [1,128,1024])cases.push({probability,group,groups});
let seed=261118562;
const random=n=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed%n;};
for(let i=0;i<200;i++)cases.push({probability:random(1001)/10,group:2+random(63),groups:1+random(1024)});
const oracle=String.raw`
import json,math,sys
from fractions import Fraction
rows=[]
for c in json.load(sys.stdin):
    p=Fraction(str(c['probability']))/100; g=c['group']; b=c['groups']
    probabilities=[math.comb(g,k)*p**k*(1-p)**(g-k) for k in range(g+1)]
    mixed=sum(probabilities[1:-1])
    result=dict(all_fail=float(probabilities[0]),all_success=float(probabilities[-1]),mixed=float(mixed),
        candidate_groups=float(b/mixed) if mixed else None,
        candidate_rollouts=float(b*g/mixed) if mixed else None,
        retained_success=float(sum(Fraction(k,g)*probabilities[k] for k in range(1,g))/mixed) if mixed else None,
        accepted_rollouts=b*g)
    rows.append(result)
json.dump(rows,sys.stdout,allow_nan=False)
`;
const child=spawnSync('python3',['-B','-c',oracle],{cwd:root,input:JSON.stringify(cases),encoding:'utf8',timeout:30000,
  env:{...process.env,OMP_NUM_THREADS:'1',OPENBLAS_NUM_THREADS:'1',MKL_NUM_THREADS:'1',NUMEXPR_NUM_THREADS:'1'}});
assert.equal(child.status,0,child.stderr);
const expected=JSON.parse(child.stdout);
cases.forEach((input,i)=>{
 const actual=lab.calculate(input);
 for(const[key,value]of Object.entries(expected[i])){
  if(value===null)assert.equal(actual[key],null);
  else assert(Math.abs(actual[key]-value)<=1e-12+Math.abs(value)*2e-11,JSON.stringify({input,key,actual:actual[key],expected:value}));
 }
});
const invalid=[{probability:-1},{probability:101},{probability:null},{probability:true},{probability:NaN},
 {probability:Infinity},{group:1},{group:65},{group:8.5},{group:true},{groups:0},{groups:1025},{groups:1.1}];
for(const changes of invalid)assert.throws(()=>lab.calculate({...lab.defaults,...changes}),RangeError);
for(const p of [1e-200,1e-12,99.9999999999]){
 const result=lab.calculate({...lab.defaults,probability:p});assert(result.mixed>0);
 assert(result.candidate_rollouts>0);assert(Number.isFinite(result.retained_success));
}
assert.equal(lab.formatResult(lab.calculate({...lab.defaults,probability:0}),'mixed'),'0%');
assert.equal(lab.formatResult(lab.calculate({...lab.defaults,probability:100}),'candidate_rollouts'),'无混合组');
for(const p of [.4,2,50,99.9])assert(Math.abs(lab.calculate({probability:p,group:2,groups:1}).retained_success-.5)<1e-13);
const fixture=JSON.parse(fs.readFileSync(path.join(root,'content/posts/ai/pear/assets/sampling-default.json'),'utf8'));
assert.deepEqual(fixture.input,lab.defaults);
const computed=lab.calculate(fixture.input);
for(const[key,value]of Object.entries(fixture.result))assert(Math.abs(computed[key]-value)<=1e-12+Math.abs(value)*1e-12);
for(const[key,value]of Object.entries(fixture.display))assert.equal(lab.formatResult(computed,key),value);
console.log(JSON.stringify({cases:cases.length,invalidCases:invalid.length,oracle:'exact independent Python binomial enumeration',staticDefault:'matched'}));
