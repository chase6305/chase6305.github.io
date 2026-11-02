#!/usr/bin/env node
// Independent identities and Python-oracle comparisons; no browser or npm deps.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {execFileSync}=require('node:child_process');
const core=require('../assets/js/pion-spectrum.js');
const bundle=path.join(__dirname,'../content/posts/ai/pion-muon-vla-rlvr');
function near(actual,expected,relative=2e-10,absolute=Number.MIN_VALUE) {
  assert(Number.isFinite(actual)&&Number.isFinite(expected));
  assert(Math.abs(actual-expected)<=Math.max(absolute,Math.abs(expected)*relative),JSON.stringify({actual,expected}));
}
let seed=20261030;
function random(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;}
let checks=0;
for(let index=0;index<300;index++) {
  const raw=Array.from({length:4},()=>random()+.01),kp=index%6;
  const original=core.calculate(raw,kp,0);
  for(const scale of [1e-200,1e200]) {
    const changed=core.calculate(raw.map(v=>v*scale),kp,0);
    for(const series of ['normalized','muon','pion'])for(let i=0;i<4;i++)near(changed[series][i],original[series][i]);
    near(changed.input_erank,original.input_erank);checks++;
  }
  const permuted=core.calculate([raw[2],raw[0],raw[3],raw[1]],kp);
  for(const series of ['normalized','muon','pion'])[2,0,3,1].forEach((source,i)=>near(permuted[series][i],original[series][source]));
  near(permuted.pion_erank,original.pion_erank);checks++;
}
const flat=core.calculate([1,1,1,1],0);
near(flat.pion_erank,4);assert(flat.pion_norm<1e-30);
const tiny=core.presets.tiny;
assert(core.calculate(tiny.raw,tiny.kp,1e-7).pion_norm<1e-50);
assert(core.calculate(tiny.raw,tiny.kp,0).pion_norm>.9);
assert.deepEqual(core.calculate([0,0,0,0]).pion,[0,0,0,0]);
assert(core.calculate([0,0,0,0]).all_zero);
for(const value of core.normalize([1e308,1e308,1e308,1e308],1e308))near(value,1/3);
near(core.effectiveRank([Number.MIN_VALUE,Number.MIN_VALUE]),2);
const invalid=[null,undefined,[],Array(4),[NaN],[Infinity],[-1],[true],['1'],{},new Float64Array([1,2])];
for(const values of invalid)assert.throws(()=>core.calculate(values));
for(const kp of [-1,6,1.5,true,'1',NaN])assert.throws(()=>core.calculate([1,1,1,1],kp));
for(const epsilon of [-1,true,'0',Infinity,NaN])assert.throws(()=>core.calculate([1,1,1,1],1,epsilon));
assert.throws(()=>core.response(.5,1,'yes'));
const cases=[];
for(let index=0;index<120;index++)cases.push({raw:Array.from({length:4},()=>random()*10**((random()-.5)*600)),kp:index%6,epsilon:[0,1e-7,1e308][index%3]});
cases.push(...Object.values(core.presets).map(p=>({raw:p.raw,kp:p.kp,epsilon:p.epsilon?1e-7:0})));
const code='import sys,json\nsys.path.insert(0,'+JSON.stringify(bundle)+')\nfrom pion_spectral import evaluate_spectrum\nprint(json.dumps([evaluate_spectrum(c["raw"],c["kp"],c["epsilon"]) for c in json.load(sys.stdin)],allow_nan=False))';
const python=JSON.parse(execFileSync(process.env.PYTHON||'python3',['-B','-c',code],{input:JSON.stringify(cases),encoding:'utf8'}));
for(const [index,expected]of python.entries()) {
  const c=cases[index],actual=core.calculate(c.raw,c.kp,c.epsilon);
  for(const series of ['normalized','muon','pion'])for(let i=0;i<4;i++)near(actual[series][i],expected[series][i]);
  for(const name of ['input_erank','muon_erank','pion_erank','muon_norm','pion_norm'])near(actual[name],expected[name]);
}
const reference=JSON.parse(fs.readFileSync(path.join(bundle,'assets/spectral-reference.json'))).interactive_default;
const actual=core.calculate(core.presets.concentrated.raw,1,0);
for(const name of ['input_erank','pion_erank','muon_norm','pion_norm'])near(actual[name],reference[name]);
console.log(JSON.stringify({metamorphicCases:checks,pythonOracles:python.length,invalidCases:invalid.length+12,flatAmplitudeAndRank:true,zeroAndEpsilon:true}));
