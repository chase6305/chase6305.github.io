// Compare browser arithmetic with the independent Python article reference.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const lab = require('../assets/js/muon-batch.js');
const root = path.resolve(__dirname, '..');
const python = process.env.PYTHON || 'python3';
const rows = JSON.parse(execFileSync(python, ['-B', '-c', `
import importlib.util,json,random
from pathlib import Path
p=Path('content/posts/ai/muon-learning-rate-batch-size/batch_math.py')
spec=importlib.util.spec_from_file_location('batch_math',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=random.Random(20261103);rows=[]
cases=[(256,1024,4096,.001,.1,.95),(4096,16384,256,.001,.1,.95),(1024,256,4096,.001,.1,.95),
       (1,100000000,1000000000000,.00000001,1000,.9999999),(100000000,1,0,1,0,0),
       (256,256,4096,.001,.1,.95),(256,1024,0,.001,.1,.95)]
for _ in range(200):
 lr=10**r.uniform(-8,0)
 cases.append((r.randint(1,100000000),r.randint(1,100000000),10**r.uniform(-8,12),lr,r.random()*min(1000,.99/lr),r.uniform(0,.9999999)))
for a,b,c,lr,wd,beta in cases:
 rows.append({'input':dict(zip(['batchOld','batchNew','criticalBatch','lrOld','decayOld','betaOld'],[a,b,c,lr,wd,beta])),
              'result':m.migration_example(a,b,c,lr,wd,beta)})
print(json.dumps(rows,allow_nan=False))
`], {cwd: root, encoding: 'utf8'}));
for (const {input, result} of rows) {
  const actual = lab.calculate(input);
  assert.deepEqual(Object.keys(actual).sort(), Object.keys(result).sort());
  for (const [key, expected] of Object.entries(result)) {
    assert(Number.isFinite(actual[key]) && actual[key] >= 0);
    assert(Math.abs(actual[key]-expected) <= Math.max(1e-300, Math.abs(expected)*2e-12), JSON.stringify({input,key,actual:actual[key],expected}));
  }
}
const saved = JSON.parse(fs.readFileSync(path.join(root, 'content/posts/ai/muon-learning-rate-batch-size/assets/batch-interactive-default.json')));
for (const [key, expected] of Object.entries(saved.result)) {
  assert(Math.abs(lab.calculate(saved.input)[key]-expected) <= Math.max(1e-14, Math.abs(expected)*1e-12));
}
for (const [name, [min, max]] of Object.entries(lab.limits)) {
  for (const value of [NaN, Infinity, -Infinity, min-1, max+1, '', null, undefined]) {
    assert.throws(() => lab.calculate({...lab.presets.fourfold, [name]: value}), RangeError);
  }
}
for (const name of ['batchOld','batchNew']) assert.throws(() => lab.calculate({...lab.presets.fourfold,[name]:1.5}),RangeError);
assert.throws(() => lab.calculate({...lab.presets.fourfold,lrOld:1,decayOld:1}),RangeError);
const zeroNoise = lab.calculate({...lab.presets.fourfold,criticalBatch:0});
assert.equal(zeroNoise.lr_ratio,1);
const sameBatch = lab.calculate({...lab.presets.fourfold,batchNew:256});
assert.equal(sameBatch.lr_ratio,1);
assert.equal(sameBatch.beta_keep_data,0.95);
assert(Math.abs(sameBatch.decay_keep_data-0.1)<1e-14);
const changed = lab.calculate(lab.presets.fourfold);
assert(Math.abs(changed.beta_keep_data**25-0.95**100)<1e-14);
assert(Math.abs((1-changed.lr_new*changed.decay_keep_data)**25-(1-0.001*0.1)**100)<1e-13);
assert(Math.abs(changed.lr_new*changed.decay_keep_step-0.001*0.1)<1e-16);
assert(Object.values(lab.presets).every(p=>lab.calculate(p).lr_new>0));
assert(Number(lab.format(0.9999999994))<1);
console.log(JSON.stringify({pythonOracleCases:rows.length,staticReference:'matched',checks:'domains, noiseless limit, identity, old-state retention and cumulative shrinkage passed'}));
