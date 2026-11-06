#!/usr/bin/env node
const assert = require('node:assert/strict');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const {calculate} = require('../assets/js/last-whitening.js');
const cases = [];
for (const ratio of [1,1.001,1.5,2,3,5,10]) for (const angle of [0,.001,15,30,45,60,89.999,90]) for (const epsilon of [0,1e-8,.1,1,3,10]) cases.push({ratio,angle,epsilon});
const source = path.resolve(__dirname,'../content/posts/ai/last-action-tokenizer/geometry_sanity.py');
const python = `import importlib.util,json,sys
spec=importlib.util.spec_from_file_location('oracle',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
print(json.dumps([m.whitening_choice(x['ratio'],x['angle'],x['epsilon']) for x in json.load(sys.stdin)]))`;
const expected = JSON.parse(execFileSync('python3',['-B','-c',python,source],{input:JSON.stringify(cases),encoding:'utf8',maxBuffer:8*1024*1024}));
let maxError = 0;
for (let i=0;i<cases.length;i++) {
  const actual = calculate(cases[i].ratio,cases[i].angle,cases[i].epsilon);
  for (const key of ['euclideanA','euclideanB','metricA','metricB','condition']) {
    const error = Math.abs(actual[key]-expected[i][key]);maxError=Math.max(error,maxError);
    assert(error<2e-12,JSON.stringify({case:cases[i],key,actual,expected:expected[i]}));
  }
  for (const key of ['euclideanWinner','metricWinner']) assert.equal(actual[key],expected[i][key],JSON.stringify(cases[i]));
}
const invalid = [[NaN,0,0],[Infinity,0,0],[.999,0,0],[10.001,0,0],[3,NaN,0],[3,-.1,0],[3,90.01,0],[3,0,NaN],[3,0,-.01],[3,0,10.01]];
for (const args of invalid) assert.throws(()=>calculate(...args),RangeError);
assert.equal(calculate().metricWinner,'A');
assert.equal(calculate(1,0,0).metricWinner,'B');
assert.equal(calculate(3,90,0).metricWinner,'B');
assert.equal(calculate(2,0,0).metricWinner,'tie');
assert.equal(calculate(3,0,3).metricWinner,'B');
console.log(JSON.stringify({cases:cases.length,numbers:cases.length*5,selections:cases.length*2,invalidCases:invalid.length,maximumAbsoluteError:maxError}));
