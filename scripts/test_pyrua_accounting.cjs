#!/usr/bin/env node
// Independent direct-prompt replay oracle for the article's browser arithmetic.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const root = path.resolve(__dirname, '..');
const lab = require('../assets/js/pyrua-accounting.js');
let state = 26101939;
const random = limit => {state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state % limit;};
const cases = [{...lab.defaults}];
for (const operations of [1,2,3,11,12,13,100]) for (const group of [1,3,7,100]) for (const retained of [0,1,20,99,100]) {
  cases.push({operations,group,retained,reply:137});
}
for (let i=0;i<400;i++) cases.push({operations:1+random(100),group:1+random(100),reply:random(100001),retained:random(101)});
const oracle = String.raw`
import json,sys
from fractions import Fraction
from math import ceil
def replay(prefix, events):
    prompt=prefix
    total=0
    for event in events:
        total += prompt
        prompt += event
    return total
rows=[]
for c in json.load(sys.stdin):
    left=c['operations']; batches=[]
    while left:
        count=min(c['group'],left); batches.append(count); left-=count
    raw=[count*c['reply'] for count in batches]
    selected=[ceil(Fraction(value*c['retained'],100)) for value in raw]
    rows.append(dict(primitive_events=c['operations'],tool_calls=c['operations'],code_calls=len(batches),
        tool_input=replay(2000,[c['reply']]*c['operations']),grouped_input=replay(2400,raw),
        selected_input=replay(2400,selected)))
json.dump(rows,sys.stdout)
`;
const child = spawnSync('python3',['-B','-c',oracle],{cwd:root,input:JSON.stringify(cases),encoding:'utf8',timeout:30000,
  env:{...process.env,OMP_NUM_THREADS:'1',OPENBLAS_NUM_THREADS:'1',MKL_NUM_THREADS:'1',NUMEXPR_NUM_THREADS:'1'}});
assert.equal(child.status,0,child.stderr);
const expected = JSON.parse(child.stdout);
cases.forEach((values,i)=>assert.deepEqual(lab.calculate(values),expected[i],JSON.stringify(values)));
for (const changes of [{operations:0},{operations:101},{operations:true},{group:1.1},{group:0},
  {reply:-1},{reply:Infinity},{reply:NaN},{retained:-1},{retained:101},{retained:null}]) {
  assert.throws(()=>lab.calculate({...lab.defaults,...changes}),RangeError);
}
const reference = JSON.parse(fs.readFileSync(path.join(root,'content/posts/ai/pyrua-lean/assets/accounting-default.json'),'utf8'));
assert.deepEqual(reference.input,lab.defaults);
assert.deepEqual(reference.result,lab.calculate(reference.input));
assert.equal(lab.calculate({...lab.defaults,operations:1}).selected_input,2400);
assert.equal(lab.calculate({...lab.defaults,operations:1}).tool_input,2000);
console.log(JSON.stringify({cases:cases.length,invalidCases:11,oracle:'independent Python rational feedback rounding and direct prompt replay',staticDefault:'matched'}));
