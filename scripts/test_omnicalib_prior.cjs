#!/usr/bin/env node
const assert = require('node:assert/strict');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const {calculate} = require('../assets/js/omnicalib-prior.js');
const cases = [];
for (const mean of [-.5,-.3,-.1,0,.1,.2,.3,.4,.5]) for (const std of [.005,.01,.02,.05,.1,.2,.3]) for (const excitation of [0,1e-7,.001,.01,.1,.3,.7,1]) cases.push({mean,std,excitation});
const source = path.resolve(__dirname,'../content/posts/calibration/omnicalib/observability_sanity.py');
const python = `import importlib.util,json,math,sys
spec=importlib.util.spec_from_file_location('oracle',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=[]
for item in json.load(sys.stdin):
 k,a,cov=m.excited_joint_fit(item['mean'],item['std'],item['excitation'])
 out.append(dict(knee=k,ankle=a,kneeError=k-.2,ankleError=a+.1,kneeStd=math.sqrt(cov[0][0]),ankleStd=math.sqrt(cov[1][1])))
print(json.dumps(out))`;
const expected = JSON.parse(execFileSync('python3',['-B','-c',python,source],{input:JSON.stringify(cases),encoding:'utf8'}));
for (let i=0;i<cases.length;i++) {
  const actual = calculate(cases[i].mean,cases[i].std,cases[i].excitation);
  for (const key of Object.keys(expected[i])) assert(Math.abs(actual[key]-expected[i][key])<2e-14,JSON.stringify({case:cases[i],key,actual,expected:expected[i]}));
}
for (const args of [[NaN,.02],[Infinity,.02],[-.50001,.02],[.50001,.02],[.2,0],[.2,-.01],[.2,NaN],[.2,.00499],[.2,.30001],[.2,.02,-.001],[.2,.02,1.001],[.2,.02,NaN]]) assert.throws(()=>calculate(...args),RangeError);
const weak = calculate(.4,.3), strong = calculate(.4,.005);
assert(Math.abs(weak.knee-strong.knee)<1e-14);assert(Math.abs(weak.kneeError-strong.kneeError)<1e-14);
assert(strong.kneeStd<weak.kneeStd&&strong.ankleStd<weak.ankleStd);
assert.equal(calculate(.4,.02).dataRank,1);
assert.equal(calculate(.4,.02,.0001).dataRank,2);
assert(calculate(.4,.02,.0001).dataCondition>1e7);
assert(Math.abs(calculate(.4,.02,1).knee-4/15)<1e-14);
console.log(JSON.stringify({cases:cases.length,valuesCompared:cases.length*6,invalidCases:12,priorStrengthInvariant:'passed',independentExcitation:'passed'}));
