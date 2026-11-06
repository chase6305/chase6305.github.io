#!/usr/bin/env node
// Compare the interactive calculation with the independent Python geometry example.
const assert = require('node:assert/strict');
const {spawnSync} = require('node:child_process');
const path = require('node:path');
const {calculate, roundEven} = require('../assets/js/ultralytics-geometry.js');
const root=path.resolve(__dirname,'..');
const cases=[];
for(const [width,height] of [[1280,720],[720,1280],[1,3000],[3000,1],[8192,8191],[641,639],[1280,721],[1280,719],[1,1]]){
 for(const target of [32,64,640,2048]) for(const minimal of [false,true]) cases.push({width,height,target,minimal});
}
let state=386941;
for(let i=0;i<300;i++){
 const next=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state};
 cases.push({width:next()%8192+1,height:next()%8192+1,target:(next()%64+1)*32,minimal:!!((next()>>>20)&1)});
}
const oracle=spawnSync(process.env.PYTHON || 'python3',['-B','-c',`
import json,sys
sys.path.insert(0,'content/posts/ai/ultralytics')
from detection_geometry import Box,letterbox
result=[]
for values in json.load(sys.stdin):
    w,h,t=values['width'],values['height'],values['target']
    p=letterbox(w,h,t,t,minimal=values['minimal'])
    original=Box(w*.25,h*.25,w*.75,h*.75)
    network=p.forward(original);restored=p.inverse(network)
    result.append(dict(gain=p.gain,resizedWidth=p.resized_width,resizedHeight=p.resized_height,
                       outputWidth=p.output_size[0],outputHeight=p.output_size[1],
                       left=p.left,top=p.top,right=p.right,bottom=p.bottom,
                       original=list(vars(original).values()),network=list(vars(network).values()),restored=list(vars(restored).values()),
                       paddingPercent=100*(1-p.resized_width*p.resized_height/(p.output_size[0]*p.output_size[1]))))
print(json.dumps(result,allow_nan=False))
`],{cwd:root,input:JSON.stringify(cases),encoding:'utf8',maxBuffer:4*1024*1024,timeout:30000});
assert.equal(oracle.status,0,oracle.stderr);
const expected=JSON.parse(oracle.stdout);
function close(actual,expected,message){assert(Math.abs(actual-expected)<=1e-10*Math.max(1,Math.abs(expected)),message+': '+actual+' != '+expected)}
cases.forEach((values,i)=>{
 const actual=calculate(values);
 for(const key of Object.keys(expected[i])){
  if(Array.isArray(expected[i][key]))actual[key].forEach((v,j)=>close(v,expected[i][key][j],`${i}/${key}/${j}`));
  else close(actual[key],expected[i][key],`${i}/${key}`);
 }
 assert(actual.outputWidth%32===0&&actual.outputHeight%32===0);
 assert(actual.paddingPercent>=0&&actual.paddingPercent<100);
});
assert.equal(roundEven(2.5),2);assert.equal(roundEven(3.5),4);
assert.equal(roundEven(-2.5),-2);assert.equal(roundEven(-3.5),-4);
const valid={width:1280,height:720,target:640,minimal:false};
for(const bad of [NaN,Infinity,-1,0,.5,8193,true,'1280',null])assert.throws(()=>calculate({...valid,width:bad}));
for(const bad of [NaN,Infinity,0,31,33,2049,true,'640'])assert.throws(()=>calculate({...valid,target:bad}));
for(const bad of [null,0,1,'false'])assert.throws(()=>calculate({...valid,minimal:bad}));
console.log(JSON.stringify({cases:cases.length,checks:'independent Python coordinates, rounding, dimensions, area and invalid inputs passed'},null,2));
