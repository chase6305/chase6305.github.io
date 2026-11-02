#!/usr/bin/env node
// Actual Chrome touch gestures; run serially with every other CDP suite.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {connect}=require('./lib/browser.cjs');
const core=require('../assets/js/pion-spectrum.js');
const base=(process.argv[2]||'http://127.0.0.1:13360').replace(/\/+$/,''),port=Number(process.argv[3]||9236),output=process.argv[4]||'/tmp/pion-touch';
const near=(a,b)=>assert(Number.isFinite(a)&&Math.abs(a-b)<=Math.max(1e-300,Math.abs(b)*6e-6),JSON.stringify({a,b}));
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(port),report={contexts:[],taps:[],gestures:[],generatedTaps:0,inputMethod:'Input.dispatchTouchEvent',exceptions:b.exceptions};
 const save=()=>fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(report,null,2)+'\n');
 async function tap(selector) {
  await b.evaluate(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView({block:'center',behavior:'instant'})`);await b.pause(80);
  const before=await b.evaluate(`(()=>{const r=document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect();return{x:r.x+r.width/2,y:r.y+r.height/2,top:r.top}})()`);
  assert(before.x>0&&before.x<390&&before.y>0&&before.y<1000,JSON.stringify(before));
  await b.cdp('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:before.x,y:before.y,id:1}]});await b.pause(60);
  await b.cdp('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await b.pause(80);report.generatedTaps++;
  const after=await b.evaluate(`document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect().top`);
  assert(Math.abs(before.top-after)<2,'A touch update moved the tapped control');
 }
 async function swipe(point,dx,dy) {
  await b.cdp('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:point.x,y:point.y,id:1}]});
  for(let step=1;step<=10;step++){await b.cdp('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:point.x+dx*step/10,y:point.y+dy*step/10,id:1}]});await b.pause(16);}
  await b.cdp('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await b.pause(180);
 }
 async function resultMatches(expected) {
  const row=await b.evaluate(`(()=>{const e=document.querySelector('[data-pion-spectrum]');return{ready:e.dataset.ready==='true',outputs:[...e.querySelectorAll('output')].map(o=>o.value),raw:[...e.querySelectorAll('input[type="number"]')].map(i=>i.valueAsNumber),epsilon:e.querySelector('[type="checkbox"]').checked,overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1,error:e.querySelector('[data-error]').hidden}})()`);
  assert(row.ready&&row.error&&!row.overflow,JSON.stringify(row));
  ['input_erank','pion_erank','muon_norm','pion_norm'].forEach((key,i)=>near(Number(row.outputs[i]),expected[key]));
  assert.deepEqual(row.raw,expected.raw);return row;
 }
 try {
  await b.cdp('Emulation.setTouchEmulationEnabled',{enabled:true,maxTouchPoints:1});
  for(const width of [320,390])for(const size of [100,200])for(const theme of ['light','dark','warm']) {
   const context={width,size,theme};await b.viewport(width);await b.navigate(base+'/posts/ai/pion-muon-vla-rlvr/');
   await b.evaluate(`document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.colorScheme='${theme==='dark'?'dark':'light'}';document.documentElement.style.fontSize='${size}%';Promise.all([document.fonts.ready,...[...document.querySelectorAll('.content img')].map(i=>{i.loading='eager';return i.decode()})])`);await b.pause(120);
   for(const [name,preset]of Object.entries(core.presets)) {
    await tap(`[data-preset="${name}"]`);const values=await resultMatches(core.calculate(preset.raw,preset.kp,preset.epsilon?1e-7:0));report.taps.push({...context,preset:name,...values});
   }
   await tap('[data-preset="tiny"]');await tap('[data-pion-spectrum] input[type="checkbox"]');const values=await resultMatches(core.calculate(core.presets.tiny.raw,1,0));assert(!values.epsilon);report.taps.push({...context,operation:'epsilon checkbox',...values});
   await b.evaluate(`window.__touchTable=document.querySelector('[data-pion-spectrum] table');__touchTable.scrollIntoView({block:'center',behavior:'instant'});__touchTable.scrollLeft=0`);await b.pause(100);
   const point=await b.evaluate(`(()=>{const r=__touchTable.getBoundingClientRect();return{x:r.x+r.width/2,y:Math.max(100,Math.min(850,r.y+r.height/2)),scrollY,extent:__touchTable.scrollWidth-__touchTable.clientWidth}})()`);
   await swipe(point,-140,0);
   const horizontal=await b.evaluate(`({left:__touchTable.scrollLeft,scrollY,scrollX})`);assert((point.extent>20?horizontal.left>20:horizontal.left<=point.extent+1)&&Math.abs(horizontal.scrollY-point.scrollY)<3&&horizontal.scrollX===0,JSON.stringify({context,point,horizontal}));
   await swipe(point,0,-140);
   const vertical=await b.evaluate('scrollY');assert(vertical>horizontal.scrollY+30,JSON.stringify({context,horizontal,vertical}));report.gestures.push({...context,horizontal,vertical,tableOverflow:point.extent>20,verticalPageScroll:true});
   report.contexts.push(context);save();
  }
  assert.deepEqual(b.exceptions,[]);report.summary={contexts:report.contexts.length,generatedTaps:report.generatedTaps,verifiedTaps:report.taps.length,horizontalAndVerticalGesturePairs:report.gestures.length,exceptions:b.exceptions.length};console.log(JSON.stringify(report.summary));
 }finally{save();await b.cdp('Emulation.setTouchEmulationEnabled',{enabled:false}).catch(()=>{});await b.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
