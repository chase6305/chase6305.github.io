#!/usr/bin/env node
// Browser-emulated touch, not a physical-device validation.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {connect}=require('../../../scripts/lib/browser.cjs');
const lab=require('../../../assets/js/pear-sampling.js');
const base=process.argv[2].replace(/\/$/,''),port=Number(process.argv[3]),out=process.argv[4];
fs.mkdirSync(out,{recursive:true});
(async()=>{
 const b=await connect(port),report={contexts:[],taps:[],gestures:[]};
 const save=()=>fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');
 async function tap(selector){
  await b.evaluate(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView({block:'center',behavior:'instant'})`);await b.pause(80);
  const p=await b.evaluate(`(()=>{const r=document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect();return{x:r.x+r.width/2,y:r.y+r.height/2,top:r.top}})()`);
  assert(p.x>0&&p.x<390&&p.y>128&&p.y<1000,JSON.stringify(p));
  await b.cdp('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:p.x,y:p.y,id:1}]});await b.pause(60);
  await b.cdp('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await b.pause(80);
  const after=await b.evaluate(`document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect().top`);
  assert(Math.abs(after-p.top)<2,'Touch activation displaced the tapped control');
 }
 async function swipe(p,dx,dy){
  await b.cdp('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:p.x,y:p.y,id:1}]});
  for(let i=1;i<=10;i++){
   await b.cdp('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:p.x+dx*i/10,y:p.y+dy*i/10,id:1}]});await b.pause(16);
  }
  await b.cdp('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await b.pause(180);
 }
 try{
  await b.cdp('Emulation.setTouchEmulationEnabled',{enabled:true,maxTouchPoints:1});
  for(const scripts of [true,false])for(const width of [320,390])for(const size of [100,200])for(const theme of ['light','dark','warm']){
   const context={scripts,width,size,theme};await b.viewport(width,1000);
   await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.navigate(base+'/posts/ai/pear/');
   await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});
   await b.evaluate(`document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='${size}%';Promise.all([document.fonts.ready,...[...document.querySelectorAll('.content img')].map(i=>{i.loading='eager';return i.decode()})])`);
   await b.pause(120);
   if(scripts)for(const[preset,probability]of [['rare',.4],['balanced',50],['default',2]]){
    await tap('[data-preset="'+preset+'"]');
    const actual=await b.evaluate(`Object.fromEntries([...document.querySelectorAll('[data-pear-sampling] output')].map(e=>[e.dataset.value,e.textContent]))`);
    const result=lab.calculate({...lab.defaults,probability});for(const[key,value]of Object.entries(actual))assert.equal(value,lab.formatResult(result,key));
    report.taps.push({...context,preset,passed:true});
   }
   await b.evaluate(`window.__touchTable=[...document.querySelectorAll('.content table')].find(t=>t.textContent.includes('当前观测需要说明什么'));__touchTable.scrollIntoView({block:'center',behavior:'instant'});__touchTable.scrollLeft=0`);await b.pause(100);
   const p=await b.evaluate(`(()=>{const r=__touchTable.getBoundingClientRect();return{x:r.x+r.width/2,y:Math.max(200,Math.min(820,r.y+r.height/2)),scrollY,extent:__touchTable.scrollWidth-__touchTable.clientWidth}})()`);
   await swipe(p,-140,0);
   const horizontal=await b.evaluate(`({left:__touchTable.scrollLeft,scrollY,scrollX})`);
   assert((p.extent>20?horizontal.left>20:horizontal.left<=p.extent+1)&&Math.abs(horizontal.scrollY-p.scrollY)<3&&horizontal.scrollX===0,JSON.stringify({context,p,horizontal}));
   await swipe(p,0,-140);
   const vertical=await b.evaluate('scrollY');assert(vertical>horizontal.scrollY+30,JSON.stringify({context,horizontal,vertical}));
   report.gestures.push({...context,horizontal,vertical,passed:true});report.contexts.push(context);save();
  }
  assert.deepEqual(b.exceptions,[]);report.exceptions=b.exceptions;
  console.log(JSON.stringify({contexts:report.contexts.length,taps:report.taps.length,gesturePairs:report.gestures.length,exceptions:b.exceptions.length}));
 }finally{save();await b.cdp('Emulation.setTouchEmulationEnabled',{enabled:false}).catch(()=>{});await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
