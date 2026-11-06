#!/usr/bin/env node
// Chrome-emulated touch only; this does not claim physical-device validation.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {connect,pause}=require('../../../scripts/lib/browser.cjs');
const [base='http://127.0.0.1:13247',port='9337',output='/tmp/three-paper-touch']=process.argv.slice(2);
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(Number(port));
 const report={layouts:[],gestures:[],exceptions:b.exceptions};
 async function tap(selector){
  const point=await b.evaluate(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});e.scrollIntoView({block:'center',behavior:'instant'});const r=e.getBoundingClientRect();return{x:r.left+r.width/2,y:r.top+r.height/2}})()`);
  await b.cdp('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[point]});await pause(50);
  await b.cdp('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await pause(100);
 }
 async function drag(x,y,dx,dy){
  await b.cdp('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x,y}]});
  for(let i=1;i<=12;i++){await b.cdp('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:x+dx*i/12,y:y+dy*i/12}]});await pause(20)}
  await b.cdp('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await pause(220);
 }
 try{
  for(const width of [320,390])for(const size of [100,200])for(const theme of ['light','dark','warm'])for(const kind of ['last','omni']){
   const root=kind==='last'?'[data-last-whitening]':'[data-omnicalib-prior]';
   await b.viewport(width,1000);await b.cdp('Emulation.setTouchEmulationEnabled',{enabled:true,maxTouchPoints:1});
   await b.navigate(base+(kind==='last'?'/posts/ai/last-action-tokenizer/':'/posts/calibration/omnicalib/'));
   await b.evaluate(`document.documentElement.classList.remove('dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='${size}%';document.fonts.ready`);
   if(kind==='last'){
    for(const [preset,winner]of [['rotated','B'],['tie','并列'],['default','A']]){
     await tap(root+` [data-preset="${preset}"]`);
     assert.equal(await b.evaluate(`document.querySelector('${root} [data-value="metricWinner"]').textContent`),winner);
    }
   }else{
    for(const [preset,knee]of [['excited','0.2667'],['confident','0.4000'],['aligned','0.2000']]){
     await tap(root+` [data-preset="${preset}"]`);
     assert.equal(await b.evaluate(`document.querySelector('${root} [data-value="knee"]').textContent`),knee);
    }
   }
   const name=kind==='last'?'ratio':'mean';await tap(root+` input[name="${name}"]`);
   assert(await b.evaluate(`document.activeElement.matches('${root} input[name="${name}"]')`));
   await b.cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'a',code:'KeyA',windowsVirtualKeyCode:65,modifiers:2});
   await b.cdp('Input.dispatchKeyEvent',{type:'keyUp',key:'a',code:'KeyA',windowsVirtualKeyCode:65,modifiers:2});
   await b.key('Backspace','Backspace',8);
   assert(await b.evaluate(`document.querySelector('${root} [data-computed]').hidden && document.activeElement.matches('input')`));
   await b.cdp('Input.insertText',{text:kind==='last'?'3':'.4'});
   assert(await b.evaluate(`!document.querySelector('${root} [data-computed]').hidden && document.activeElement.matches('input')`));
   await b.evaluate(`document.activeElement.blur();document.querySelector('${root} [data-computed]').scrollIntoView({block:'start',behavior:'instant'});scrollBy(0,-180)`);await pause(80);
   const before=await b.evaluate('scrollY');await drag(width/2,700,0,-210);const after=await b.evaluate('scrollY');assert(after>before+5,JSON.stringify({kind,width,size,theme,before,after}));
   const svgPoint=await b.evaluate(`(()=>{const e=document.querySelector('${root} svg');e.scrollIntoView({block:'center',behavior:'instant'});const r=e.getBoundingClientRect();return{x:r.left+r.width/2,y:r.top+r.height/2,before:scrollY}})()`);
   await drag(svgPoint.x,svgPoint.y,0,-160);assert(await b.evaluate('scrollY')>svgPoint.before+20,JSON.stringify({kind,width,size,theme,svgPoint}));
   assert(await b.evaluate('document.documentElement.scrollWidth<=document.documentElement.clientWidth+1'));
   report.layouts.push({kind,width,size,theme,presets:3,nativeInput:true,verticalDrag:true,svgVerticalDrag:true});
   if(theme==='light'){
    const table=await b.evaluate(`(()=>{const t=[...document.querySelectorAll('article table')].find(e=>e.scrollWidth>e.clientWidth+2);if(!t)return null;t.dataset.touchTest='true';t.scrollIntoView({block:'center',behavior:'instant'});t.scrollLeft=0;const r=t.getBoundingClientRect();return {x:Math.min(${width}-35,r.right-25),y:Math.max(220,Math.min(740,r.top+r.height/2))}})()`);
    if(table){
     const beforeY=await b.evaluate('scrollY');await drag(table.x,table.y,-180,0);
     const state=await b.evaluate(`({left:document.querySelector('[data-touch-test]').scrollLeft,y:scrollY,x:scrollX})`);assert(state.left>5&&Math.abs(state.y-beforeY)<3&&state.x===0,JSON.stringify({kind,width,size,state,table}));
     report.gestures.push({kind,width,size,tableScroll:state.left,verticalDrift:state.y-beforeY});
    }
   }
  }
  assert(!b.exceptions.length,JSON.stringify(b.exceptions));console.log(JSON.stringify({layouts:report.layouts.length,horizontalTableGestures:report.gestures.length,exceptions:0}));
 }finally{await b.cdp('Emulation.setTouchEmulationEnabled',{enabled:false,maxTouchPoints:1});fs.writeFileSync(path.join(output,'touch.json'),JSON.stringify(report,null,2)+'\n');await b.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
