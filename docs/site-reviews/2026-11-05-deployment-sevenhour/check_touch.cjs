#!/usr/bin/env node
// Browser-emulated touch taps and gestures; no physical-device claim.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {connect,pause}=require('../../../scripts/lib/browser.cjs');
const [base='http://127.0.0.1:13327',port='9337',output='/tmp/deployment-touch']=process.argv.slice(2);
const posts=[
 {kind:'timing',route:'/posts/ai/rma-umi-rtc-deployment/',presets:{default:['delay','4'],latencies:['delay','6'],short:['schedule','不满足'],exhausted:['schedule','不满足']}},
 {kind:'residual',route:'/posts/ai/residual-reinforcement-learning/',presets:{default:['finalError','0.0018'],baseline:['finalError','0.2014'],wrong:['pole','1.0200'],sampled:['pole','-1.4000']}},
 {kind:'contact',route:'/posts/calibration/interactive-contact-calibration/',presets:{default:['meanX','0.0000'],parallel:['stdY','1.4142'],oblique:['meanX','4.3208'],orthogonal:['meanX','6.0000'],broad:['meanX','5.3745']}}
];
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(Number(port));
 const root='[data-deployment-lab]',report={contexts:[],taps:[],gestures:[],exceptions:b.exceptions};
 const save=()=>fs.writeFileSync(path.join(output,'touch.json'),JSON.stringify(report,null,2)+'\n');
 async function tap(selector){
  await b.evaluate(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView({block:'center',behavior:'instant'})`);await pause(90);
  const point=await b.evaluate(`(()=>{const r=document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect();return{x:r.x+r.width/2,y:r.y+r.height/2,top:r.top}})()`);
  assert(point.x>0&&point.x<390&&point.y>128&&point.y<1000,JSON.stringify(point));
  await b.cdp('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:point.x,y:point.y,id:1}]});await pause(40);
  await b.cdp('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await pause(100);
  const after=await b.evaluate(`document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect().top`);
  assert(Math.abs(after-point.top)<2,'A touch update displaced the activated control');
 }
 async function swipe(point,dx,dy){
  await b.cdp('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:point.x,y:point.y,id:1}]});
  for(let i=1;i<=10;i++){
   await b.cdp('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:point.x+dx*i/10,y:point.y+dy*i/10,id:1}]});await pause(16);
  }
  await b.cdp('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await pause(200);
 }
 try{
  await b.cdp('Emulation.setTouchEmulationEnabled',{enabled:true,maxTouchPoints:1});
  for(const post of posts)for(const width of [320,390])for(const size of [100,200])for(const theme of ['light','dark','warm']){
   const context={kind:post.kind,width,size,theme};await b.viewport(width,1000);await b.navigate(base+post.route);
   await b.evaluate(`document.documentElement.classList.remove('dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='${size}%';Promise.all([document.fonts.ready,...[...document.querySelectorAll('article img')].map(i=>{i.loading='eager';return i.decode()})])`);await pause(80);
   for(const [name,[key,value]]of Object.entries(post.presets)){
    await tap(`${root} [data-preset="${name}"]`);
    const observed=await b.evaluate(`document.querySelector('${root} [data-value="${key}"]').textContent`);
    assert.equal(observed,value,JSON.stringify({...context,name,key}));report.taps.push({...context,preset:name,key,observed});
   }
   if(post.kind==='contact'){
    await tap(`${root} [data-preset="default"]`);await tap(`${root} label:has(input[name="second"])`);
    assert(await b.evaluate(`document.querySelector('${root} input[name="second"]').checked`));
    assert.equal(await b.evaluate(`document.querySelector('${root} [data-value="meanX"]').textContent`),'6.0000');
    report.taps.push({...context,operation:'native checkbox',passed:true});
   }
   // A vertical drag beginning on the SVG must scroll the page. Plots do
   // not handle dragging, so they should not trap a reader's touch gesture.
   await tap(`${root} [data-preset="default"]`);
   await b.evaluate(`document.querySelector('${root} svg').scrollIntoView({block:'center',behavior:'instant'})`);await pause(100);
   const chart=await b.evaluate(`(()=>{const r=document.querySelector('${root} svg').getBoundingClientRect();return{x:r.x+r.width/2,y:r.y+r.height/2,scroll:scrollY}})()`);
   await swipe(chart,0,-100);const chartScroll=await b.evaluate('scrollY');
   assert(chartScroll>chart.scroll+25,JSON.stringify({context,chart,chartScroll}));
   // Existing article tables retain horizontal touch panning, while a
   // vertical drag over the same table continues page reading.
   const point=await b.evaluate(`(()=>{const t=[...document.querySelectorAll('article table')].find(t=>t.scrollWidth>t.clientWidth+20);if(!t)return null;window.__touchReviewTable=t;t.scrollLeft=0;t.scrollIntoView({block:'center',behavior:'instant'});const r=t.getBoundingClientRect();return{x:r.x+r.width/2,y:Math.min(850,Math.max(180,r.y+r.height/2)),extent:t.scrollWidth-t.clientWidth,scroll:scrollY}})()`);
   assert(point,'No horizontally readable article table');await pause(100);await swipe(point,-120,0);
   const horizontal=await b.evaluate(`({left:__touchReviewTable.scrollLeft,scroll:scrollY,x:scrollX})`);
   assert(horizontal.left>20&&Math.abs(horizontal.scroll-point.scroll)<3&&horizontal.x===0,JSON.stringify({context,point,horizontal}));
   await swipe(point,0,-100);const tableScroll=await b.evaluate('scrollY');assert(tableScroll>horizontal.scroll+25);
   assert(await b.evaluate('document.documentElement.scrollWidth<=document.documentElement.clientWidth+1'));
   report.gestures.push({...context,chartVerticalDelta:chartScroll-chart.scroll,tableHorizontal:horizontal.left,tableVerticalDelta:tableScroll-horizontal.scroll});
   report.contexts.push(context);save();
  }
  assert(!b.exceptions.length,JSON.stringify(b.exceptions));console.log(JSON.stringify({contexts:report.contexts.length,verifiedTaps:report.taps.length,gestureTriples:report.gestures.length,exceptions:0}));
 }finally{save();await b.cdp('Emulation.setTouchEmulationEnabled',{enabled:false}).catch(()=>{});await b.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
