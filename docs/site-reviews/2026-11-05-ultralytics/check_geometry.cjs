#!/usr/bin/env node
// Article-scoped checks; use a private CDP endpoint and run browser suites serially.
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const {connect,pause}=require(path.resolve(__dirname,'../../../scripts/lib/browser.cjs'));
const {contrastAudit}=require(path.resolve(__dirname,'../../../scripts/lib/contrast.cjs'));
const base=process.argv[2]||'http://127.0.0.1:13217';
const port=Number(process.argv[3]||9337);
const output=process.argv[4]||'/tmp/ultralytics-geometry-browser';
const axe=process.argv[5];
const route=base+'/posts/ai/ultralytics/';
fs.mkdirSync(output,{recursive:true});
(async()=>{
 const b=await connect(port),report=[];
 const state=()=>b.evaluate(`(()=>{const r=document.querySelector('[data-ultralytics-geometry]');return {ready:!r.querySelector('[data-interactive]').hidden,error:!r.querySelector('[data-error]').hidden,errorText:r.querySelector('[data-error]').textContent,computed:!r.querySelector('[data-computed]').hidden,values:Object.fromEntries([...r.querySelectorAll('output')].map(e=>[e.dataset.value,e.value])),overflow:document.documentElement.scrollWidth>innerWidth+1,focus:document.activeElement.name||document.activeElement.dataset.preset,clipped:[...r.querySelectorAll('dt,dd,label')].filter(e=>e.scrollWidth>e.clientWidth+2).map(e=>e.textContent)}})()`);
 const load=async(width,theme,scale=100,forced=false)=>{
  await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});
  await b.cdp('Network.setBlockedURLs',{urls:[]});
  await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[{name:'forced-colors',value:forced?'active':'none'},{name:'prefers-color-scheme',value:theme}]});
  await b.viewport(width,1000);await b.navigate(route);
  await b.evaluate(`document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.colorScheme='${theme==='dark'?'dark':'light'}';document.documentElement.style.fontSize='${scale}%'`);
  await pause(100);
 };
 try{
  for(const width of [320,390,768,1440])for(const theme of ['light','dark','warm'])for(const scale of [100,200]){
   await load(width,theme,scale);let current=await state();
   assert(current.ready&&!current.overflow&&!current.clipped.length,JSON.stringify({width,theme,scale,current}));
   const colors=await b.evaluate('('+contrastAudit.toString()+')('+JSON.stringify('[data-ultralytics-geometry]')+')');assert.equal(colors.issues.length,0,JSON.stringify(colors));
   assert.equal(current.values.size,'640 × 640');assert.equal(current.values.network,'160, 230, 480, 410');
   await b.evaluate(`document.querySelector('[data-preset="minimal"]').focus();document.activeElement.click()`);
   current=await state();assert.equal(current.values.size,'640 × 384');assert.equal(current.values.network,'160, 102, 480, 282');assert.equal(current.focus,'minimal');
   // Invalid edits clear stale numeric outputs without moving the input's focus.
   await b.evaluate(`(()=>{const f=document.querySelector('[data-ultralytics-geometry] input[name="width"]');f.focus();f.value='';f.dispatchEvent(new Event('input',{bubbles:true}));})()`);
   current=await state();assert(current.error&&!current.computed&&current.focus==='width');assert(await b.evaluate(`document.querySelector('[data-ultralytics-geometry] input[name=width]').getAttribute('aria-invalid')==='true'`));assert(Object.values(current.values).every(v=>v===''));
   await b.evaluate(`(()=>{const f=document.querySelector('[data-ultralytics-geometry] input[name="width"]');f.value='720';f.dispatchEvent(new Event('input',{bubbles:true}));})()`);
   current=await state();assert(!current.error&&current.computed&&current.focus==='width');
   await b.evaluate(`(()=>{const r=document.querySelector('[data-ultralytics-geometry]');r.style.letterSpacing='.12em';r.style.wordSpacing='.16em';r.style.lineHeight='1.5';r.scrollIntoView({block:'start'});})()`);
   assert(!(await state()).overflow);await b.screenshot(path.join(output,`${width}-${theme}-${scale}.png`));
   report.push({width,theme,scale,status:'passed'});
  }
  // The controls keep their native tab order and checkbox Space handling.
  await load(390,'light');await b.evaluate(`document.querySelector('[data-preset=square]').focus()`);
  const order=['minimal','width','height','target','minimal'];
  for(const expected of order){await b.key('Tab','Tab',9);assert.equal((await state()).focus,expected)}
  await b.key(' ','Space',32);assert.equal((await state()).values.size,'640 × 384');
  // Actual native arrow-key editing respects step=32 and keeps focus.
  await load(390,'light');await b.evaluate(`document.querySelector('[data-ultralytics-geometry] input[name="target"]').focus()`);
  await b.key('ArrowUp','ArrowUp',38);await pause(100);
  assert.equal(await b.evaluate(`document.querySelector('[data-ultralytics-geometry] input[name="target"]').value`),'672');assert.equal((await state()).focus,'target');
  // SVG failure and a blocked script retain the native default prose.
  for(const mode of ['blocked','scripts-off','unsupported-svg']){
   await load(320,'light');
   if(mode==='blocked')await b.cdp('Network.setBlockedURLs',{urls:['*ultralytics-geometry*.js']});
   if(mode==='scripts-off')await b.cdp('Emulation.setScriptExecutionDisabled',{value:true});
   let init;
   if(mode==='unsupported-svg')init=await b.cdp('Page.addScriptToEvaluateOnNewDocument',{source:"SVGElement.prototype.replaceChildren=undefined"});
   await b.navigate(route);await pause(100);
   assert(await b.evaluate(`(()=>{const r=document.querySelector('[data-ultralytics-geometry]');return !r.querySelector('[data-static-result]').hidden&&r.querySelector('[data-interactive]').hidden&&document.documentElement.scrollWidth<=innerWidth+1})()`),mode);
   if(init)await b.cdp('Page.removeScriptToEvaluateOnNewDocument',{identifier:init.identifier});
   report.push({fallback:mode,status:'passed'});
  }
  for(const theme of ['light','dark']){
   await load(390,theme,200,true);assert((await state()).ready&&!(await state()).overflow);
   await b.evaluate(`document.querySelector('[data-ultralytics-geometry] input').focus()`);
   assert(await b.evaluate(`getComputedStyle(document.activeElement).outlineStyle!=='none'`));
   await b.screenshot(path.join(output,`forced-${theme}.png`));
   report.push({theme,forcedColors:'passed'});
  }
  if(axe){
   await load(1440,'light');await b.evaluate(fs.readFileSync(axe,'utf8'));
   const audit=await b.evaluate(`axe.run(document.querySelector('[data-ultralytics-geometry]'),{rules:{'color-contrast':{enabled:false}}})`);
   fs.writeFileSync(path.join(output,'axe.json'),JSON.stringify(audit,null,2));assert.equal(audit.violations.length,0,JSON.stringify(audit.violations));
   await b.evaluate(`(()=>{const f=document.querySelector('[data-ultralytics-geometry] input[name=width]');f.value='';f.dispatchEvent(new Event('input',{bubbles:true}));})()`);
   const invalid=await b.evaluate(`axe.run(document.querySelector('[data-ultralytics-geometry]'),{rules:{'color-contrast':{enabled:false}}})`);assert.equal(invalid.violations.length,0,JSON.stringify(invalid.violations));fs.writeFileSync(path.join(output,'axe-invalid.json'),JSON.stringify(invalid,null,2));
  }
  assert.equal(b.exceptions.length,0,JSON.stringify(b.exceptions));
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify({cases:report,exceptions:b.exceptions},null,2));
  console.log(`Passed ${report.length} geometry cases, native editing, fallback, focus and exception checks.`);
 }finally{await b.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
