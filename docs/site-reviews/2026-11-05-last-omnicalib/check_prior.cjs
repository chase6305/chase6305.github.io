#!/usr/bin/env node
// Prior experiment: arithmetic UI, native controls, errors, fallbacks and contrast.
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const {connect,pause}=require('../../../scripts/lib/browser.cjs');
const {contrastAudit}=require('../../../scripts/lib/contrast.cjs');
const base=process.argv[2]||'http://127.0.0.1:13227';
const port=Number(process.argv[3]||9337);
const output=process.argv[4]||'/tmp/omnicalib-prior-review';
const axePath=process.argv[5];
const route='/posts/calibration/omnicalib/';
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(port);const report={layouts:[],fallbacks:[],semantics:[],exceptions:b.exceptions};
 const values=()=>b.evaluate(`Object.fromEntries([...document.querySelectorAll('[data-omnicalib-prior] output')].map(e=>[e.dataset.value,Number(e.textContent.replace('−','-'))]))`);
 const set=async(name,value)=>{await b.evaluate(`(()=>{const f=document.querySelector('[data-omnicalib-prior] input[name="${name}"]');f.focus();f.value=${JSON.stringify(value)};f.dispatchEvent(new Event('input',{bubbles:true}))})()`);};
 const preset=async(name)=>{await b.evaluate(`document.querySelector('[data-omnicalib-prior] [data-preset="${name}"]').focus()`);await b.key('Enter','Enter',13);};
 try{
  for(const width of [320,1440])for(const size of [100,200])for(const theme of ['light','dark','warm']){
   await b.viewport(width,1000);await b.navigate(base+route);
   await b.evaluate(`document.documentElement.classList.remove('dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='${size}%';document.fonts.ready`);await pause(100);
   const defaults=await values();assert.equal(defaults.knee,.4);assert.equal(defaults.ankle,-.3);assert.equal(defaults.kneeError,.2);assert.equal(defaults.kneeStd,.02);assert.equal(defaults.ankleStd,.0283);
   await preset('aligned');assert.equal((await values()).kneeError,0);
   await preset('biased');const biased=await values();await preset('confident');const confident=await values();
   assert.equal(biased.knee,confident.knee);assert.equal(biased.kneeError,confident.kneeError);assert(confident.kneeStd<biased.kneeStd);
   for(const [name,value]of [['std',''],['std','0'],['std','.3001'],['mean','.5001']]){
    await set(name,value);
    const invalid=await b.evaluate(`(()=>{const r=document.querySelector('[data-omnicalib-prior]');return {focused:document.activeElement.matches('input'),error:!r.querySelector('[data-error]').hidden,computed:r.querySelector('[data-computed]').hidden,empty:[...r.querySelectorAll('output')].every(e=>!e.textContent),charts:[...r.querySelectorAll('svg')].every(e=>!e.children.length),named:!!r.querySelector('input[aria-invalid="true"][aria-errormessage]')}})()`);
    assert(Object.values(invalid).every(Boolean),JSON.stringify(invalid));
    await set('std','.02');await set('mean','.4');
   }
   await set('std','.3');await set('mean','-.5');assert.equal((await values()).ankle,.6);
   await preset('aligned');
   // Use a real keyboard insertion into the native number input.
   await b.evaluate(`document.querySelector('[data-omnicalib-prior] input[name="mean"]').focus()`);
   await b.cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'a',code:'KeyA',windowsVirtualKeyCode:65,modifiers:2});
   await b.cdp('Input.dispatchKeyEvent',{type:'keyUp',key:'a',code:'KeyA',windowsVirtualKeyCode:65,modifiers:2});
   await b.cdp('Input.insertText',{text:'.25'});assert.equal((await values()).knee,.25);
   assert(await b.evaluate(`document.activeElement.matches('input[name="mean"]')`));
   await b.key('Tab','Tab',9);assert(await b.evaluate(`document.activeElement.matches('input[name="std"]')`));
   const state=await b.evaluate(`(()=>{const r=document.querySelector('[data-omnicalib-prior]');return {overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1,labels:[...r.querySelectorAll('input')].every(i=>i.labels.length>0),clipped:[...r.querySelectorAll('label,dt,output')].filter(e=>e.scrollWidth>e.clientWidth+2).map(e=>e.textContent)}})()`);
   const contrast=await b.evaluate('('+contrastAudit.toString()+')('+JSON.stringify('[data-omnicalib-prior]')+')');
   assert(!state.overflow&&state.labels&&!state.clipped.length&&!contrast.issues.length,JSON.stringify({state,contrast}));
   report.layouts.push({width,size,theme,...state,contrastIssues:contrast.issues});
   if(width===320&&size===200&&theme==='dark'){
    await b.evaluate(`document.querySelector('[data-omnicalib-prior]').scrollIntoView({block:'start'})`);await b.screenshot(path.join(output,'prior-mobile-dark-large.png'));
   }
  }
  if(axePath){
   await b.viewport(1440,1000);await b.navigate(base+route);await b.evaluate(fs.readFileSync(axePath,'utf8'));
   for(const invalid of [false,true]){
    if(invalid)await set('std','');
    const result=await b.evaluate(`axe.run(document.querySelector('[data-omnicalib-prior]'),{rules:{'color-contrast':{enabled:false}}}).then(r=>({violations:r.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),incomplete:r.incomplete.map(v=>v.id)}))`);
    report.semantics.push({invalid,...result});assert(!result.violations.length,JSON.stringify(result));
   }
  }
  for(const mode of ['scripts-off','blocked-script','setup-failure']){
   let injected;
   if(mode==='scripts-off')await b.cdp('Emulation.setScriptExecutionDisabled',{value:true});
   if(mode==='blocked-script')await b.cdp('Network.setBlockedURLs',{urls:['*omnicalib-prior*.js*']});
   if(mode==='setup-failure')injected=await b.cdp('Page.addScriptToEvaluateOnNewDocument',{source:`(()=>{const original=SVGSVGElement.prototype.replaceChildren;SVGSVGElement.prototype.replaceChildren=function(...args){if(this.matches('[data-chart="knee"],[data-chart="ankle"]'))throw Error('controlled plot failure');return original.apply(this,args)}})()`});
   await b.viewport(320,1000);await b.navigate(base+route);
   const fallback=await b.evaluate(`(()=>{const r=document.querySelector('[data-omnicalib-prior]');return{staticVisible:!r.querySelector('[data-static-result]').hidden,interactiveHidden:r.querySelector('[data-interactive]').hidden,text:r.querySelector('[data-static-result]').textContent,overflow:document.documentElement.scrollWidth>innerWidth+1}})()`);
   assert(fallback.staticVisible&&fallback.interactiveHidden&&!fallback.overflow&&fallback.text.includes('0.0283'),JSON.stringify({mode,fallback}));
   report.fallbacks.push({mode,...fallback});
   await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});await b.cdp('Network.setBlockedURLs',{urls:[]});
   if(injected)await b.cdp('Page.removeScriptToEvaluateOnNewDocument',{identifier:injected.identifier});
  }
  assert(!b.exceptions.length,JSON.stringify(b.exceptions));console.log(JSON.stringify({layouts:report.layouts.length,fallbacks:report.fallbacks.length,semantics:report.semantics.length,exceptions:0}));
 }finally{fs.writeFileSync(path.join(output,'prior.json'),JSON.stringify(report,null,2)+'\n');await b.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
