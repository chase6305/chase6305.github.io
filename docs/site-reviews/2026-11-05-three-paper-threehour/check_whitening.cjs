#!/usr/bin/env node
// The calculator's arithmetic has a separate Python oracle; this checks its UI.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {connect,pause}=require('../../../scripts/lib/browser.cjs');
const {contrastAudit}=require('../../../scripts/lib/contrast.cjs');
const [base='http://127.0.0.1:13247',port='9337',output='/tmp/last-whitening-review',axePath]=process.argv.slice(2);
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(Number(port));
 const report={layouts:[],fallbacks:[],semantics:[],exceptions:b.exceptions};
 const root='[data-last-whitening]';
 const values=()=>b.evaluate(`Object.fromEntries([...document.querySelectorAll('${root} output')].map(e=>[e.dataset.value,e.textContent]))`);
 const preset=async(name)=>{await b.evaluate(`document.querySelector('${root} [data-preset="${name}"]').focus()`);await b.key('Enter','Enter',13)};
 const set=async(name,value)=>b.evaluate(`(()=>{const f=document.querySelector('${root} input[name="${name}"]');f.focus();f.value=${JSON.stringify(value)};f.dispatchEvent(new Event('input',{bubbles:true}))})()`);
 try{
  for(const width of [320,768,1440])for(const size of [100,200])for(const theme of ['light','dark','warm']){
   await b.viewport(width,1000);await b.navigate(base+'/posts/ai/last-action-tokenizer/');
   await b.evaluate(`document.documentElement.classList.remove('dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='${size}%';document.fonts.ready`);await pause(100);
   let result=await values();assert.equal(result.metricWinner,'A');assert.equal(result.euclideanWinner,'B');assert.equal(result.metricA,'0.6667');
   for(const [name,winner]of [['isotropic','B'],['rotated','B'],['tie','并列'],['regularized','B'],['default','A']]){await preset(name);assert.equal((await values()).metricWinner,winner)}
   for(const [name,value]of [['ratio',''],['ratio','.9'],['ratio','10.001'],['angle','-1'],['angle','90.1'],['epsilon','-1'],['epsilon','10.01']]){
    await set(name,value);
    const invalid=await b.evaluate(`(()=>{const r=document.querySelector('${root}');return{focused:document.activeElement.matches('input'),error:!r.querySelector('[data-error]').hidden,computed:r.querySelector('[data-computed]').hidden,empty:[...r.querySelectorAll('output')].every(e=>!e.textContent),charts:[...r.querySelectorAll('svg')].every(e=>!e.children.length),named:!!r.querySelector('input[aria-invalid="true"][aria-errormessage]')}})()`);
    assert(Object.values(invalid).every(Boolean),JSON.stringify(invalid));await preset('default');
   }
   await b.evaluate(`document.querySelector('${root} input[name="ratio"]').focus()`);
   await b.cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'a',code:'KeyA',windowsVirtualKeyCode:65,modifiers:2});
   await b.cdp('Input.dispatchKeyEvent',{type:'keyUp',key:'a',code:'KeyA',windowsVirtualKeyCode:65,modifiers:2});
   await b.cdp('Input.insertText',{text:'2'});assert.equal((await values()).metricWinner,'并列');
   await b.key('Tab','Tab',9);assert(await b.evaluate(`document.activeElement.matches('input[name="angle"]')`));
   await b.key('Tab','Tab',9);assert(await b.evaluate(`document.activeElement.matches('input[name="epsilon"]')`));
   const state=await b.evaluate(`(()=>{const r=document.querySelector('${root}');return{overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1,labels:[...r.querySelectorAll('input')].every(i=>i.labels.length>0),clipped:[...r.querySelectorAll('label,dt,output')].filter(e=>e.scrollWidth>e.clientWidth+2).map(e=>e.textContent),charts:[...r.querySelectorAll('svg')].every(e=>e.children.length===3)}})()`);
   const contrast=await b.evaluate('('+contrastAudit.toString()+')('+JSON.stringify(root)+')');
   assert(!state.overflow&&state.labels&&!state.clipped.length&&state.charts&&!contrast.issues.length,JSON.stringify({width,size,theme,state,contrast}));
   report.layouts.push({width,size,theme,...state,contrastIssues:contrast.issues});
   if(width===320&&size===200&&theme==='dark'){await b.evaluate(`document.querySelector('${root}').scrollIntoView({block:'start'})`);await b.screenshot(path.join(output,'whitening-mobile-dark-large.png'))}
  }
  if(axePath){
   await b.viewport(1440,1000);await b.navigate(base+'/posts/ai/last-action-tokenizer/');await b.evaluate(fs.readFileSync(axePath,'utf8'));
   for(const invalid of [false,true]){
    if(invalid)await set('ratio','');
    const result=await b.evaluate(`axe.run(document.querySelector('${root}'),{rules:{'color-contrast':{enabled:false}}}).then(r=>({violations:r.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),incomplete:r.incomplete.map(v=>v.id)}))`);
    report.semantics.push({invalid,...result});assert(!result.violations.length,JSON.stringify(result));
   }
  }
  for(const mode of ['scripts-off','blocked-script','setup-failure']){
   let injected;
   if(mode==='scripts-off')await b.cdp('Emulation.setScriptExecutionDisabled',{value:true});
   if(mode==='blocked-script')await b.cdp('Network.setBlockedURLs',{urls:['*last-whitening*.js*']});
   if(mode==='setup-failure')injected=await b.cdp('Page.addScriptToEvaluateOnNewDocument',{source:`(()=>{const original=SVGSVGElement.prototype.replaceChildren;SVGSVGElement.prototype.replaceChildren=function(...args){if(this.matches('[data-chart="A"],[data-chart="B"]'))throw Error('controlled plot failure');return original.apply(this,args)}})()`});
   await b.viewport(320,1000);await b.navigate(base+'/posts/ai/last-action-tokenizer/');
   const fallback=await b.evaluate(`(()=>{const r=document.querySelector('${root}');return{staticVisible:!r.querySelector('[data-static-result]').hidden,interactiveHidden:r.querySelector('[data-interactive]').hidden,text:r.querySelector('[data-static-result]').textContent,overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1}})()`);
   assert(fallback.staticVisible&&fallback.interactiveHidden&&!fallback.overflow&&fallback.text.includes('0.6667'),JSON.stringify({mode,fallback}));report.fallbacks.push({mode,...fallback});
   await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});await b.cdp('Network.setBlockedURLs',{urls:[]});
   if(injected)await b.cdp('Page.removeScriptToEvaluateOnNewDocument',{identifier:injected.identifier});
  }
  assert(!b.exceptions.length,JSON.stringify(b.exceptions));
  console.log(JSON.stringify({layouts:report.layouts.length,fallbacks:report.fallbacks.length,semantics:report.semantics.length,exceptions:0}));
 }finally{fs.writeFileSync(path.join(output,'whitening.json'),JSON.stringify(report,null,2)+'\n');await b.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
