#!/usr/bin/env node
// Arithmetic has independent Python/Decimal references; this covers browser UI.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {connect,pause}=require('../../../scripts/lib/browser.cjs');
const {contrastAudit}=require('../../../scripts/lib/contrast.cjs');
const [base='http://127.0.0.1:13327',port='9337',output='/tmp/deployment-lab-review',axePath]=process.argv.slice(2);
const posts=[
 {kind:'timing',route:'/posts/ai/rma-umi-rtc-deployment/',field:'inference',keyboard:'120',next:'execution',
  presets:{default:{delay:'4',schedule:'满足',regions:'4／7／5'},latencies:{delay:'6',workDelay:'3',regions:'6／4／6'},short:{schedule:'不满足'},exhausted:{schedule:'不满足'}},
  invalid:[['horizon','7'],['stride','0'],['period','20.01'],['observation','-1'],['inference','']]},
 {kind:'residual',route:'/posts/ai/residual-reinforcement-learning/',field:'correction',keyboard:'0',next:'bias',
  presets:{default:{finalError:'0.0018'},baseline:{finalError:'0.2014'},wrong:{pole:'1.0200',stable:'不满足 |p|＜1'},sampled:{pole:'-1.4000',stable:'不满足 |p|＜1'}},
  invalid:[['gain','0'],['feedback','5'],['correction','2'],['bias',''],['period','0']]},
 {kind:'contact',route:'/posts/calibration/interactive-contact-calibration/',field:'sigma',keyboard:'4',next:'angle',
  presets:{default:{meanX:'0.0000',stdX:'11.8322'},parallel:{stdX:'11.8322',stdY:'1.4142'},oblique:{meanX:'4.3208',stdX:'8.8869'},orthogonal:{meanX:'6.0000',stdX:'2.0000'},broad:{meanX:'5.3745',stdX:'7.3463'}},
  invalid:[['sigma','.99'],['angle','91'],['sigma',''],['angle','-1']]}
];
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(Number(port));
 const root='[data-deployment-lab]';
 const report={layouts:[],fallbacks:[],runtimeRecovery:[],semantics:[],exceptions:b.exceptions};
 const values=()=>b.evaluate(`Object.fromEntries([...document.querySelectorAll('${root} output')].map(o=>[o.dataset.value,o.textContent]))`);
 const preset=async name=>{await b.evaluate(`document.querySelector('${root} [data-preset="${name}"]').focus()`);await b.key('Enter','Enter',13)};
 const set=async(name,value)=>b.evaluate(`(()=>{const f=document.querySelector('${root} input[name="${name}"]');f.focus();f.value=${JSON.stringify(value)};f.dispatchEvent(new Event('input',{bubbles:true}))})()`);
 try{
  for(const post of posts) for(const width of [320,768,1440]) for(const mode of ['ordinary','large','spacing']) for(const theme of ['light','dark','warm']){
   await b.viewport(width,1000);await b.navigate(base+post.route);
   await b.evaluate(`document.documentElement.classList.remove('dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='${mode==='ordinary'?100:200}%';${mode==='spacing'?`(()=>{const s=document.createElement('style');s.textContent='${root} p,${root} label,${root} dt,${root} output{line-height:1.5!important;letter-spacing:.12em!important;word-spacing:.16em!important}';document.head.append(s)})()` :''};document.fonts.ready`);
   await pause(60);assert(await b.evaluate(`document.querySelector('${root}').dataset.ready==='true'`));
   for(const [name,expected] of Object.entries(post.presets)){
    await preset(name);const result=await values();
    Object.entries(expected).forEach(([k,v])=>assert.equal(result[k],v,post.kind+'/'+name+'/'+k));
    assert(await b.evaluate(`document.activeElement.matches('[data-preset="${name}"]')`));
    if(post.kind==='timing'&&expected.schedule==='不满足')assert(await b.evaluate(`document.querySelector('${root} [data-plot]').hidden&&document.querySelector('${root} [data-chart]').children.length===0`));
   }
   await preset('default');
   for(const [name,value]of post.invalid){
    await set(name,value);
    const state=await b.evaluate(`(()=>{const r=document.querySelector('${root}');return{focus:document.activeElement.matches('input[name="${name}"]'),error:!r.querySelector('[data-error]').hidden,hidden:r.querySelector('[data-computed]').hidden,empty:[...r.querySelectorAll('output')].every(o=>!o.textContent),chart:!r.querySelector('[data-chart]').children.length,named:!!r.querySelector('input[aria-invalid="true"][aria-errormessage]')}})()`);
    assert(Object.values(state).every(Boolean),JSON.stringify({post:post.kind,name,state}));await preset('default');
   }
   await b.evaluate(`document.querySelector('${root} input[name="${post.field}"]').focus()`);
   await b.cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'a',code:'KeyA',windowsVirtualKeyCode:65,modifiers:2});
   await b.cdp('Input.dispatchKeyEvent',{type:'keyUp',key:'a',code:'KeyA',windowsVirtualKeyCode:65,modifiers:2});
   await b.cdp('Input.insertText',{text:post.keyboard});
   if(post.kind==='timing')assert.equal((await values()).delay,'6');
   if(post.kind==='residual')assert.equal((await values()).finalError,'0.2014');
   if(post.kind==='contact')assert.equal((await values()).stdX,'11.8322');
   await b.key('Tab','Tab',9);assert(await b.evaluate(`document.activeElement.matches('input[name="${post.next}"]')`));
   if(post.kind==='contact'){
    await preset('default');await b.evaluate(`document.querySelector('${root} input[name="second"]').focus()`);
    await b.key(' ','Space',32);assert.equal((await values()).meanX,'6.0000');
   }
   await preset('default');
   const state=await b.evaluate(`(()=>{const r=document.querySelector('${root}');return{overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1,labels:[...r.querySelectorAll('input')].every(i=>i.labels.length),clipped:[...r.querySelectorAll('label,dt,output')].filter(e=>e.scrollWidth>e.clientWidth+2).map(e=>e.textContent),chart:r.querySelector('[data-chart]').children.length>0}})()`);
   const contrast=await b.evaluate('('+contrastAudit.toString()+')('+JSON.stringify(root)+')');
   assert(!state.overflow&&state.labels&&!state.clipped.length&&state.chart&&!contrast.issues.length,JSON.stringify({post:post.kind,width,mode,theme,state,contrast}));
   report.layouts.push({kind:post.kind,width,mode,theme,...state,contrastIssues:contrast.issues});
   if(width===320&&mode==='large'&&theme==='dark'){
    await b.evaluate(`document.querySelector('${root} [data-computed]').scrollIntoView({block:'start'})`);
    await b.screenshot(path.join(output,post.kind+'-mobile-dark-large-results.png'));
   }
  }
  for(const post of posts){
   await b.viewport(768,1000);await b.navigate(base+post.route);
   // A later SVG failure must clear stale results, preserve focus and recover.
   await b.evaluate(`window.__labReplace=SVGSVGElement.prototype.replaceChildren;SVGSVGElement.prototype.replaceChildren=function(...a){if(this.matches('[data-chart]'))throw Error('controlled later SVG failure');return window.__labReplace.apply(this,a)}`);
   await set(post.field,post.keyboard);
   assert(await b.evaluate(`document.querySelector('${root} [data-computed]').hidden&&[...document.querySelectorAll('${root} output')].every(o=>!o.textContent)&&!document.querySelector('${root} svg').children.length&&document.activeElement.matches('input')`));
   await b.evaluate('SVGSVGElement.prototype.replaceChildren=window.__labReplace;delete window.__labReplace');await preset('default');
   assert(await b.evaluate(`!document.querySelector('${root} [data-computed]').hidden`));report.runtimeRecovery.push({kind:post.kind,passed:true});
   if(axePath){
    await b.evaluate(fs.readFileSync(axePath,'utf8'));
    for(const invalid of [false,true]){
     if(invalid)await set(post.field,'');
     const result=await b.evaluate(`axe.run(document.querySelector('${root}'),{rules:{'color-contrast':{enabled:false}}}).then(r=>({violations:r.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),incomplete:r.incomplete.map(v=>v.id)}))`);
     assert(!result.violations.length,JSON.stringify(result));report.semantics.push({kind:post.kind,invalid,...result});
    }
   }
   for(const mode of ['scripts-off','blocked-script','setup-failure']){
    let injected;
    if(mode==='scripts-off')await b.cdp('Emulation.setScriptExecutionDisabled',{value:true});
    if(mode==='blocked-script')await b.cdp('Network.setBlockedURLs',{urls:['*deployment-labs*.js*']});
    if(mode==='setup-failure')injected=await b.cdp('Page.addScriptToEvaluateOnNewDocument',{source:`(()=>{const original=SVGSVGElement.prototype.replaceChildren;SVGSVGElement.prototype.replaceChildren=function(...a){if(this.matches('[data-chart]'))throw Error('controlled initial SVG failure');return original.apply(this,a)}})()`});
    await b.viewport(320,1000);await b.navigate(base+post.route);
    const fallback=await b.evaluate(`(()=>{const r=document.querySelector('${root}');return{visible:!r.querySelector('[data-static-result]').hidden,controlsHidden:r.querySelector('[data-interactive]').hidden,text:r.querySelector('[data-static-result]').textContent,overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1}})()`);
    assert(fallback.visible&&fallback.controlsHidden&&fallback.text.length>40&&!fallback.overflow,JSON.stringify({post:post.kind,mode,fallback}));report.fallbacks.push({kind:post.kind,mode,...fallback});
    await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});await b.cdp('Network.setBlockedURLs',{urls:[]});
    if(injected)await b.cdp('Page.removeScriptToEvaluateOnNewDocument',{identifier:injected.identifier});
   }
  }
  assert(!b.exceptions.length,JSON.stringify(b.exceptions));
  console.log(JSON.stringify({layouts:report.layouts.length,fallbacks:report.fallbacks.length,recoveries:report.runtimeRecovery.length,semantics:report.semantics.length,exceptions:0}));
 }finally{fs.writeFileSync(path.join(output,'labs.json'),JSON.stringify(report,null,2)+'\n');await b.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
