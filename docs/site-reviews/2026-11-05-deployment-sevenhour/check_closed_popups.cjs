#!/usr/bin/env node
// Resolve axe's closed-popup manual-review cases through real keyboard opening.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {connect,pause}=require('../../../scripts/lib/browser.cjs');
const [base='http://127.0.0.1:38515',port='9337',output='/tmp/deployment-closed-popups',axePath='/tmp/chase-a11y/node_modules/axe-core/axe.min.js']=process.argv.slice(2);
const routes=['/posts/ai/rma-umi-rtc-deployment/','/posts/ai/residual-reinforcement-learning/','/posts/calibration/interactive-contact-calibration/'];
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(Number(port)),report={pages:[],controls:[],emptySearchReview:[],exceptions:b.exceptions};
 const wait=async expression=>{for(let i=0;i<100;i++){if(await b.evaluate(expression))return;await pause(20)}throw Error('Popup condition timed out: '+expression)};
 try{
  for(const width of [320,1440])for(const route of routes){
   await b.viewport(width,1000);await b.navigate(base+route);await b.evaluate('document.fonts.ready');
   await b.evaluate(fs.readFileSync(axePath,'utf8'));
   const initial=await b.evaluate(`axe.run(document,{rules:{'color-contrast':{enabled:false}}}).then(r=>({violations:r.violations.map(v=>v.id),incomplete:r.incomplete.map(v=>({id:v.id,nodes:v.nodes.map(n=>({target:n.target,summary:n.failureSummary}))}))}))`);
   assert(!initial.violations.length,JSON.stringify(initial));
   assert(initial.incomplete.every(r=>r.id==='aria-valid-attr-value'&&r.nodes.every(n=>n.summary.includes('aria-controls'))),JSON.stringify(initial));
   const controls=await b.evaluate(`(()=>{const result=[];document.querySelectorAll('[aria-haspopup][aria-controls]').forEach((e,i)=>{if(!e.checkVisibility()||e.closest('[inert]'))return;e.dataset.popupReview=String(i);result.push({selector:'[data-popup-review="'+i+'"]',id:e.getAttribute('aria-controls'),type:e.getAttribute('aria-haspopup')})});return result})()`);
   report.pages.push({route,width,...initial,controls:controls.length});
   for(const item of controls){
    await b.evaluate(`(()=>{window.__popupReviewOpener=document.querySelector(${JSON.stringify(item.selector)});__popupReviewOpener.scrollIntoView({block:'center',behavior:'instant'});__popupReviewOpener.focus()})()`);
    const closed=await b.evaluate(`(()=>{const id=${JSON.stringify(item.id)},e=document.getElementById(id);return {exists:!!e,role:e?.getAttribute('role')||e?.tagName.toLowerCase(),closed:e&&!e.checkVisibility(),idCount:[...document.querySelectorAll('[id]')].filter(x=>x.id===id).length}})()`);
    assert(closed.exists&&closed.closed&&closed.idCount===1,JSON.stringify({route,width,item,closed}));
    assert.equal(closed.role,item.type);
    await b.key('Enter','Enter',13);await wait(`document.getElementById(${JSON.stringify(item.id)}).checkVisibility()`);
    const opened=await b.evaluate(`(()=>{const e=document.getElementById(${JSON.stringify(item.id)});return{visible:e.checkVisibility(),dialogOpen:e.tagName==='DIALOG'?e.open:null,focusWithin:e.contains(document.activeElement),expanded:__popupReviewOpener.getAttribute('aria-expanded')}})()`);
    assert(opened.visible&&opened.focusWithin,JSON.stringify({route,width,item,opened}));
    if(item.type==='menu')assert.equal(opened.expanded,'true');else assert.equal(opened.dialogOpen,true);
    if(item.id==='hextra-search-dialog'){
     const empty=await b.evaluate(`({query:document.querySelector('.hextra-search-input').value,options:document.querySelectorAll('.hextra-search-results [role="option"]').length})`);
     const emptyAudit=await b.evaluate(`axe.run(document.getElementById('hextra-search-dialog'),{rules:{'color-contrast':{enabled:false}}}).then(r=>({violations:r.violations.map(v=>v.id),incomplete:r.incomplete.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)}))}))`);
     assert.equal(empty.query,'');assert.equal(empty.options,0);assert(!emptyAudit.violations.length);
     assert(emptyAudit.incomplete.every(v=>v.id==='aria-required-children'&&v.nodes.every(n=>n.length===1&&n[0]==='#hextra-search-results')),JSON.stringify(emptyAudit));
     report.emptySearchReview.push({route,width,...empty,...emptyAudit,interpretation:'An empty query has no result options; verify the populated listbox next.'});
     const query=route.includes('rma-')?'RMA':route.includes('residual-')?'残差强化学习':'自标定';
     await b.evaluate(`(()=>{const input=document.querySelector('.hextra-search-input');input.value=${JSON.stringify(query)};input.dispatchEvent(new Event('input',{bubbles:true}))})()`);
     await wait(`document.querySelector('.hextra-search-results [role="option"]')!==null`);
     assert(await b.evaluate(`[...document.querySelectorAll('.hextra-search-results a[role="option"]')].some(a=>new URL(a.href).pathname===${JSON.stringify(route)})`),'New article missing from search results: '+route);
    }
    const audit=await b.evaluate(`axe.run(document.getElementById(${JSON.stringify(item.id)}),{rules:{'color-contrast':{enabled:false}}}).then(r=>({violations:r.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),incomplete:r.incomplete.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)}))}))`);
    assert(!audit.violations.length&&!audit.incomplete.length,JSON.stringify({route,width,item,audit}));
    await b.key('Escape','Escape',27);await wait(`!document.getElementById(${JSON.stringify(item.id)}).checkVisibility()`);await pause(100);
    assert(await b.evaluate('document.activeElement===__popupReviewOpener'),'Escape did not return focus: '+JSON.stringify(item));
    report.controls.push({route,width,...item,closed,opened,audit,escapeFocusReturned:true});
   }
  }
  assert(!b.exceptions.length,JSON.stringify(b.exceptions));console.log(JSON.stringify({pages:report.pages.length,controls:report.controls.length,emptySearchStatesReviewed:report.emptySearchReview.length,closedReferencesResolved:true,populatedPopupViolations:0,populatedPopupIncomplete:0,exceptions:0}));
 }finally{fs.writeFileSync(path.join(output,'closed-popups.json'),JSON.stringify(report,null,2)+'\n');await b.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
