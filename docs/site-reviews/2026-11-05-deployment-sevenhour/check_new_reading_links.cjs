#!/usr/bin/env node
// New body-fragment/cross-post links and body-only search terms.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {connect,pause}=require('../../../scripts/lib/browser.cjs');
const [base,port='9337',output='/tmp/deployment-new-links']=process.argv.slice(2);
const deployment='/posts/ai/rma-umi-rtc-deployment/',residual='/posts/ai/residual-reinforcement-learning/';
const td3='104-td3twin-delayed-deep-deterministic-policy-gradient';
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(Number(port));
 const report={fragments:[],searches:[],exceptions:b.exceptions};
 const wait=async expression=>{
  for(let i=0;i<120;i++){
   try{if(await b.evaluate(expression))return}catch(e){if(!/context|navigat|closed/i.test(e.message))throw e}
   await pause(30);
  }
  throw Error('Condition timed out: '+expression);
 };
 const follow=async(selector,pathname,id)=>{
  assert(await b.evaluate(`(()=>{const a=document.querySelector(${JSON.stringify(selector)});if(!a||!a.checkVisibility())throw Error('Missing visible body link');a.scrollIntoView({block:'center'});a.focus();return document.activeElement===a})()`),'Body link did not receive keyboard focus');
  await b.key('Enter','Enter',13);
  await wait(`document.readyState==='complete'&&location.pathname===${JSON.stringify(pathname)}&&location.hash===${JSON.stringify('#'+id)}`);
  await b.evaluate('document.fonts.ready');
  await wait(`(()=>{const r=document.getElementById(${JSON.stringify(id)}).getBoundingClientRect();return r.bottom>0&&r.top<innerHeight})()`);await pause(150);
  const before=await b.evaluate(`(()=>{const h=document.getElementById(${JSON.stringify(id)}),r=h.getBoundingClientRect();return{top:r.top,bottom:r.bottom,visible:r.bottom>0&&r.top<innerHeight,rootFont:getComputedStyle(document.documentElement).fontSize,activeTag:document.activeElement.tagName}})()`);
  assert(before.visible,JSON.stringify({pathname,id,before}));
  await b.key('Tab','Tab',9);await pause(100);
  const after=await b.evaluate(`(()=>{const h=document.getElementById(${JSON.stringify(id)}),a=document.activeElement,r=a.getBoundingClientRect();return{tag:a.tagName,href:a.getAttribute('href'),following:!!(h.compareDocumentPosition(a)&Node.DOCUMENT_POSITION_FOLLOWING),inContent:!!a.closest('article .content'),visible:r.bottom>0&&r.top<innerHeight}})()`);
  assert(after.following&&after.inContent&&after.visible,JSON.stringify({pathname,id,after}));
  return {pathname,id,before,after};
 };
 try{
  for(const scripts of [true,false])for(const width of [320,1440])for(const font of [100,200]){
   await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.viewport(width,1000);await b.navigate(base+deployment);
   await b.evaluate(`document.documentElement.style.fontSize='${font}%';document.fonts.ready`);
   const local=await follow('article .content p a[href="#waypoint-timing"]',deployment,'waypoint-timing');
   await b.navigate(base+residual);await b.evaluate(`document.documentElement.style.fontSize='${font}%';document.fonts.ready`);
   const cross=await follow('article .content a[href="/posts/rl/#'+td3+'"]','/posts/rl/',td3);
   report.fragments.push({scripts,width,font,local,cross});
  }
  await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});
  for(const width of [320,1440])for(const [route,query]of [[deployment,'插值终点'],[residual,'双曲正切']]){
   await b.viewport(width,1000);await b.navigate(base+route);await b.evaluate('document.fonts.ready');
   await b.evaluate(`(()=>{const a=[...document.querySelectorAll('[aria-controls="hextra-search-dialog"]')].find(e=>e.checkVisibility()&&!e.closest('[inert]'));if(!a)throw Error('Missing search opener');window.__sourceSearchOpener=a;a.focus()})()`);
   await b.key('Enter','Enter',13);await wait(`document.querySelector('#hextra-search-dialog').open`);
   await b.evaluate(`(()=>{const i=document.querySelector('.hextra-search-input');i.value=${JSON.stringify(query)};i.dispatchEvent(new Event('input',{bubbles:true}))})()`);
   await wait(`[...document.querySelectorAll('.hextra-search-results a[role="option"]')].some(a=>new URL(a.href).pathname===${JSON.stringify(route)})`);
   const links=await b.evaluate(`[...document.querySelectorAll('.hextra-search-results a[role="option"]')].map(a=>({id:a.id,href:a.href,pathname:new URL(a.href).pathname,child:a.parentElement.classList.contains('hextra-search-child'),parentTitle:a.getAttribute('aria-describedby'),text:a.textContent.trim()}))`);
   // Page groups and their chapter excerpts intentionally share a pathname.
   // Check actual option IDs, parent page groups and per-group excerpt URLs.
   assert.equal(new Set(links.map(a=>a.id)).size,links.length,'Duplicate DOM option IDs');
   const parents=links.filter(a=>!a.child);
   assert.equal(new Set(parents.map(a=>a.pathname)).size,parents.length,'Duplicate parent page groups');
   for(const title of new Set(links.filter(a=>a.child).map(a=>a.parentTitle))){
    const children=links.filter(a=>a.child&&a.parentTitle===title);
    assert.equal(new Set(children.map(a=>a.href)).size,children.length,'Duplicate chapter excerpt within page group');
   }
   await b.key('Escape','Escape',27);await wait(`!document.querySelector('#hextra-search-dialog').open`);
   assert(await b.evaluate('document.activeElement===__sourceSearchOpener'));
   report.searches.push({route,query,width,links,escapeFocusReturned:true});
  }
  assert(!b.exceptions.length,JSON.stringify(b.exceptions));
  console.log(JSON.stringify({fragmentContexts:report.fragments.length,keyboardLinkActivations:report.fragments.length*2,bodySearchCases:report.searches.length,exceptions:0}));
 }finally{fs.writeFileSync(path.join(output,'reading-links.json'),JSON.stringify(report,null,2)+'\n');await b.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
