#!/usr/bin/env node
// Serial keyboard review of the two added chapters, with native fallback.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {connect,pause}=require('../../../scripts/lib/browser.cjs');
const [base,port='9337',output='/tmp/deployment-followup-chapters']=process.argv.slice(2);
const posts=[['/posts/ai/rma-umi-rtc-deployment/','action-metric'],['/posts/calibration/interactive-contact-calibration/','multimodal-mean']];
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(Number(port));const report={cases:[],exceptions:b.exceptions};
 try{
  for(const scripts of [true,false])for(const width of [320,768,1440])for(const font of [100,200])for(const [route,id]of posts){
   await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.viewport(width,1000);await b.navigate(base+route);
   await b.evaluate(`document.documentElement.style.fontSize='${font}%';document.fonts.ready`);
   const selector=`a[href="#${id}"]`;
   assert(await b.evaluate(`(()=>{const d=document.querySelector('.blog-mobile-toc');if(d?.checkVisibility())d.open=true;const a=[...document.querySelectorAll(${JSON.stringify(selector)})].find(a=>!a.classList.contains('subheading-anchor')&&a.checkVisibility());if(!a)return false;window.__chapterFromMobile=!!a.closest('.blog-mobile-toc');a.scrollIntoView({block:'center'});a.focus();return document.activeElement===a})()`),'Missing visible chapter link');
   await b.key('Enter','Enter',13);await pause(130);
   const state=await b.evaluate(`(()=>{const h=document.getElementById(${JSON.stringify(id)}),r=h.getBoundingClientRect();return{hash:location.hash,visible:r.top>=-1&&r.top<innerHeight-24,focused:document.activeElement===h,mobile:__chapterFromMobile,overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1}})()`);
   // Mobile enhancement explicitly focuses the heading. Desktop links retain
   // native fragment behavior; both must preserve subsequent keyboard reading.
   assert(state.hash==='#'+id&&state.visible&&!state.overflow&&(!scripts||!state.mobile||state.focused),JSON.stringify({scripts,width,font,route,state}));
   await b.key('Tab','Tab',9);await pause(100);
   const after=await b.evaluate(`(()=>{const h=document.getElementById(${JSON.stringify(id)}),a=document.activeElement,r=a.getBoundingClientRect();return{following:!!(h.compareDocumentPosition(a)&Node.DOCUMENT_POSITION_FOLLOWING),content:!!a.closest('article .content'),visible:r.bottom>0&&r.top<innerHeight}})()`);
   assert(after.following&&after.content&&after.visible,JSON.stringify({scripts,width,font,route,after}));
   report.cases.push({scripts,width,font,route,id,...state,after});
  }
  assert(!b.exceptions.length,JSON.stringify(b.exceptions));console.log(JSON.stringify({keyboardChapterCases:report.cases.length,exceptions:0}));
 }finally{fs.writeFileSync(path.join(output,'chapters.json'),JSON.stringify(report,null,2)+'\n');await b.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
