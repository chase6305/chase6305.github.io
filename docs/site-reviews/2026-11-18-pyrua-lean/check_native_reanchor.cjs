#!/usr/bin/env node
// Diagnose browser-owned no-script print reanchoring against an existing article.
const fs=require('node:fs'),path=require('node:path');
const {connect}=require('../../../scripts/lib/browser.cjs');
const base=process.argv[2],port=Number(process.argv[3]),out=process.argv[4];
fs.mkdirSync(out,{recursive:true});
(async()=>{
 const b=await connect(port),rows=[];
 try{
  for(const route of ['/posts/ai/pyrua-lean/','/posts/ai/muon-optimizer/']){
   await b.viewport(1440,1100);await b.cdp('Emulation.setScriptExecutionDisabled',{value:true});
   await b.navigate(base+route);await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});
   await b.evaluate(`Promise.all([document.fonts.ready,...[...document.querySelectorAll('.content img')].map(i=>{i.loading='eager';return i.decode()})])`);
   await b.evaluate(`window.__target=[...document.querySelectorAll('.content p,.content li')].filter(e=>!e.closest('nav,.blog-related,.blog-mobile-toc')&&e.getBoundingClientRect().height>0&&e.getBoundingClientRect().height<200).at(-1);__target.tabIndex=-1;__target.scrollIntoView({block:'center',behavior:'instant'});__target.focus({preventScroll:true})`);
   await b.pause(200);
   const snapshot=`({y:scrollY,height:document.documentElement.scrollHeight,top:__target.getBoundingClientRect().top,bottom:__target.getBoundingClientRect().bottom,focused:document.activeElement===__target,viewport:innerHeight})`;
   const before=await b.evaluate(snapshot);
   await b.cdp('Page.printToPDF',{printBackground:true,generateTaggedPDF:true,paperWidth:8.2677,paperHeight:11.6929,marginTop:.4,marginBottom:.4,marginLeft:.4,marginRight:.4});
   await b.pause(400);const after=await b.evaluate(snapshot);
   rows.push({route,before,after,scrollDelta:after.y-before.y});
  }
  console.log(JSON.stringify(rows));
 }finally{fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(rows,null,2)+'\n');await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
