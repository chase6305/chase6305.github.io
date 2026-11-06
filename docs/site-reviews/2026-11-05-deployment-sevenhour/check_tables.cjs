#!/usr/bin/env node
// Inspect every table in the three articles; keyboard events use the live CDP tab.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {connect,pause}=require('../../../scripts/lib/browser.cjs');
const [base,port='9337',output='/tmp/deployment-sevenhour-tables']=process.argv.slice(2);
const posts=[['deployment','/posts/ai/rma-umi-rtc-deployment/'],['residual','/posts/ai/residual-reinforcement-learning/'],['contact','/posts/calibration/interactive-contact-calibration/']];
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(Number(port));
 const report={pages:[],tables:[],exceptions:b.exceptions,scope:'All article tables at narrow viewports, native keyboard scrolling, and recorded long-cell width for manual review.'};
 try{
  for(const scripts of [true,false])for(const [post,route]of posts)for(const width of [320,390])for(const font of [100,200]){
   await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.viewport(width,1000);await b.navigate(base+route);
   await b.evaluate(`document.documentElement.style.fontSize='${font}%';document.fonts.ready`);await pause(120);
   const page=await b.evaluate(`({width:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth,count:document.querySelectorAll('article table').length})`);
   assert(page.count>0&&page.scroll<=page.width+1,JSON.stringify({post,width,font,scripts,page}));report.pages.push({post,width,font,scripts,...page});
   for(let index=0;index<page.count;index++){
    const state=await b.evaluate(`(()=>{const t=document.querySelectorAll('article table')[${index}];t.scrollIntoView({block:'center'});t.scrollLeft=0;const cells=[...t.querySelectorAll('th,td')].map(c=>({text:c.textContent.trim(),width:c.clientWidth,font:parseFloat(getComputedStyle(c).fontSize),header:c.tagName==='TH',contentWidth:c.clientWidth-parseFloat(getComputedStyle(c).paddingLeft)-parseFloat(getComputedStyle(c).paddingRight),cjk:(c.textContent.match(/[\u3400-\u9fff]/g)||[]).length}));return {width:t.clientWidth,scroll:t.scrollWidth,tab:t.tabIndex,readable:t.classList.contains('table-readable'),cells,longNarrow:cells.filter(c=>(c.cjk>=12||c.header&&c.cjk>=4)&&c.contentWidth<c.font*2.5)}})()`);
    let keyboard=false;
    if(state.scroll>state.width+2){
     assert.equal(state.tab,0,JSON.stringify({post,width,font,scripts,index,state}));
     assert(await b.evaluate(`(()=>{const t=document.querySelectorAll('article table')[${index}];t.focus();return document.activeElement===t})()`));
     await b.key('ArrowRight','ArrowRight',39);await pause(140);
     keyboard=await b.evaluate(`document.querySelectorAll('article table')[${index}].scrollLeft>0`);assert(keyboard,JSON.stringify({post,width,font,scripts,index}));
    }
    if(scripts&&width===320&&font===200){
     await b.evaluate(`document.querySelectorAll('article table')[${index}].scrollLeft=0`);await pause(50);
     await b.screenshot(path.join(output,`${post}-table-${index}-320-200.png`));
    }
    report.tables.push({post,viewportWidth:width,font,scripts,index,keyboard,...state});
   }
  }
  assert(!b.exceptions.length,JSON.stringify(b.exceptions));
  console.log(JSON.stringify({pages:report.pages.length,tables:report.tables.length,keyboardTables:report.tables.filter(t=>t.keyboard).length,longNarrowCandidates:report.tables.filter(t=>t.longNarrow.length).length,exceptions:0}));
 }finally{fs.writeFileSync(path.join(output,'tables.json'),JSON.stringify(report,null,2)+'\n');await b.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
