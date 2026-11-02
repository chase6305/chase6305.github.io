// Real arrow-key access and focus preservation for the new/revised equations.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const{connect}=require('../../../scripts/lib/browser.cjs');
const{applyReviewTheme}=require('./theme_state.cjs');
const[base,portText,out]=process.argv.slice(2),ids=['vla-gradient-rank','snr-estimation','snr-correlated-groups','grpo-group-gradient','flat-rank-study','vla-padding-recovery'];
(async()=>{fs.mkdirSync(out,{recursive:true});const b=await connect(Number(portText)),report={cases:[],exceptions:b.exceptions};
 async function transition(mode,narrow){await b.viewport(mode==='resize'?(narrow?320:1440):390);await b.evaluate(`document.documentElement.style.fontSize='${mode==='resize'||narrow?200:100}%';document.fonts.ready`);await b.pause(180);}
 try{
  for(const scripts of[true,false])for(const theme of['light','dark','warm'])for(const mode of['resize','font'])for(const id of ids){
   await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.navigate(base+'/posts/ai/pion-muon-vla-rlvr/');
   const palette=await b.evaluate('('+applyReviewTheme.toString()+')('+JSON.stringify(theme)+','+scripts+')');
   await b.evaluate(`document.getElementById('${id}').open=true;`);
   await transition(mode,true);const initial=await b.evaluate(`(()=>{window.__math=[...document.querySelectorAll('#${id} .katex-display')].sort((a,b)=>b.scrollWidth-a.scrollWidth)[0];__math.scrollIntoView({block:'center'});__math.scrollLeft=0;__math.focus();return{focus:document.activeElement===__math,tab:__math.tabIndex,width:__math.clientWidth,scroll:__math.scrollWidth,tex:__math.querySelector('annotation').textContent}})()`);assert(initial.tex.trim());assert(initial.focus&&initial.tab===0&&initial.scroll>initial.width+2,JSON.stringify({scripts,theme,mode,id,initial}));
   for(let n=0;n<5;n++)await b.key('ArrowRight','ArrowRight',39);await b.pause(200);const movement=await b.evaluate('__math.scrollLeft');assert(movement>0);
   await transition(mode,false);const fitting=await b.evaluate(`(()=>{const hint=__math.previousElementSibling;return{focus:document.activeElement===__math,tab:__math.tabIndex,width:__math.clientWidth,scroll:__math.scrollWidth,hintHidden:!hint?.matches('.blog-math-scroll-hint')||hint.hidden,bodyOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1}})()`);assert(fitting.focus&&fitting.tab===0&&fitting.scroll<=fitting.width+1&&fitting.hintHidden&&!fitting.bodyOverflow,JSON.stringify({scripts,theme,mode,id,fitting}));
   await b.key('Tab','Tab',9);await b.pause(180);const blur=await b.evaluate(`window.__afterMathTab=document.activeElement;({left:document.activeElement!==__math,hasTabIndex:__math.hasAttribute('tabindex')})`);assert(blur.left);if(scripts)assert(!blur.hasTabIndex);
   await transition(mode,true);assert(await b.evaluate(`__math.tabIndex===0&&document.activeElement===__afterMathTab`));report.cases.push({scripts,theme,mode,id,palette,initial,movement,fitting,blur,restoredEntry:true,laterFocusPreserved:true});
  }
  assert.deepEqual(b.exceptions,[]);console.log(JSON.stringify({cases:report.cases.length,exceptions:0}));
 }finally{fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
