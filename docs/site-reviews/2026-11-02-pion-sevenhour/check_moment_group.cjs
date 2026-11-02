// Check the local print grouping without changing content or keyboard access.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {connect}=require('../../../scripts/lib/browser.cjs');
const {contrastAudit}=require('../../../scripts/lib/contrast.cjs');
const {applyReviewTheme}=require('./theme_state.cjs');
const [base,prior,portText,out]=process.argv.slice(2),route='/posts/ai/pion-muon-vla-rlvr/';
(async()=>{
 fs.mkdirSync(out,{recursive:true});const b=await connect(Number(portText)),report={layouts:[],exceptions:b.exceptions};
 try{
  const source=await(await fetch(base+route)).text(),old=await(await fetch(prior+route)).text();
  report.sourceHtmlSha256=crypto.createHash('sha256').update(source).digest('hex');
  await b.navigate(base+route);
  report.contentComparison=await b.evaluate(`(()=>{const parse=s=>new DOMParser().parseFromString(s,'text/html');const pick=d=>({prose:[...d.querySelectorAll('.content p,.content details>summary')].map(e=>e.textContent.trim()),headings:[...d.querySelectorAll('.content h2,.content h3')].map(e=>[e.id,e.textContent.trim()]),formulas:[...d.querySelectorAll('.content annotation[encoding="application/x-tex"]')].map(e=>e.textContent),code:[...d.querySelectorAll('.hextra-code-block code')].map(e=>e.textContent),figures:[...d.querySelectorAll('figure.article-figure')].map(e=>[e.id,e.querySelector('img')?.getAttribute('src'),e.querySelector('figcaption')?.textContent.trim()]),ids:[...d.querySelectorAll('[id]')].map(e=>e.id),links:[...d.querySelectorAll('.content a[href]')].map(e=>e.getAttribute('href')),styles:[...d.querySelectorAll('link[rel="stylesheet"]')].map(e=>e.getAttribute('href'))});const a=pick(parse(${JSON.stringify(old)})),c=pick(parse(${JSON.stringify(source)}));return Object.fromEntries(Object.keys(a).map(k=>[k,{same:JSON.stringify(a[k])===JSON.stringify(c[k]),before:a[k].length,after:c[k].length}]))})()`);
  assert(Object.values(report.contentComparison).every(r=>r.same),JSON.stringify(report.contentComparison));
  for(const scripts of[true,false])for(const width of[320,1440])for(const theme of['light','dark','warm'])for(const size of[100,200]){
   await b.viewport(width,1100);await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.navigate(base+route);
   const palette=await b.evaluate('('+applyReviewTheme.toString()+')('+JSON.stringify(theme)+','+scripts+')');
   await b.evaluate(`document.documentElement.style.fontSize='${size}%';document.fonts.ready`);
   await b.evaluate(`document.querySelector('#grpo-group-gradient>summary').focus()`);await b.key('Enter','Enter',13);
   await b.evaluate(`window.__group=[...document.querySelectorAll('#grpo-group-gradient div[style]')].find(e=>e.style.breakInside==='avoid');window.__momentMath=__group?.querySelector('.katex-display');__group.scrollIntoView({block:'center'});`);await b.pause(150);
   const before=await b.evaluate(`(()=>{if(!__group||!__momentMath)throw Error('Moment group missing');__momentMath.scrollLeft=0;const overflow=__momentMath.scrollWidth>__momentMath.clientWidth+1;if(overflow)__momentMath.focus();return{overflow,width:__momentMath.clientWidth,scroll:__momentMath.scrollWidth,focus:document.activeElement===__momentMath,tex:__momentMath.querySelector('annotation').textContent,breakInside:getComputedStyle(__group).breakInside,bodyOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1}})()`);
   assert(!before.bodyOverflow&&before.breakInside==='avoid'&&before.tex.includes('operatorname{Var}'));
   const contrast=await b.evaluate('('+contrastAudit.toString()+')('+JSON.stringify('#grpo-group-gradient')+')');assert.deepEqual(contrast.issues,[]);
   const stem=`moment-${scripts}-${width}-${theme}-${size}`;
   if(width===320&&size===200)await b.screenshot(path.join(out,stem+'-start.png'));
   if(before.overflow){assert(before.focus);for(let n=0;n<30;n++)await b.key('ArrowRight','ArrowRight',39);await b.pause(200);}
   const after=await b.evaluate(`({left:__momentMath.scrollLeft,lastTermReachable:__momentMath.scrollLeft>=__momentMath.scrollWidth-__momentMath.clientWidth-1,focus:document.activeElement===__momentMath})`);
   assert(after.lastTermReachable);if(before.overflow)assert(after.focus&&after.left>0);
   if(width===320&&size===200)await b.screenshot(path.join(out,stem+'-end.png'));
   report.layouts.push({scripts,width,theme,size,palette,before,after,contrast});
  }
  assert.deepEqual(b.exceptions,[]);console.log(JSON.stringify({layouts:report.layouts.length,contentUnchanged:true,exceptions:0}));
 }finally{
  fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');
  await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});await b.close();
 }
})().catch(e=>{console.error(e);process.exitCode=1});
