// Review the paper/example connection and its new native backlink.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const{connect}=require('../../../scripts/lib/browser.cjs');
const{contrastAudit}=require('../../../scripts/lib/contrast.cjs');
const{applyReviewTheme}=require('./theme_state.cjs');
const[base,prior,portText,out]=process.argv.slice(2),route='/posts/ai/pion-muon-vla-rlvr/';
(async()=>{
 fs.mkdirSync(out,{recursive:true});const b=await connect(Number(portText)),report={layouts:[],exceptions:b.exceptions};
 try{
  const source=await(await fetch(base+route)).text(),old=await(await fetch(prior+route)).text();
  report.sourceHtmlSha256=crypto.createHash('sha256').update(source).digest('hex');await b.navigate(base+route);
  report.preserved=await b.evaluate(`(()=>{const parse=s=>new DOMParser().parseFromString(s,'text/html'),pick=d=>({disclosures:['vla-gradient-rank','snr-correlated-groups','grpo-group-gradient'].map(id=>d.getElementById(id).outerHTML),headings:[...d.querySelectorAll('.content h2,.content h3')].map(e=>[e.id,e.textContent.trim()]),code:[...d.querySelectorAll('.hextra-code-block code')].map(e=>e.textContent),figures:[...d.querySelectorAll('figure.article-figure')].map(e=>e.outerHTML),ids:[...d.querySelectorAll('[id]')].map(e=>e.id),styles:[...d.querySelectorAll('link[rel="stylesheet"]')].map(e=>e.getAttribute('href'))}),a=pick(parse(${JSON.stringify(old)})),c=pick(parse(${JSON.stringify(source)}));return Object.fromEntries(Object.keys(a).map(k=>[k,{same:JSON.stringify(a[k])===JSON.stringify(c[k]),before:a[k].length,after:c[k].length}]))})()`);
  assert(Object.values(report.preserved).every(r=>r.same),JSON.stringify(report.preserved));
  for(const scripts of[true,false])for(const width of[320,1440])for(const theme of['light','dark','warm'])for(const size of[100,200]){
   await b.viewport(width,1100);await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.navigate(base+route);
   const palette=await b.evaluate('('+applyReviewTheme.toString()+')('+JSON.stringify(theme)+','+scripts+')');
   await b.evaluate(`document.documentElement.style.fontSize='${size}%';document.fonts.ready`);
   const prose=await b.evaluate(`(()=>{window.__bridge=[...document.querySelectorAll('.content p')].find(p=>p.querySelector('a[href="https://arxiv.org/pdf/2605.19282v1#page=20"]')?.textContent.includes('A13'));if(!__bridge)throw Error('Paper bridge missing');window.__bridgeParas=[__bridge,__bridge.nextElementSibling,__bridge.nextElementSibling.nextElementSibling];__bridgeParas.forEach(p=>p.dataset.reviewResidual='true');window.__backlink=__bridgeParas[1].querySelector('a[href="#grpo-group-gradient"]');if(!__backlink)throw Error('Example backlink missing');return __bridgeParas.map(p=>({text:p.textContent.trim(),visible:p.checkVisibility(),width:p.clientWidth}))})()`);
   assert(prose.length===3&&prose.every(p=>p.visible&&p.width>0));assert(prose[1].text.includes('条件残差恒为零'));assert(prose[2].text.includes('逐项检查这些条件'));
   const contrast=await b.evaluate('('+contrastAudit.toString()+')('+JSON.stringify('[data-review-residual]')+')');assert(contrast.checked>0);assert.deepEqual(contrast.issues,[]);
   assert(await b.evaluate('document.documentElement.scrollWidth<=document.documentElement.clientWidth+1'));
   if((width===1440&&size===100)||(width===320&&size===200))for(let index=0;index<3;index++){
    await b.evaluate(`__bridgeParas[${index}].scrollIntoView({block:'start'});scrollBy(0,-document.querySelector('.hextra-nav-container nav').getBoundingClientRect().height-20)`);await b.pause(80);await b.screenshot(path.join(out,`bridge-${scripts}-${width}-${theme}-${size}-p${index}.png`));
   }
   assert(await b.evaluate('(__backlink.focus(),document.activeElement===__backlink)'));await b.key('Enter','Enter',13);await b.pause(160);
   const reached=await b.evaluate(`(()=>{const d=document.getElementById('grpo-group-gradient'),r=d.querySelector('summary').getBoundingClientRect();return{hash:location.hash,open:d.open,top:r.top,bottom:r.bottom,nav:document.querySelector('.hextra-nav-container nav').getBoundingClientRect().bottom,height:innerHeight}})()`);
   assert(reached.hash==='#grpo-group-gradient'&&!reached.open&&reached.top>=reached.nav-1&&reached.bottom<=reached.height,JSON.stringify(reached));
   await b.key('Tab','Tab',9);assert(await b.evaluate(`document.activeElement===document.querySelector('#grpo-group-gradient>summary')`));await b.key('Enter','Enter',13);assert(await b.evaluate(`document.getElementById('grpo-group-gradient').open`));await b.key('Enter','Enter',13);assert(await b.evaluate(`!document.getElementById('grpo-group-gradient').open`));
   report.layouts.push({scripts,width,theme,size,palette,prose,contrast,reached,keyboardExpandedAndClosed:true});
  }
  assert.deepEqual(b.exceptions,[]);console.log(JSON.stringify({layouts:report.layouts.length,preservedDisclosures:3,exceptions:0}));
 }finally{fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
