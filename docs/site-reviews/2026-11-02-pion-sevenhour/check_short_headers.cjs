// Review two deliberately shortened table headers; all other article HTML must match.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const{connect}=require('../../../scripts/lib/browser.cjs');
const{contrastAudit}=require('../../../scripts/lib/contrast.cjs');
const{applyReviewTheme}=require('./theme_state.cjs');
const[base,prior,portText,out]=process.argv.slice(2),route='/posts/ai/pion-muon-vla-rlvr/';
(async()=>{
 fs.mkdirSync(out,{recursive:true});const b=await connect(Number(portText)),report={layouts:[],exceptions:b.exceptions};
 try{
  const source=await(await fetch(base+route)).text(),old=await(await fetch(prior+route)).text();
  const replacements=[['逐行套用独立校正的功率期望','逐行校正的功率期望'],['以四个独立组校正的功率期望','按四组校正的功率期望']];let normalizedOld=old;for(const[from,to]of replacements){assert.equal(old.split(from).length-1,1);normalizedOld=normalizedOld.replaceAll(from,to);}report.intentionalHeaderChanges=replacements;
  report.sourceHtmlSha256=crypto.createHash('sha256').update(source).digest('hex');await b.navigate(base+route);
  report.preserved=await b.evaluate(`(()=>{const parse=s=>new DOMParser().parseFromString(s,'text/html'),pick=d=>({body:[d.querySelector('.content').outerHTML],disclosures:['vla-gradient-rank','snr-correlated-groups','grpo-group-gradient'].map(id=>d.getElementById(id).outerHTML),headings:[...d.querySelectorAll('.content h2,.content h3')].map(e=>[e.id,e.textContent.trim()]),code:[...d.querySelectorAll('.hextra-code-block code')].map(e=>e.textContent),figures:[...d.querySelectorAll('figure.article-figure')].map(e=>e.outerHTML),ids:[...d.querySelectorAll('[id]')].map(e=>e.id),styles:[...d.querySelectorAll('link[rel="stylesheet"]')].map(e=>e.getAttribute('href'))}),a=pick(parse(${JSON.stringify(normalizedOld)})),c=pick(parse(${JSON.stringify(source)}));return Object.fromEntries(Object.keys(a).map(k=>[k,{same:JSON.stringify(a[k])===JSON.stringify(c[k]),before:a[k].length,after:c[k].length}]))})()`);
  assert(Object.values(report.preserved).every(r=>r.same),JSON.stringify(report.preserved));
  report.headers=await b.evaluate(`[...document.querySelectorAll('#snr-correlated-groups th')].slice(2).map(e=>e.textContent.trim())`);assert.deepEqual(report.headers,replacements.map(r=>r[1]));
  for(const scripts of[true,false])for(const width of[320,1440])for(const theme of['light','dark','warm'])for(const size of[100,200]){
   await b.viewport(width,1100);await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.navigate(base+route);
   const palette=await b.evaluate('('+applyReviewTheme.toString()+')('+JSON.stringify(theme)+','+scripts+')');
   await b.evaluate(`document.documentElement.style.fontSize='${size}%';document.fonts.ready`);
   // Font readiness precedes the resize observer that updates overflow focus.
   await b.pause(250);
   const prose=await b.evaluate(`(()=>{window.__layout=[...document.querySelectorAll('.content p')].find(p=>p.querySelector('a[href$="/muon.py#L271-L299"]'));if(!__layout)throw Error('Layout paragraph missing');__layout.dataset.reviewLayout='true';window.__intro=document.getElementById('optimizer-assignment').nextElementSibling;window.__entry=[...__intro.querySelectorAll('a[href]')].find(a=>decodeURIComponent(a.getAttribute('href'))==='#113-官方训练代码从哪里进入');if(!__entry)throw Error('Official-entry backlink missing');const tables=[...document.querySelectorAll('.content table')];const module=tables.find(t=>t.tHead?.rows[0]?.cells[0]?.textContent.trim()==='配置名称');return{text:__layout.textContent.trim(),visible:__layout.checkVisibility(),width:__layout.clientWidth,tex:[...__layout.querySelectorAll('annotation')].map(e=>e.textContent),auxiliaryHeader:module?.tHead.rows[0].cells[4].textContent.trim(),overflowTablesFocusable:tables.every(t=>t.scrollWidth<=t.clientWidth+1||t.tabIndex===0)}})()`);
   assert(prose.visible&&prose.width>0&&prose.text.includes('(2,3,5,7)')&&prose.text.includes('(2,105)')&&prose.auxiliaryHeader==='辅助参数'&&prose.tex.length===2&&prose.overflowTablesFocusable,JSON.stringify(prose));
   const contrast=await b.evaluate('('+contrastAudit.toString()+')('+JSON.stringify('[data-review-layout]')+')');assert(contrast.checked>0);assert.deepEqual(contrast.issues,[]);
   assert(await b.evaluate('document.documentElement.scrollWidth<=document.documentElement.clientWidth+1'));
   if((width===1440&&size===100)||(width===320&&size===200)){
    await b.evaluate(`__layout.scrollIntoView({block:'start'});scrollBy(0,-document.querySelector('.hextra-nav-container nav').getBoundingClientRect().height-20)`);await b.pause(100);await b.screenshot(path.join(out,`layout-${scripts}-${width}-${theme}-${size}.png`));
   }
   assert(await b.evaluate('(__entry.focus(),document.activeElement===__entry)'));await b.key('Enter','Enter',13);await b.pause(180);
   const reached=await b.evaluate(`(()=>{window.__target=document.getElementById('113-官方训练代码从哪里进入');const r=__target.getBoundingClientRect();return{hash:decodeURIComponent(location.hash),top:r.top,bottom:r.bottom,nav:document.querySelector('.hextra-nav-container nav').getBoundingClientRect().bottom,height:innerHeight,headingFocus:document.activeElement===__target}})()`);
   assert(reached.hash==='#113-官方训练代码从哪里进入'&&reached.top>=reached.nav-1&&reached.bottom<=reached.height,JSON.stringify(reached));
   await b.key('Tab','Tab',9);const later=await b.evaluate(`window.__later=document.activeElement;({inContent:!!__later.closest('.content'),visible:__later.checkVisibility()})`);assert(later.inContent&&later.visible);await b.pause(200);assert(await b.evaluate('document.activeElement===__later'));
   report.layouts.push({scripts,width,theme,size,palette,prose,contrast,reached,laterKeyboardFocusPreserved:true});
  }
  assert.deepEqual(b.exceptions,[]);console.log(JSON.stringify({layouts:report.layouts.length,preservedDisclosures:3,exceptions:0}));
 }finally{fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
