// Native print exports of the Pion article. Run serially on a dedicated CDP endpoint.
const assert=require('node:assert/strict'),crypto=require('node:crypto'),fs=require('node:fs'),path=require('node:path');
const {execFileSync}=require('node:child_process');
const {connect}=require('../../../scripts/lib/browser.cjs');
const [base,portText,out]=process.argv.slice(2),route='/posts/ai/pion-muon-vla-rlvr/';
(async()=>{
 fs.mkdirSync(out,{recursive:true});const b=await connect(Number(portText)),report={pdfs:[],exceptions:b.exceptions};
 try{
  const response=await fetch(base+route);assert(response.ok);const sourceHTML=await response.text();
  report.sourceHtmlSha256=crypto.createHash('sha256').update(sourceHTML).digest('hex');report.cacheControl=response.headers.get('cache-control');
  for(const [scripts,size,theme,paper]of [[true,100,'light','a4'],[true,200,'dark','a4'],[false,100,'warm','a4'],[false,200,'warm','a4'],[true,200,'light','letter']]){
   await b.viewport(1440);await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.navigate(base+route);
   await b.evaluate(`document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='${size}%';Promise.all([document.fonts.ready,...[...document.querySelectorAll('.content img')].map(i=>{i.loading='eager';return i.decode()})])`);
   const sourceCheck=await b.evaluate(`(()=>{const expected=new DOMParser().parseFromString(${JSON.stringify(sourceHTML)},'text/html');const clean=e=>{const c=e.cloneNode(true);c.querySelectorAll('.blog-math-scroll-hint').forEach(n=>n.remove());return c.textContent.trim()};const pick=d=>({prose:[...d.querySelectorAll('.content p,.content details>summary')].filter(e=>!e.closest('[data-pion-spectrum]')).map(clean),headings:[...d.querySelectorAll('.content h2,.content h3')].map(e=>[e.id,e.textContent.trim()]),formulas:[...d.querySelectorAll('.content annotation[encoding="application/x-tex"]')].map(e=>e.textContent),code:[...d.querySelectorAll('.hextra-code-block code')].map(e=>e.textContent),figures:[...d.querySelectorAll('.content figure.article-figure')].map(e=>[e.id,e.querySelector('img')?.getAttribute('src'),e.querySelector('figcaption')?.textContent.trim()]),styles:[...d.querySelectorAll('link[rel="stylesheet"]')].map(e=>e.getAttribute('href')),entries:[...d.querySelectorAll('.content a[href="#snr-estimation"],.content a[href="#flat-rank-study"],.content a[href="#vla-padding-recovery"]')].map(e=>e.getAttribute('href'))});const a=pick(expected),c=pick(document);return Object.fromEntries(Object.keys(a).map(k=>[k,{same:JSON.stringify(a[k])===JSON.stringify(c[k]),expectedCount:a[k].length,actualCount:c[k].length}]))})()`);
   assert(Object.values(sourceCheck).every(r=>r.same),JSON.stringify(sourceCheck));
   await b.evaluate(`document.querySelector('#vla-padding-recovery>summary').focus();window.__printBefore={scroll:scrollY,open:[...document.querySelectorAll('.content details')].map(d=>d.open),active:document.activeElement};window.__printInk=[];window.addEventListener('beforeprint',()=>{const s=getComputedStyle(document.activeElement);__printInk.push({outline:s.outlineStyle,shadow:s.boxShadow,print:matchMedia('print').matches})});`);await b.pause(100);
   const result=await b.cdp('Page.printToPDF',{printBackground:true,paperWidth:paper==='letter'?8.5:8.27,paperHeight:paper==='letter'?11:11.69,marginTop:.4,marginBottom:.4,marginLeft:.4,marginRight:.4,generateTaggedPDF:true});
   const file=path.join(out,`pion-${scripts}-${size}-${theme}-${paper}.pdf`);fs.writeFileSync(file,Buffer.from(result.data,'base64'));await b.pause(180);
   const restored=await b.evaluate(`({scrollError:Math.abs(scrollY-__printBefore.scroll),focus:document.activeElement===__printBefore.active,details:JSON.stringify([...document.querySelectorAll('.content details')].map(d=>d.open))===JSON.stringify(__printBefore.open)})`);
   assert(restored.focus&&restored.details&&restored.scrollError<=1,JSON.stringify({scripts,size,theme,paper,restored}));
   // Chrome can dispatch beforeprint before applying print media. Retain
   // these timing observations; printed ink is checked in the actual PDF
   // and in the separate print-media geometry/focus regression suite.
   const beforeprintStyles=await b.evaluate('__printInk');if(scripts)assert(beforeprintStyles.length);else assert.equal(beforeprintStyles.length,0);
   const info=execFileSync('pdfinfo',[file],{encoding:'utf8'});assert(/Tagged:\s+yes/.test(info));
   const text=execFileSync('pdftotext',['-layout',file,'-'],{encoding:'utf8',maxBuffer:8*1024*1024});
   for(const token of ['pion_snr.py','snr-result.json','7.491901','flat_spectrum.py','probe_official_vla_padding.py','2.000828427','243'])assert(text.includes(token),'Missing PDF text: '+token);
   const compact=text.replace(/[\s\u200b]/g,'');for(const phrase of['两个相反梯度','同步后各进程上的同一梯度副本'])assert(compact.includes(phrase),'Missing current prose: '+phrase);
   fs.writeFileSync(file.replace(/\.pdf$/,'.txt'),text);
   report.pdfs.push({scripts,size,theme,paper,file,pages:Number(info.match(/Pages:\s+(\d+)/)[1]),tagged:true,restored,beforeprintStyles,sourceCheck,scriptExecutionDisabledEntireCase:!scripts});
  }
  assert.deepEqual(b.exceptions,[]);console.log(JSON.stringify(report));
 }finally{fs.writeFileSync(path.join(out,'exports.json'),JSON.stringify(report,null,2)+'\n');await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
