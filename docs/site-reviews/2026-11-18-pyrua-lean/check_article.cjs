#!/usr/bin/env node
// Serial article reading, keyboard, fallback and native-print checks.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
const {connect}=require('../../../scripts/lib/browser.cjs');
const {contrastAudit}=require('../../../scripts/lib/contrast.cjs');
const arithmetic=require('../../../assets/js/pyrua-accounting.js');
const base=process.argv[2].replace(/\/$/,''),port=Number(process.argv[3]),out=process.argv[4];
const flag=name=>process.argv.includes(name),option=name=>process.argv[process.argv.indexOf(name)+1];
const route='/posts/ai/pyrua-lean/',forced=flag('--forced-colors');
const behaviorOnly=flag('--behavior-only'),screenOnly=flag('--screen-only');
const printOnly=flag('--print-only')||flag('--letter-only')||(forced&&!behaviorOnly);
const axePath=flag('--axe')?option('--axe'):null;
let axe='';
if(axePath){axe=fs.readFileSync(axePath,'utf8');assert.equal(crypto.createHash('sha256').update(axe).digest('hex'),'20c09fe157a8a34a30e241aaa1fcdade657734f08ab379ecfbeb7d45cc46e878');}
fs.mkdirSync(out,{recursive:true});
const report={reading:[],semantics:[],behavior:[],fallbacks:[],pdfs:[],screenshots:[]};
let b;
const save=()=>fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');
async function wait(expression){for(let i=0;i<150;i++){if(await b.evaluate(expression))return;await b.pause(100);}throw Error('Timed out: '+expression);}
async function load(width,theme,size,scripts=true,spacing=false){
 await b.viewport(width,width<768?900:1100);
 await b.cdp('Emulation.setEmulatedMedia',{media:'',features:forced?[{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:theme==='dark'?'dark':'light'}]:[]});
 await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});
 await b.navigate(base+route);await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});
 await b.evaluate(`document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add(${JSON.stringify(theme)});document.documentElement.style.colorScheme=${JSON.stringify(theme==='dark'?'dark':'light')};document.documentElement.style.fontSize='${size}%';document.documentElement.style.letterSpacing=${spacing?"'.12em'":"''"};document.documentElement.style.wordSpacing=${spacing?"'.16em'":"''"};Promise.all([document.fonts.ready,...[...document.querySelectorAll('.content img')].map(i=>{i.loading='eager';return i.decode()})])`);
 await b.pause(180);
}
async function semantic(label){
 if(!axe)return;
 await b.evaluate(axe);
 const result=await b.evaluate(`axe.run(document.querySelector('main'),{rules:{'color-contrast':{enabled:false}}}).then(r=>({violations:r.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>({html:n.html,summary:n.failureSummary}))})),incomplete:r.incomplete.map(v=>({id:v.id,count:v.nodes.length,nodes:v.nodes.map(n=>({html:n.html,reasons:[...n.any,...n.all,...n.none].map(c=>({id:c.id,message:c.message}))}))}))}))`);
 report.semantics.push({label,...result});assert.equal(result.violations.length,0,JSON.stringify(result));
}
async function screens(){
 for(const scripts of [true,false])for(const width of [320,390,768,1440])for(const theme of ['light','dark','warm'])for(const size of [100,200])for(const spacing of [false,true]){
  await load(width,theme,size,scripts,spacing);
  const measured=await b.evaluate(`(()=>{
   const article=document.querySelector('.content'),bounds=article.getBoundingClientRect();
   const math=[...article.querySelectorAll('.katex-display')],tables=[...article.querySelectorAll('table')];
   const scrollers=[...math,...tables].filter(e=>e.scrollWidth>e.clientWidth+2);
   return {overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+2,
    brokenImages:[...article.querySelectorAll('img')].filter(i=>!i.complete||!i.naturalWidth).map(i=>i.src),
    focusIssues:scrollers.filter(e=>e.tabIndex!==0).map(e=>({tag:e.tagName,class:e.className})),
    inlineMath:[...article.querySelectorAll('.katex')].filter(e=>!e.closest('.katex-display,table,pre,code')).filter(e=>{const r=e.getBoundingClientRect();return r.left<bounds.left-2||r.right>bounds.right+2}).map(e=>e.textContent.slice(0,80)),
    labReady:document.querySelector('[data-pyrua-accounting]').dataset.ready==='true',
    math:math.length,tables:tables.length,overflowRegions:scrollers.length,
    contrast:(${contrastAudit.toString()})('.content')};})()`);
  const row={scripts,width,theme,size,spacing,...measured};report.reading.push(row);
  assert(!measured.overflow,JSON.stringify(row));assert.deepEqual(measured.brokenImages,[]);
  assert.deepEqual(measured.focusIssues,[],JSON.stringify(row));assert.deepEqual(measured.inlineMath,[],JSON.stringify(row));
  assert.equal(measured.labReady,scripts);assert.equal(measured.contrast.issues.length,0,JSON.stringify(row));
  if(scripts&&width===320&&theme==='dark'&&size===200&&spacing){
   await b.evaluate(`document.querySelector('[data-pyrua-accounting]').scrollIntoView({block:'start'})`);
   const file=path.join(out,'lab-320-dark-200-spacing.png');await b.screenshot(file);report.screenshots.push(file);
  }
  if(report.reading.length%24===0){console.log('Reading layouts '+report.reading.length);save();}
 }
 await load(1440,'light',100,true);await semantic('enhanced');
 await load(320,'dark',200,false,true);await semantic('scripts disabled, enlarged');
 for(const id of ['handoff','placement','token-accounting','recovery-examples','robodojo','coder','source-probe','design-checklist','self-check','fig-pyrua-runtime','fig-pyrua-outcomes']){
  await load(320,'light',200,true);
  await b.cdp('Page.navigate',{url:base+route+'#'+id});await b.pause(200);
  const anchor=await b.evaluate(`(()=>{const target=document.getElementById(${JSON.stringify(id)});const e=target.closest('h2,h3')||target;const r=e.getBoundingClientRect();return {exists:!!e,top:r.top,bottom:r.bottom,height:innerHeight}})()`);
  assert(anchor.exists&&anchor.top>=45&&anchor.top<anchor.height-40,JSON.stringify({id,...anchor}));report.behavior.push({case:'native fragment anchor',id,...anchor});
 }
 await b.viewport(1300,1100);await b.navigate(base+route+'assets/main-outcomes.svg');
 const chart=path.join(out,'main-outcomes.png');await b.screenshot(chart);report.screenshots.push(chart);
}
async function replaceInput(name,text){
 await b.evaluate(`document.querySelector('[data-pyrua-accounting] input[name="${name}"]').focus()`);
 await b.cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'a',code:'KeyA',windowsVirtualKeyCode:65,modifiers:2});
 await b.cdp('Input.dispatchKeyEvent',{type:'keyUp',key:'a',code:'KeyA',windowsVirtualKeyCode:65,modifiers:2});
 await b.key('Backspace','Backspace',8);if(text)await b.cdp('Input.insertText',{text});
}
async function values(){return b.evaluate(`Object.fromEntries([...document.querySelectorAll('[data-pyrua-accounting] output')].map(e=>[e.dataset.value,e.textContent]))`);}
async function behavior(){
 await load(320,'dark',200,true,true);
 await b.evaluate(`document.querySelector('[data-pyrua-accounting]').scrollIntoView({block:'start'});document.querySelector('[data-preset="default"]').focus()`);
 await b.key('Tab','Tab',9);assert.equal(await b.evaluate(`document.activeElement.dataset.preset`),'single');
 await b.key('Tab','Tab',9);assert.equal(await b.evaluate('document.activeElement.name'),'operations');
 await b.key('ArrowUp','ArrowUp',38);assert.equal(await b.evaluate('document.activeElement.value'),'13');
 assert(Object.values(await values()).every(v=>v==='—'));
 await b.key('Enter','Enter',13);assert.equal(await b.evaluate('document.activeElement.name'),'operations');
 let expected=arithmetic.calculate({...arithmetic.defaults,operations:13});
 for(const [k,v]of Object.entries(await values()))assert.equal(Number(v),expected[k]);
 report.behavior.push({case:'native number arrow, stale clearing, Enter, focus',passed:true});
 await replaceInput('group','0');await b.key('Enter','Enter',13);
 assert.equal(await b.evaluate(`document.activeElement.getAttribute('aria-invalid')`),'true');
 assert.equal(await b.evaluate('document.activeElement.name'),'group');assert(Object.values(await values()).every(v=>v==='—'));
 await semantic('invalid input');
 await replaceInput('group','3');await b.key('Enter','Enter',13);
 assert.equal(await b.evaluate(`document.querySelector('[data-pyrua-accounting]').dataset.result`),'valid');
 await b.key('Tab','Tab',9);assert.equal(await b.evaluate('document.activeElement.name'),'reply');
 await b.evaluate(`document.querySelector('[data-preset="default"]').focus()`);await b.key(' ','Space',32);
 assert.equal(await b.evaluate('document.activeElement.dataset.preset'),'default');
 expected=arithmetic.calculate(arithmetic.defaults);for(const[k,v]of Object.entries(await values()))assert.equal(Number(v),expected[k]);
 report.behavior.push({case:'invalid recovery, Tab, Space preset, focus',passed:true});
 // Every native table/formula that overflows receives actual ArrowRight input.
 const scrollers=await b.evaluate(`[...document.querySelectorAll('.content table,.content .katex-display')].map((e,i)=>({i,overflow:e.scrollWidth>e.clientWidth+3})).filter(r=>r.overflow).map(r=>r.i)`);
 for(const index of scrollers){
  const before=await b.evaluate(`(()=>{const e=[...document.querySelectorAll('.content table,.content .katex-display')][${index}];e.scrollLeft=0;e.focus();return e.scrollLeft})()`);
  for(let j=0;j<4;j++)await b.key('ArrowRight','ArrowRight',39);await b.pause(150);
  const after=await b.evaluate(`[...document.querySelectorAll('.content table,.content .katex-display')][${index}].scrollLeft`);
  assert(after>before,JSON.stringify({index,before,after}));
 }
 report.behavior.push({case:'actual keyboard horizontal scrolling',regions:scrollers.length});
 await load(390,'light',100,true);
 await b.evaluate(`document.querySelector('.post-image-link').focus();document.querySelector('.post-image-link').click()`);
 await wait(`!!document.querySelector('dialog[open]')`);
 const references=await b.evaluate(`({viewerIds:document.querySelectorAll('#post-image-viewer').length,allResolved:[...document.querySelectorAll('.post-image-link[aria-controls]')].every(e=>!!document.getElementById(e.getAttribute('aria-controls'))),nativeDialog:document.getElementById('post-image-viewer').localName==='dialog'})`);
 assert.equal(references.viewerIds,1);assert(references.allResolved&&references.nativeDialog);
 report.behavior.push({case:'manual review of closed-dialog axe incomplete references',...references});
 await b.key('Escape','Escape',27);
 assert.equal(await b.evaluate(`document.activeElement.classList.contains('post-image-link')`),true);
 report.behavior.push({case:'image viewer open/Escape/focus',passed:true});
 for(const resource of ['interface_sanity.py','probe_runtime.py','reference-output.json','README.txt','assets/main-outcomes.svg']){
  const response=await fetch(base+route+resource);assert.equal(response.status,200);assert((await response.arrayBuffer()).byteLength>100);
 }
 // Block just this lab script: native static default must remain readable.
 await b.cdp('Network.setBlockedURLs',{urls:['*pyrua-accounting*.js*']});await load(320,'light',200,true);
 assert.equal(await b.evaluate(`document.querySelector('[data-controls]').hidden`),true);
 for(const[k,v]of Object.entries(await values()))assert.equal(Number(v),arithmetic.calculate(arithmetic.defaults)[k]);
 report.fallbacks.push({case:'blocked lab script, static default',passed:true});
 const src=await b.evaluate(`document.querySelector('script[src*="pyrua-accounting"]').src`);
 await b.evaluate(`window.__readingTarget=document.getElementById('main-results');__readingTarget.tabIndex=-1;__readingTarget.focus();__readingTarget.scrollIntoView({block:'start'});window.__readingY=__readingTarget.getBoundingClientRect().top`);
 await b.cdp('Network.setBlockedURLs',{urls:[]});
 await b.evaluate(`fetch(${JSON.stringify(src)}).then(r=>r.text()).then(code=>{const script=document.createElement('script');script.textContent=code;document.body.append(script)})`);
 await wait(`document.querySelector('[data-pyrua-accounting]').dataset.ready==='true'`);
 const late=await b.evaluate(`({focus:document.activeElement===__readingTarget,delta:Math.abs(__readingTarget.getBoundingClientRect().top-__readingY)})`);
 assert(late.focus&&late.delta<3,JSON.stringify(late));report.fallbacks.push({case:'late enhancement preserves reader',...late});
 await b.cdp('Network.setBlockedURLs',{urls:['*pyrua-accounting*.js*']});await load(320,'light',200,true);
 let labSource=await b.evaluate(`document.querySelector('script[src*="pyrua-accounting"]').src`);
 await b.evaluate(`document.querySelector('input[name="group"]').value='0'`);
 await b.cdp('Network.setBlockedURLs',{urls:[]});
 await b.evaluate(`fetch(${JSON.stringify(labSource)}).then(r=>r.text()).then(code=>{const script=document.createElement('script');script.textContent=code;document.body.append(script)})`);
 await wait(`document.querySelector('[data-pyrua-accounting]').dataset.ready==='true'`);
 assert.equal(await b.evaluate(`document.querySelector('[data-controls]').hidden`),false);
 assert.equal(await b.evaluate(`document.querySelector('[data-pyrua-accounting]').dataset.result`),'invalid');
 await replaceInput('group','3');await b.key('Enter','Enter',13);
 assert.equal(await b.evaluate(`document.querySelector('[data-pyrua-accounting]').dataset.result`),'valid');
 report.fallbacks.push({case:'browser-restored invalid input remains correctable',passed:true});
 await b.cdp('Network.setBlockedURLs',{urls:['*pyrua-accounting*.js*']});await load(320,'light',200,true);
 labSource=await b.evaluate(`document.querySelector('script[src*="pyrua-accounting"]').src`);
 await b.cdp('Network.setBlockedURLs',{urls:[]});
 await b.evaluate(`fetch(${JSON.stringify(labSource)}).then(r=>r.text()).then(code=>{const script=document.createElement('script');script.textContent='(()=>{const Number={};'+code+'})();';document.body.append(script)})`);
 assert.equal(await b.evaluate(`document.querySelector('[data-controls]').hidden`),true);
 for(const[k,v]of Object.entries(await values()))assert.equal(Number(v),arithmetic.calculate(arithmetic.defaults)[k]);
 report.fallbacks.push({case:'unavailable numeric feature preserves static default',passed:true});
 for(const entrance of ['/','/zh-cn/']){
  await b.navigate(base+entrance);await b.evaluate(`document.querySelector('[data-search-open]').click()`);
  await wait(`!!document.querySelector('dialog[open] .hextra-search-input')`);
  await b.evaluate(`const input=document.querySelector('.hextra-search-input');input.value='PyRUA';input.dispatchEvent(new Event('input',{bubbles:true}))`);
  await wait(`!!document.querySelector('#hextra-search-results a[href="${route}"]')`);
  report.behavior.push({case:'new article searchable',entrance,passed:true});await b.key('Escape','Escape',27);
 }
}
async function printing(){
 for(const paper of(flag('--letter-only')?['Letter']:['A4','Letter']))for(const theme of ['light','dark','warm'])for(const size of [100,200])for(const scripts of [true,false]){
  await load(size===200?320:1440,theme,size,scripts);
  const configured={operations:17,group:4,reply:1237,retained:37};
  if(scripts)await b.evaluate(`(()=>{const f=document.querySelector('[data-pyrua-accounting] form');for(const[n,v]of Object.entries(${JSON.stringify(configured)}))f.elements.namedItem(n).value=v;f.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}))})()`);
  const cards=await b.evaluate(`[...document.querySelectorAll('.pyrua-accounting__results>div')].map(e=>({label:e.querySelector('dt').textContent,value:e.querySelector('output').textContent}))`);
  await b.evaluate(`window.__printFocus=${scripts}?document.querySelector('[data-pyrua-accounting] input[name="retained"]'):(document.getElementById('token-accounting').closest('h2')||document.getElementById('token-accounting'));__printFocus.tabIndex=__printFocus.tabIndex<0?-1:__printFocus.tabIndex;__printFocus.scrollIntoView({block:'center',behavior:'instant'});__printFocus.focus({preventScroll:true});new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))`);
  await b.pause(120);
  const snapshot=`({width:innerWidth,scroll:scrollY,height:document.documentElement.scrollHeight,targetTop:__printFocus.getBoundingClientRect().top,focus:document.activeElement===__printFocus,ready:document.querySelector('[data-pyrua-accounting]').dataset.ready||'',inputs:[...document.querySelectorAll('[data-pyrua-accounting] input')].map(e=>e.value),outputs:[...document.querySelectorAll('[data-pyrua-accounting] output')].map(e=>e.textContent)})`;
  const before=await b.evaluate(snapshot);
  const filename=`${paper}-${theme}-${size}-${scripts?'scripts':'native'}${forced?'-forced':''}.pdf`;
  const pdf=await b.cdp('Page.printToPDF',{printBackground:true,generateTaggedPDF:true,preferCSSPageSize:false,
   paperWidth:paper==='A4'?8.2677:8.5,paperHeight:paper==='A4'?11.6929:11,marginTop:.4,marginBottom:.4,marginLeft:.4,marginRight:.4});
  const file=path.join(out,filename);fs.writeFileSync(file,Buffer.from(pdf.data,'base64'));
  const text=execFileSync('pdftotext',['-layout',file,'-'],{encoding:'utf8',maxBuffer:20*1024*1024});
  const pages=text.split('\f').map(p=>p.replace(/\s+/g,''));
  const bbox=execFileSync('pdftotext',['-bbox',file,'-'],{encoding:'utf8',maxBuffer:20*1024*1024});
  const pagination=JSON.parse(execFileSync('python3',['-B',path.join(__dirname,'check_pdf_cards.py'),JSON.stringify(cards)],{input:bbox,encoding:'utf8',maxBuffer:5*1024*1024}));
  assert(pagination.passed,JSON.stringify({file,pagination}));
  for(const required of ['1,061','2,100','442','502','3840','1.047'])assert(text.replace(/,/g,'').includes(required.replace(/,/g,'')),JSON.stringify({file,required}));
  const info=execFileSync('pdfinfo',[file],{encoding:'utf8'});assert(/Tagged:\s+yes/.test(info));
  const sameState=state=>['scroll','height','targetTop'].every(k=>Math.abs(state[k]-before[k])<=1)&&
   ['width','focus','ready','inputs','outputs'].every(k=>JSON.stringify(state[k])===JSON.stringify(before[k]));
  let after;for(let i=0;i<30;i++){after=await b.evaluate(snapshot);if(sameState(after))break;await b.pause(50);}
  assert(sameState(after),JSON.stringify({file,before,after}));report.pdfs.push({file,paper,theme,size,scripts,forced,pages:pages.filter(p=>p.length).length,cards,pagination,stateRestored:true});
  console.log('PDF '+filename);save();
 }
}
(async()=>{
 b=await connect(port);try{
  if(!printOnly&&!behaviorOnly)await screens();
  if(!printOnly&&!screenOnly)await behavior();
  if(!screenOnly&&!behaviorOnly)await printing();
  assert.equal(b.exceptions.length,0,JSON.stringify(b.exceptions));report.exceptions=b.exceptions;save();
  console.log(JSON.stringify({reading:report.reading.length,behavior:report.behavior.length,fallbacks:report.fallbacks.length,semantics:report.semantics.length,pdfs:report.pdfs.length,exceptions:b.exceptions.length}));
 }finally{save();await b.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
