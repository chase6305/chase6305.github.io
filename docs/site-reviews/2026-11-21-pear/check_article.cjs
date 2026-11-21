#!/usr/bin/env node
// Serial PEAR article reading, fallback, keyboard and native PDF checks.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
const {connect}=require('../../../scripts/lib/browser.cjs');
const {contrastAudit}=require('../../../scripts/lib/contrast.cjs');
const lab=require('../../../assets/js/pear-sampling.js');
const base=process.argv[2].replace(/\/$/,''),port=Number(process.argv[3]),out=process.argv[4];
const flag=name=>process.argv.includes(name),option=name=>process.argv[process.argv.indexOf(name)+1];
const route='/posts/ai/pear/',forced=flag('--forced-colors');
const screenOnly=flag('--screen-only'),behaviorOnly=flag('--behavior-only'),printOnly=flag('--print-only');
const report={reading:[],semantics:[],behavior:[],fallbacks:[],pdfs:[],screenshots:[]};
fs.mkdirSync(out,{recursive:true});
let axe='';
if(flag('--axe')){axe=fs.readFileSync(option('--axe'),'utf8');assert.equal(crypto.createHash('sha256').update(axe).digest('hex'),'20c09fe157a8a34a30e241aaa1fcdade657734f08ab379ecfbeb7d45cc46e878');}
const save=()=>fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');
let b;
async function wait(expression){for(let i=0;i<120;i++){if(await b.evaluate(expression))return;await b.pause(100);}throw Error('Timed out: '+expression);}
async function load(width,theme,size,scripts=true,spacing=false){
 await b.viewport(width,width<768?1000:1150);
 await b.cdp('Emulation.setEmulatedMedia',{media:'',features:forced?[{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:theme==='dark'?'dark':'light'}]:[]});
 await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});
 await b.navigate(base+route);await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});
 await b.evaluate(`document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add(${JSON.stringify(theme)});document.documentElement.style.colorScheme=${JSON.stringify(theme==='dark'?'dark':'light')};document.documentElement.style.fontSize='${size}%';document.documentElement.style.letterSpacing=${spacing?"'.12em'":"''"};document.documentElement.style.wordSpacing=${spacing?"'.16em'":"''"};Promise.all([document.fonts.ready,...[...document.querySelectorAll('.content img')].map(i=>{i.loading='eager';return i.decode()})])`);
 await b.pause(160);
}
async function semantic(label){
 if(!axe)return;await b.evaluate(axe);
 const r=await b.evaluate(`axe.run(document.querySelector('main'),{rules:{'color-contrast':{enabled:false}}}).then(r=>({violations:r.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>({html:n.html,summary:n.failureSummary}))})),incomplete:r.incomplete.map(v=>({id:v.id,nodes:v.nodes.map(n=>({html:n.html,summary:n.failureSummary}))}))}))`);
 report.semantics.push({label,...r});assert.equal(r.violations.length,0,JSON.stringify(r));
}
async function screens(){
 for(const scripts of [true,false])for(const width of [320,390,768,1440])for(const theme of ['light','dark','warm'])for(const size of [100,200])for(const spacing of [false,true]){
  await load(width,theme,size,scripts,spacing);
  const state=await b.evaluate(`(()=>{
   const a=document.querySelector('.content'),bounds=a.getBoundingClientRect();
   const regions=[...a.querySelectorAll('.katex-display,table')].filter(e=>e.scrollWidth>e.clientWidth+2);
   return {bodyOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+2,
    brokenImages:[...a.querySelectorAll('img')].filter(i=>!i.complete||!i.naturalWidth).map(i=>i.src),
    inaccessibleScrollers:regions.filter(e=>e.tabIndex!==0).map(e=>({tag:e.tagName,class:e.className})),
    inlineMath:[...a.querySelectorAll('.katex')].filter(e=>!e.closest('.katex-display,table,pre,code')).filter(e=>{const r=e.getBoundingClientRect();return r.left<bounds.left-2||r.right>bounds.right+2}).map(e=>e.querySelector('annotation')?.textContent||e.textContent.slice(0,70)),
    ready:document.querySelector('[data-pear-sampling]').dataset.ready==='true',
    contrast:(${contrastAudit.toString()})('.content')};})()`);
  report.reading.push({scripts,width,theme,size,spacing,...state});
  assert(!state.bodyOverflow,JSON.stringify(report.reading.at(-1)));assert.deepEqual(state.brokenImages,[]);
  assert.deepEqual(state.inaccessibleScrollers,[]);assert.deepEqual(state.inlineMath,[],JSON.stringify(report.reading.at(-1)));
  assert.equal(state.ready,scripts);assert.equal(state.contrast.issues.length,0,JSON.stringify(state.contrast));
  if(report.reading.length%24===0){console.log('Reading layouts '+report.reading.length);save();}
 }
 await load(1440,'light',100,true);await semantic('enhanced');
 await load(320,'dark',200,false,true);await semantic('native enlarged');
 for(const id of ['boundary','action','action-loop','robot-description','objective','sampling','task-mixture','curriculum','engine','evidence','real-world','latency','general-capabilities','reproduce','design','self-check','fig-pear-loop','fig-pear-groups','fig-pear-trials']){
  await load(320,'light',200,true);await b.cdp('Page.navigate',{url:base+route+'#'+id});await b.pause(150);
  const anchor=await b.evaluate(`(()=>{const t=document.getElementById(${JSON.stringify(id)});const r=(t.closest('h2,h3')||t).getBoundingClientRect();return {top:r.top,height:innerHeight}})()`);
  assert(anchor.top>=45&&anchor.top<anchor.height-40,JSON.stringify({id,...anchor}));
  report.behavior.push({case:'native chapter/figure fragment',id,...anchor});
 }
 for(const c of [{width:1440,theme:'light',size:100,id:'boundary',name:'architecture-desktop'},
                 {width:390,theme:'dark',size:200,id:'sampling',name:'sampling-mobile-large'},
                 {width:1440,theme:'light',size:100,id:'real-world',name:'trials-desktop'}]){
  await load(c.width,c.theme,c.size,true);await b.evaluate(`document.getElementById('${c.id}').scrollIntoView({block:'start'})`);
  const file=path.join(out,c.name+'.png');await b.screenshot(file);report.screenshots.push(file);
 }
 for(const name of ['group-signal','real-trials']){
  await b.viewport(1400,700);await b.navigate(base+route+'assets/'+name+'.svg');
  const file=path.join(out,name+'.png');await b.screenshot(file);report.screenshots.push(file);
 }
}
async function replaceInput(name,text){
 await b.evaluate(`document.querySelector('[data-pear-sampling] input[name="${name}"]').focus()`);
 await b.cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'a',code:'KeyA',windowsVirtualKeyCode:65,modifiers:2});
 await b.cdp('Input.dispatchKeyEvent',{type:'keyUp',key:'a',code:'KeyA',windowsVirtualKeyCode:65,modifiers:2});
 await b.key('Backspace','Backspace',8);if(text)await b.cdp('Input.insertText',{text});
}
async function values(){return b.evaluate(`Object.fromEntries([...document.querySelectorAll('[data-pear-sampling] output')].map(e=>[e.dataset.value,e.textContent]))`);}
function checkValues(actual,input){const r=lab.calculate(input);for(const[key,value]of Object.entries(actual))assert.equal(value,lab.formatResult(r,key),JSON.stringify({key,value,input}));}
async function behavior(){
 await load(320,'dark',200,true,true);
 await b.evaluate(`document.querySelector('[data-preset="default"]').focus()`);
 await b.key('Tab','Tab',9);assert.equal(await b.evaluate('document.activeElement.dataset.preset'),'rare');
 await b.key('Tab','Tab',9);assert.equal(await b.evaluate('document.activeElement.dataset.preset'),'balanced');
 await b.key('Tab','Tab',9);assert.equal(await b.evaluate('document.activeElement.name'),'probability');
 await b.key('ArrowUp','ArrowUp',38);assert.equal(await b.evaluate('document.activeElement.value'),'2.1');
 assert(Object.values(await values()).every(v=>v==='—'));await b.key('Enter','Enter',13);
 checkValues(await values(),{...lab.defaults,probability:2.1});
 const announcement=await b.evaluate(`document.querySelector('[data-status]').textContent`);
 assert(announcement.includes(lab.formatResult(lab.calculate({...lab.defaults,probability:2.1}),'mixed')));
 assert(announcement.includes(lab.formatResult(lab.calculate({...lab.defaults,probability:2.1}),'candidate_rollouts')));
 assert.equal(await b.evaluate('document.activeElement.name'),'probability');
 report.behavior.push({case:'native number, Tab order, stale clearing, Enter, focus',passed:true});
 await replaceInput('group','1');await b.key('Enter','Enter',13);
 assert.equal(await b.evaluate('document.activeElement.name'),'group');
 assert.equal(await b.evaluate(`document.activeElement.getAttribute('aria-invalid')`),'true');
 assert(Object.values(await values()).every(v=>v==='—'));await semantic('invalid group');
 await replaceInput('group','8');await b.key('Enter','Enter',13);
 await b.key('Tab','Tab',9);assert.equal(await b.evaluate('document.activeElement.name'),'groups');
 report.behavior.push({case:'invalid recovery preserves focus and later Tab',passed:true});
 for(const probability of ['0','100']){
  await replaceInput('probability',probability);await b.key('Enter','Enter',13);
  checkValues(await values(),{...lab.defaults,probability:Number(probability)});
  assert.equal((await values()).candidate_rollouts,'无混合组');
 }
 await b.evaluate(`document.querySelector('[data-preset="rare"]').focus()`);await b.key(' ','Space',32);
 checkValues(await values(),{...lab.defaults,probability:.4});
 report.behavior.push({case:'zero/one probability, Space preset, finite display',passed:true});
 const regions=await b.evaluate(`[...document.querySelectorAll('.content table,.content .katex-display')].map((e,i)=>({i,overflow:e.scrollWidth>e.clientWidth+3})).filter(r=>r.overflow).map(r=>r.i)`);
 for(const index of regions){
  await b.evaluate(`(()=>{const e=[...document.querySelectorAll('.content table,.content .katex-display')][${index}];e.scrollLeft=0;e.focus()})()`);
  for(let i=0;i<4;i++)await b.key('ArrowRight','ArrowRight',39);await b.pause(120);
  assert(await b.evaluate(`[...document.querySelectorAll('.content table,.content .katex-display')][${index}].scrollLeft>0`),String(index));
 }
 report.behavior.push({case:'actual keyboard formula/table scrolling',regions:regions.length});
 await load(390,'light',100,true);
 const refs=await b.evaluate(`({dialogs:document.querySelectorAll('#post-image-viewer').length,resolved:[...document.querySelectorAll('.post-image-link[aria-controls]')].every(e=>!!document.getElementById(e.getAttribute('aria-controls'))),images:document.querySelectorAll('.post-image-link').length})`);
 assert.equal(refs.dialogs,1);assert(refs.resolved);assert.equal(refs.images,3);
 for(let index=0;index<3;index++){
  await b.evaluate(`(()=>{const a=document.querySelectorAll('.post-image-link')[${index}];a.focus();a.click()})()`);
  await wait(`!!document.querySelector('dialog[open]')`);await b.key('Escape','Escape',27);
  assert(await b.evaluate(`document.activeElement===document.querySelectorAll('.post-image-link')[${index}]`));
 }
 report.behavior.push({case:'closed-dialog incomplete references manually resolved; all image open/Escape/focus',...refs});
 for(const file of ['pear_sanity.py','probe_pear_contracts.py','reference-output.json','README.txt']){
  const response=await fetch(base+route+file);assert.equal(response.status,200);assert((await response.arrayBuffer()).byteLength>100);
 }
 await b.cdp('Network.setBlockedURLs',{urls:['*pear-sampling*.js*']});await load(320,'light',200,true);
 assert(await b.evaluate(`document.querySelector('[data-controls]').hidden`));checkValues(await values(),lab.defaults);
 const src=await b.evaluate(`document.querySelector('script[src*="pear-sampling"]').src`);
 await b.evaluate(`window.__reading=document.getElementById('evidence');__reading.tabIndex=-1;__reading.focus();__reading.scrollIntoView({block:'start'});window.__readingTop=__reading.getBoundingClientRect().top`);
 await b.cdp('Network.setBlockedURLs',{urls:[]});
 await b.evaluate(`fetch(${JSON.stringify(src)}).then(r=>r.text()).then(code=>{const s=document.createElement('script');s.textContent=code;document.body.append(s)})`);
 await wait(`document.querySelector('[data-pear-sampling]').dataset.ready==='true'`);
 const late=await b.evaluate(`({focus:document.activeElement===__reading,delta:Math.abs(__reading.getBoundingClientRect().top-__readingTop)})`);
 assert(late.focus&&late.delta<3,JSON.stringify(late));report.fallbacks.push({case:'blocked/default then late enhancement',...late});
 await b.cdp('Network.setBlockedURLs',{urls:['*pear-sampling*.js*']});await load(320,'light',200,true);
 const again=await b.evaluate(`document.querySelector('script[src*="pear-sampling"]').src`);
 await b.evaluate(`document.querySelector('input[name="group"]').value='1'`);
 await b.cdp('Network.setBlockedURLs',{urls:[]});
 await b.evaluate(`fetch(${JSON.stringify(again)}).then(r=>r.text()).then(code=>{const s=document.createElement('script');s.textContent=code;document.body.append(s)})`);
 await wait(`document.querySelector('[data-pear-sampling]').dataset.ready==='true'`);
 assert.equal(await b.evaluate(`document.querySelector('[data-controls]').hidden`),false);
 await replaceInput('group','8');await b.key('Enter','Enter',13);checkValues(await values(),lab.defaults);
 report.fallbacks.push({case:'restored invalid input remains correctable',passed:true});
 await b.cdp('Network.setBlockedURLs',{urls:['*pear-sampling*.js*']});await load(320,'light',200,true);
 await b.cdp('Network.setBlockedURLs',{urls:[]});
 await b.evaluate(`fetch(${JSON.stringify(again)}).then(r=>r.text()).then(code=>{const s=document.createElement('script');s.textContent='(()=>{const Math={};'+code+'})();';document.body.append(s)})`);
 assert(await b.evaluate(`document.querySelector('[data-controls]').hidden`));checkValues(await values(),lab.defaults);
 report.fallbacks.push({case:'missing numeric feature retains native defaults',passed:true});
 for(const entrance of ['/','/zh-cn/']){
  await b.navigate(base+entrance);await b.evaluate(`document.querySelector('[data-search-open]').click()`);
  await wait(`!!document.querySelector('dialog[open] .hextra-search-input')`);
  await b.evaluate(`const i=document.querySelector('.hextra-search-input');i.value='PEAR';i.dispatchEvent(new Event('input',{bubbles:true}))`);
  await wait(`!!document.querySelector('#hextra-search-results a[href="${route}"]')`);
  report.behavior.push({case:'article found in search',entrance,passed:true});await b.key('Escape','Escape',27);
 }
}
async function printing(){
 for(const paper of ['A4','Letter'])for(const theme of ['light','dark','warm'])for(const size of [100,200])for(const scripts of [true,false]){
  await load(size===200?320:1440,theme,size,scripts);
  if(scripts)await b.evaluate(`const f=document.querySelector('[data-pear-sampling] form');for(const[n,v]of Object.entries({probability:.4,group:16,groups:64}))f.elements.namedItem(n).value=v;f.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}))`);
  const cards=await b.evaluate(`[...document.querySelectorAll('.pear-sampling__results>div')].map(e=>({label:e.querySelector('dt').textContent,value:e.querySelector('output').textContent}))`);
  await b.evaluate(`window.__printTarget=${scripts}?document.querySelector('input[name="groups"]'):document.getElementById('sampling');__printTarget.tabIndex=__printTarget.tabIndex<0?-1:__printTarget.tabIndex;__printTarget.scrollIntoView({block:'center',behavior:'instant'});__printTarget.focus({preventScroll:true});new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))`);
  await b.pause(100);
  const snapshot=`({width:innerWidth,scroll:scrollY,height:document.documentElement.scrollHeight,targetTop:__printTarget.getBoundingClientRect().top,focus:document.activeElement===__printTarget,inputs:[...document.querySelectorAll('[data-pear-sampling] input')].map(e=>e.value),outputs:[...document.querySelectorAll('[data-pear-sampling] output')].map(e=>e.textContent)})`;
  const before=await b.evaluate(snapshot);
  const file=path.join(out,`${paper}-${theme}-${size}-${scripts?'scripts':'native'}${forced?'-forced':''}.pdf`);
  const pdf=await b.cdp('Page.printToPDF',{printBackground:true,generateTaggedPDF:true,preferCSSPageSize:false,paperWidth:paper==='A4'?8.2677:8.5,paperHeight:paper==='A4'?11.6929:11,marginTop:.4,marginBottom:.4,marginLeft:.4,marginRight:.4});
  fs.writeFileSync(file,Buffer.from(pdf.data,'base64'));
  const bbox=execFileSync('pdftotext',['-bbox',file,'-'],{encoding:'utf8',maxBuffer:20*1024*1024});
  const pagination=JSON.parse(execFileSync('python3',['-B',path.join(__dirname,'../2026-11-18-pyrua-lean/check_pdf_cards.py'),JSON.stringify(cards)],{input:bbox,encoding:'utf8',maxBuffer:5*1024*1024}));
  assert(pagination.passed,JSON.stringify({file,pagination}));
  assert(/Tagged:\s+yes/.test(execFileSync('pdfinfo',[file],{encoding:'utf8'})));
  let after;for(let i=0;i<30;i++){after=await b.evaluate(snapshot);if(Math.abs(after.targetTop-before.targetTop)<=1)break;await b.pause(50);}
  for(const key of ['width','focus','inputs','outputs'])assert.deepEqual(after[key],before[key],JSON.stringify({file,key,before,after}));
  if(scripts)for(const key of ['scroll','height','targetTop'])assert(Math.abs(after[key]-before[key])<=1,JSON.stringify({file,key,before,after}));
  report.pdfs.push({file,paper,theme,size,scripts,forced,cards,pagination,scrollDelta:after.scroll-before.scroll,stateRestored:true});
  console.log('PDF '+path.basename(file));save();
 }
}
(async()=>{
 b=await connect(port);try{
  if(!printOnly&&!behaviorOnly)await screens();
  if(!printOnly&&!screenOnly)await behavior();
  if(!screenOnly&&!behaviorOnly)await printing();
  assert.equal(b.exceptions.length,0,JSON.stringify(b.exceptions));report.exceptions=b.exceptions;save();
  console.log(JSON.stringify({reading:report.reading.length,behavior:report.behavior.length,fallbacks:report.fallbacks.length,pdfs:report.pdfs.length,exceptions:b.exceptions.length}));
 }finally{save();await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
