#!/usr/bin/env node
// Run serially against loopback CDP; native PDFs require Poppler and Python.
const assert=require('node:assert/strict'),crypto=require('node:crypto'),fs=require('node:fs'),path=require('node:path');
const {execFileSync}=require('node:child_process');
const {connect}=require('./lib/browser.cjs');
const {contrastAudit}=require('./lib/contrast.cjs');
const core=require('../assets/js/pion-spectrum.js');
const base=(process.argv[2]||'http://127.0.0.1:13351').replace(/\/+$/,''),port=Number(process.argv[3]||9236),output=process.argv[4]||'/tmp/chase-pion-spectrum';
const option=name=>{const index=process.argv.indexOf(name);return index<0?null:process.argv[index+1];};
const axePath=option('--axe');assert(axePath,'Supply the pinned axe-core 4.14.0 file with --axe');
const axe=fs.readFileSync(axePath,'utf8');assert.equal(crypto.createHash('sha256').update(axe).digest('hex'),'20c09fe157a8a34a30e241aaa1fcdade657734f08ab379ecfbeb7d45cc46e878');
const flag=name=>process.argv.includes(name),forced=flag('--forced-colors'),letter=flag('--letter-only'),readingOnly=flag('--reading-only'),behaviorOnly=flag('--behavior-only');
const nativeOnly=flag('--native-only'),printOnly=flag('--print-only')||forced||letter||nativeOnly,route='/posts/ai/pion-muon-vla-rlvr/';
const near=(a,b)=>assert(Number.isFinite(a)&&Math.abs(a-b)<=Math.max(1e-300,Math.abs(b)*6e-6),JSON.stringify({a,b}));
const pdfInspect=String.raw`
import json,sys,xml.etree.ElementTree as ET
root=ET.fromstring(sys.stdin.buffer.read());ns={'x':'http://www.w3.org/1999/xhtml'}
cards=json.loads(sys.argv[1]);compact=lambda s:''.join(s.split()).replace('\u2212','-')
pages=[[dict(text=w.text or '',**{k:float(v)for k,v in w.attrib.items()})for w in p.findall('.//x:word',ns)]for p in root.findall('.//x:page',ns)]
def matches(words,text):
 text=compact(text)
 def walk(group,remaining):
  if not remaining:
   yield {k:op(w[k]for w in group)for k,op in [('xMin',min),('yMin',min),('xMax',max),('yMax',max)]};return
  if len(group)>=24:return
  previous=group[-1]if group else None
  for word in words:
   part=compact(word['text'])
   if not part or not remaining.startswith(part)or word in group:continue
   if previous:
    center=lambda w:(w['yMin']+w['yMax'])/2
    height=max(word['yMax']-word['yMin'],previous['yMax']-previous['yMin']);dy=center(word)-center(previous)
    same=abs(dy)<height*.3 and -1<=word['xMin']-previous['xMax']<height*.7
    following=height*.3<=dy<height*1.8 and abs(word['xMin']-group[0]['xMin'])<4
    if not(same or following):continue
   yield from walk(group+[word],remaining[len(part):])
 yield from walk([],text)
rows=[]
for card in cards:
 locations=[]
 for number,words in enumerate(pages,1):
  for label in matches(words,card['label']):
   for value in matches(words,card['value']):
    if abs(value['xMin']-label['xMin'])<4 and label['yMin']+2<value['yMin']<label['yMax']+100:locations.append(number)
 rows.append(dict(card,pages=locations,passed=bool(locations)))
card_pages=[p for row in rows for p in row['pages']]
heights=[w['yMax']-w['yMin']for i,words in enumerate(pages,1)if card_pages and min(card_pages)<=i<=max(card_pages)+3 for w in words if w['text']=='1.25']
spans=[]
for i,words in enumerate(pages,1):
 if not card_pages or not(min(card_pages)<=i<=max(card_pages)+3):continue
 for top in (w for w in words if w['text']=='1.25'):
  zeros=[w for w in words if w['text']=='0'and top['yMax']<w['yMin']<top['yMax']+400]
  ones=[w for w in words if w['text']=='1'and top['yMax']<w['yMin']<top['yMax']+400]
  mids=[w for w in words if w['text']=='0.5'and top['yMax']<w['yMin']<top['yMax']+400]
  spans.extend(one['xMin']-zero['xMin']for zero in zeros for one in ones if abs(zero['yMin']-one['yMin'])<1 and one['xMin']-zero['xMin']>100 and any(abs(mid['yMin']-zero['yMin'])<1 and .35<(mid['xMin']-zero['xMin'])/(one['xMin']-zero['xMin'])<.65 for mid in mids))
print(json.dumps({'pages':len(pages),'cards':rows,'passed':all(r['passed']for r in rows),'axis_height_pt':min(heights)if heights else None,'axis_span_pt':max(spans)if spans else None},ensure_ascii=False))
`;
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(port),report={screens:[],interactions:[],keyboard:[],tableScroll:[],numberWrapping:[],semantics:[],fallbacks:[],outlines:[],pdfs:[],exceptions:b.exceptions};let focusIndex=0;
 const save=()=>fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(report,null,2)+'\n');
 async function load(width,theme,size,scripts=true,spacing=false) {
  await b.viewport(width);await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]});await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.navigate(base+route);await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});
  await b.evaluate(`document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.colorScheme='${theme==='dark'?'dark':'light'}';document.documentElement.style.fontSize='${size}%';document.documentElement.style.letterSpacing=${spacing?"'.12em'":"''"};document.documentElement.style.wordSpacing=${spacing?"'.16em'":"''"};Promise.all([document.fonts.ready,...[...document.querySelectorAll('.content img')].map(i=>{i.loading='eager';return i.decode()})])`);
  await b.pause(160);await b.evaluate(`window.__pion=document.getElementById('pion-spectrum-lab');__pion.scrollIntoView({block:'start'})`);
 }
 async function state() {
  return b.evaluate(`(()=>{const e=__pion,s=e.querySelector('svg'),visible=e.querySelector('[data-chart]').checkVisibility();return{ready:e.dataset.ready==='true',controlsHidden:e.querySelector('[data-controls]').hidden,staticVisible:!e.querySelector('[data-static]').hidden,chartVisible:visible,error:e.querySelector('[data-error]').hidden?null:e.querySelector('[data-error]').textContent,outputs:[...e.querySelectorAll('output')].map(o=>o.value),rows:[...e.querySelectorAll('tbody tr')].map(r=>[...r.querySelectorAll('td')].map(c=>c.textContent)),overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1,svg:{width:s.getBoundingClientRect().width,height:s.getBoundingClientRect().height,texts:visible?[...s.querySelectorAll('text')].map(t=>{const r=t.getBBox();return{text:t.textContent,x:r.x,y:r.y,right:r.x+r.width,bottom:r.y+r.height}}):[],paths:s.querySelectorAll('path').length,markers:s.querySelectorAll('rect,circle').length,bad:[...s.querySelectorAll('path')].some(p=>/NaN|Infinity/.test(p.getAttribute('d')))}}})()`);
 }
 function bounds(row,context) {
  assert(!row.overflow&&!row.svg.bad,JSON.stringify({context,row}));assert.equal(row.svg.paths,3);assert.equal(row.svg.markers,8);
  for(const t of row.svg.texts)assert(t.x>=-1&&t.y>=-1&&t.right<=row.svg.width+1&&t.bottom<=row.svg.height+1,JSON.stringify({context,t,svg:row.svg}));
 }
 function resultMatches(row,result) {
  ['input_erank','pion_erank','muon_norm','pion_norm'].forEach((key,i)=>near(Number(row.outputs[i]),result[key]));
  row.rows.forEach((values,i)=>['normalized','muon','pion'].forEach((name,j)=>near(Number(values[j]),result[name][i])));
 }
 async function preset(name) {await b.evaluate(`__pion.querySelector('[data-preset="${name}"]').click()`);await b.pause(30);}
 async function setInput(name,value) {await b.evaluate(`(()=>{const i=__pion.querySelector('[name="${name}"]');i.focus();i.value=${JSON.stringify(value)};i.dispatchEvent(new Event('input',{bubbles:true}));})()`);await b.pause(25);}
 async function typeNumber(text) {
  for(const type of ['keyDown','keyUp'])await b.cdp('Input.dispatchKeyEvent',{type,key:'a',code:'KeyA',windowsVirtualKeyCode:65,modifiers:2});
  await b.key('Backspace','Backspace',8);await b.cdp('Input.insertText',{text});await b.pause(30);
 }
 async function semantics(context) {
  await b.evaluate(axe+'\n;void 0');const result=await b.evaluate(`axe.run(__pion,{rules:{'color-contrast':{enabled:false}},resultTypes:['violations','incomplete']}).then(r=>({version:r.testEngine.version,violations:r.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),incomplete:r.incomplete.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)}))}))`);
  assert.equal(result.version,'4.14.0');assert.deepEqual(result.violations,[]);assert.deepEqual(result.incomplete,[]);report.semantics.push({...context,...result});
 }
 async function printCase({scripts=true,theme='light',size=100,palette='none',paper='a4',printState='concentrated',native=false,width=1440}) {
  const screenWidth=native?width:paper==='letter'?739:716;
  await load(screenWidth,theme,size,scripts);if(scripts){await preset(printState==='invalid'?'concentrated':printState);if(printState==='invalid')await setInput('sigma0','-1');}
  if(scripts)await b.evaluate(`(()=>{const e=[...__pion.querySelectorAll('[data-controls] button,input,select')][${focusIndex++%10}];e.scrollIntoView({block:'center'});e.focus({preventScroll:true});window.__pionPrintFocus=e;})()`);
  const inputs=await b.evaluate(`JSON.stringify([...__pion.querySelectorAll('input,select')].map(e=>[e.value,e.checked]))`),outputs=await b.evaluate(`JSON.stringify([...__pion.querySelectorAll('output,[data-series]')].map(e=>e.value||e.textContent))`);
  const cards=await b.evaluate(`[...__pion.querySelectorAll('dl>div')].map(e=>({label:e.querySelector('dt').textContent.trim(),value:e.querySelector('dd').textContent.trim()}))`);
  const focusedControl=scripts?await b.evaluate('__pionPrintFocus.name||__pionPrintFocus.dataset.preset'):null;
  if(native)await b.evaluate(`window.__pionPrintEvents=[];window.__pionShape=()=>{const s=__pion.querySelector('svg');return{viewBox:s.getAttribute('viewBox'),width:s.getBoundingClientRect().width,font:getComputedStyle(s).fontSize}};window.__pionScreenShape=__pionShape();for(const event of ['beforeprint','afterprint'])window.addEventListener(event,()=>__pionPrintEvents.push({event,...__pionShape()}))`);
  await b.cdp('Emulation.setEmulatedMedia',{media:native?'':'print',features:palette==='none'?[]:[{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:palette}]});await b.pause(80);
  const chartVisible=await b.evaluate(`__pion.querySelector('[data-chart]').checkVisibility()`);assert.equal(chartVisible,scripts&&printState!=='invalid');
  if(chartVisible)bounds(await state(),{scripts,theme,size,printState,palette,paper});
  const cellOverflow=native?[]:await b.evaluate(`([...__pion.querySelectorAll('table th,table td')].flatMap(e=>{const r=document.createRange();r.selectNodeContents(e);const box=e.getBoundingClientRect(),s=getComputedStyle(e),content=r.getBoundingClientRect();return content.left<box.left+parseFloat(s.paddingLeft)-1||content.right>box.right-parseFloat(s.paddingRight)+1?[{text:e.textContent,cell:box.toJSON(),content:content.toJSON()}]:[]}))`);
  assert.deepEqual(cellOverflow,[],JSON.stringify({size,theme,paper,printState,cellOverflow}));
  if(scripts&&!native)assert(await b.evaluate(`document.activeElement===__pionPrintFocus&&getComputedStyle(__pionPrintFocus).outlineStyle==='none'&&[...__pion.querySelectorAll('input')].every(e=>getComputedStyle(e).caretColor==='rgba(0, 0, 0, 0)')`));
  const colors=chartVisible?await b.evaluate(`[...__pion.querySelectorAll('path')].map(p=>getComputedStyle(p).stroke.match(/\\d+/g).slice(0,3).map(Number))`):[];
  if(chartVisible&&!native)assert.deepEqual(colors,[[75,85,99],[140,69,17],[31,79,136]]);
  const pdf=await b.cdp('Page.printToPDF',{printBackground:true,paperWidth:paper==='letter'?8.5:8.27,paperHeight:paper==='letter'?11:11.69,marginTop:.4,marginBottom:.4,marginLeft:.4,marginRight:.4,generateTaggedPDF:true});
  const file=path.join(output,`pion-${report.pdfs.length}-${scripts}-${theme}-${size}-${palette}-${paper}.pdf`);fs.writeFileSync(file,Buffer.from(pdf.data,'base64'));
  const bbox=execFileSync('pdftotext',['-bbox',file,'-']);const pagination=JSON.parse(execFileSync('python3',['-B','-c',pdfInspect,JSON.stringify(cards)],{input:bbox,maxBuffer:64*1024*1024}));
  assert(pagination.passed,JSON.stringify({scripts,theme,size,printState,pagination}));if(chartVisible)assert(pagination.axis_height_pt>=8*size/100,JSON.stringify({scripts,theme,size,pagination}));
  if(chartVisible)assert(pagination.axis_span_pt>=280,JSON.stringify({native,width,theme,size,pagination}));
  const info=execFileSync('pdfinfo',[file],{encoding:'utf8'});assert(/Tagged:\s+yes/.test(info));
  await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]});await b.pause(100);
  assert.equal(await b.evaluate(`JSON.stringify([...__pion.querySelectorAll('input,select')].map(e=>[e.value,e.checked]))`),inputs);assert.equal(await b.evaluate(`JSON.stringify([...__pion.querySelectorAll('output,[data-series]')].map(e=>e.value||e.textContent))`),outputs);
  if(scripts)assert(await b.evaluate('document.activeElement===__pionPrintFocus'));
  const events=native?await b.evaluate(`({events:__pionPrintEvents,original:__pionScreenShape,restored:__pionShape()})`):null;
  if(native&&chartVisible){assert.deepEqual(events.restored,events.original);assert.equal(events.events.find(e=>e.event==='beforeprint').viewBox.split(' ')[2],'624');}
  report.pdfs.push({scripts,theme,size,palette,paper,printState,file,focusedControl,chartVisible,colors,colorsContext:native?'screen before native print':'print media',...pagination,native,width:screenWidth,events,cellOverflow,tagged:true,inputsPreserved:true,outputsPreserved:true,focusPreserved:scripts});save();
 }
 try {
  if(!printOnly&&!behaviorOnly) {
   const contexts=[];
   for(const scripts of [true,false])for(const width of [320,768,1440])for(const theme of ['light','dark','warm'])for(const size of [100,200])for(const spacing of width===768?[false]:[false,true])contexts.push({scripts,width,theme,size,spacing});
   for(const c of contexts) {
    await load(c.width,c.theme,c.size,c.scripts,c.spacing);const row=await state();assert(!row.overflow,JSON.stringify({c,row}));assert.equal(row.ready,c.scripts);assert.equal(row.controlsHidden,!c.scripts);assert.equal(row.staticVisible,!c.scripts);assert.equal(row.chartVisible,c.scripts);if(c.scripts)bounds(row,c);resultMatches(row,core.calculate(core.presets.concentrated.raw,1,0));
    const contrast=await b.evaluate('('+contrastAudit.toString()+')('+JSON.stringify('#pion-spectrum-lab')+')');assert.deepEqual(contrast.issues,[],JSON.stringify({c,contrast}));
    report.screens.push({...c,contrastSamples:contrast.checked,...row});
    if(c.width===320&&c.theme==='dark'&&c.size===100&&!c.spacing){await b.evaluate(`__pion.querySelector('table').scrollIntoView({block:'center'})`);await b.screenshot(path.join(output,'mobile-results-'+c.scripts+'.png'));}
   }save();
  }
  if(!printOnly&&!readingOnly) {
   for(const width of [320,1440])for(const theme of ['light','dark','warm']) {
    await load(width,theme,100);
    for(const name of Object.keys(core.presets)) {
     await preset(name);const p=core.presets[name],row=await state();bounds(row,{width,theme,name});resultMatches(row,core.calculate(p.raw,p.kp,p.epsilon?1e-7:0));report.interactions.push({width,theme,name,...row});await semantics({width,theme,state:name});
    }
    await preset('concentrated');await setInput('sigma0','');let row=await state();assert(!row.chartVisible&&row.outputs.every(v=>v==='—')&&row.rows.every(r=>r.every(v=>v==='—'))&&row.error);assert(await b.evaluate(`document.activeElement===__pion.querySelector('[name="sigma0"]')`));await semantics({width,theme,state:'invalid'});
    await preset('concentrated');for(let kp=0;kp<=5;kp++){await setInput('promotionSteps',String(kp));row=await state();resultMatches(row,core.calculate(core.presets.concentrated.raw,kp));report.interactions.push({width,theme,kp,...row});}
    await preset('concentrated');await b.evaluate(`__pion.querySelector('[name="sigma0"]').focus()`);await typeNumber('-');row=await state();
    assert(row.error&&!row.chartVisible&&row.outputs.every(v=>v==='—'));assert(await b.evaluate(`document.activeElement.name==='sigma0'&&document.activeElement.getAttribute('aria-invalid')==='true'`));
    report.keyboard.push({width,theme,operation:'partial negative entry',cleared:true,focusPreserved:true});
    await typeNumber('0.5');row=await state();resultMatches(row,core.calculate([.5,.4,.2,.02]));assert(!row.error&&row.chartVisible);report.keyboard.push({width,theme,operation:'correct numeric entry',recovered:true});
    await b.evaluate(`__pion.querySelector('select').focus()`);await b.key('End','End',35);await b.key('ArrowUp','ArrowUp',38);await b.key('Enter','Enter',13);await b.pause(40);
    resultMatches(await state(),core.calculate([.5,.4,.2,.02],4));assert(await b.evaluate(`document.activeElement.name==='promotionSteps'`));report.keyboard.push({width,theme,operation:'native select keys',promotionSteps:4});
    await b.evaluate(`__pion.querySelector('[type="checkbox"]').focus()`);await b.key(' ','Space',32);await b.pause(30);resultMatches(await state(),core.calculate([.5,.4,.2,.02],4,1e-7));assert(await b.evaluate(`document.activeElement.name==='includeEpsilon'&&document.activeElement.checked`));report.keyboard.push({width,theme,operation:'checkbox Space',epsilon:1e-7});
    await setInput('promotionSteps','99');row=await state();assert(row.error&&!row.chartVisible&&row.outputs.every(v=>v==='—'));report.keyboard.push({width,theme,operation:'missing select option',cleared:true});
    await preset('concentrated');await setInput('sigma0','10000');resultMatches(await state(),core.calculate([10000,.4,.2,.02]));
    for(const value of ['10001','-1','1e309']){await setInput('sigma0',value);row=await state();assert(row.error&&!row.chartVisible&&row.outputs.every(v=>v==='—'));report.keyboard.push({width,theme,operation:'invalid amplitude',value,cleared:true});}
    if(width===320){
     await preset('concentrated');await b.evaluate(`document.documentElement.style.fontSize='200%';window.__pionTable=__pion.querySelector('table');__pionTable.scrollIntoView({block:'center'});__pionTable.focus();__pionTable.scrollLeft=0`);await b.pause(100);for(let key=0;key<8;key++)await b.key('ArrowRight','ArrowRight',39);await b.pause(180);const scroll=await b.evaluate(`({focused:document.activeElement===__pionTable,left:__pionTable.scrollLeft,extent:__pionTable.scrollWidth-__pionTable.clientWidth,bodyOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1})`);assert(scroll.focused&&scroll.left>0&&scroll.extent>0&&!scroll.bodyOverflow,JSON.stringify(scroll));report.tableScroll.push({width,theme,...scroll});
     await preset('flat');const wrapping=await b.evaluate(`(()=>{const e=__pion.querySelector('[data-value="pion_norm"]'),node=e.querySelector('.pion-spectrum__exponent')?.firstChild,ys=[];if(node)for(let i=0;i<node.length;i++){const r=document.createRange();r.setStart(node,i);r.setEnd(node,i+1);ys.push(r.getBoundingClientRect().y)}const range=document.createRange();range.selectNodeContents(e);return{value:e.value,selectedText:range.toString(),hasSoftBreak:!!e.querySelector('wbr'),exponentY:ys}})()`);assert(wrapping.hasSoftBreak&&wrapping.value==='1.18160e-31'&&wrapping.selectedText===wrapping.value&&wrapping.exponentY.length===4&&Math.max(...wrapping.exponentY)-Math.min(...wrapping.exponentY)<1,JSON.stringify(wrapping));report.numberWrapping.push({width,theme,...wrapping});
    }
   }
   for(const width of [320,1440])for(const failure of ['blocked','svg']) {
    let identifier;
    try {
     if(failure==='blocked')await b.cdp('Network.setBlockedURLs',{urls:['*pion-spectrum*.js*']});
     else identifier=(await b.cdp('Page.addScriptToEvaluateOnNewDocument',{source:`SVGGraphicsElement.prototype.getBBox=function(){throw new Error('controlled SVG failure')}` })).identifier;
     await load(width,'light',100);const row=await state();assert(!row.ready&&row.controlsHidden&&row.staticVisible&&!row.chartVisible);resultMatches(row,core.calculate(core.presets.concentrated.raw));report.fallbacks.push({width,failure,...row});
    }finally{await b.cdp('Network.setBlockedURLs',{urls:[]});if(identifier)await b.cdp('Page.removeScriptToEvaluateOnNewDocument',{identifier});}
   }
   for(const width of [320,1440])for(const failure of ['late-input','late-resize','late-print']) {
    await load(width,'light',100);await b.evaluate(`__pion.querySelector('[name="sigma0"]').focus();window.__pionBBox=SVGGraphicsElement.prototype.getBBox;SVGGraphicsElement.prototype.getBBox=function(){throw new Error('controlled later SVG failure')}`);
    let raw=core.presets.concentrated.raw;
    try {
     if(failure==='late-input'){await setInput('sigma0','.6');raw=[.6,.4,.2,.02];}
     else if(failure==='late-resize'){await b.viewport(width===320?390:768);await b.pause(120);}
     else {const pdf=await b.cdp('Page.printToPDF',{printBackground:true,paperWidth:8.27,paperHeight:11.69});assert(pdf.data.length>1000);}
     const row=await state();assert(row.ready&&row.error&&!row.chartVisible&&!row.controlsHidden&&!row.staticVisible);resultMatches(row,core.calculate(raw));assert(await b.evaluate(`document.activeElement.name==='sigma0'`));
     report.fallbacks.push({width,failure,currentNumbersPreserved:true,focusPreserved:true,...row});
    }finally {await b.evaluate(`SVGGraphicsElement.prototype.getBBox=__pionBBox`);}
    await setInput('sigma0','.7');const recovered=await state();assert(recovered.chartVisible&&!recovered.error);resultMatches(recovered,core.calculate([.7,.4,.2,.02]));assert(await b.evaluate(`document.activeElement.name==='sigma0'`));report.fallbacks.at(-1).recovered=true;
   }
   for(const width of [320,1440])for(const size of [100,200])for(const theme of ['light','dark','warm'])for(const palette of ['none','light','dark']) {
    await load(width,theme,size);await b.cdp('Emulation.setEmulatedMedia',{media:'',features:palette==='none'?[]:[{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:palette}]});await b.key('Tab','Tab',9);await b.evaluate(`__pion.querySelector('button').focus()`);
    for(let index=0;index<10;index++) {
     if(index)await b.key('Tab','Tab',9);
     const row=await b.evaluate(`(()=>{const e=[...__pion.querySelectorAll('[data-controls] button,input,select')][${index}],s=getComputedStyle(e),c=document.createElement('canvas');c.width=c.height=1;const ctx=c.getContext('2d',{willReadFrequently:true});const rgba=color=>{ctx.clearRect(0,0,1,1);ctx.fillStyle=color;ctx.fillRect(0,0,1,1);return[...ctx.getImageData(0,0,1,1).data].map(v=>v/255)};const over=(a,b)=>[0,1,2].map(i=>a[i]*a[3]+b[i]*(1-a[3]));const background=n=>n?over(rgba(getComputedStyle(n).backgroundColor),background(n.parentElement)):[1,1,1];const bg=background(e.parentElement),fg=over(rgba(s.outlineColor),bg),lum=rgb=>rgb.map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((n,v,i)=>n+v*[.2126,.7152,.0722][i],0),values=[lum(fg),lum(bg)].sort((a,b)=>b-a),r=e.getBoundingClientRect();return{index:${index},name:e.name||e.dataset.preset,focused:document.activeElement===e,visible:e.matches(':focus-visible'),outline:s.outlineColor,outlineWidth:parseFloat(s.outlineWidth),outlineStyle:s.outlineStyle,ratio:(values[0]+.05)/(values[1]+.05),inside:r.left>=-1&&r.right<=document.documentElement.clientWidth+1}})()`);
     assert(row.focused&&row.visible&&row.outlineWidth>=2&&row.outlineStyle!=='none'&&row.ratio>=3&&row.inside,JSON.stringify({width,size,theme,palette,row}));report.outlines.push({width,size,theme,palette,...row});
    }
   }save();
  }
  if(!readingOnly&&!behaviorOnly) {
   if(nativeOnly){for(const width of [320,1440])for(const theme of ['light','dark','warm'])for(const size of [100,200])await printCase({native:true,width,theme,size,printState:['concentrated','flat','tiny','zero'][focusIndex%4]});for(const size of [100,200])await printCase({native:true,width:320,size,paper:'letter',printState:'flat'});}
   else if(letter)for(const size of [100,200])await printCase({size,paper:'letter',printState:'flat'});
   else if(forced)for(const palette of ['light','dark'])for(const theme of ['light','dark','warm'])for(const size of [100,200])await printCase({palette,theme,size,printState:['concentrated','flat','tiny','zero','invalid'][focusIndex%5]});
   else {
    for(const scripts of [true,false])for(const theme of ['light','dark','warm'])for(const size of [100,200])await printCase({scripts,theme,size,printState:['concentrated','flat','tiny','zero','invalid'][focusIndex%5]});
    for(const printState of ['flat','tiny','zero','invalid'])await printCase({printState});
   }
  }
  assert.deepEqual(b.exceptions,[]);report.summary={screens:report.screens.length,interactions:report.interactions.length,keyboardCases:report.keyboard.length,nativeTableScrolls:report.tableScroll.length,scientificNumberWraps:report.numberWrapping.length,semanticStates:report.semantics.length,fallbacks:report.fallbacks.length,actualTabOutlines:report.outlines.length,nativePdfs:report.pdfs.length,exceptions:b.exceptions.length};console.log(JSON.stringify(report.summary));
 } finally {save();await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]}).catch(()=>{});await b.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
