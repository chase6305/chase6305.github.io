#!/usr/bin/env node
// Inspect the inline force-lag experiment; run serially with other CDP suites.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const {connect} = require('./lib/browser.cjs');
const {contrastAudit} = require('./lib/contrast.cjs');
const base = (process.argv[2] || 'http://127.0.0.1:13321').replace(/\/+$/, '');
const port = Number(process.argv[3] || 9236);
const output = process.argv[4] || '/tmp/chase-identification-timing';
const option = name => { const index = process.argv.indexOf(name); return index < 0 ? null : process.argv[index + 1]; };
const axePath = option('--axe');
assert(axePath, 'Supply --axe /path/to/axe-core/axe.min.js (version 4.14.0)');
const axe = fs.readFileSync(axePath, 'utf8');
assert.equal(crypto.createHash('sha256').update(axe).digest('hex'), '20c09fe157a8a34a30e241aaa1fcdade657734f08ab379ecfbeb7d45cc46e878');
const route = '/posts/robotics/dynamics/parameter-identification/';
const forcedColors = process.argv.includes('--forced-colors');
const letterOnly = process.argv.includes('--letter-only');
const printOnly = process.argv.includes('--print-only') || forcedColors || letterOnly;
const behaviorOnly = process.argv.includes('--behavior-only');
const skipReading = printOnly || behaviorOnly;
const near = (actual, expected, tolerance = 2e-5) => assert(Math.abs(actual - expected) < tolerance, actual + ' != ' + expected);
const pdfInspect = String.raw`
import json,sys,xml.etree.ElementTree as ET
root=ET.fromstring(sys.stdin.buffer.read());ns={'x':'http://www.w3.org/1999/xhtml'}
cards=json.loads(sys.argv[1]);compact=lambda s:''.join(s.split()).replace('\u2212','-')
pages=[]
for page in root.findall('.//x:page',ns):
 pages.append([dict(text=w.text or '',**{k:float(v) for k,v in w.attrib.items()}) for w in page.findall('.//x:word',ns)])
def matches(words,text):
 text=compact(text)
 def walk(group,remaining):
  if not remaining:
   yield {k:op(w[k] for w in group) for k,op in [('xMin',min),('yMin',min),('xMax',max),('yMax',max)]};return
  if len(group)>=24:return
  previous=group[-1] if group else None
  for word in words:
   part=compact(word['text'])
   if not part or not remaining.startswith(part) or word in group:continue
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
    if abs(value['xMin']-label['xMin'])<4 and label['yMin']+2<value['yMin']<label['yMax']+80:
     locations.append(number)
 rows.append(dict(card,pages=locations,passed=bool(locations)))
axis_heights=[w['yMax']-w['yMin'] for words in pages for w in words if w['text']=='2.99']
print(json.dumps({'pages':len(pages),'cards':rows,'passed':all(r['passed'] for r in rows),'axis_label_height_pt':min(axis_heights) if axis_heights else None},ensure_ascii=False))
`;
(async () => {
  fs.mkdirSync(output, {recursive: true});
  const b = await connect(port), report = {screens: [], interactions: [], semantics: [], fallbacks: [], outlines: [], pdfs: [], exceptions: b.exceptions};
  let printFocusIndex = 0;
  const save = () => fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(report, null, 2) + '\n');
  async function load(width, theme, size, scripts = true, spacing = false) {
    await b.viewport(width);
    await b.cdp('Emulation.setEmulatedMedia', {media: '', features: []});
    await b.cdp('Emulation.setScriptExecutionDisabled', {value: !scripts});
    await b.navigate(base + route);
    await b.cdp('Emulation.setScriptExecutionDisabled', {value: false});
    await b.evaluate(`document.documentElement.classList.remove('light','dark','warm');
      document.documentElement.classList.add(${JSON.stringify(theme)});
      document.documentElement.style.colorScheme=${JSON.stringify(theme === 'dark' ? 'dark' : 'light')};
      document.documentElement.style.fontSize='${size}%';
      document.documentElement.style.letterSpacing=${spacing ? "'.12em'" : "''"};
      document.documentElement.style.wordSpacing=${spacing ? "'.16em'" : "''"};
      Promise.all([document.fonts.ready,...[...document.querySelectorAll('.content img')].map(i=>{i.loading='eager';return i.decode()})])`);
    await b.pause(200);
    await b.evaluate(`window.__timing=document.getElementById('timing-lab');__timing.scrollIntoView({block:'start'})`);
  }
  async function state() {
    return b.evaluate(`(()=>{const e=__timing,s=e.querySelector('svg'),r=s.getBoundingClientRect();
      const texts=[...s.querySelectorAll('text')].map(t=>{const b=t.getBBox();return{text:t.textContent,axis:t.hasAttribute('dominant-baseline')?'y':'x',x:b.x,y:b.y,right:b.x+b.width,bottom:b.y+b.height}});
      return{ready:e.dataset.ready==='true',overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1,
        outputs:[...e.querySelectorAll('output')].map(o=>o.value),svg:{width:r.width,height:r.height,texts,paths:s.querySelectorAll('path').length},
        hidden:e.querySelector('[data-interactive]').hidden,staticVisible:!e.querySelector('[data-static-result]').hidden,
        bad:[...e.querySelectorAll('path')].some(p=>/NaN|Infinity/.test(p.getAttribute('d')))};})()`);
  }
  function bounds(row, context) {
    assert(!row.overflow && !row.bad, JSON.stringify({context,row}));
    assert.equal(row.svg.paths, 3);
    for (const text of row.svg.texts) assert(text.x >= -1 && text.right <= row.svg.width + 1 && text.y >= -1 && text.bottom <= row.svg.height + 1,
      JSON.stringify({context,text,svg:row.svg}));
    const ticks = row.svg.texts.filter(t => t.axis==='x').sort((a,c)=>a.x-c.x);
    for (let index = 1; index < ticks.length; index++) assert(ticks[index].x >= ticks[index-1].right + 2, JSON.stringify({context,ticks}));
  }
  async function setInput(name, value) {
    return b.evaluate(`(()=>{const i=__timing.querySelector('[name="${name}"]');i.focus({preventScroll:true});i.value=${JSON.stringify(value)};i.dispatchEvent(new Event('input',{bubbles:true}));return document.activeElement===i})()`);
  }
  async function preset(name) {
    await b.evaluate(`(()=>{const p=__timing.querySelector('[data-preset="${name}"]');p.scrollIntoView({block:'center'});p.focus({preventScroll:true})})()`);
    await b.key('Enter','Enter',13);
    assert(await b.evaluate(`document.activeElement===__timing.querySelector('[data-preset="${name}"]')`));
  }
  async function semantics(context) {
    await b.evaluate(axe + ';void 0');
    const result = await b.evaluate(`axe.run('#timing-lab',{rules:{'color-contrast':{enabled:false}},resultTypes:['violations','incomplete']})
      .then(r=>({violations:r.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),incomplete:r.incomplete.map(v=>v.id)}))`);
    assert.deepEqual(result.violations, [], JSON.stringify({context,result}));
    assert.deepEqual(result.incomplete, [], JSON.stringify({context,result}));
    report.semantics.push({...context,...result});
  }
  try {
    for (const scripts of skipReading ? [] : [true,false]) for (const width of [320,768,1440]) for (const theme of ['light','dark','warm']) for (const size of [100,200]) {
      const context = {scripts,width,theme,size}; await load(width,theme,size,scripts);
      const row = await state(); assert(!row.overflow, JSON.stringify({context,row}));
      const contrast = await b.evaluate('('+contrastAudit.toString()+')("#timing-lab")'); assert.deepEqual(contrast.issues, [], JSON.stringify({context,contrast}));
      if (scripts) {
        assert(row.ready && !row.hidden && !row.staticVisible); bounds(row,context);
        near(Number(row.outputs[0]),1.984277531322008); near(Number(row.outputs[1]),1.1712329299973558);
        if (size===100) await semantics({...context,state:'default'});
        await preset('aligned'); let corrected=await state(); near(Number(corrected.outputs[0]),2);near(Number(corrected.outputs[1]),.4);
        assert(corrected.outputs[4].startsWith('<') && Number(corrected.outputs[3])>.27);
        assert(await setInput('trainHz','0'));
        const invalid = await b.evaluate(`({invalid:__timing.querySelector('[name=trainHz]').getAttribute('aria-invalid'),error:!__timing.querySelector('[data-error]').hidden,
          empty:[...__timing.querySelectorAll('output')].every(o=>o.value==='—'),chart:__timing.querySelector('[data-chart]').hidden})`);
        assert(invalid.invalid==='true'&&invalid.error&&invalid.empty&&invalid.chart,JSON.stringify({context,invalid}));
        if (size===100) await semantics({...context,state:'invalid'});
        assert(await setInput('trainHz','.7')); await preset('delayed'); assert(await setInput('delayMs','-20'));
        assert(await b.evaluate(`Number(__timing.querySelector('[data-value=damping]').value)<0 && __timing.querySelector('[data-note]').textContent.includes('负阻尼')`));
        await preset('delayed'); assert(await setInput('validationHz','.7')); const same=await state();
        assert(same.outputs[3].startsWith('<')&&Number(same.outputs[4])>.01);
        await preset('delayed'); report.interactions.push({...context,corrected:true,invalidInput:true,focusPreserved:true,negativeEstimate:true,sameFrequency:true});
      } else assert(!row.ready&&row.hidden&&row.staticVisible);
      if (width===320&&theme==='dark'&&size===100) {
        await b.evaluate(`__timing.querySelector('[data-chart]').scrollIntoView({block:'center'})`);
        await b.screenshot(path.join(output,'mobile-chart-'+scripts+'.png'));
      }
      report.screens.push({...context,...row,textSamples:contrast.checked});save();console.log('screen',report.screens.length);
    }
    for(const width of skipReading ? [] : [320,1440])for(const size of [100,200]){
      const context={width,size,spacing:true};await load(width,'light',size,true,true);
      for(const [name,value] of [['trainHz','2'],['validationHz','2'],['delayMs','-50']])assert(await setInput(name,value));
      bounds(await state(),context);report.screens.push({...context,passed:true});save();
    }
    if(!printOnly) {
      for(const failure of ['blocked-script','svg-setup']) {
        let identifier;
        if(failure==='blocked-script') await b.cdp('Network.setBlockedURLs',{urls:['*identification-timing*.js*']});
        else ({identifier}=await b.cdp('Page.addScriptToEvaluateOnNewDocument',{source:`{
          const original=SVGGraphicsElement.prototype.getBBox;
          SVGGraphicsElement.prototype.getBBox=function(...args){
            if(this.closest('[data-identification-timing]'))throw Error('Controlled SVG setup failure');
            return original.apply(this,args);
          };
        }`}));
        try {
          for(const width of [320,1440]) {
            await load(width,'light',100);
            const row=await b.evaluate(`({ready:__timing.dataset.ready==='true',hidden:__timing.querySelector('[data-interactive]').hidden,
              staticVisible:__timing.querySelector('[data-static-result]').checkVisibility(),
              readable:__timing.querySelector('[data-static-result]').textContent.includes('1.98428'),
              visibleControls:[...__timing.querySelectorAll('input,button')].filter(e=>e.checkVisibility()).length,
              sourceLink:!!document.querySelector('a[href$="timing_counterexample.py"]')})`);
            assert(!row.ready&&row.hidden&&row.staticVisible&&row.readable&&!row.visibleControls&&row.sourceLink,JSON.stringify({failure,width,row}));
            report.fallbacks.push({failure,width,...row});save();
          }
        } finally {
          await b.cdp('Network.setBlockedURLs',{urls:[]});
          if(identifier)await b.cdp('Page.removeScriptToEvaluateOnNewDocument',{identifier});
        }
      }
      for(const width of [320,1440])for(const size of [100,200])for(const theme of ['light','dark','warm'])for(const palette of ['none','light','dark']) {
        await load(width,theme,size);
        await b.cdp('Emulation.setEmulatedMedia',{features:palette==='none'?[]:[{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:palette}]});
        await b.key('Tab','Tab',9);
        await b.evaluate(`__timing.querySelector('button').focus()`);
        for(let index=0;index<6;index++) {
          if(index)await b.key('Tab','Tab',9);
          const row=await b.evaluate(`(()=>{
            const e=[...__timing.querySelectorAll('button,input')][${index}],s=getComputedStyle(e),c=document.createElement('canvas');c.width=c.height=1;
            const ctx=c.getContext('2d',{willReadFrequently:true});
            const rgba=color=>{ctx.clearRect(0,0,1,1);ctx.fillStyle=color;ctx.fillRect(0,0,1,1);return [...ctx.getImageData(0,0,1,1).data].map(v=>v/255)};
            const over=(a,b)=>[0,1,2].map(i=>a[i]*a[3]+b[i]*(1-a[3]));
            const background=node=>node?over(rgba(getComputedStyle(node).backgroundColor),background(node.parentElement)):[1,1,1];
            const bg=background(e.parentElement),fg=over(rgba(s.outlineColor),bg);
            const lum=rgb=>rgb.map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((n,v,i)=>n+v*[.2126,.7152,.0722][i],0);
            const values=[lum(fg),lum(bg)].sort((a,b)=>b-a),r=e.getBoundingClientRect();
            return {index:${index},name:e.name||e.dataset.preset,focused:document.activeElement===e,visible:e.matches(':focus-visible'),
              outline:s.outlineColor,outlineWidth:parseFloat(s.outlineWidth),outlineStyle:s.outlineStyle,ratio:(values[0]+.05)/(values[1]+.05),
              inside:r.left>=-1&&r.right<=document.documentElement.clientWidth+1};
          })()`);
          assert(row.focused&&row.visible&&row.outlineWidth>=2&&row.outlineStyle!=='none'&&row.ratio>=3&&row.inside,JSON.stringify({width,size,theme,palette,row}));
          report.outlines.push({width,size,theme,palette,...row});
        }
        if(width===320&&size===100&&palette==='none')await semantics({width,size,theme,state:'keyboard-controls'});
        save();
      }
    }
    for(const palette of behaviorOnly ? [] : forcedColors ? ['light','dark'] : ['none'])
      for(const scripts of forcedColors || letterOnly ? [true] : [true,false])
        for(const theme of letterOnly ? ['light'] : ['light','dark','warm'])for(const size of [100,200]){
      await load(1440,theme,size,scripts);
      const printState=scripts?['delayed','aligned','negative','invalid','aligned','negative'][printFocusIndex%6]:'static';
      if(printState==='aligned')await preset('aligned');
      if(printState==='negative')assert(await setInput('delayMs','-20'));
      if(printState==='invalid')assert(await setInput('trainHz','0'));
      if(scripts)await b.evaluate(`(()=>{const i=[...__timing.querySelectorAll('button,input')][${printFocusIndex++ % 6}];i.scrollIntoView({block:'center'});i.focus({preventScroll:true});window.__printFocus=i;})()`);
      const focusedControl=scripts?await b.evaluate('__printFocus.name||__printFocus.dataset.preset'):null;
      const inputs=await b.evaluate(`JSON.stringify([...__timing.querySelectorAll('input')].map(e=>[e.value,e.checked]))`);
      const outputs=await b.evaluate(`JSON.stringify([...__timing.querySelectorAll('output')].map(e=>e.value))`);
      const cards=scripts?await b.evaluate(`[...__timing.querySelectorAll('dl > div')].map(e=>({label:e.querySelector('dt').textContent.trim(),value:e.querySelector('dd').textContent.trim()}))`):[];
      const features=forcedColors?[{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:palette}]:[];
      await b.cdp('Emulation.setEmulatedMedia',{media:'print',features});await b.pause(100);
      await b.evaluate(`document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add(${JSON.stringify(theme)});document.documentElement.style.colorScheme=${JSON.stringify(theme === 'dark' ? 'dark' : 'light')}`);
      await b.pause(100);
      const chartVisible=await b.evaluate(`__timing.querySelector('[data-chart]').checkVisibility()`);
      assert.equal(chartVisible,scripts&&printState!=='invalid');
      if(scripts)assert(await b.evaluate(`getComputedStyle(__printFocus).outlineStyle==='none' && [...__timing.querySelectorAll('input')].every(i=>getComputedStyle(i).caretColor==='rgba(0, 0, 0, 0)')`),'Print retained a caret or focus outline');
      if(chartVisible)bounds(await state(),{scripts,theme,size,print:true});
      const colors=chartVisible?await b.evaluate(`[...__timing.querySelectorAll('path')].map(p=>getComputedStyle(p).stroke.match(/\\d+/g).slice(0,3).map(Number))`):[];
      if(chartVisible)assert.deepEqual(colors,[[32,32,32],[140,69,17],[31,79,136]],JSON.stringify({palette,theme,size,colors}));
      const paper=letterOnly?{name:'letter',width:8.5,height:11}:{name:'a4',width:8.27,height:11.69};
      const result=await b.cdp('Page.printToPDF',{printBackground:true,generateTaggedPDF:true,paperWidth:paper.width,paperHeight:paper.height,marginTop:.4,marginBottom:.4,marginLeft:.4,marginRight:.4});
      const file=path.join(output,'timing-'+theme+'-'+size+'-'+scripts+'-'+palette+'-'+paper.name+'.pdf'),bytes=Buffer.from(result.data,'base64');fs.writeFileSync(file,bytes);
      assert(bytes.includes(Buffer.from('/StructTreeRoot'))&&bytes.includes(Buffer.from('/MarkInfo')));
      const xml=execFileSync('pdftotext',['-bbox',file,'-'],{encoding:'utf8'}),pagination=JSON.parse(execFileSync('python3',['-B','-c',pdfInspect,JSON.stringify(cards)],{input:xml,encoding:'utf8'}));
      if(scripts)assert(pagination.passed,JSON.stringify({scripts,theme,size,pagination}));
      if(chartVisible)assert(pagination.axis_label_height_pt>=8*size/100,JSON.stringify({scripts,theme,size,pagination}));
      else assert.equal(pagination.axis_label_height_pt,null,'Hidden chart leaked into the native PDF');
      await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]});await b.pause(150);
      assert.equal(await b.evaluate(`JSON.stringify([...__timing.querySelectorAll('input')].map(e=>[e.value,e.checked]))`),inputs);
      assert.equal(await b.evaluate(`JSON.stringify([...__timing.querySelectorAll('output')].map(e=>e.value))`),outputs);
      if(scripts)assert(await b.evaluate('document.activeElement===__printFocus'),'Printing moved the current experiment control focus: '+focusedControl);
      report.pdfs.push({scripts,theme,size,palette,paper:paper.name,printState,chartVisible,colors,file,pages:pagination.pages,cards:pagination.cards,axisLabelHeightPt:pagination.axis_label_height_pt,tagged:true,inputsPreserved:true,outputsPreserved:true,focusPreserved:scripts,focusedControl,printCaretHidden:scripts});save();console.log('pdf',report.pdfs.length);
    }
    assert.deepEqual(b.exceptions,[]);
    report.summary={screens:report.screens.length,interactiveStates:report.interactions.length,semantics:report.semantics.length,fallbacks:report.fallbacks.length,outlines:report.outlines.length,pdfs:report.pdfs.length,exceptions:b.exceptions.length};
    console.log(JSON.stringify(report.summary));
  }finally{save();await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]}).catch(()=>{});await b.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
