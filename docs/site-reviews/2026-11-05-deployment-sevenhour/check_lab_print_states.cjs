#!/usr/bin/env node
// Native article PDFs after changed, unschedulable and invalid lab inputs.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const {connect,pause}=require('../../../scripts/lib/browser.cjs');
const [base='http://127.0.0.1:13327',port='9337',output='/tmp/deployment-lab-print-states']=process.argv.slice(2);
const examples=[
 {kind:'timing',route:'/posts/ai/rma-umi-rtc-deployment/',fields:['horizon','stride','period','observation','inference','execution'],
  states:[['decimal',[16,5,20,.1,79.9,0]],['separate',[16,6,20,40,60,20]],['unschedulable',[16,5,20,0,240,0]],['invalid',[16,5,20,0,'',0]]]},
 {kind:'residual',route:'/posts/ai/residual-reinforcement-learning/',fields:['gain','feedback','correction','bias','period'],
  states:[['uncorrected',[1,0,0,.2,.1]],['wrong',[1,1.2,0,0,.1]],['sampled',[3,0,0,0,.8]],['invalid',[1,0,'',.2,.1]]]},
 {kind:'contact',route:'/posts/calibration/interactive-contact-calibration/',fields:['sigma','angle'],second:true,
  states:[['parallel',[2,0]],['oblique',[2,15]],['broad',[8,90]],['invalid',['',90]]]}
];
const normalize=s=>s.normalize('NFKC').replace(/\s/g,'').replace(/−/g,'-');
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(Number(port));const root='[data-deployment-lab]';
 const report={exports:[],palettes:[],exceptions:b.exceptions};
 const snapshot=()=>b.evaluate(`(()=>{const r=document.querySelector('${root}');return{values:Object.fromEntries([...r.querySelectorAll('output')].map(o=>[o.dataset.value,o.textContent])),inputs:[...r.querySelectorAll('input')].map(i=>[i.name,i.value,i.checked]),computed:r.querySelector('[data-computed]').hidden,error:r.querySelector('[data-error]').hidden,scroll:scrollY,focus:document.activeElement.id}})()`);
 try{
  for(const example of examples)for(const [name,inputs]of example.states)for(const forced of [false,true]){
   const features=forced?[{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:'dark'}]:[];
   await b.cdp('Emulation.setEmulatedMedia',{media:'',features});await b.viewport(390,1000);await b.navigate(base+example.route);
   await b.evaluate(`document.documentElement.classList.add('dark');document.documentElement.style.fontSize='200%';document.fonts.ready`);
   await b.evaluate(`(()=>{const r=document.querySelector('${root}'),names=${JSON.stringify(example.fields)},values=${JSON.stringify(inputs)};names.forEach((n,i)=>r.querySelector('input[name="'+n+'"]').value=values[i]);const checkbox=r.querySelector('[type="checkbox"]');if(checkbox)checkbox.checked=${!!example.second};const f=r.querySelector('input');f.focus();f.dispatchEvent(new Event('input',{bubbles:true}));r.querySelector('[data-${name==='invalid'?'error':'computed'}]').scrollIntoView({block:'center',behavior:'instant'})})()`);
   const before=await snapshot();assert.equal(before.computed,name==='invalid');assert.equal(before.error,name!=='invalid');
   const pairs=await b.evaluate(`[...document.querySelectorAll('${root} .deployment-lab__results>div')].map(d=>({label:d.querySelector('dt').textContent,value:d.querySelector('output').textContent}))`);
   const hasPlot=name!=='invalid'&&name!=='unschedulable';
   assert.equal(await b.evaluate(`!document.querySelector('${root} [data-plot]').hidden&&!document.querySelector('${root} [data-computed]').hidden`),hasPlot);
   if(hasPlot){
    await b.cdp('Emulation.setEmulatedMedia',{media:'print',features});
    const palette=await b.evaluate(`(()=>{const r=document.querySelector('${root}'),paths=[...r.querySelectorAll('svg path')];return{background:getComputedStyle(r).backgroundColor,text:getComputedStyle(r).color,curves:paths.map(p=>({stroke:getComputedStyle(p).stroke,dash:getComputedStyle(p).strokeDasharray}))}})()`);
    assert.equal(palette.background,'rgb(255, 255, 255)');
    // Repository print rules may choose #111 over the widget's #202020.
    assert(/^rgb\(\d+, \d+, \d+\)$/.test(palette.text)&&palette.text.match(/\d+/g).every(v=>Number(v)<=32),JSON.stringify(palette));
    assert(palette.curves.length>=1&&palette.curves[0].stroke==='rgb(21, 94, 157)',JSON.stringify(palette));
    if(example.kind!=='timing'){assert.equal(palette.curves[1].stroke,'rgb(109, 40, 217)');assert.notEqual(palette.curves[1].dash,'none')}
    report.palettes.push({kind:example.kind,name,forced,...palette});
    await b.cdp('Emulation.setEmulatedMedia',{media:'',features});await pause(180);
   }
   const settled=await snapshot();
   for(const paper of ['A4','Letter']){
    const file=path.join(output,`${example.kind}-${name}-${forced?'forced':'normal'}-${paper}.pdf`);
    const pdf=await b.cdp('Page.printToPDF',{printBackground:true,paperWidth:paper==='A4'?8.27:8.5,paperHeight:paper==='A4'?11.69:11,generateTaggedPDF:true});
    fs.writeFileSync(file,Buffer.from(pdf.data,'base64'));
    const raw=execFileSync('pdftotext',[file,'-'],{encoding:'utf8',maxBuffer:12*1024*1024}),pages=raw.split('\f').map(normalize);
    let resultPairs=0;
    for(const pair of pairs){
     const label=normalize(pair.label),value=normalize(pair.value);
     if(name==='invalid'){assert(!value);continue}
     assert.equal(pages.filter(p=>p.includes(label+value)).length,1,file+' missing or split label/value '+label+'='+value);resultPairs++;
    }
    if(name==='invalid')assert(normalize(raw).includes('请填写各字段标示范围内的有限数值'),file+' missing invalid-input message');
    if(name==='unschedulable')assert(normalize(raw).includes('旧块在新结果能够执行前已用尽'),file+' missing schedule diagnosis');
    const bbox=execFileSync('pdftotext',['-bbox',file,'-'],{encoding:'utf8',maxBuffer:24*1024*1024});
    const geometry=JSON.parse(execFileSync('python3',['-B','-c',`import sys,json,xml.etree.ElementTree as E
r=E.fromstring(sys.stdin.buffer.read());ns={'x':'http://www.w3.org/1999/xhtml'};pages=r.findall('.//x:page',ns);bad=[]
for i,p in enumerate(pages,1):
 for w in p.findall('.//x:word',ns):
  if float(w.attrib['xMin'])<-.5 or float(w.attrib['xMax'])>float(p.attrib['width'])+.5:bad.append([i,w.text])
print(json.dumps({'pages':len(pages),'outside':bad}))`],{input:bbox,encoding:'utf8'}));
    assert(!geometry.outside.length,JSON.stringify({file,geometry}));
    assert(/Tagged:\s+yes/.test(execFileSync('pdfinfo',[file],{encoding:'utf8'})));
    await pause(180);const after=await snapshot();
    assert.deepEqual(after.inputs,settled.inputs);assert.deepEqual(after.values,settled.values);assert.equal(after.focus,settled.focus);
    assert.equal(after.computed,settled.computed);assert.equal(after.error,settled.error);
    assert(Math.abs(after.scroll-settled.scroll)<=2,JSON.stringify({file,settled,after}));
    report.exports.push({kind:example.kind,name,forced,paper,file,...geometry,resultPairs,tagged:true,focusPreserved:true,inputsPreserved:true,screenScrollDelta:after.scroll-settled.scroll});
   }
  }
  assert(!b.exceptions.length,JSON.stringify(b.exceptions));console.log(JSON.stringify({exports:report.exports.length,pages:report.exports.reduce((n,r)=>n+r.pages,0),resultPairs:report.exports.reduce((n,r)=>n+r.resultPairs,0),palettes:report.palettes.length,exceptions:0}));
 }finally{await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]});fs.writeFileSync(path.join(output,'lab-print-states.json'),JSON.stringify(report,null,2)+'\n');await b.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
