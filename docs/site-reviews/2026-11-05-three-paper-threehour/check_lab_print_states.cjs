#!/usr/bin/env node
// Native PDFs after edited calculator inputs, including invalid and forced colors.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const {connect,pause}=require('../../../scripts/lib/browser.cjs');
const [base='http://127.0.0.1:13247',port='9337',output='/tmp/three-paper-lab-print-states']=process.argv.slice(2);
const examples=[
 {kind:'last',route:'/posts/ai/last-action-tokenizer/',root:'[data-last-whitening]',fields:['ratio','angle','epsilon'],states:[['tie',[2,0,0]],['regularized',[3,0,3]],['rotated',[10,90,10]],['invalid',['',0,0]]],labels:{metricA:'A的白化距离',metricB:'B的白化距离',euclideanWinner:'原坐标选择',metricWinner:'白化度量选择',condition:'协方差条件数'}},
 {kind:'omni',route:'/posts/calibration/omnicalib/',root:'[data-omnicalib-prior]',fields:['mean','std','excitation'],states:[['excited',[.4,.02,1]],['weak',[.4,.02,.0001]],['strong',[.4,.005,1]],['invalid',[.4,.02,'']]],labels:{knee:'膝偏置估计（°）',ankle:'踝偏置估计（°）',kneeError:'膝实际误差（°）',ankleError:'踝实际误差（°）',kneeStd:'膝局部标准差（°）',ankleStd:'踝局部标准差（°）'}}
];
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(Number(port));const report={exports:[],exceptions:b.exceptions};
 try{
  for(const example of examples)for(const [name,inputs]of example.states)for(const forced of [false,true]){
   await b.cdp('Emulation.setEmulatedMedia',{media:'',features:forced?[{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:'dark'}]:[]});
   await b.viewport(390,1000);await b.navigate(base+example.route);
   await b.evaluate(`document.documentElement.classList.add('dark');document.documentElement.style.fontSize='200%';document.fonts.ready`);
   await b.evaluate(`(()=>{const r=document.querySelector('${example.root}'),names=${JSON.stringify(example.fields)},values=${JSON.stringify(inputs)};names.forEach((name,i)=>{r.querySelector('input[name="'+name+'"]').value=values[i]});const f=r.querySelector('input');f.focus();f.dispatchEvent(new Event('input',{bubbles:true}));r.querySelector('[data-${name==='invalid'?'error':'computed'}]').scrollIntoView({block:'center',behavior:'instant'})})()`);
   const before=await b.evaluate(`(()=>{const r=document.querySelector('${example.root}');return{values:Object.fromEntries([...r.querySelectorAll('output')].map(e=>[e.dataset.value,e.textContent])),inputs:[...r.querySelectorAll('input')].map(e=>e.value),computed:r.querySelector('[data-computed]').hidden,error:r.querySelector('[data-error]').hidden,scroll:scrollY,focus:document.activeElement.id}})()`);
   assert.equal(before.computed,name==='invalid');assert.equal(before.error,name!=='invalid');
   for(const paper of ['A4','Letter']){
    const file=path.join(output,`${example.kind}-${name}-${forced?'forced':'normal'}-${paper}.pdf`);
    const pdf=await b.cdp('Page.printToPDF',{printBackground:true,paperWidth:paper==='A4'?8.27:8.5,paperHeight:paper==='A4'?11.69:11,generateTaggedPDF:true});
    fs.writeFileSync(file,Buffer.from(pdf.data,'base64'));
    const raw=execFileSync('pdftotext',[file,'-'],{encoding:'utf8'}),pages=raw.split('\f').map(p=>p.replace(/\s/g,''));
    let pairs=0;
    for(const [key,label]of Object.entries(example.labels)){
     const matches=pages.filter(page=>page.includes(label));
     if(name==='invalid'){assert.equal(matches.length,0,file+' stale result label '+label);continue}
     assert.equal(matches.length,1,file+' ambiguous label '+label);
     assert(matches[0].slice(matches[0].indexOf(label)+label.length).startsWith(before.values[key]),file+' split or stale result '+key);pairs++;
    }
    if(name==='invalid')assert(raw.replace(/\s/g,'').includes('请填写范围内的有限数值'),file+' missing error');
    const bbox=execFileSync('pdftotext',['-bbox',file,'-'],{encoding:'utf8'});
    const geometry=JSON.parse(execFileSync('python3',['-B','-c',`import sys,json,xml.etree.ElementTree as E
r=E.fromstring(sys.stdin.buffer.read());ns={'x':'http://www.w3.org/1999/xhtml'};pages=r.findall('.//x:page',ns);bad=[]
for i,p in enumerate(pages,1):
 for w in p.findall('.//x:word',ns):
  if float(w.attrib['xMin'])<-.5 or float(w.attrib['xMax'])>float(p.attrib['width'])+.5:bad.append([i,w.text])
print(json.dumps({'pages':len(pages),'outside':bad}))`],{input:bbox,encoding:'utf8'}));
    assert(!geometry.outside.length,JSON.stringify({file,geometry}));await pause(160);
    const after=await b.evaluate(`(()=>{const r=document.querySelector('${example.root}');return{values:Object.fromEntries([...r.querySelectorAll('output')].map(e=>[e.dataset.value,e.textContent])),inputs:[...r.querySelectorAll('input')].map(e=>e.value),computed:r.querySelector('[data-computed]').hidden,error:r.querySelector('[data-error]').hidden,scroll:scrollY,focus:document.activeElement.id}})()`);
    assert.deepEqual(after.inputs,before.inputs);assert.deepEqual(after.values,before.values);assert.equal(after.focus,before.focus);assert(Math.abs(after.scroll-before.scroll)<=2,JSON.stringify({file,before,after}));
    report.exports.push({kind:example.kind,name,forced,paper,file,...geometry,pairs,focusPreserved:true,inputsPreserved:true,screenScrollDelta:after.scroll-before.scroll});
   }
  }
  assert(!b.exceptions.length,JSON.stringify(b.exceptions));console.log(JSON.stringify({exports:report.exports.length,pages:report.exports.reduce((n,e)=>n+e.pages,0),resultPairs:report.exports.reduce((n,e)=>n+e.pairs,0),exceptions:0}));
 }finally{await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]});fs.writeFileSync(path.join(output,'lab-print-states.json'),JSON.stringify(report,null,2)+'\n');await b.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
