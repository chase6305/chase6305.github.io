#!/usr/bin/env node
// Native PDF export: bounds, article tail, result pairs and screen restoration.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const {connect,pause}=require('../../../scripts/lib/browser.cjs');
const [base='http://127.0.0.1:13327',port='9337',output='/tmp/deployment-native-print']=process.argv.slice(2);
const posts=[
 {name:'deployment',route:'/posts/ai/rma-umi-rtc-deployment/',key:'d_deadline',ending:'动作块预测与表示的相关方法'},
 {name:'residual',route:'/posts/ai/residual-reinforcement-learning/',key:'8000',ending:'稳定性、约束和总体性能指标'},
 {name:'contact',route:'/posts/calibration/interactive-contact-calibration/',key:'2.0448',ending:'尺度、秩与留出验证的相关方法'}
].filter(p=>!process.argv.includes('--post')||p.name===process.argv[process.argv.indexOf('--post')+1]);
const normalize=s=>s.normalize('NFKC').replace(/\s/g,'').replace(/−/g,'-');
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(Number(port));const report={exports:[],exceptions:b.exceptions};
 try{
  for(const scripts of [true,false]) for(const post of posts) for(const size of [100,200]) for(const theme of ['light','dark','warm']){
   await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.viewport(390,1000);await b.navigate(base+post.route);
   await b.evaluate(`document.documentElement.classList.remove('dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='${size}%';Promise.all([document.fonts.ready,...[...document.querySelectorAll('article img')].map(i=>{i.loading='eager';return i.decode()})])`);
   await b.evaluate(`document.querySelector('#figure-2').scrollIntoView({block:'center'})`);await pause(120);
   const before=await b.evaluate(`({scroll:scrollY,width:document.documentElement.scrollWidth,font:document.documentElement.style.fontSize})`);
   const pairs=scripts?await b.evaluate(`[...document.querySelectorAll('.deployment-lab__results > div')].map(d=>({label:d.querySelector('dt').textContent,value:d.querySelector('output').textContent}))`):[];
   for(const paper of theme==='light'&&size===200?['A4','Letter']:['A4']){
    const file=path.join(output,`${post.name}-${scripts?'scripts':'native'}-${theme}-${size}-${paper}.pdf`);
    const pdf=await b.cdp('Page.printToPDF',{printBackground:true,paperWidth:paper==='A4'?8.27:8.5,paperHeight:paper==='A4'?11.69:11,generateTaggedPDF:true});
    fs.writeFileSync(file,Buffer.from(pdf.data,'base64'));
    const xml=execFileSync('pdftotext',['-bbox',file,'-'],{encoding:'utf8',maxBuffer:24*1024*1024});
    const geometry=JSON.parse(execFileSync('python3',['-B','-c',`import sys,json,xml.etree.ElementTree as E
r=E.fromstring(sys.stdin.buffer.read());ns={'x':'http://www.w3.org/1999/xhtml'};pages=r.findall('.//x:page',ns);bad=[]
for i,p in enumerate(pages,1):
 for w in p.findall('.//x:word',ns):
  if float(w.attrib['xMin'])<-.5 or float(w.attrib['xMax'])>float(p.attrib['width'])+.5:bad.append([i,w.text,w.attrib])
print(json.dumps({'pages':len(pages),'outside':bad}))`],{input:xml,encoding:'utf8'}));
    assert(!geometry.outside.length,JSON.stringify({file,geometry}));
    const raw=execFileSync('pdftotext',[file,'-'],{encoding:'utf8',maxBuffer:12*1024*1024}),text=normalize(raw);
    assert(text.includes('阅读自测与验收')&&text.includes(normalize(post.key))&&text.includes(normalize(post.ending)),file+' missing key content or tail');
    let resultPairs=0;const pages=raw.split('\f').map(normalize);
    for(const pair of pairs){
     const label=normalize(pair.label),value=normalize(pair.value),matches=pages.filter(p=>p.includes(label+value));
     assert.equal(matches.length,1,file+' missing or split result label/value '+label+'='+value);
     resultPairs++;
    }
    const info=execFileSync('pdfinfo',[file],{encoding:'utf8'});assert(/Tagged:\s+yes/.test(info),file+' is not tagged');
    await pause(200);const after=await b.evaluate(`({scroll:scrollY,width:document.documentElement.scrollWidth,font:document.documentElement.style.fontSize})`);
    const sameGeometry=before.width===after.width&&before.font===after.font,restored=Math.abs(before.scroll-after.scroll)<=2;
    const targetVisible=await b.evaluate(`(()=>{const r=document.querySelector('#figure-2').getBoundingClientRect();return r.bottom>0&&r.top<innerHeight})()`);
    assert(sameGeometry&&targetVisible&&(!scripts||restored),JSON.stringify({file,before,after}));
    report.exports.push({post:post.name,scripts,size,theme,paper,file,...geometry,tagged:true,tailPresent:true,resultPairs,sameGeometry,targetVisible,restored,nativeScrollDelta:after.scroll-before.scroll});
   }
  }
  assert(!b.exceptions.length,JSON.stringify(b.exceptions));
  console.log(JSON.stringify({exports:report.exports.length,pages:report.exports.reduce((s,r)=>s+r.pages,0),resultPairs:report.exports.reduce((s,r)=>s+r.resultPairs,0),outsideText:0,exceptions:0}));
 }finally{fs.writeFileSync(path.join(output,'native-print.json'),JSON.stringify(report,null,2)+'\n');await b.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
