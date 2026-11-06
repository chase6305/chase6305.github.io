#!/usr/bin/env node
// Native A4/Letter PDFs and compact result-card pagination for this article.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {execFileSync}=require('node:child_process');
const {connect}=require(path.resolve(__dirname,'../../../scripts/lib/browser.cjs'));
const {contrastAudit}=require(path.resolve(__dirname,'../../../scripts/lib/contrast.cjs'));
const base=process.argv[2]||'http://127.0.0.1:13217';
const port=Number(process.argv[3]||9337);
const output=process.argv[4]||'/tmp/ultralytics-print';
fs.mkdirSync(output,{recursive:true});
const inspector=String.raw`
import json,sys,xml.etree.ElementTree as ET
root=ET.fromstring(sys.stdin.buffer.read())
ns={'x':'http://www.w3.org/1999/xhtml'}
compact=lambda s:''.join(s.split())
pages=[]
for page in root.findall('.//x:page',ns):
 words=[dict(text=w.text or '',**{k:float(v)for k,v in w.attrib.items()})for w in page.findall('.//x:word',ns)]
 pages.append(words)
def locate(words,text):
 text=compact(text)
 for start in range(len(words)):
  value='';group=[]
  for w in words[start:start+64]:
   value+=compact(w['text']);group.append(w)
   if value==text:
    yield {k:f(w[k]for w in group)for k,f in [('xMin',min),('yMin',min),('xMax',max),('yMax',max)]}
    break
   if not text.startswith(value):break
rows=[]
for card in json.loads(sys.argv[1]):
 found=[]
 for index,words in enumerate(pages,1):
  for label in locate(words,card['label']):
   for value in locate(words,card['value']):
    if abs(label['xMin']-value['xMin'])<4 and label['yMin']+2<value['yMin']<label['yMax']+70:
     found.append({'page':index,'label':label,'value':value})
 rows.append(dict(card,passed=bool(found),locations=found))
print(json.dumps({'pages':len(pages),'cards':rows,'passed':all(r['passed']for r in rows)},ensure_ascii=False))
`;
(async()=>{
 const b=await connect(port),report={cases:[],exceptions:b.exceptions};
 try{
  for(const theme of ['light','dark','warm'])for(const size of [100,200]){
   await b.viewport(390,1000);await b.navigate(base+'/posts/ai/ultralytics/');
   await b.evaluate(`document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='${size}%';document.querySelector('[data-preset=minimal]').click();Promise.all([document.fonts.ready,...[...document.querySelectorAll('article img')].map(i=>{i.loading='eager';return i.decode()})])`);
   const cards=await b.evaluate(`Array.from(document.querySelectorAll('.ultralytics-geometry__results > div')).map(e=>({label:e.querySelector('dt').textContent.trim(),value:e.querySelector('dd').textContent.trim()}))`);
   assert.equal(cards.length,8);
   const original=await b.evaluate(`JSON.stringify([...document.querySelectorAll('[data-ultralytics-geometry] input')].map(e=>[e.value,e.checked]))`);
   // Measure print geometry, then return to screen before native export.
   await b.cdp('Emulation.setEmulatedMedia',{media:'print',features:[]});
   const colors=await b.evaluate('('+contrastAudit.toString()+')('+JSON.stringify('[data-ultralytics-geometry]')+')');
   assert.equal(colors.issues.length,0,JSON.stringify(colors));
   const geometry=await b.evaluate(`(()=>{const r=document.querySelector('[data-ultralytics-geometry]'),svg=r.querySelector('svg');return {width:r.getBoundingClientRect().width,svgWidth:svg.getBoundingClientRect().width,overflow:document.documentElement.scrollWidth>innerWidth+1,fill:getComputedStyle(svg.querySelector('.ultralytics-geometry__box')).fill}})()`);
   await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]});
   for(const paper of [{name:'a4',width:8.27,height:11.69},{name:'letter',width:8.5,height:11}]){
    const file=path.join(output,`${theme}-${size}-${paper.name}.pdf`);
    const pdf=await b.cdp('Page.printToPDF',{printBackground:true,paperWidth:paper.width,paperHeight:paper.height,preferCSSPageSize:false,generateTaggedPDF:true});
    fs.writeFileSync(file,Buffer.from(pdf.data,'base64'));
    const xml=execFileSync('pdftotext',['-bbox',file,'-'],{encoding:'utf8',timeout:10000});
    const pagination=JSON.parse(execFileSync('python3',['-B','-c',inspector,JSON.stringify(cards)],{input:xml,encoding:'utf8',timeout:10000}));
    report.cases.push({theme,size,paper:paper.name,file,geometry,...pagination});
    fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(report,null,2));
    assert(pagination.passed,JSON.stringify({file,cards:pagination.cards.filter(r=>!r.passed)}));
    assert.equal(await b.evaluate(`JSON.stringify([...document.querySelectorAll('[data-ultralytics-geometry] input')].map(e=>[e.value,e.checked]))`),original);
   }
  }
  // Print colors remain independent of a dark forced-colors screen palette.
  await b.cdp('Emulation.setEmulatedMedia',{media:'print',features:[{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:'dark'}]});
  const forced=await b.evaluate(`(()=>{const r=document.querySelector('[data-ultralytics-geometry]');return{bg:getComputedStyle(r).backgroundColor,fg:getComputedStyle(r).color,fill:getComputedStyle(r.querySelector('.ultralytics-geometry__box')).fill}})()`);
  assert.equal(forced.bg,'rgb(255, 255, 255)');assert.equal(forced.fill,'rgb(191, 219, 254)');
  report.forcedPrintPalette=forced;
  assert.equal(b.exceptions.length,0,JSON.stringify(b.exceptions));
  console.log(JSON.stringify({cases:report.cases.length,cardPairs:report.cases.reduce((n,c)=>n+c.cards.length,0),passed:true,forcedPrintPalette:forced}));
 }finally{
  fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(report,null,2));
  await b.close();
 }
})().catch(e=>{console.error(e);process.exitCode=1});
