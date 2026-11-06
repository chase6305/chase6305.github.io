#!/usr/bin/env node
// Native PDF export, text bounds, content presence and screen restoration.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const {connect, pause} = require('../../../scripts/lib/browser.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13217';
const port = Number(process.argv[3] || 9337);
const output = process.argv[4] || '/tmp/last-omnicalib-native-print';
const posts = [
  {name:'last',route:'/posts/ai/last-action-tokenizer/',fragment:'Mahalanobis',ending:'合法SE(3)位姿为什么不能替代运动和接触约束检查'},
  {name:'omnicalib',route:'/posts/calibration/omnicalib/',fragment:'0.062881',ending:'为什么最好序列的0.775°不能证明整体优于iKalibr'},
  {name:'robotics-ml-review',route:'/posts/ai/machine-learning-for-robotics/',fragment:'13.910834',ending:'哪些部分仍需要具体算法和验证'},
].filter(post=>(!process.argv.includes('--omnicalib-only')||post.name==='omnicalib')&&(!process.argv.includes('--last-only')||post.name==='last')&&(!process.argv.includes('--review-only')||post.name==='robotics-ml-review'));

(async()=>{
  fs.mkdirSync(output,{recursive:true});
  const b = await connect(port);
  const report = {exports:[],exceptions:b.exceptions};
  try {
    for (const scripts of [true,false]) for (const post of posts) for (const size of [100,200]) for (const theme of ['light','dark','warm']) {
      await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});
      await b.viewport(390,1000);await b.navigate(base+post.route);
      await b.evaluate(`document.documentElement.classList.remove('dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='${size}%';Promise.all([document.fonts.ready,...[...document.querySelectorAll('article img')].map(i=>{i.loading='eager';return i.decode()})])`);
      await b.evaluate(`document.querySelector('#figure-2').scrollIntoView({block:'center'})`);await pause(120);
      const before = await b.evaluate(`({scroll:scrollY,width:document.documentElement.scrollWidth,font:document.documentElement.style.fontSize})`);
      for (const paper of (theme==='light'&&size===200 ? ['A4','Letter'] : ['A4'])) {
        const file = path.join(output,`${post.name}-${scripts?'scripts':'native'}-${theme}-${size}-${paper}.pdf`);
        const pdf = await b.cdp('Page.printToPDF',{printBackground:true,paperWidth:paper==='A4'?8.27:8.5,paperHeight:paper==='A4'?11.69:11,generateTaggedPDF:true});
        fs.writeFileSync(file,Buffer.from(pdf.data,'base64'));
        const xml = execFileSync('pdftotext',['-bbox',file,'-'],{encoding:'utf8'});
        const geometry = JSON.parse(execFileSync('python3',['-B','-c',`import sys,json,xml.etree.ElementTree as E
r=E.fromstring(sys.stdin.buffer.read());ns={'x':'http://www.w3.org/1999/xhtml'};pages=r.findall('.//x:page',ns);bad=[]
for i,p in enumerate(pages,1):
 for w in p.findall('.//x:word',ns):
  if float(w.attrib['xMin'])<-.5 or float(w.attrib['xMax'])>float(p.attrib['width'])+.5:bad.append([i,w.text,w.attrib])
print(json.dumps({'pages':len(pages),'outside':bad}))`],{input:xml,encoding:'utf8'}));
        assert(!geometry.outside.length,JSON.stringify({file,geometry}));
        const rawText = execFileSync('pdftotext',[file,'-'],{encoding:'utf8'});
        const text = rawText.replace(/\s/g,'');
        assert(text.includes('阅读自测与验收')&&text.includes(post.fragment)&&text.includes(post.ending),file+' missing article tail or key content');
        let resultCardPairs=0;
        if(post.name==='omnicalib'&&scripts){
          const pages=rawText.split('\f').map(p=>p.replace(/\s/g,''));
          for(const [label,value]of [['膝偏置估计（°）','0.4000'],['踝偏置估计（°）','−0.3000'],['膝实际误差（°）','+0.2000'],['踝实际误差（°）','−0.2000'],['膝局部标准差（°）','0.0200'],['踝局部标准差（°）','0.0283']]){
            const matches=pages.filter(p=>p.includes(label));assert.equal(matches.length,1,file+' label '+label);
            assert(matches[0].slice(matches[0].indexOf(label)+label.length).startsWith(value),file+' split result '+label);resultCardPairs++;
          }
        }
        if(post.name==='last'&&scripts){
          const pages=rawText.split('\f').map(p=>p.replace(/\s/g,''));
          for(const [label,value]of [['A的白化距离','0.6667'],['B的白化距离','1.0000'],['原坐标选择','B'],['白化度量选择','A'],['协方差条件数','9.0000']]){
            const matches=pages.filter(p=>p.includes(label));assert.equal(matches.length,1,file+' label '+label);
            assert(matches[0].slice(matches[0].indexOf(label)+label.length).startsWith(value),file+' split result '+label);resultCardPairs++;
          }
        }
        const info = execFileSync('pdfinfo',[file],{encoding:'utf8'});
        assert(/Tagged:\s+yes/.test(info),file+' is not tagged');
        await pause(200);
        const after = await b.evaluate(`({scroll:scrollY,width:document.documentElement.scrollWidth,font:document.documentElement.style.fontSize})`);
        const sameGeometry=before.width===after.width&&before.font===after.font;
        const restored=Math.abs(before.scroll-after.scroll)<=2;
        const targetVisible=await b.evaluate(`(()=>{const r=document.querySelector('#figure-2').getBoundingClientRect();return r.bottom>0&&r.top<innerHeight})()`);
        assert(sameGeometry&&targetVisible&&(!scripts||restored),JSON.stringify({file,before,after}));
        // Without page scripts Chrome owns scroll anchoring; record any native
        // drift rather than attributing a JS restoration guarantee to the page.
        report.exports.push({post:post.name,scripts,size,theme,paper,file,...geometry,tagged:true,tailPresent:true,sameGeometry,targetVisible,restored,resultCardPairs,nativeScrollDelta:after.scroll-before.scroll});
      }
    }
    assert(!b.exceptions.length,JSON.stringify(b.exceptions));
    console.log(JSON.stringify({nativeExports:report.exports.length,pages:report.exports.reduce((n,e)=>n+e.pages,0),outsideText:0,exceptions:0}));
  } finally {
    fs.writeFileSync(path.join(output,'native-print.json'),JSON.stringify(report,null,2)+'\n');
    await b.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1});
