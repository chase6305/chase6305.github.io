#!/usr/bin/env node
// Direct native PDF export from screen media: inspect actual painted paper edges.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const {connect,pause}=require('../../../scripts/lib/browser.cjs');
const [base='http://127.0.0.1:13327',port='9337',output='/tmp/deployment-sevenhour-print-canvas']=process.argv.slice(2);
const posts=[['deployment','/posts/ai/rma-umi-rtc-deployment/'],['contact','/posts/calibration/interactive-contact-calibration/'],['residual','/posts/ai/residual-reinforcement-learning/']];
const palettes=[['light',[]],['dark',[]],['warm',[]],['forced-light',[{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:'light'}]],['forced-dark',[{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:'dark'}]]];
function paperPixels(file){
 const prefix=file.replace(/\.pdf$/,'');execFileSync('pdftoppm',['-f','1','-l','1','-scale-to','300','-png','-singlefile',file,prefix]);
 return JSON.parse(execFileSync('python3',['-B','-c',`import sys,json
from PIL import Image
p=Image.open(sys.argv[1]).convert('RGB');w,h=p.size
points=[(0,0),(w//2,0),(w-1,0),(0,h//2),(w-1,h//2),(0,h-1),(w//2,h-1),(w-1,h-1)]
colors=[list(p.getpixel(point)) for point in points]
print(json.dumps({'width':w,'height':h,'edge_colors':colors,'white':all(min(color)>=250 for color in colors)}))`,prefix+'.png'],{encoding:'utf8'}));
}
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(Number(port));const report={exports:[],negativeControl:null,exceptions:b.exceptions};
 try{
  for(const [name,route]of posts)for(const [palette,features]of palettes){
   await b.cdp('Emulation.setEmulatedMedia',{media:'',features});await b.viewport(390,1000);await b.navigate(base+route);
   await b.evaluate(`document.documentElement.classList.remove('dark','warm');document.documentElement.classList.add('${palette==='dark'||palette==='forced-dark'?'dark':palette==='warm'?'warm':'light'}');Promise.all([document.fonts.ready,...[...document.querySelectorAll('article img')].map(i=>{i.loading='eager';return i.decode()})])`);
   await b.evaluate(`document.querySelector('#figure-2').scrollIntoView({block:'center',behavior:'instant'})`);await pause(80);
   const before=await b.evaluate('({scroll:scrollY,width:document.documentElement.scrollWidth})');
   for(const paper of ['A4','Letter']){
    const file=path.join(output,`${name}-${palette}-${paper}.pdf`);
    const pdf=await b.cdp('Page.printToPDF',{printBackground:true,paperWidth:paper==='A4'?8.27:8.5,paperHeight:paper==='A4'?11.69:11,generateTaggedPDF:true});fs.writeFileSync(file,Buffer.from(pdf.data,'base64'));
    const pixels=paperPixels(file);assert(pixels.white,JSON.stringify({file,pixels}));await pause(120);
    const after=await b.evaluate('({scroll:scrollY,width:document.documentElement.scrollWidth})');assert.equal(after.width,before.width);assert(Math.abs(after.scroll-before.scroll)<=2,JSON.stringify({file,before,after}));
    report.exports.push({name,palette,paper,file,...pixels,screenRestored:true});
   }
  }
  // Remove the explicit page paint in a controlled fixture. The oracle must
  // detect the previous dark margin, independently of body/root style checks.
  await b.cdp('Emulation.setEmulatedMedia',{media:'',features:palettes.at(-1)[1]});
  const fixture='<!doctype html><html lang="en"><meta charset="utf-8"><title>Print canvas negative control</title><style>html{background:#111;color:#eee;color-scheme:dark}body{margin:3rem;font:30px sans-serif}@media print{html,body{background:#fff!important;color:#111!important}html{forced-color-adjust:none;color-scheme:light!important}}</style><h1>Print canvas</h1><p>Body and root print white; no explicit page paint.</p></html>';
  await b.navigate('data:text/html;charset=utf-8,'+encodeURIComponent(fixture));
  const controlDir=path.join(output,'control');fs.mkdirSync(controlDir,{recursive:true});const file=path.join(controlDir,'root-only-page.pdf');
  const pdf=await b.cdp('Page.printToPDF',{printBackground:true,paperWidth:8.27,paperHeight:11.69});fs.writeFileSync(file,Buffer.from(pdf.data,'base64'));
  report.negativeControl={file,...paperPixels(file)};assert(!report.negativeControl.white,'The negative-control canvas was unexpectedly white');
  assert(!b.exceptions.length,JSON.stringify(b.exceptions));console.log(JSON.stringify({exports:report.exports.length,paperEdgeSamples:report.exports.length*8,whiteEdges:true,negativeControlDetected:true,exceptions:0}));
 }finally{await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]});fs.writeFileSync(path.join(output,'print-canvas.json'),JSON.stringify(report,null,2)+'\n');await b.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
