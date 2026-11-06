#!/usr/bin/env node
// New-article A4 geometry at 200%, including native HTML without page scripts.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {connect,pause} = require('../../../scripts/lib/browser.cjs');
const {contrastAudit} = require('../../../scripts/lib/contrast.cjs');
const base=process.argv[2]||'http://127.0.0.1:13227';
const port=Number(process.argv[3]||9337);
const output=process.argv[4]||'/tmp/last-omnicalib-print-geometry';
const forced=process.argv.includes('--forced-colors');
const features=forced?[{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:'dark'}]:[];
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(port);const report={layouts:[],exceptions:b.exceptions,forced};
 try {
  await b.viewport(716,1000);
  for(const scripts of [true,false])for(const route of ['/posts/ai/last-action-tokenizer/','/posts/calibration/omnicalib/','/posts/ai/machine-learning-for-robotics/','/posts/ai/ultralytics/','/posts/robotics/curobo-v2-fast/'])for(const theme of ['light','dark','warm']){
   await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.navigate(base+route);
   await b.evaluate(`document.documentElement.classList.remove('dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='200%';document.fonts.ready`);
   await b.cdp('Emulation.setEmulatedMedia',{media:'print',features});await pause(80);
   const geometry=await b.evaluate(`(()=>{const math=[...document.querySelectorAll('.content .katex')].flatMap(e=>{const h=e.querySelector('.katex-html');if(!h)return[];const p=e.closest('.katex-display,td,th,p,li')||e.parentElement;const r=h.getBoundingClientRect(),bounds=p.getBoundingClientRect();if(!r.width)return[];const bases=[...h.querySelectorAll('.base')].map(b=>b.getBoundingClientRect());const left=Math.min(r.left,...bases.map(b=>b.left)),right=Math.max(r.right,...bases.map(b=>b.right));return right>bounds.right+1||left<bounds.left-1?[{tex:e.querySelector('annotation')?.textContent,width:right-left,available:bounds.width}]:[]});return {articlePresent:!!document.querySelector('article .content'),formulaCount:document.querySelectorAll('article .katex').length,math,code:[...document.querySelectorAll('.content pre')].filter(e=>e.clientWidth>0&&e.scrollWidth>e.clientWidth+1).length,tables:[...document.querySelectorAll('.content table')].filter(e=>e.getBoundingClientRect().right>document.documentElement.clientWidth+1).length,paper:getComputedStyle(document.body).backgroundColor==='rgb(255, 255, 255)',labInk:[...document.querySelectorAll('.omnicalib-prior__estimate')].map(e=>getComputedStyle(e).stroke)}})()`);
   const contrast=await b.evaluate('('+contrastAudit.toString()+')('+JSON.stringify('.content')+')');
   report.layouts.push({route,scripts,theme,...geometry,contrastIssues:contrast.issues});
   assert(geometry.articlePresent&&geometry.formulaCount>0&&!geometry.math.length&&!geometry.code&&!geometry.tables&&geometry.paper&&!contrast.issues.length,JSON.stringify(report.layouts.at(-1)));
   if(route.includes('omnicalib')&&scripts)assert(geometry.labInk.every(c=>c==='rgb(21, 94, 157)'),JSON.stringify(geometry));
   await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]});
  }
  assert(!b.exceptions.length);console.log(JSON.stringify({layouts:report.layouts.length,fontSize:200,failures:0,forced}));
 }finally{fs.writeFileSync(path.join(output,'geometry.json'),JSON.stringify(report,null,2)+'\n');await b.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
