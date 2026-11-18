#!/usr/bin/env node
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {connect}=require('../../../scripts/lib/browser.cjs');
const {contrastAudit}=require('../../../scripts/lib/contrast.cjs');
const base=process.argv[2].replace(/\/$/,''),port=Number(process.argv[3]),out=process.argv[4];
const forced=process.argv.includes('--forced-colors');fs.mkdirSync(out,{recursive:true});
(async()=>{
 const b=await connect(port),rows=[];
 try{
  for(const width of [718,739])for(const size of [100,200])for(const scripts of [true,false])for(const theme of ['light','dark','warm']){
   await b.viewport(width);await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]});
   await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.navigate(base+'/posts/ai/pyrua-lean/');
   await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});
   await b.evaluate(`document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='${size}%';document.fonts.ready`);
   await b.cdp('Emulation.setEmulatedMedia',{media:'print',features:forced?[{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:theme==='dark'?'dark':'light'}]:[]});
   await b.pause(100);
   const state=await b.evaluate(`(()=>{
    const article=document.querySelector('.content'),bounds=article.getBoundingClientRect();
    const math=[...article.querySelectorAll('.katex')].flatMap(e=>{
     const html=e.querySelector('.katex-html');if(!html)return [];
     const parent=e.closest('.katex-display,td,th,p,li')||e.parentElement;
     const r=html.getBoundingClientRect(),p=parent.getBoundingClientRect();if(!r.width)return [];
     const bases=[...html.querySelectorAll('.base')].map(e=>e.getBoundingClientRect());
     const left=Math.min(r.left,...bases.map(e=>e.left)),right=Math.max(r.right,...bases.map(e=>e.right));
     return right>p.right+1||left<p.left-1?[{tex:e.querySelector('annotation')?.textContent,width:right-left,available:p.width}]:[];
    });
    const code=[...article.querySelectorAll('pre')].filter(e=>e.clientWidth&&e.scrollWidth>e.clientWidth+1).map(e=>e.textContent.slice(0,80));
    const tables=[...article.querySelectorAll('table')].filter(e=>e.getBoundingClientRect().right>bounds.right+2).length;
    return {math,code,tables,paper:getComputedStyle(document.body).backgroundColor==='rgb(255, 255, 255)',
      contrast:(${contrastAudit.toString()})('.content')};})()`);
   rows.push({width,size,scripts,theme,forced,...state});
  }
  const failures=rows.filter(r=>r.math.length||r.code.length||r.tables||!r.paper||r.contrast.issues.length);
  fs.writeFileSync(path.join(out,'results.json'),JSON.stringify({rows,failures,exceptions:b.exceptions},null,2)+'\n');
  console.log(JSON.stringify({layouts:rows.length,failures,exceptions:b.exceptions.length}));
  assert.equal(failures.length,0);assert.equal(b.exceptions.length,0);
 }finally{await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
