// Capture the revised request/response and training entrances on a fresh build.
const fs=require('node:fs'),path=require('node:path');
const {connect}=require('../../../scripts/lib/browser.cjs');
const [base,port,out]=process.argv.slice(2);
(async()=>{
 const b=await connect(Number(port)),report={captures:[],exceptions:b.exceptions};
 fs.mkdirSync(out,{recursive:true});
 try{
  for(const [width,size,theme] of [[320,100,'light'],[320,200,'dark'],[1440,100,'light']]){
   await b.viewport(width,1000);await b.navigate(base+'/posts/ai/jev-decision-layer/');
   await b.evaluate(`document.querySelector('[data-item=${theme}]').click();
    document.documentElement.style.colorScheme='${theme==='dark'?'dark':'light'}';
    document.documentElement.style.fontSize='${size}%';
    document.querySelectorAll('.content details:not(.blog-mobile-toc)').forEach(d=>d.open=true);
    Promise.all([document.fonts.ready,...[...document.querySelectorAll('.content img')].map(i=>{i.loading='eager';return i.decode()})])`);
   for(const id of ['onejev-response-example','onejev-startup-troubleshooting','onejev-data-and-evaluation']){
    await b.evaluate(`document.getElementById('${id}').scrollIntoView({block:'start'})`);await b.pause(150);
    const file=path.join(out,`${id}-${width}-${size}-${theme}.png`);await b.screenshot(file);
    report.captures.push({id,width,size,theme,file});
   }
   await b.evaluate(`window.__responseTable=[...document.querySelectorAll('.content table')].find(t=>t.textContent.includes('响应中的原始位置'));
    __responseTable.scrollIntoView({block:'start'});__responseTable.scrollLeft=0;
    scrollBy(0,-(document.querySelector('.hextra-nav-container nav')?.getBoundingClientRect().height||64)-24)`);
   for(const side of ['left','right']){
    if(side==='right')await b.evaluate('__responseTable.scrollLeft=__responseTable.scrollWidth');
    const file=path.join(out,`response-table-${side}-${width}-${size}-${theme}.png`);await b.screenshot(file);
    report.captures.push({id:'response-table',side,width,size,theme,file});
   }
  }
 }finally{fs.writeFileSync(path.join(out,'captures.json'),JSON.stringify(report,null,2)+'\n');await b.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
