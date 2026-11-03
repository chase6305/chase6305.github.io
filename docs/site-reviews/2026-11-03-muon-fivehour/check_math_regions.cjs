// Actual native arrow keys in the two supplemental matrix/filter derivations.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {connect}=require('../../../scripts/lib/browser.cjs');
const [base,portText,out]=process.argv.slice(2);
const articles=[
  {slug:'muon-optimizer',summary:'Muon 与 Shampoo'},
  {slug:'muon-learning-rate-batch-size',summary:'保持 EMA 的稳态噪声方差'},
];
(async()=>{
  fs.mkdirSync(out,{recursive:true});const b=await connect(Number(portText)),report={cases:[],exceptions:b.exceptions};
  const run=(fn,...args)=>b.evaluate('('+fn.toString()+')(...'+JSON.stringify(args)+')');
  try{
    for(const article of articles)for(const scripts of [true,false])for(const size of [100,200]){
      await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.viewport(320,1000);
      await b.navigate('about:blank');await b.navigate(base+'/posts/ai/'+article.slug+'/');
      await run((summary,size)=>{
        window.__fold=[...document.querySelectorAll('.content details')].find(d=>d.querySelector('summary')?.textContent.includes(summary));
        if(!__fold)throw Error('Missing derivation');
        __fold.open=true;document.documentElement.style.fontSize=size+'%';return document.fonts.ready;
      },article.summary,size);
      await b.pause(180);
      const before=await run(()=>{
        window.__math=[...__fold.querySelectorAll('.katex-display')].find(e=>e.scrollWidth>e.clientWidth+1)||__fold.querySelector('.katex-display');
        if(!__math)return null;
        __math.scrollLeft=0;__math.scrollIntoView({block:'center'});__math.focus();
        return {focused:document.activeElement===__math,tabIndex:__math.tabIndex,width:__math.clientWidth,scroll:__math.scrollWidth,fits:__math.scrollWidth<=__math.clientWidth+1};
      });
      if(before?.fits){
        // Chrome can focus a scroll container with a one-pixel rounding
        // overflow even after the repository's temporary tabindex is removed.
        assert(scripts?before.tabIndex===-1:(before.focused&&before.tabIndex===0),JSON.stringify(before));
        report.cases.push({article:article.slug,scripts,size,before,fitting:true});continue;
      }
      assert(before&&before.focused&&before.tabIndex===0,JSON.stringify({article,scripts,size,before}));
      await b.key('ArrowRight','ArrowRight',39);await b.pause(180);
      const after=await run(()=>({focused:document.activeElement===__math,left:__math.scrollLeft,pageX:scrollX}));
      assert(after.focused&&after.left>0&&after.pageX===0,JSON.stringify(after));
      await run(()=>{document.documentElement.style.fontSize='100%';});await b.viewport(1440,1000);await b.pause(180);
      const wide=await run(()=>({focused:document.activeElement===__math,tabIndex:__math.tabIndex,fits:__math.scrollWidth<=__math.clientWidth+1}));
      assert(wide.focused&&wide.fits&&wide.tabIndex===0,JSON.stringify(wide));
      await b.key('Tab','Tab',9);await b.pause(120);
      const blurred=await run(()=>({focused:document.activeElement===__math,tabIndex:__math.tabIndex}));
      assert(!blurred.focused&&blurred.tabIndex===(scripts?-1:0),JSON.stringify(blurred));
      report.cases.push({article:article.slug,scripts,size,before,after,wide,blurred});
    }
    assert.deepEqual(b.exceptions,[]);
    console.log(JSON.stringify({arrowContexts:report.cases.filter(c=>!c.fitting).length,fittingContexts:report.cases.filter(c=>c.fitting).length,exceptions:0}));
  }finally{
    fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');
    await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});await b.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1});
