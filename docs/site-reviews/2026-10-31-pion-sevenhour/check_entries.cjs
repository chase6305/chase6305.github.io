// Native reading-route links should expose the summary and preserve keyboard use.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {connect}=require('../../../scripts/lib/browser.cjs');
const[base,portText,out]=process.argv.slice(2);
(async()=>{fs.mkdirSync(out,{recursive:true});const b=await connect(Number(portText)),report={entries:[],exceptions:b.exceptions};
 try{
  for(const scripts of[true,false])for(const width of[320,1440])for(const size of[100,200])for(const id of['snr-estimation','flat-rank-study','vla-padding-recovery']){
   await b.viewport(width);await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.navigate(base+'/posts/ai/pion-muon-vla-rlvr/');
   await b.evaluate(`document.documentElement.style.fontSize='${size}%';document.fonts.ready`);await b.pause(180);
   await b.evaluate(`document.querySelector('.content a[href="#${id}"]').focus()`);await b.key('Enter','Enter',13);await b.pause(150);
   const before=await b.evaluate(`(()=>{const d=document.getElementById('${id}'),r=d.querySelector('summary').getBoundingClientRect();return{hash:location.hash,top:r.top,bottom:r.bottom,open:d.open,nav:document.querySelector('.hextra-nav-container nav').getBoundingClientRect().bottom,height:innerHeight,overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1}})()`);
   assert.equal(before.hash,'#'+id);assert(!before.open&&!before.overflow&&before.top>=Math.max(0,before.nav)-1&&before.bottom<=before.height,JSON.stringify({scripts,width,size,id,before}));
   await b.evaluate(`Promise.all([...document.querySelectorAll('.content img')].map(i=>{i.loading='eager';return i.decode()}))`);await b.pause(100);
   const after=await b.evaluate(`document.querySelector('#${id}>summary').getBoundingClientRect().top`);assert(Math.abs(after-before.top)<=1);
   await b.key('Tab','Tab',9);assert(await b.evaluate(`document.activeElement===document.querySelector('#${id}>summary')`));await b.key('Enter','Enter',13);await b.pause(100);assert(await b.evaluate(`document.getElementById('${id}').open`));
   await b.key('Enter','Enter',13);assert(await b.evaluate(`!document.getElementById('${id}').open&&document.activeElement===document.querySelector('#${id}>summary')`));
   report.entries.push({scripts,width,size,id,...before,imageLoadingShift:after-before.top,keyboardExpandedAndClosed:true});
  }
  assert.deepEqual(b.exceptions,[]);console.log(JSON.stringify({entries:report.entries.length,exceptions:0}));
 }finally{fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
