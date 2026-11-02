// Inspect the three new Pion details at real mobile/desktop sizes. Run CDP serially.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {connect}=require('../../../scripts/lib/browser.cjs');
const base=process.argv[2],port=Number(process.argv[3]),out=process.argv[4];
const ids=['snr-estimation','flat-rank-study','vla-padding-recovery'];
(async()=>{
 fs.mkdirSync(out,{recursive:true});const b=await connect(port),report={layouts:[],tables:[],exceptions:b.exceptions};
 try{
  for(const scripts of [true,false])for(const width of [320,768,1440])for(const theme of ['light','dark','warm'])for(const size of [100,200])for(const spacing of width===768?[false]:[false,true]){
   await b.viewport(width);await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});
   await b.navigate(base+'/posts/ai/pion-muon-vla-rlvr/');
   await b.evaluate(`(()=>{document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='${size}%';for(const id of ${JSON.stringify(ids)})document.getElementById(id).open=true;${spacing?"const s=document.createElement('style');s.textContent='body :is(p,li,dt,dd,th,td,summary,h1,h2,h3,h4,h5,h6,a,button,label,input,select){line-height:1.5!important;letter-spacing:.12em!important;word-spacing:.16em!important} .katex {letter-spacing:normal!important;word-spacing:normal!important}';document.head.append(s);":""}})()`);
   await b.evaluate(`Promise.all([document.fonts.ready,...[...document.querySelectorAll('#flat-rank-study img')].map(i=>{i.loading='eager';return i.decode()})])`);await b.pause(180);
   const state=await b.evaluate(`(()=>{const ds=${JSON.stringify(ids)}.map(id=>document.getElementById(id));const issues=[];for(const d of ds){for(const e of d.querySelectorAll('table,pre,.katex-display')){if(e.scrollWidth>e.clientWidth+1&&e.tabIndex!==0&&${scripts})issues.push({tag:e.tagName,reason:'overflow is not focusable'});}for(const e of d.querySelectorAll('figcaption>span')){const box=e.getBoundingClientRect(),r=document.createRange();r.selectNodeContents(e);for(const t of r.getClientRects())if(t.left<box.left-1||t.right>box.right+1||t.top<box.top-1||t.bottom>box.bottom+1)issues.push({reason:'caption clipping',text:e.textContent,box:box.toJSON(),textBox:t.toJSON()});}}const i=document.querySelector('#flat-rank-study img');return{overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1,issues,image:{complete:i.complete,width:i.naturalWidth,reserved:i.hasAttribute('width')&&i.hasAttribute('height')},captions:ds.flatMap(d=>[...d.querySelectorAll('figcaption')].map(c=>c.textContent.trim()))}})()`);
   assert(!state.overflow&&state.issues.length===0&&state.image.complete&&state.image.width>0&&state.image.reserved,JSON.stringify({scripts,width,theme,size,spacing,state}));
   report.layouts.push({scripts,width,theme,size,spacing,...state});
   if(width===320&&size===200&&!spacing&&theme==='dark'){
    await b.evaluate(`document.getElementById('fig-pion-flat-rms').scrollIntoView({block:'start'})`);await b.screenshot(path.join(out,'flat-mobile-dark-'+scripts+'.png'));
   }
   if(width===320&&!spacing&&theme==='light')for(const id of ids){
    const count=await b.evaluate(`document.getElementById('${id}').querySelectorAll('table').length`);
    for(let index=0;index<count;index++){
     const overflowing=await b.evaluate(`(()=>{window.__table=document.getElementById('${id}').querySelectorAll('table')[${index}];__table.scrollIntoView({block:'center'});__table.scrollLeft=0;if(__table.scrollWidth<=__table.clientWidth+1)return false;__table.focus();return true})()`);
     if(overflowing){for(let n=0;n<5;n++)await b.key('ArrowRight','ArrowRight',39);await b.pause(180);const row=await b.evaluate(`({focus:document.activeElement===__table,left:__table.scrollLeft,tabIndex:__table.tabIndex})`);assert(row.focus&&row.left>0,JSON.stringify({scripts,size,id,index,row}));report.tables.push({scripts,size,id,index,...row});}
    }
   }
  }
  assert.deepEqual(b.exceptions,[]);console.log(JSON.stringify({layouts:report.layouts.length,keyboardTables:report.tables.length,exceptions:b.exceptions.length}));
 }finally{fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
