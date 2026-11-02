// Compare keyboard-copied article code with the actual Markdown fences.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {connect}=require('../../../scripts/lib/browser.cjs');
const [base,portText,out]=process.argv.slice(2);
const markdown=fs.readFileSync(path.resolve(__dirname,'../../../content/posts/ai/pion-muon-vla-rlvr/index.md'),'utf8');
const expected=[...markdown.matchAll(/^```[^\n]*\n([\s\S]*?)^```[ \t]*$/gm)].map(m=>m[1].replace(/\n$/,''));
(async()=>{fs.mkdirSync(out,{recursive:true});const b=await connect(Number(portText)),report={copies:[],contexts:[],exceptions:b.exceptions};
 try{
  for(const width of[320,1440])for(const size of[100,200]){
   await b.viewport(width);await b.navigate(base+'/posts/ai/pion-muon-vla-rlvr/');await b.evaluate(`document.documentElement.style.fontSize='${size}%';document.fonts.ready`);
   const opened=[];
   for(const id of['snr-estimation','spectral-step-trace','flat-rank-study','vla-buffer-order','vla-padding-recovery']){
    assert.equal(await b.evaluate(`document.getElementById('${id}').open`),false);
    await b.evaluate(`(()=>{const s=document.querySelector('#${id}>summary');s.scrollIntoView({block:'center'});s.focus()})()`);await b.key('Enter','Enter',13);await b.pause(80);
    assert.equal(await b.evaluate(`document.getElementById('${id}').open`),true);opened.push(id);
   }
   assert.deepEqual(await b.evaluate(`[...document.querySelectorAll('.hextra-code-block code')].map(e=>e.textContent)`),expected);
   await b.evaluate(`window.__copies=[];Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:t=>{__copies.push(t);return Promise.resolve()}}})`);
   for(let index=0;index<expected.length;index++){
    await b.evaluate(`window.__copyButton=document.querySelectorAll('.hextra-code-copy-btn')[${index}];__copyButton.scrollIntoView({block:'center'});__copyButton.focus()`);await b.key('Enter','Enter',13);await b.pause(45);
    const state=await b.evaluate(`({text:__copies.at(-1),count:__copies.length,focus:document.activeElement===__copyButton,busy:__copyButton.hasAttribute('aria-busy')})`);
    assert.equal(state.text,expected[index]);assert.equal(state.count,index+1);assert(state.focus&&!state.busy);
    report.copies.push({width,size,index,bytes:Buffer.byteLength(state.text),exact:true,focusPreserved:true});
   }
   report.contexts.push({width,size,openedByKeyboard:opened,blocks:expected.length});
  }
  assert.deepEqual(b.exceptions,[]);console.log(JSON.stringify({blocks:expected.length,copies:report.copies.length,contexts:report.contexts.length,exceptions:0}));
 }finally{fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
