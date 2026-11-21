#!/usr/bin/env node
// Apply the repository's painted-outline inspection to this article's controls.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {connect}=require('../../../scripts/lib/browser.cjs');
const base=process.argv[2].replace(/\/$/,''),port=Number(process.argv[3]),out=process.argv[4];
const shared=fs.readFileSync(path.resolve(__dirname,'../../../scripts/check_site_focus.cjs'),'utf8');
const start=shared.indexOf('function inspectOutline('),end=shared.indexOf('\n(async () =>',start);
assert(start>=0&&end>start);const inspect=shared.slice(start,end);
const selectors=['[data-preset="default"]','[data-preset="rare"]','[data-preset="balanced"]',
 'input[name="probability"]','input[name="group"]','input[name="groups"]','button[type="submit"]'];
fs.mkdirSync(out,{recursive:true});
(async()=>{
 const b=await connect(port),rows=[];
 try{
  for(const mode of ['normal','forced-light','forced-dark'])for(const width of [320,1440])for(const size of [100,200])for(const theme of ['light','dark','warm']){
   await b.viewport(width,1000);await b.cdp('Emulation.setEmulatedMedia',{media:'',features:mode==='normal'?[]:[
    {name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:mode==='forced-dark'?'dark':'light'}]});
   await b.navigate(base+'/posts/ai/pear/');
   await b.evaluate(`document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='${size}%';document.fonts.ready`);
   await b.key('Tab','Tab',9);
   for(const suffix of selectors){
    const selector='[data-pear-sampling] '+suffix;
    await b.evaluate(`document.querySelector(${JSON.stringify(selector)}).focus()`);
    const state=await b.evaluate('('+inspect+')('+JSON.stringify(selector)+')');
    const passed=state.focused&&state.visible&&state.outlineWidth>=2&&state.outlineStyle!=='none'&&state.ratio>=3&&state.insideViewport;
    rows.push({mode,width,size,theme,selector,passed,...state});
    assert(passed,JSON.stringify(rows.at(-1)));
   }
   if(mode!=='normal'&&width===320&&size===200&&theme==='dark')await b.screenshot(path.join(out,mode+'.png'));
  }
  assert.equal(b.exceptions.length,0);
  console.log(JSON.stringify({outlines:rows.length,failures:rows.filter(r=>!r.passed).length,exceptions:b.exceptions.length}));
 }finally{fs.writeFileSync(path.join(out,'results.json'),JSON.stringify({rows,exceptions:b.exceptions},null,2)+'\n');await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
