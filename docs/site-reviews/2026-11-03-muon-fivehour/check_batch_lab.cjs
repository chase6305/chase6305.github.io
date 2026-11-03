// Serial private-CDP checks for the article's conditional Batch calculator.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('../../../scripts/lib/browser.cjs');
const {contrastAudit} = require('../../../scripts/lib/contrast.cjs');
const lab = require('../../../assets/js/muon-batch.js');
const [base, portText, out] = process.argv.slice(2);
const route = '/posts/ai/muon-learning-rate-batch-size/';
const source = fs.readFileSync(path.join(__dirname, '../../../assets/js/muon-batch.js'), 'utf8');
const report = {layouts: [], behavior: [], fallbacks: []};
function readState() {
  const e = document.querySelector('[data-muon-batch]');
  return {ready: e.dataset.ready || null, result: e.dataset.result || null,
    outputs: Object.fromEntries([...e.querySelectorAll('output')].map(o => [o.dataset.value,o.textContent])),
    controls: !e.querySelector('[data-controls]').hidden,
    error: e.querySelector('[data-error]').hidden ? '' : e.querySelector('[data-error]').textContent,
    invalid: [...e.querySelectorAll('[aria-invalid=true]')].map(i => i.name),
    focused: document.activeElement.name || document.activeElement.dataset.preset || document.activeElement.tagName,
    hash: location.hash};
}
function numbers(actual, expected) {
  for (const [key,value] of Object.entries(expected))
    assert(Math.abs(Number(actual[key])-value) <= Math.max(1e-12,Math.abs(value)*6e-8), JSON.stringify({key,actual:actual[key],value}));
}
(async()=>{
  fs.mkdirSync(out,{recursive:true});
  const b = await connect(Number(portText)); report.exceptions = b.exceptions;
  const run = (fn,...args) => b.evaluate('('+fn.toString()+')(...'+JSON.stringify(args)+')');
  const state = () => run(readState);
  const visit = async hash => {
    // Same-fragment Page.navigate can retain an already enhanced document.
    await b.navigate('about:blank');
    await b.navigate(base+route+hash);
    await b.evaluate('document.fonts.ready');
  };
  const edit = values => run(values=>{
    const f = document.querySelector('[data-muon-batch] form');
    for (const [name,value] of Object.entries(values)) {
      const input=f.elements.namedItem(name); input.value=value; input.focus();
      input.dispatchEvent(new Event('input',{bubbles:true}));
    }
  }, values);
  const preset = name => run(name=>document.querySelector('[data-preset='+name+']').click(),name);
  try {
    for (const scripts of [true,false]) for (const width of [320,1440]) {
      await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts}); await b.viewport(width,1000);
      await visit('#batch-calculator');
      for (const size of [100,200]) for (const spaced of [false,true]) for (const theme of ['light','dark','warm']) {
        await run((size,theme,spaced)=>{
          document.documentElement.style.fontSize=size+'%';
          let stress=document.getElementById('muon-spacing-test');if(!stress){stress=document.createElement('style');stress.id='muon-spacing-test';document.head.append(stress)}
          stress.textContent=spaced?'body :is(p,li,dt,dd,th,td,summary,h1,h2,h3,h4,h5,h6,a,button,label,input,select){line-height:1.5!important;letter-spacing:.12em!important;word-spacing:.16em!important}main p{margin-bottom:2em!important}.katex{letter-spacing:normal!important;word-spacing:normal!important}':'';
          document.documentElement.classList.remove('dark','warm');
          if (theme!=='light') document.documentElement.classList.add(theme);
          document.documentElement.style.colorScheme=theme==='dark'?'dark':'light';
          return document.fonts.ready;
        },size,theme,spaced);
        await b.pause(150);
        const geometry=await run(()=>{
          const e=document.querySelector('[data-muon-batch]'),r=e.getBoundingClientRect(),issues=[];
          if(document.documentElement.scrollWidth>document.documentElement.clientWidth+1)issues.push('page overflow');
          for(const n of e.querySelectorAll('input,button,dt,dd,output')){
            if(!n.checkVisibility())continue;const q=n.getBoundingClientRect();
            if(q.left<r.left-1||q.right>r.right+1)issues.push(n.tagName+' outside lab');
            if(n.matches('dt,dd,output')&&n.scrollWidth>n.clientWidth+1)issues.push(n.tagName+' clips text');
          }
          return {issues,width:r.width,visibleInputs:[...e.querySelectorAll('input')].filter(i=>i.checkVisibility()).length};
        });
        numbers((await state()).outputs,lab.calculate(lab.presets.fourfold));
        assert.deepEqual(geometry.issues,[],JSON.stringify({scripts,width,size,spaced,theme,geometry}));
        assert.equal(geometry.visibleInputs,scripts?6:0);
        const text=await run(contrastAudit,'.muon-batch');assert.deepEqual(text.issues,[]);
        report.layouts.push({scripts,width,size,spaced,theme,...geometry,textSamples:text.checked});
      }
    }
    await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});await b.viewport(320,1000);
    await visit('#batch-calculator');
    await edit({batchNew:'2048'});let current=await state();
    assert.equal(current.result,'pending');assert(Object.values(current.outputs).every(v=>v==='—'));assert.equal(current.focused,'batchNew');
    await b.key('Enter','Enter',13);current=await state();
    assert.equal(current.result,'valid');assert.equal(current.focused,'batchNew');
    numbers(current.outputs,lab.calculate({...lab.presets.fourfold,batchNew:2048}));
    report.behavior.push({case:'actual Enter on edited input',...current});
    for(const bad of [{batchNew:'0'},{batchNew:'1.5'},{criticalBatch:''},{lrOld:'1',decayOld:'1'}]){
      await preset('fourfold');await edit(bad);await b.key('Enter','Enter',13);current=await state();
      assert.equal(current.result,'invalid');assert(current.error&&current.invalid.length);
      assert(Object.values(current.outputs).every(v=>v==='—'));report.behavior.push({case:'invalid input',input:bad,...current});
    }
    await preset('fourfold');await edit({criticalBatch:'0'});await b.key('Enter','Enter',13);
    current=await state();numbers(current.outputs,lab.calculate({...lab.presets.fourfold,criticalBatch:0}));
    assert.deepEqual(current.invalid,[]);assert.equal(current.error,'');report.behavior.push({case:'noiseless limit after recovery',...current});
    for(const name of ['saturation','smaller','fourfold']){
      await run(name=>{const button=document.querySelector('[data-preset='+name+']');button.scrollIntoView({block:'center'});button.focus();},name);
      await b.key('Enter','Enter',13);current=await state();numbers(current.outputs,lab.calculate(lab.presets[name]));
      assert.equal(current.focused,name);report.behavior.push({case:'keyboard preset',preset:name,...current});
    }
    await b.evaluate("window.__changes=0;window.__observer=new MutationObserver(rows=>__changes+=rows.length);__observer.observe(document.querySelector('.muon-batch__results'),{childList:true,subtree:true})");
    await b.evaluate(source);await b.pause(50);assert.equal(await b.evaluate('__changes'),0);
    await b.key('Enter','Enter',13);await b.pause(50);assert.equal(await b.evaluate('__changes'),7);
    report.behavior.push({case:'duplicate mount',changesPerCalculation:7});
    await run(()=>{const i=document.querySelector('[name=batchNew]');i.focus();i.select();});
    await b.key('ArrowUp','ArrowUp',38);current=await state();assert.equal(current.result,'pending');
    assert.equal(await b.evaluate("document.querySelector('[name=batchNew]').value"),'1025');
    report.behavior.push({case:'native number stepper',...current});
    await b.cdp('Network.setBlockedURLs',{urls:['*muon-batch*.js']});
    await visit('#procedure');current=await state();
    assert(!current.controls&&!current.ready);numbers(current.outputs,lab.calculate(lab.presets.fourfold));
    report.fallbacks.push({case:'blocked component script',...current});
    await run(()=>{const e=document.getElementById('procedure');e.tabIndex=-1;e.focus({preventScroll:true});
      window.scrollTo({top:scrollY+e.getBoundingClientRect().top-80,behavior:'instant'});window.__later=e;});
    await b.pause(120);const top=await b.evaluate('__later.getBoundingClientRect().top');
    await b.evaluate(source);await b.pause(200);
    const late=await b.evaluate('({top:__later.getBoundingClientRect().top,focused:document.activeElement===__later,hash:location.hash})');
    assert(late.focused&&Math.abs(late.top-top)<=1&&late.hash==='#procedure',JSON.stringify({top,late}));
    report.fallbacks.push({case:'late component enhancement',top,...late});
    await b.cdp('Network.setBlockedURLs',{urls:[]});
    const hook=await b.cdp('Page.addScriptToEvaluateOnNewDocument',{source:'Math.expm1=undefined;'});
    await visit('#batch-calculator');current=await state();
    assert(!current.controls&&!current.ready);numbers(current.outputs,lab.calculate(lab.presets.fourfold));
    report.fallbacks.push({case:'missing numeric feature',...current});
    await b.cdp('Page.removeScriptToEvaluateOnNewDocument',{identifier:hook.identifier});
    assert.deepEqual(b.exceptions,[]);
    console.log(JSON.stringify({layouts:report.layouts.length,behavior:report.behavior.length,fallbacks:report.fallbacks.length,exceptions:0}));
  } finally {
    fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');
    await b.cdp('Network.setBlockedURLs',{urls:[]}).catch(()=>{});
    await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});await b.close();
  }
})().catch(e=>{console.error(e);process.exitCode=1});
