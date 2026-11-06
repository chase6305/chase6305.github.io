#!/usr/bin/env node
// Native Tab order, painted outlines and chart contrast in all three labs.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {connect,pause}=require('../../../scripts/lib/browser.cjs');
const {inspectOutline}=require('./outline_oracle.cjs');
const [base='http://127.0.0.1:39783',port='9337',output='/tmp/deployment-lab-focus']=process.argv.slice(2);
const posts=[['timing','/posts/ai/rma-umi-rtc-deployment/'],['residual','/posts/ai/residual-reinforcement-learning/'],['contact','/posts/calibration/interactive-contact-calibration/']];
function graphColors(){
 const root=document.querySelector('[data-deployment-lab]'),canvas=document.createElement('canvas'),ctx=canvas.getContext('2d');
 canvas.width=canvas.height=1;
 const rgb=color=>{ctx.clearRect(0,0,1,1);ctx.fillStyle=color;ctx.fillRect(0,0,1,1);return [...ctx.getImageData(0,0,1,1).data].slice(0,3).map(v=>v/255)};
 const luminance=c=>c.map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((s,v,i)=>s+v*[.2126,.7152,.0722][i],0);
 const background=luminance(rgb(getComputedStyle(root).backgroundColor));
 return [...root.querySelectorAll('svg path')].map(p=>{const css=getComputedStyle(p),value=luminance(rgb(css.stroke));return{stroke:css.stroke,ratio:(Math.max(value,background)+.05)/(Math.min(value,background)+.05),dash:css.strokeDasharray,width:parseFloat(css.strokeWidth)}});
}
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(Number(port)),root='[data-deployment-lab]';
 const report={contexts:[],outlines:[],charts:[],nativeSpin:[],exceptions:b.exceptions};
 try{
  for(const mode of ['normal','forced-light','forced-dark']){
   await b.cdp('Emulation.setEmulatedMedia',{media:'',features:mode==='normal'?[]:[{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:mode==='forced-dark'?'dark':'light'}]});
   for(const [kind,route]of posts)for(const width of [320,1440])for(const size of [100,200])for(const theme of ['light','dark','warm']){
    const context={kind,width,size,theme,mode};await b.viewport(width,1000);await b.navigate(base+route);
    await b.evaluate(`document.documentElement.classList.remove('dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='${size}%';document.fonts.ready`);
    await b.key('Tab','Tab',9);
    const controls=await b.evaluate(`[...document.querySelectorAll('${root} [data-preset],${root} input')].map(e=>e.matches('input')?'${root} input[name="'+e.name+'"]':'${root} [data-preset="'+e.dataset.preset+'"]')`);
    await b.evaluate(`document.querySelector(${JSON.stringify(controls[0])}).focus()`);
    for(let i=0;i<controls.length;i++){
     const selector=controls[i];
     const state=await b.evaluate('('+inspectOutline.toString()+')('+JSON.stringify(selector)+')');
     assert(state.focused&&state.visible&&state.outlineStyle!=='none'&&state.outlineWidth>=2&&state.ratio>=3&&state.insideViewport,JSON.stringify({context,selector,state}));
     assert(!state.gradients.length,'A gradient needs a separate painted outline review');
     report.outlines.push({...context,selector,...state});if(i<controls.length-1)await b.key('Tab','Tab',9);
    }
    const colors=await b.evaluate('('+graphColors.toString()+')()');
    assert(colors.every(c=>c.ratio>=3&&c.width>=2),JSON.stringify({context,colors}));
    if(kind!=='timing')assert.equal(colors.length,2);else assert.equal(colors.length,1);
    if(colors.length===2)assert(colors[0].dash==='none'&&colors[1].dash!=='none');
    report.charts.push({...context,colors});
    if(kind==='timing'){
     await b.evaluate(`document.querySelector('${root} input[name="inference"]').focus()`);
     await b.key('ArrowUp','ArrowUp',38);
     const up=await b.evaluate(`({input:document.querySelector('${root} input[name="inference"]').valueAsNumber,delay:document.querySelector('${root} [data-value="delay"]').textContent,focus:document.activeElement.name})`);
     assert.equal(up.input,80.1);assert.equal(up.delay,'5');assert.equal(up.focus,'inference');
     await b.key('ArrowDown','ArrowDown',40);assert.equal(await b.evaluate(`document.querySelector('${root} [data-value="delay"]').textContent`),'4');
     report.nativeSpin.push({...context,up,restoredDelay:4});
    }
    if(width===320&&size===200&&theme==='dark'){
     await b.evaluate(`document.querySelector('${root} input').scrollIntoView({block:'center'})`);await pause(60);
     await b.screenshot(path.join(output,`${kind}-${mode}-large-focus.png`));
    }
    assert(await b.evaluate('document.documentElement.scrollWidth<=document.documentElement.clientWidth+1'));
    report.contexts.push(context);
   }
  }
  assert(!b.exceptions.length,JSON.stringify(b.exceptions));console.log(JSON.stringify({contexts:report.contexts.length,outlines:report.outlines.length,charts:report.charts.length,nativeSpin:report.nativeSpin.length,minimumOutlineContrast:Math.min(...report.outlines.map(x=>x.ratio)),exceptions:0}));
 }finally{await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]});fs.writeFileSync(path.join(output,'lab-focus.json'),JSON.stringify(report,null,2)+'\n');await b.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
