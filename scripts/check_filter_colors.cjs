#!/usr/bin/env node
// Actual canvas paint colors, patterned curves and forced-color transitions.
// Run serially with the other CDP browser checks.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('./lib/browser.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13182';
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-filter-colors';
const modes = {normal: [], light: [{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:'light'}],
  dark: [{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:'dark'}]};
function inspectPaint() {
  const root = document.querySelector('.robot-filter-lab'), canvas = root.querySelector('canvas');
  const probe = document.createElement('canvas');probe.width = probe.height = 1;
  const ctx = probe.getContext('2d', {willReadFrequently:true});
  const rgba = color => {ctx.clearRect(0,0,1,1);ctx.fillStyle=color;ctx.fillRect(0,0,1,1);return [...ctx.getImageData(0,0,1,1).data].map(v=>v/255)};
  const over = (a,b) => [0,1,2].map(i=>a[i]*a[3]+b[i]*(1-a[3]));
  const background = node => node ? over(rgba(getComputedStyle(node).backgroundColor),background(node.parentElement)) : [1,1,1];
  const bg = background(canvas), luminance = c => c.map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((s,v,i)=>s+v*[.2126,.7152,.0722][i],0);
  const contrast = color => {const a=luminance(over(rgba(color),bg)),b=luminance(bg);return (Math.max(a,b)+.05)/(Math.min(a,b)+.05)};
  const labels = window.__paint.text.slice(-10), curves = window.__paint.stroke.slice(-3);
  const selected = root.querySelector('[data-view][aria-pressed=true]'), style = getComputedStyle(selected);
  return {labels:labels.map(row=>({...row,contrast:contrast(row.color)})),curves:curves.map(row=>({...row,contrast:contrast(row.color)})),
    foreground:getComputedStyle(canvas).color,background:bg,selected:selected.dataset.view,selectedDecoration:style.textDecorationLine,
    selectedWeight:Number(style.fontWeight),metrics:[...root.querySelectorAll('[data-metric]')].map(e=>e.textContent),
    forced:matchMedia('(forced-colors:active)').matches,legend:root.querySelector('.robot-filter-lab__legend').textContent,
    pageOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1};
}
(async()=>{
  fs.mkdirSync(output,{recursive:true});const b=await connect(port),report={layouts:[],print:[],exceptions:b.exceptions};let savedTheme;
  const wait=async expression=>{for(let i=0;i<100;i++){if(await b.evaluate(expression))return;await b.pause(30)}throw Error('Condition timeout: '+expression)};
  const measure=()=>b.evaluate('('+inspectPaint.toString()+')()');
  const references = new Map();
  try{
    for(const width of [320,1440]){
      await b.viewport(width);await b.navigate(base+'/posts/robotics/control/robot-filters/');
      if(savedTheme===undefined)savedTheme=await b.evaluate("localStorage.getItem('color-theme')");
      await wait("!document.querySelector('[data-filter-interactive]').hidden");
      await b.evaluate(`(()=>{window.__paint={text:[],stroke:[]};const c=document.querySelector('.robot-filter-lab canvas').getContext('2d'),text=c.fillText,stroke=c.stroke;
        c.fillText=function(...args){__paint.text.push({color:this.fillStyle,alpha:this.globalAlpha});return text.apply(this,args)};
        c.stroke=function(...args){__paint.stroke.push({color:this.strokeStyle,dash:this.getLineDash(),width:this.lineWidth,alpha:this.globalAlpha});return stroke.apply(this,args)};})()`);
      // Switching only a media feature must repaint without a resize or input.
      await b.cdp('Emulation.setEmulatedMedia',{features:[]});await b.evaluate("document.querySelector('[data-item=dark]').click()");await b.pause(80);
      await b.evaluate('__paint.text=[];__paint.stroke=[]');await b.cdp('Emulation.setEmulatedMedia',{features:modes.light});
      await wait('__paint.text.length>=10');assert((await measure()).labels.every(row=>row.contrast>=4.5));
      await b.evaluate('__paint.text=[];__paint.stroke=[]');await b.cdp('Emulation.setEmulatedMedia',{features:modes.dark});
      await wait('__paint.text.length>=10');assert((await measure()).labels.every(row=>row.contrast>=4.5));
      report.mediaOnlyRedraw=true;
      for(const theme of ['light','dark','warm']){
        await b.evaluate(`document.querySelector('[data-item=${theme}]').click()`);
        for(const [mode,features] of Object.entries(modes)){
          await b.cdp('Emulation.setEmulatedMedia',{features});await b.pause(30);
          for(const method of ['oneEuro','butterworth','wma','ema','msd'])for(const view of ['all','still','onset']){
            await b.evaluate(`(()=>{__paint.text=[];__paint.stroke=[];const root=document.querySelector('.robot-filter-lab'),input=root.querySelector('[name=method]');
              input.value=${JSON.stringify(method)};input.dispatchEvent(new Event('input',{bubbles:true}));root.querySelector('[data-view=${view}]').click();})()`);
            const state=await measure();
            assert.equal(state.labels.length,10);assert(state.labels.every(row=>row.alpha===1&&row.contrast>=4.5),JSON.stringify({theme,mode,method,view,state}));
            assert.equal(state.curves.length,3);assert(state.curves.every(row=>row.alpha===1&&row.contrast>=3),JSON.stringify({theme,mode,method,view,state}));
            assert.deepEqual(state.curves.map(row=>row.dash),[[1,3],[5,4],[]]);
            assert(!state.pageOverflow&&state.selected===view);
            if(mode!=='normal')assert(state.selectedDecoration.includes('underline')&&state.selectedWeight>=700);
            if(references.has(method))assert.deepEqual(state.metrics,references.get(method));else references.set(method,state.metrics);
            report.layouts.push({width,theme,mode,method,view,passed:true,...state});
          }
        }
      }
      await b.evaluate("document.querySelector('.robot-filter-lab canvas').scrollIntoView({block:'center'})");
      if(width===1440)await b.screenshot(path.join(output,'forced-dark-chart.png'));
    }
    await b.cdp('Emulation.setEmulatedMedia',{features:[]});await b.evaluate("document.querySelector('[data-item=dark]').click()");await b.pause(30);
    for(const features of [[],modes.light,modes.dark]){
      await b.cdp('Emulation.setEmulatedMedia',{media:'print',features});await b.pause(100);
      const state=await measure();report.print.push(state);
      assert(state.labels.every(row=>row.contrast>=4.5),JSON.stringify(state));
      assert(state.curves.every(row=>row.contrast>=3),JSON.stringify(state));
    }
    await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]});
    report.summary={layouts:report.layouts.length,printModes:report.print.length,exceptions:b.exceptions.length};
    console.log(JSON.stringify(report.summary,null,2));assert.equal(b.exceptions.length,0,JSON.stringify(b.exceptions));
  }finally{
    await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]}).catch(()=>{});
    if(savedTheme!==undefined)await b.evaluate(savedTheme===null?"localStorage.removeItem('color-theme')":'localStorage.setItem("color-theme",'+JSON.stringify(savedTheme)+')').catch(()=>{});
    fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(report,null,2)+'\n');await b.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1});
