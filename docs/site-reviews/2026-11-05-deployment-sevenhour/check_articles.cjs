#!/usr/bin/env node
// Adapted from the LAST/OmniCalib article review; new deployment article routes.
// Run serially on a private loopback CDP endpoint.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {connect, pause} = require('../../../scripts/lib/browser.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13317';
const port = Number(process.argv[3] || 9337);
const output = process.argv[4] || '/tmp/deployment-sevenhour-review';
const axePath = process.argv[5];
const posts = [
  {name:'deployment',route:'/posts/ai/rma-umi-rtc-deployment/',download:'deployment_contracts.py',figures:['figure-1','figure-rma-stages','figure-umi-anchors','figure-2'],svg:['figure-umi-anchors','figure-2']},
  {name:'residual',route:'/posts/ai/residual-reinforcement-learning/',download:'residual_control_sanity.py',figures:['figure-1','figure-2'],svg:['figure-2']},
  {name:'contact',route:'/posts/calibration/interactive-contact-calibration/',download:'contact_calibration_sanity.py',figures:['figure-1','figure-2','figure-3'],svg:['figure-2','figure-3']},
].filter(p => !process.argv.includes('--post') ||
  p.name === process.argv[process.argv.indexOf('--post') + 1]);

(async () => {
  fs.mkdirSync(output, {recursive:true});
  const b = await connect(port);
  const report = {layouts:[], arrows:[], svg:[], semantics:[], zoom:[], downloads:[], exceptions:b.exceptions};
  try {
    for (const scripts of [true, false]) {
      await b.cdp('Emulation.setScriptExecutionDisabled', {value:!scripts});
      for (const post of posts) for (const width of [320, 768, 1440]) for (const theme of ['light', 'dark', 'warm']) {
        await b.viewport(width, 1000);
        await b.navigate(base + post.route);
        await b.evaluate(`document.documentElement.classList.remove('dark','warm');document.documentElement.classList.add('${theme}');Promise.all([document.fonts.ready,...[...document.querySelectorAll('article img')].map(i=>{i.loading='eager';return i.decode()})])`);
        for (const mode of ['ordinary', 'large', 'spacing']) {
          await b.evaluate(`document.documentElement.style.fontSize='${mode==='ordinary'?100:200}%';(()=>{let s=document.querySelector('#review-spacing');if(s)s.remove();${mode==='spacing'?`s=document.createElement('style');s.id='review-spacing';s.textContent='.content p,.content li,.content td,.content th{line-height:1.5!important;letter-spacing:.12em!important;word-spacing:.16em!important}.content p{margin-bottom:2em!important}';document.head.append(s);`:''}})()`);
          await pause(100);
          const state = await b.evaluate(`(()=>{const a=document.querySelector('article');return {
            overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1,
            layoutWidth:document.documentElement.clientWidth,scrollWidth:document.documentElement.scrollWidth,visualWidth:innerWidth,
            images:[...a.querySelectorAll('img')].map(i=>({ok:i.complete&&i.naturalWidth>0,alt:i.alt})),
            mathErrors:a.querySelectorAll('.katex-error').length,formulas:a.querySelectorAll('.katex').length,
            captions:[...a.querySelectorAll('figcaption')].map(c=>({text:c.innerText,clipped:[...c.querySelectorAll('span')].some(s=>s.scrollWidth>s.clientWidth+2)})),
            missingAnchors:[...a.querySelectorAll('a[href^="#"]')].filter(e=>!document.getElementById(decodeURIComponent(e.hash.slice(1)))).map(e=>e.hash)
          }})()`);
          assert(!state.overflow, JSON.stringify({post,width,theme,scripts,mode,state}));
          assert(state.images.length===post.figures.length && state.images.every(i=>i.ok&&i.alt));
          assert(state.formulas>20 && !state.mathErrors);
          assert(state.captions.length===post.figures.length && state.captions.every(c=>!c.clipped));
          assert(!state.missingAnchors.length, JSON.stringify(state));
          report.layouts.push({post:post.name,width,theme,scripts,mode,...state});
        }
        await b.evaluate(`document.querySelector('#review-spacing')?.remove();document.documentElement.style.fontSize=''`);
        if (width===320 && theme==='light') {
          const regions = await b.evaluate(`[...document.querySelectorAll('article table')].map((e,index)=>({index,width:e.clientWidth,scroll:e.scrollWidth,tab:e.tabIndex})).filter(e=>e.scroll>e.width+2)`);
          assert(regions.length>0, 'Expected a horizontally readable table');
          const index = regions[0].index;
          const focused = await b.evaluate(`(()=>{const t=document.querySelectorAll('article table')[${index}];t.scrollIntoView({block:'center'});t.scrollLeft=0;t.focus();return document.activeElement===t&&t.tabIndex===0})()`);
          assert(focused);
          await b.key('ArrowRight','ArrowRight',39);await pause(160);
          assert(await b.evaluate(`document.querySelectorAll('article table')[${index}].scrollLeft>0`));
          report.arrows.push({post:post.name,scripts,index,passed:true});
        }
        if (width===1440 && theme==='light') {
          if (axePath) {
            // Load the native page with scripts disabled, then allow only the
            // injected audit engine's asynchronous tasks to execute.
            if (!scripts) await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});
            await b.evaluate(fs.readFileSync(axePath,'utf8'));
            const audit = await b.evaluate(`axe.run(document,{rules:{'color-contrast':{enabled:false}}}).then(r=>({violations:r.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),incomplete:r.incomplete.map(v=>v.id)}))`);
            report.semantics.push({post:post.name,scripts,...audit});
            assert(!audit.violations.length, JSON.stringify(audit));
            if (!scripts) await b.cdp('Emulation.setScriptExecutionDisabled',{value:true});
          }
          const download = await b.evaluate(`fetch(${JSON.stringify(post.download)}).then(async r=>({status:r.status,source:(await r.text()).includes('def check():')}))`);
          assert(download.status===200&&download.source);report.downloads.push({post:post.name,scripts,...download});
        }
        if (scripts && width===320 && theme==='dark') {
          for (const id of post.figures) {
            await b.evaluate(`document.querySelector('#${id}').scrollIntoView({block:'center'})`);
            await b.screenshot(path.join(output,`${post.name}-${theme}-${width}-${id}.png`));
            await b.evaluate(`document.querySelector('#${id} img').click()`);await pause(150);
            assert(await b.evaluate(`!!document.querySelector('dialog[open]')`));
            await b.key('Escape','Escape',27);await pause(100);
            assert(await b.evaluate(`!document.querySelector('dialog[open]')`));
            report.zoom.push({post:post.name,figure:id,passed:true});
          }
        }
      }
    }
    await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});
    for (const post of posts) {
      await b.viewport(1440,1000);await b.navigate(base+post.route);
      for (const figure of post.svg) {
        const labels = await b.evaluate(`(async()=>{const i=document.querySelector('#${figure} img');const parsed=new DOMParser().parseFromString(await(await fetch(i.src)).text(),'image/svg+xml');const svg=document.importNode(parsed.documentElement,true);svg.style.cssText='position:absolute;left:-9000px;top:0;width:1120px;height:620px;max-width:none';document.body.append(svg);const bounds=svg.viewBox.baseVal;const labels=[...svg.querySelectorAll('text')].map(n=>{const r=n.getBBox();return{text:n.textContent,x:r.x,y:r.y,right:r.x+r.width,bottom:r.y+r.height}});const result={labels,outside:labels.filter(r=>r.x<0||r.y<0||r.right>bounds.width||r.bottom>bounds.height)};svg.remove();return result})()`);
        report.svg.push({post:post.name,figure,...labels});assert(!labels.outside.length,JSON.stringify(labels));
      }
    }
    assert.equal(b.exceptions.length,0,JSON.stringify(b.exceptions));
    console.log(JSON.stringify({layouts:report.layouts.length,arrows:report.arrows.length,zoom:report.zoom.length,semantics:report.semantics.length,svg:report.svg.length,exceptions:0}));
  } finally {
    fs.writeFileSync(path.join(output,'articles.json'),JSON.stringify(report,null,2)+'\n');
    await b.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1});
