// Native figure fragments, reserved image geometry and full caption bounds.
// Use serially with all other checks on the private CDP endpoint.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('../../../scripts/lib/browser.cjs');
const [base, port, out] = process.argv.slice(2);
const route = '/posts/ai/jev-decision-layer/';
(async () => {
  fs.mkdirSync(out, {recursive:true});
  const b = await connect(Number(port));
  const report = {figures:[], exceptions:b.exceptions};
  try {
    await b.navigate(base + route);
    const ids = await b.evaluate("[...document.querySelectorAll('.article-figure')].map(e=>e.id)");
    assert.equal(ids.length, 8);
    for (const scripts of [true, false]) for (const width of [320, 1440]) for (const size of [100, 200]) {
      await b.cdp('Emulation.setScriptExecutionDisabled', {value:!scripts});
      await b.viewport(width);
      for (const id of ids) {
        await b.navigate(base + route + '#' + id);
        await b.evaluate(`document.documentElement.style.fontSize='${size}%';document.fonts.ready`);
        if (scripts) for (let i=0;i<100;i++) {
          if (await b.evaluate("document.querySelector('[data-jev-routing-lab]')?.dataset.loadState==='ready'")) break;
          await b.pause(30);
        }
        // Resizing text is an explicit reader action; put its selected figure back in view.
        const before = await b.evaluate(`(()=>{const f=document.getElementById('${id}');
          f.scrollIntoView({block:'start'});const i=f.querySelector('img'),r=f.getBoundingClientRect();
          return {top:r.top,width:Number(i.getAttribute('width')),height:Number(i.getAttribute('height'))}})()`);
        assert(before.width>0 && before.height>0);
        await b.evaluate("Promise.all([...document.querySelectorAll('.content img')].map(i=>{i.loading='eager';return i.decode()}))");
        await b.pause(70);
        const after = await b.evaluate(`(()=>{const f=document.getElementById('${id}'),i=f.querySelector('img'),
          c=f.querySelector('figcaption'),text=c.querySelector('.article-figure__text'),r=f.getBoundingClientRect(),
          ir=i.getBoundingClientRect(),cr=c.getBoundingClientRect(),tr=text.getBoundingClientRect();
          return {top:r.top,imageLoaded:i.complete&&i.naturalWidth>0,
            aspectDifference:Math.abs(ir.height/ir.width-i.naturalHeight/i.naturalWidth),
            captionComplete:tr.left>=cr.left-1&&tr.right<=cr.right+1&&text.scrollWidth<=text.clientWidth+1&&text.scrollHeight<=text.clientHeight+1,
            label:f.querySelector('.article-figure__number').textContent.trim(),
            pageOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1}})()`);
        assert(after.imageLoaded && after.aspectDifference<.01 && after.captionComplete && !after.pageOverflow, JSON.stringify({id,scripts,width,size,after}));
        assert(Math.abs(after.top-before.top)<=2, JSON.stringify({id,before,after}));
        report.figures.push({id,scripts,width,size,before,after});
      }
    }
    assert.deepEqual(b.exceptions, []);
    console.log(JSON.stringify({figureContexts:report.figures.length,exceptions:0}));
  } finally {
    fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');
    await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});
    await b.close();
  }
})().catch(e=>{console.error(e);process.exitCode=1});
