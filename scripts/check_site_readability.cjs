#!/usr/bin/env node
// Inspect rendered text and print output against an existing loopback CDP browser.
// Browser suites share focus; run them serially.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('./lib/browser.cjs');
const {contrastAudit} = require('./lib/contrast.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13143';
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-site-readability';
const full = process.argv.includes('--all-articles');
const captureOnly = process.argv.includes('--capture-only');


(async () => {
  fs.mkdirSync(output, {recursive: true});
  const browser = await connect(port);
  const report = {contrast: [], print: [], exceptions: browser.exceptions};
  try {
    const routes = new Set(['/posts/', '/learning-paths/', '/tags/', '/archives/', '/404.html', '/zh-cn/404.html', '/posts/ai/supermimic/', '/posts/cpp/gcc/']);
    if (full) {
      const posts = JSON.parse(fs.readFileSync(path.join(__dirname, '../docs/blog-editorial-review.json'))).posts;
      posts.forEach(post => routes.add('/' + post.path.replace(/^content\//, '').replace(/\/index\.md$|\.md$/, '').toLowerCase() + '/'));
    }
    await browser.viewport(1440);
    for (const route of routes) {
      await browser.navigate(base + route);
      await browser.evaluate('document.fonts.ready');
      for (const theme of ['light', 'dark', 'warm']) {
        await browser.evaluate(`document.querySelector('[data-item=${theme}]').click()`);
        await browser.pause(400);
        const audit = await browser.evaluate('(' + contrastAudit.toString() + ')()');
        report.contrast.push({route, theme, ...audit});
      }
    }

    for (const route of ['/posts/ai/supermimic/', '/posts/cpp/gcc/', '/posts/ai/cosmos3/', '/posts/robotics/control/robot-filters/', '/posts/ai/jev-decision-layer/', '/posts/open3d/introduction/']) {
      await browser.navigate(base + route);
      await browser.evaluate(`Promise.all([...document.querySelectorAll('.content img')].map(img=>{img.loading='eager';return img.decode().catch(()=>null)}))`);
      // Keep one originally open disclosure, to check exact state restoration.
      const original = await browser.evaluate(`(() => {const ds=[...document.querySelectorAll('.content details:not(.blog-mobile-toc)')];if(ds[0])ds[0].open=true;return ds.map(d=>d.open);})()`);
      let canvasSignature;
      for (const theme of ['light', 'dark', 'warm']) {
        await browser.evaluate(`document.querySelector('[data-item=${theme}]').click()`);
        await browser.pause(350);
        await browser.cdp('Emulation.setEmulatedMedia', {media: 'print'});
        await browser.pause(100);
        const state = await browser.evaluate(`({
          background:getComputedStyle(document.body).backgroundColor,
          rootBackground:getComputedStyle(document.documentElement).backgroundColor,
          colorScheme:getComputedStyle(document.documentElement).colorScheme,
          text:getComputedStyle(document.querySelector('.content p')).color,
          details:[...document.querySelectorAll('.content details:not(.blog-mobile-toc)')].map(d=>d.open),
          code:[...document.querySelectorAll('.content pre')].map(p=>({wrap:getComputedStyle(p).whiteSpace,overflow:getComputedStyle(p).overflow,maxHeight:getComputedStyle(p).maxHeight})),
          tables:[...document.querySelectorAll('.content table')].map(t=>({display:getComputedStyle(t).display,overflow:getComputedStyle(t).overflow})),
          surfaces:[...document.querySelectorAll('.robot-filter-lab,.jev-routing-lab')].map(e=>getComputedStyle(e).backgroundColor),
          tokenBackgrounds:[...document.querySelectorAll('.content .chroma :is(.err,.gd,.gi)')].map(e=>getComputedStyle(e).backgroundColor),
          hiddenControls:[...document.querySelectorAll('.hextra-nav-container,.hextra-toc,.hextra-footer,.hextra-code-copy-btn-container')].every(e=>!e.getBoundingClientRect().height)
        })`);
        assert.equal(state.background, 'rgb(255, 255, 255)');
        assert.equal(state.rootBackground, 'rgb(255, 255, 255)');
        assert.equal(state.colorScheme, 'light');
        assert.equal(state.text, 'rgb(17, 17, 17)');
        assert(state.details.every(Boolean));
        assert(state.hiddenControls);
        assert(state.code.every(c => c.wrap === 'pre-wrap' && c.overflow === 'visible' && c.maxHeight === 'none'));
        assert(state.tables.every(t => t.display === 'table' && t.overflow === 'visible'));
        assert(state.surfaces.every(color => color === 'rgb(255, 255, 255)'));
        assert(state.tokenBackgrounds.every(color => color === 'rgba(0, 0, 0, 0)'));
        const signature = await browser.evaluate(`(async()=>{const c=document.querySelector('.robot-filter-lab canvas');if(!c)return null;const bytes=c.getContext('2d').getImageData(0,0,c.width,c.height).data;return [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');})()`);
        if (signature) {
          if (canvasSignature) assert.equal(signature, canvasSignature, 'Printed canvas depends on screen theme');
          canvasSignature = signature;
        }
        await browser.cdp('Emulation.setEmulatedMedia', {media: ''});
        await browser.pause(100);
        assert.deepEqual(await browser.evaluate(`[...document.querySelectorAll('.content details:not(.blog-mobile-toc)')].map(d=>d.open)`), original);
        report.print.push({route, theme, ...state, canvasSignature: signature, stateRestored: true});
        if (route.includes('robot-filters') && theme === 'dark') {
          await browser.evaluate(`(() => {
            window.__canvasPrintTrace=[];window.__canvasStrokeColors=[];
            const context=document.querySelector('.robot-filter-lab canvas').getContext('2d'),stroke=context.stroke;
            context.stroke=function(...args){window.__canvasStrokeColors.push(this.strokeStyle);return stroke.apply(this,args)};
            for(const type of ['beforeprint','afterprint'])window.addEventListener(type,()=>window.__canvasPrintTrace.push({type,curves:window.__canvasStrokeColors.slice(-3)}));
          })()`);
          const pdf = await browser.cdp('Page.printToPDF', {printBackground: true, paperWidth: 8.27, paperHeight: 11.69, marginTop: .4, marginBottom: .4, marginLeft: .4, marginRight: .4});
          fs.writeFileSync(path.join(output, 'filter-native-dark.pdf'), Buffer.from(pdf.data, 'base64'));
          await browser.pause(100);
          const trace = await browser.evaluate('window.__canvasPrintTrace');
          assert.deepEqual(trace, [
            {type: 'beforeprint', curves: ['#6d7d90', '#26384c', '#317ebd']},
            {type: 'afterprint', curves: ['#6d7d90', '#eef3f9', '#86bdff']}
          ]);
          report.nativeCanvasPrint = trace;
          assert.deepEqual(await browser.evaluate(`[...document.querySelectorAll('.content details:not(.blog-mobile-toc)')].map(d=>d.open)`), original);
        }
      }
    }

    await browser.navigate(base + '/posts/ai/supermimic/');
    await browser.evaluate(`(() => {
      const image=document.querySelector('.post-image-link img');
      window.__printZoomOpened=false;
      image.addEventListener('medium-zoom:opened',()=>{window.__printZoomOpened=true},{once:true});
      image.closest('a').focus();image.closest('a').click();
    })()`);
    for (let attempt=0;attempt<100;attempt++) {
      if(await browser.evaluate('window.__printZoomOpened'))break;
      await browser.pause(50);
    }
    assert(await browser.evaluate('window.__printZoomOpened'));
    await browser.cdp('Emulation.setEmulatedMedia', {media: 'print'});
    const zoomPrint = await browser.evaluate(`({
      originalVisible:getComputedStyle(document.querySelector('.post-image-link img')).visibility==='visible',
      overlayHidden:[...document.querySelectorAll('.medium-zoom-overlay,.medium-zoom-image--opened')].every(e=>getComputedStyle(e).visibility==='hidden'&&getComputedStyle(e).position==='fixed')
    })`);
    assert(zoomPrint.originalVisible && zoomPrint.overlayHidden);
    const zoomPDF=await browser.cdp('Page.printToPDF', {printBackground:true,paperWidth:8.27,paperHeight:11.69,
      marginTop:.4,marginBottom:.4,marginLeft:.4,marginRight:.4});
    fs.writeFileSync(path.join(output,'zoom-open-supermimic.pdf'),Buffer.from(zoomPDF.data,'base64'));
    await browser.cdp('Emulation.setEmulatedMedia', {media: ''});
    await browser.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
    await browser.cdp('Page.bringToFront');
    await browser.key('Escape', 'Escape', 27);
    let zoomRestored=false;
    for (let attempt=0;attempt<40;attempt++) {
      zoomRestored=await browser.evaluate(`!document.querySelector('.medium-zoom-image--opened') && document.activeElement.classList.contains('post-image-link')`);
      if(zoomRestored)break;
      await browser.pause(50);
    }
    report.zoomAfterPrint=await browser.evaluate(`({
      clones:document.querySelectorAll('.medium-zoom-image--opened').length,
      overlays:document.querySelectorAll('.medium-zoom-overlay').length,
      activeElement:document.activeElement.outerHTML.slice(0,300)
    })`);
    assert(zoomRestored,JSON.stringify(report.zoomAfterPrint));
    report.zoomPrint={...zoomPrint,screenRestored:true};

    const failures = report.contrast.filter(row => row.issues.length);
    report.summary = {contrastLayouts: report.contrast.length, textSamples: report.contrast.reduce((n, row) => n + row.checked, 0),
      contrastFailures: failures.length, printLayouts: report.print.length, runtimeExceptions: browser.exceptions.length};
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({...report.summary, failures: failures.map(f => ({route: f.route, theme: f.theme, issues: f.issues.slice(0, 5)}))}, null, 2));
    if (!captureOnly) {
      assert.equal(browser.exceptions.length, 0, JSON.stringify(browser.exceptions));
      assert.equal(failures.length, 0, 'Text contrast issues; see ' + path.join(output, 'results.json'));
    }
  } finally {
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(report, null, 2) + '\n');
    await browser.cdp('Emulation.setEmulatedMedia', {media: ''}).catch(() => {});
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
