#!/usr/bin/env node
// Exercise every article image after changes to the shared viewer.
// Use a fresh Hugo build; run serially with other browser checks.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('./lib/browser.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13163';
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-blog-image-viewers';
const drafts = process.argv.includes('--drafts');
const posts = JSON.parse(fs.readFileSync(path.join(__dirname, '../docs/blog-editorial-review.json'))).posts
  .filter(post => drafts || !post.draft);

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const browser = await connect(port);
  const report = {images: [], exceptions: browser.exceptions};
  const wait = async expression => {
    const deadline = Date.now() + 15000;
    while (Date.now() < deadline) {
      if (await browser.evaluate(expression)) return;
      await browser.pause(30);
    }
    throw Error('Image viewer timed out: ' + expression);
  };
  try {
    for (const width of [320, 1440]) {
      await browser.viewport(width);
      for (const post of posts) {
        const route = '/' + post.path.replace(/^content\//, '').replace(/\/index\.md$|\.md$/, '').toLowerCase() + '/';
        await browser.navigate(base + route);
        await browser.evaluate('document.fonts.ready');
        const count = await browser.evaluate(`(() => {
          window.__viewerSources=[...document.querySelectorAll('.post-image-link img[data-zoomable]')];
          return window.__viewerSources.length;
        })()`);
        for (const theme of ['light', 'dark']) {
          await browser.evaluate(`document.querySelector('[data-item=${theme}]').click()`);
          for (let index = 0; index < count; index++) {
            const source = await browser.evaluate(`(async () => {
              const image=window.__viewerSources[${index}],link=image.closest('a');
              for(let parent=image.parentElement;parent;parent=parent.parentElement)if(parent.tagName==='DETAILS')parent.open=true;
              image.loading='eager';await image.decode();image.scrollIntoView({block:'center'});link.focus({preventScroll:true});
              return {original:new URL(image.dataset.zoomSrc||link.href||image.src,document.baseURI).href,
                alt:image.alt,scroll:scrollY,format:new URL(link.href).pathname.split('.').pop()};
            })()`);
            await browser.key('Enter', 'Enter', 13);
            await wait(`document.querySelector('[data-image-viewer]')?.open`);
            await wait(`document.querySelector('[data-image-viewer]').dataset.state!=='loading'`);
            const viewer = await browser.evaluate(`(() => {
              const dialog=document.querySelector('[data-image-viewer]'),image=dialog.querySelector('img'),r=dialog.getBoundingClientRect();
              return {state:dialog.dataset.state,source:image.currentSrc,alt:image.alt,loaded:image.naturalWidth>0,
                modal:dialog.matches(':modal'),focused:dialog.contains(document.activeElement),
                fits:r.left>=-1&&r.top>=-1&&r.right<=innerWidth+1&&r.bottom<=innerHeight+1,
                overflow:dialog.scrollWidth>dialog.clientWidth+1,
                originalVisible:getComputedStyle(window.__viewerSources[${index}]).visibility==='visible'};
            })()`);
            assert.equal(viewer.state, 'ready', JSON.stringify({route,index,width,theme,viewer}));
            assert.equal(viewer.source, source.original);
            assert.equal(viewer.alt, source.alt);
            assert(viewer.loaded&&viewer.modal&&viewer.focused&&viewer.fits&&!viewer.overflow&&viewer.originalVisible,
              JSON.stringify({route,index,width,theme,viewer}));
            await browser.key('Tab', 'Tab', 9);
            assert(await browser.evaluate(`document.querySelector('[data-image-viewer]').contains(document.activeElement)`));
            if (index===0 && ['/posts/ai/supermimic/','/posts/queue/'].includes(route)) {
              await browser.screenshot(path.join(output,route.split('/').filter(Boolean).at(-1)+'-'+width+'-'+theme+'.png'));
            }
            await browser.key('Escape', 'Escape', 27);
            const closed = await browser.evaluate(`({
              open:document.querySelector('[data-image-viewer]').open,
              focused:document.activeElement===window.__viewerSources[${index}].closest('a'),
              overflow:document.documentElement.style.overflow,scroll:scrollY,
              children:document.querySelector('[data-image-viewer-stage]').childElementCount})`);
            assert(!closed.open&&closed.focused&&closed.overflow===''&&closed.children===0,
              JSON.stringify({route,index,width,theme,closed}));
            assert(Math.abs(closed.scroll-source.scroll)<=1,'Article position changed after viewing an image');
            report.images.push({route,index,width,theme,format:source.format,passed:true});
          }
        }
      }
    }
    report.summary={cases:report.images.length,articles:new Set(report.images.map(row=>row.route)).size,
      formats:[...new Set(report.images.map(row=>row.format))].sort(),runtimeExceptions:browser.exceptions.length};
    console.log(JSON.stringify(report.summary,null,2));
    assert.equal(browser.exceptions.length,0,JSON.stringify(browser.exceptions));
  } finally {
    fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(report,null,2)+'\n');
    await browser.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1});
