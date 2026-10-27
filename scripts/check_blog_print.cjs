#!/usr/bin/env node
// A4 print geometry, complete disclosures and screen-state restoration.
// Run serially with other CDP checks. --drafts requires a draft-inclusive build.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('./lib/browser.cjs');
const {contrastAudit} = require('./lib/contrast.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13147';
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-blog-print';
const drafts = process.argv.includes('--drafts');
const features = process.argv.includes('--forced-colors') ? [
  {name: 'forced-colors', value: 'active'}, {name: 'prefers-color-scheme', value: 'dark'}
] : [];
const posts = JSON.parse(fs.readFileSync(path.join(__dirname, '../docs/blog-editorial-review.json'))).posts
  .filter(post => drafts || !post.draft);

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const browser = await connect(port);
  const report = {layouts: [], exceptions: browser.exceptions, forcedColors: features.length > 0};
  try {
    // A4 with 0.4-inch side margins leaves about 717 CSS pixels.
    await browser.viewport(716);
    for (const scripts of [true, false]) {
      await browser.cdp('Emulation.setScriptExecutionDisabled', {value: !scripts});
      for (const post of posts) {
        const route = '/' + post.path.replace(/^content\//, '').replace(/\/index\.md$|\.md$/, '').toLowerCase() + '/';
        await browser.navigate(base + route);
        await browser.evaluate('document.fonts.ready');
        const original = await browser.evaluate(`[...document.querySelectorAll('.content details:not(.blog-mobile-toc)')].map(d=>d.open)`);
        for (const theme of ['light', 'dark', 'warm']) {
          await browser.evaluate(`document.documentElement.classList.remove('dark','warm');${theme === 'light' ? '' : `document.documentElement.classList.add('${theme}')`}`);
          await browser.cdp('Emulation.setEmulatedMedia', {media: 'print', features});
          await browser.pause(60);
          const state = await browser.evaluate(`(() => {
            const math=[...document.querySelectorAll('.content .katex')].flatMap(e=>{
              const html=e.querySelector('.katex-html');if(!html)return [];
              const parent=e.closest('.katex-display,td,th,p,li')||e.parentElement;
              const r=html.getBoundingClientRect(),bounds=parent.getBoundingClientRect();if(!r.width)return [];
              const bases=[...html.querySelectorAll('.base')].map(b=>b.getBoundingClientRect());
              const left=Math.min(r.left,...bases.map(b=>b.left)),right=Math.max(r.right,...bases.map(b=>b.right));
              return right>bounds.right+1||left<bounds.left-1?[{tex:e.querySelector('annotation')?.textContent,width:right-left,available:bounds.width}]:[];
            });
            const code=[...document.querySelectorAll('.content pre')].filter(e=>e.clientWidth>0&&e.scrollWidth>e.clientWidth+1).length;
            const tables=[...document.querySelectorAll('.content table')].filter(e=>e.getBoundingClientRect().right>document.documentElement.clientWidth+1).length;
            const hiddenDetails=[...document.querySelectorAll('.content details:not(.blog-mobile-toc)')].filter(d=>getComputedStyle(d,'::details-content').contentVisibility!=='visible').length;
            const surfaces=[...document.querySelectorAll('.robot-filter-lab,.jev-routing-lab')].filter(e=>getComputedStyle(e).backgroundColor!=='rgb(255, 255, 255)').length;
            const paper=getComputedStyle(document.body).backgroundColor==='rgb(255, 255, 255)'
              &&getComputedStyle(document.documentElement).backgroundColor==='rgb(255, 255, 255)'
              &&getComputedStyle(document.documentElement).colorScheme==='light';
            return {math,code,tables,hiddenDetails,surfaces,paper};
          })()`);
          const text = await browser.evaluate('(' + contrastAudit.toString() + ')(".content")');
          await browser.cdp('Emulation.setEmulatedMedia', {media: '', features});
          const restoreStarted = Date.now();
          let restored;
          do {
            await browser.pause(20);
            restored = await browser.evaluate(`[...document.querySelectorAll('.content details:not(.blog-mobile-toc)')].map(d=>d.open)`);
          } while (JSON.stringify(restored) !== JSON.stringify(original) && Date.now() - restoreStarted < 2000);
          const stateRestored = JSON.stringify(restored) === JSON.stringify(original);
          const passed = state.paper && !state.math.length && !state.code && !state.tables && !state.hiddenDetails && !state.surfaces && !text.issues.length && stateRestored;
          report.layouts.push({route, scripts, theme, passed, stateRestored, restoreWaitMs: Date.now() - restoreStarted,
            ...state, textSamples: text.checked, contrastIssues: text.issues});
        }
      }
    }
    const failures = report.layouts.filter(row => !row.passed);
    report.summary = {layouts: report.layouts.length, failures: failures.length,
      textSamples: report.layouts.reduce((sum,row)=>sum+row.textSamples,0), runtimeExceptions: browser.exceptions.length};
    console.log(JSON.stringify({...report.summary, failedLayouts: failures}, null, 2));
    assert.equal(browser.exceptions.length, 0, JSON.stringify(browser.exceptions));
    assert.equal(failures.length, 0, 'Print geometry failures; see results.json');
  } finally {
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(report, null, 2) + '\n');
    await browser.cdp('Emulation.setScriptExecutionDisabled', {value: false}).catch(() => {});
    await browser.cdp('Emulation.setEmulatedMedia', {media: '', features: []}).catch(() => {});
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
