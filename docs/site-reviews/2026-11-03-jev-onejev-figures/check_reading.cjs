// Review the Jev and OneJev article against an immutable, fresh Hugo build.
// Serial CDP use only; no application files or existing browser tabs are changed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('../../../scripts/lib/browser.cjs');
const [base, portText, out] = process.argv.slice(2);
const articles = [
  {route: '/posts/ai/jev-decision-layer/', anchors: ['onejev', 'onejev-model-and-engine', 'onejev-probability-readout', 'onejev-state-forking',
    'onejev-quickstart', 'onejev-image-example', 'onejev-backends', 'onejev-training',
    'onejev-data-and-evaluation', 'onejev-results', 'fig-jev-onejev-deployment', 'fig-onejev-probability-readout', 'fig-onejev-state-forking', 'architecture', 'confidence-routing', 'lab']},
];
(async () => {
  fs.mkdirSync(out, {recursive: true});
  const b = await connect(Number(portText));
  const report = {layouts: [], anchors: [], exceptions: b.exceptions};
  try {
    for (const scripts of [true, false]) {
      await b.cdp('Emulation.setScriptExecutionDisabled', {value: !scripts});
      for (const {route} of articles) for (const width of [320, 1440]) {
        await b.viewport(width, 1000);
        await b.navigate(base + route);
        await b.evaluate(`Promise.all([...document.querySelectorAll('.content img')].map(i=>{
          i.loading='eager';return i.decode().catch(()=>null)}))`);
        // Include optional worked examples and answers in the geometry checks.
        await b.evaluate(`document.querySelectorAll('.content details:not(.blog-mobile-toc)').forEach(d=>d.open=true)`);
        for (const size of [100, 200]) for (const theme of ['light', 'dark', 'warm']) {
          await b.evaluate(`document.documentElement.style.fontSize='${size}%';
            document.documentElement.classList.remove('dark','warm');
            ${theme === 'light' ? '' : `document.documentElement.classList.add('${theme}');`}
            document.documentElement.style.colorScheme='${theme === 'dark' ? 'dark' : 'light'}';
            document.fonts.ready`);
          await b.pause(250);
          const state = await b.evaluate(`(() => {
            const issues=[];
            if(document.documentElement.scrollWidth>document.documentElement.clientWidth+1)issues.push('page overflow');
            const images=[...document.querySelectorAll('.content img')];
            images.forEach((i,n)=>{if(!i.complete||!i.naturalWidth)issues.push('image '+n+' failed')});
            const captions=[...document.querySelectorAll('.article-figure figcaption')];
            captions.forEach((c,n)=>{
              const t=c.querySelector('.article-figure__text'),r=t?.getBoundingClientRect(),p=c.getBoundingClientRect();
              if(!t||r.left<p.left-1||r.right>p.right+1||t.scrollWidth>t.clientWidth+1||t.scrollHeight>t.clientHeight+1)
                issues.push('caption '+n+' clipped');
            });
            const tables=[...document.querySelectorAll('.content table')].filter(t=>t.clientWidth);
            tables.forEach((t,n)=>{
              if(t.matches('.table-readable')&&[...t.querySelectorAll('th,td')].some(c=>c.getBoundingClientRect().width+1<parseFloat(getComputedStyle(c).minWidth)))
                issues.push('table '+n+' collapsed columns');
              if(t.scrollWidth>t.clientWidth+2&&t.tabIndex!==0)issues.push('table '+n+' not focusable');
            });
            const regions=[...document.querySelectorAll('.content .katex-display,.content pre')].filter(e=>e.clientWidth);
            regions.forEach((e,n)=>{if((e.scrollWidth>e.clientWidth+2||e.scrollHeight>e.clientHeight+2)&&e.tabIndex!==0)
              issues.push('math/code '+n+' not focusable')});
            const root=getComputedStyle(document.documentElement);
            return {issues,images:images.length,captions:captions.length,tables:tables.length,regions:regions.length,
              colorScheme:root.colorScheme,rootFont:root.fontSize};
          })()`);
          report.layouts.push({route, scripts, width, size, theme, ...state});
          assert.deepEqual(state.issues, [], JSON.stringify(report.layouts.at(-1)));
          assert.equal(state.colorScheme, theme === 'dark' ? 'dark' : 'light');
          if (size === 100 && (theme === 'light' || theme === 'dark')) {
            const id = 'onejev-quickstart';
            await b.evaluate(`document.getElementById('${id}').scrollIntoView({block:'start'})`);
            await b.screenshot(path.join(out, `${route.split('/').at(-2)}-${width}-${theme}-${scripts ? 'js' : 'native'}.png`));
          }
        }
      }
      // Fresh document loads exercise native fragment positioning too.
      for (const {route, anchors} of articles) for (const width of [320, 1440]) {
        await b.viewport(width, 1000);
        for (const id of anchors) {
          await b.navigate('about:blank');
          await b.navigate(base + route + '#' + id);
          await b.evaluate('document.fonts.ready');
          await b.pause(200);
          const a = await b.evaluate(`(()=>{const e=document.getElementById('${id}'),h=e?.closest('h2,h3')||e;
            if(!h)return {exists:false};const r=h.getBoundingClientRect();
            return {exists:true,top:r.top,bottom:r.bottom,hash:location.hash}})()`);
          report.anchors.push({route, scripts, width, id, ...a});
          assert(a.exists && a.top >= 55 && a.bottom <= 1000 && a.hash === '#' + id, JSON.stringify(report.anchors.at(-1)));
        }
      }
    }
    assert.deepEqual(b.exceptions, []);
    console.log(JSON.stringify({layouts: report.layouts.length, anchors: report.anchors.length, exceptions: 0}));
  } finally {
    fs.writeFileSync(path.join(out, 'results.json'), JSON.stringify(report, null, 2) + '\n');
    await b.cdp('Emulation.setScriptExecutionDisabled', {value: false}).catch(()=>{});
    await b.close();
  }
})().catch(e=>{console.error(e);process.exitCode=1});
