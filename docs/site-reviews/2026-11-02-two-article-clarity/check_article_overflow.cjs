#!/usr/bin/env node
// Derived from scripts/check_overflow_focus.cjs for the revised articles.
// Accept a fitting font-size case only after verifying every visible region.
// Report those separately from actual arrow-key scrolling.
// Run serially with the other loopback CDP checks.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('../../../scripts/lib/browser.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13188';
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-overflow-focus';

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const b = await connect(port), report = {cases: [], exceptions: b.exceptions};
  let savedTheme;
  const transition = async (mode, narrow) => {
    await b.viewport(narrow && mode === 'resize' ? 320 : 1440);
    await b.evaluate(`document.documentElement.style.fontSize='${narrow && mode === 'font' ? '200%' : '100%'}';document.fonts.ready`);
    await b.pause(180);
  };
  try {
    for (const [route, selector] of [
      ['/posts/cpp/std/', '.content table'],
      ['/posts/ai/supermimic/', '.content .katex-display'],
      ['/posts/ai/rlhf-rlvr-rlcd/', '.content .katex-display'],
    ]) for (const theme of ['light', 'dark', 'warm']) for (const mode of ['resize', 'font']) {
      await b.navigate(base + route);
      if (savedTheme === undefined) savedTheme = await b.evaluate("localStorage.getItem('color-theme')");
      await b.evaluate(`document.querySelector('[data-item=${theme}]').click()`);
      await transition(mode, true);
      const overflowing = await b.evaluate(`([...document.querySelectorAll(${JSON.stringify(selector)})]
        .map((e,index)=>({index,width:e.clientWidth,scroll:e.scrollWidth}))
        .filter(e=>e.width>0&&e.scroll>e.width+2).map(e=>e.index))`);
      await transition(mode, false);
      const fitting = await b.evaluate(`([...document.querySelectorAll(${JSON.stringify(selector)})]
        .map((e,index)=>({index,width:e.clientWidth,scroll:e.scrollWidth}))
        .filter(e=>e.width>0&&e.scroll<=e.width+1).map(e=>e.index))`);
      if (mode === 'font' && overflowing.length === 0) {
        // The editorial reflow may remove the fixture's need for a scroller.
        // Prove that all visible regions fit and are absent from the tab order.
        await transition(mode, true);
        const regions = await b.evaluate(`([...document.querySelectorAll(${JSON.stringify(selector)})]
          .filter(e=>e.clientWidth>0).map(e=>({width:e.clientWidth,scroll:e.scrollWidth,
            tabStop:e.hasAttribute('tabindex'),described:e.hasAttribute('aria-describedby'),
            hintHidden:!e.matches('.katex-display')||!e.previousElementSibling?.matches('.blog-math-scroll-hint')||e.previousElementSibling.hidden})))`);
        assert(regions.length && regions.every(r=>r.scroll<=r.width+1&&!r.tabStop&&!r.described&&r.hintHidden), JSON.stringify(regions));
        report.cases.push({route,theme,mode,allVisibleRegionsFit:true,regions,actualArrowScroll:false,passed:true});
        continue;
      }
      const index = overflowing.find(i => fitting.includes(i));
      assert(index !== undefined, JSON.stringify({route, mode, overflowing, fitting}));
      await transition(mode, true);
      const initial = await b.evaluate(`(() => {
        const e=document.querySelectorAll(${JSON.stringify(selector)})[${index}];window.__region=e;
        e.scrollIntoView({block:'center'});e.scrollLeft=0;e.focus();
        return {focused:document.activeElement===e,tabIndex:e.tabIndex,width:e.clientWidth,scroll:e.scrollWidth};
      })()`);
      assert(initial.focused && initial.tabIndex === 0);
      await b.key('ArrowRight', 'ArrowRight', 39);await b.pause(200);
      assert(await b.evaluate('__region.scrollLeft>0'), route + ' cannot scroll with the keyboard');
      await transition(mode, false);
      const after = await b.evaluate(`(() => {const e=__region;return {
        focused:document.activeElement===e,tabIndex:e.tabIndex,width:e.clientWidth,scroll:e.scrollWidth,
        described:e.hasAttribute('aria-describedby'),role:e.getAttribute('role'),
        hintHidden:!e.matches('.katex-display')||e.previousElementSibling.hidden};})()`);
      assert(after.focused && after.tabIndex === 0 && after.scroll <= after.width + 1, JSON.stringify(after));
      if (selector.includes('katex')) assert(!after.described && after.role === null && after.hintHidden);
      await b.key('Tab', 'Tab', 9);await b.pause(180);
      const cleanup = await b.evaluate(`({left:document.activeElement!==__region,
        tabStopRemoved:!__region.hasAttribute('tabindex'),active:document.activeElement.tagName})`);
      assert(cleanup.left && cleanup.tabStopRemoved, JSON.stringify(cleanup));
      await transition(mode, true);
      assert(await b.evaluate('__region.tabIndex===0'), 'Overflow must restore keyboard entry');
      report.cases.push({route, theme, mode, index, initial, after, cleanup, actualArrowScroll:true, passed: true});
    }
    assert.equal(b.exceptions.length, 0, JSON.stringify(b.exceptions));
    console.log(JSON.stringify({cases: report.cases.length, exceptions: 0}));
  } finally {
    if (savedTheme !== undefined) await b.evaluate(savedTheme === null ? "localStorage.removeItem('color-theme')"
      : 'localStorage.setItem("color-theme",'+JSON.stringify(savedTheme)+')').catch(()=>{});
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(report, null, 2)+'\n');
    await b.close();
  }
})().catch(error => { console.error(error);process.exitCode = 1; });
