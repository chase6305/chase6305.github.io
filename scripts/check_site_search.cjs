#!/usr/bin/env node
// Full-text search behavior, network recovery and native text-editing checks.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('./lib/browser.cjs');
const {contrastAudit} = require('./lib/contrast.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13143';
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-site-search';

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const browser = await connect(port);
  const report = {layouts: [], exceptions: browser.exceptions};
  const wait = async expression => {
    const deadline = Date.now() + 20000;
    while (Date.now() < deadline) {
      if (await browser.evaluate(expression)) return;
      await browser.pause(50);
    }
    throw Error('Condition timed out: ' + expression);
  };
  const query = async value => {
    await browser.evaluate(`(() => {const input=document.querySelector('.hextra-search-input');input.value=${JSON.stringify(value)};input.dispatchEvent(new Event('input',{bubbles:true}));})()`);
    await wait(`document.querySelector('[data-search-empty]').hidden || (!document.querySelector('[data-search-empty]').textContent.includes('Searching') && !document.querySelector('[data-search-empty]').textContent.includes('正在搜索'))`);
  };
  try {
    await browser.viewport(1440);
    await browser.navigate(base + '/posts/');
    const savedTheme = await browser.evaluate(`localStorage.getItem('color-theme')`);
    const api = await browser.evaluate(`(async()=>{
      const result={};for(const query of ['C++','A*','Cosmos','Ｃｏｓｍｏｓ','MorphMimic','ＭｏｒｐｈＭｉｍｉｃ','形态 描述','四元数','关节','C++ zzxxyy','不存在的随机词zzxxyy','+++'])result[query]=await window.hextraSearch.search(query);
      return result;
    })()`);
    assert(api['C++'].length && api['C++'].slice(0, 3).every(g => g.title.includes('C++')));
    assert(api['A*'].length >= 2 && api['A*'].slice(0, 2).every(g => g.title.includes('A*')));
    assert(!api['A*'].some(g => g.title.includes('Adam') || g.route.includes('distributed-training-memory')));
    assert.deepEqual(api.Cosmos.map(g => g.route), api['Ｃｏｓｍｏｓ'].map(g => g.route));
    assert.equal(api.MorphMimic[0].route, '/posts/ai/morphmimic/');
    assert.deepEqual(api.MorphMimic.map(g => g.route), api['ＭｏｒｐｈＭｉｍｉｃ'].map(g => g.route));
    assert(api['形态 描述'].some(g => g.route === '/posts/ai/morphmimic/'));
    assert(api['四元数'][0].matches.some(m => m.title.includes('四元数')));
    assert(api['关节'][0].title.includes('关节'));
    for (const text of ['C++ zzxxyy', '不存在的随机词zzxxyy', '+++']) assert.deepEqual(api[text], []);
    for (const groups of Object.values(api)) for (const group of groups) {
      assert.equal(new Set(group.matches.map(m => m.route)).size, group.matches.length, 'Repeated chapter destination');
    }
    report.querySemantics = 'technical punctuation, NFKC, CJK, all terms and title/chapter ranking passed';
    report.queries = Object.fromEntries(Object.entries(api).map(([q, groups]) => [q, groups.slice(0, 3).map(g => g.route)]));

    // An opening-frame callback must not reclaim focus after a result is chosen.
    await browser.evaluate(`(()=>{
      const frame=requestAnimationFrame;window.__heldSearchFrame=null;
      window.requestAnimationFrame=callback=>{window.__heldSearchFrame=callback;return 0};
      document.querySelector('[data-search-open]').click();window.requestAnimationFrame=frame;
    })()`);
    await query('MorphMimic');
    await browser.evaluate(`window.__chosenSearchResult=document.querySelector('#hextra-search-results a[role=option]');window.__chosenSearchResult.focus()`);
    assert(await browser.evaluate('document.activeElement===window.__chosenSearchResult'));
    await browser.evaluate('window.__heldSearchFrame?.(performance.now())');
    assert(await browser.evaluate('document.activeElement===window.__chosenSearchResult'), 'Opening frame stole result focus');
    report.openingFramePreservesFocus=true;
    await browser.key('Escape','Escape',27);await wait(`!document.getElementById('hextra-search-dialog').open`);

    for (const width of [320, 390, 768, 1440]) for (const theme of ['light', 'dark', 'warm']) {
      await browser.viewport(width);
      await browser.evaluate(`localStorage.setItem('color-theme',${JSON.stringify(theme)})`);
      await browser.navigate(base + '/posts/');
      await browser.evaluate(`document.querySelector('[data-search-open]').focus();document.querySelector('[data-search-open]').click()`);
      await query('');
      await browser.pause(400);
      const emptyContrast = await browser.evaluate('(' + contrastAudit.toString() + ')("#hextra-search-dialog")');
      await query('Ｃ＋＋');
      await browser.pause(400);
      const state = await browser.evaluate(`(() => {
        const dialog=document.getElementById('hextra-search-dialog'), rect=dialog.getBoundingClientRect();
        return {open:dialog.open,left:rect.left,right:rect.right,pageWidth:document.documentElement.scrollWidth,
          results:dialog.querySelectorAll('a[role=option]').length,
          highlights:[...dialog.querySelectorAll('.hextra-search-match')].map(e=>e.textContent),
          maxLength:dialog.querySelector('input').maxLength,
          innerOverflow:dialog.querySelector('[data-search-inner]').scrollWidth>dialog.querySelector('[data-search-inner]').clientWidth+1};
      })()`);
      assert(state.open && state.left >= -1 && state.right <= width + 1 && state.pageWidth <= width + 1);
      assert(state.results > 0 && state.highlights.some(text => text === 'C++'));
      assert(!state.innerOverflow && state.maxLength === 120);
      const contrast = await browser.evaluate('(' + contrastAudit.toString() + ')("#hextra-search-dialog")');
      assert(contrast.checked > 0);
      report.layouts.push({width, theme, ...state, contrast, emptyContrast});
      await browser.screenshot(path.join(output, `${width}-${theme}.png`));
      await browser.key('Escape', 'Escape', 27); await browser.pause(450);
      assert(await browser.evaluate(`!document.getElementById('hextra-search-dialog').open`));
      assert(await browser.evaluate(`document.activeElement===document.querySelector('[data-search-open]')`));
    }

    await browser.evaluate(`document.querySelector('[data-search-open]').click()`);
    await query('Cosmos训练');
    const highlightedTerms = await browser.evaluate(`[...document.querySelectorAll('.hextra-search-match')].map(e=>e.textContent.toLowerCase())`);
    assert(highlightedTerms.includes('cosmos') && highlightedTerms.includes('训练'));
    report.multipleTermHighlight = true;
    await query('Cosmos');
    const expectedTitles = await browser.evaluate(`(async()=> (await window.hextraSearch.search('SuperMimic')).map(group=>group.title))()`);
    await browser.evaluate(`const input=document.querySelector('.hextra-search-input');input.focus();input.setSelectionRange(3,3)`);
    await browser.key('Home', 'Home', 36);
    assert.equal(await browser.evaluate(`document.querySelector('.hextra-search-input').selectionStart`), 0);
    await browser.key('End', 'End', 35);
    assert.equal(await browser.evaluate(`document.querySelector('.hextra-search-input').selectionStart`), 6);
    await browser.key('ArrowDown', 'ArrowDown', 40);
    assert(await browser.evaluate(`document.activeElement===document.querySelector('.hextra-search-input') && !!document.querySelector('.hextra-search-input').getAttribute('aria-activedescendant')`));
    await browser.evaluate(`document.querySelectorAll('#hextra-search-results a[role=option]')[1].focus()`);
    await browser.key('Home', 'Home', 36);
    assert(await browser.evaluate(`document.activeElement===document.querySelector('#hextra-search-results a[role=option]')`));
    await browser.key('End', 'End', 35);
    assert(await browser.evaluate(`document.activeElement===[...document.querySelectorAll('#hextra-search-results a[role=option]')].at(-1)`));
    report.nativeHomeEndAndResultNavigation = true;

    // IME Enter commits the composition rather than activating a result.
    assert(await browser.evaluate(`(() => {const input=document.querySelector('.hextra-search-input');input.focus();const before=location.href;const event=new KeyboardEvent('keydown',{key:'Enter',code:'Enter',isComposing:true,bubbles:true,cancelable:true});input.dispatchEvent(event);return !event.defaultPrevented&&location.href===before;})()`));
    // Clearing an input removes the active option immediately, even while the
    // visual result panel is finishing its collapse animation.
    const cleared = await browser.evaluate(`(() => {const input=document.querySelector('.hextra-search-input');input.value='';input.dispatchEvent(new Event('input',{bubbles:true}));return {active:input.getAttribute('aria-activedescendant'),inert:document.getElementById('hextra-search-results').inert};})()`);
    assert.equal(cleared.active, null); assert(cleared.inert);
    await browser.key('ArrowDown', 'ArrowDown', 40); await browser.key('Enter', 'Enter', 13);
    assert.equal(await browser.evaluate('location.pathname'), '/posts/');
    report.imeAndClearedInput = true;

    await query('Cosmos');
    await browser.evaluate(`(() => {const original=window.hextraSearch.search;window.hextraSearch.search=async q=>{await new Promise(r=>setTimeout(r,q==='Cosmos'?500:50));return original(q);};const input=document.querySelector('.hextra-search-input');input.value='Cosmos';input.dispatchEvent(new Event('input',{bubbles:true}));})()`);
    assert.equal(await browser.evaluate(`document.querySelectorAll('#hextra-search-results a[role=option]').length`), 0);
    await query('SuperMimic'); await browser.pause(550);
    const actualTitles = await browser.evaluate(`[...document.querySelectorAll('#hextra-search-results > li:not(.hextra-search-child) > a > .hextra-search-title')].map(e=>e.textContent)`);
    assert.deepEqual(actualTitles, expectedTitles);
    report.staleResponsesDropped = true;

    // Fail both the on-open preload and first query, then retry in the same tab.
    const interceptor = await browser.cdp('Page.addScriptToEvaluateOnNewDocument', {source: `window.__failSearch=true;const originalFetch=window.fetch;window.fetch=(...args)=>String(args[0]).includes('search-data')&&window.__failSearch?Promise.reject(new TypeError('simulated unavailable index')):originalFetch(...args);`});
    await browser.navigate(base + '/posts/');
    await browser.evaluate(`document.querySelector('[data-search-open]').click()`);
    await query('Cosmos');
    assert(await browser.evaluate(`document.querySelector('[data-search-empty]').textContent.includes('could not load') && document.querySelectorAll('#hextra-search-results a').length===0`));
    await browser.evaluate('window.__failSearch=false');
    await query('Cosmos');
    assert(await browser.evaluate(`document.querySelectorAll('#hextra-search-results a').length>0`));
    report.networkFailureAndRetry = true;
    await browser.cdp('Page.removeScriptToEvaluateOnNewDocument', {identifier: interceptor.identifier});

    await browser.navigate(base + '/zh-cn/');
    const shared = await browser.evaluate(`(async()=>({
      cosmos:(await window.hextraSearch.search('Cosmos')).map(g=>g.route),
      cpp:(await window.hextraSearch.search('C++')).map(g=>g.route),
      latest:document.querySelectorAll('.latest-card').length
    }))()`);
    assert(shared.cosmos.includes('/posts/ai/cosmos3/') && shared.cpp.includes('/posts/cpp/gcc/') && shared.latest === 3);
    await browser.evaluate(`document.querySelector('[data-search-open]').click()`);
    await query('SuperMimic');
    assert(await browser.evaluate(`document.querySelector('#hextra-search-results a').getAttribute('href')==='/posts/ai/supermimic/'`));
    report.chineseDiscovery = true;

    const restore = savedTheme === null ? `localStorage.removeItem('color-theme')` : `localStorage.setItem('color-theme',${JSON.stringify(savedTheme)})`;
    await browser.evaluate(restore);
    const failures = report.layouts.filter(row => row.contrast.issues.length || row.emptyContrast.issues.length);
    report.summary = {layouts: report.layouts.length, contrastFailures: failures.length, runtimeExceptions: browser.exceptions.length};
    assert.equal(browser.exceptions.length, 0, JSON.stringify(browser.exceptions));
    assert.equal(failures.length, 0, 'Search dialog contrast issues; see results.json');
    console.log(JSON.stringify({...report.summary, querySemantics: report.querySemantics, nativeEditing: true, networkRetry: true, ime: true, staleResponses: true}));
  } finally {
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(report, null, 2) + '\n');
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
