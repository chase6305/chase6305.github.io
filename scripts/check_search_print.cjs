#!/usr/bin/env node
// Print the page while search is ready, pending or dismissing. Run serially.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('./lib/browser.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13200';
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-search-print';
const edgesOnly = process.argv.includes('--edges-only');

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const b = await connect(port), report = {cases: [], pdfs: [], exceptions: b.exceptions};
  let savedTheme;
  const wait = async expression => {
    const deadline = Date.now() + 15000;
    while (Date.now() < deadline) {
      if (await b.evaluate(expression)) return;
      await b.pause(40);
    }
    throw Error('Timed out: ' + expression);
  };
  try {
    for (const route of edgesOnly ? [] : ['/posts/ai/supermimic/', '/zh-cn/']) {
      for (const width of [320, 1440]) for (const theme of ['light', 'dark', 'warm']) {
        for (const reduced of [false, true]) for (const state of ['ready', 'pending', 'closing']) {
          await b.viewport(width);
          await b.cdp('Emulation.setEmulatedMedia', {media: '', features: [
            {name: 'prefers-reduced-motion', value: reduced ? 'reduce' : 'no-preference'}
          ]});
          await b.navigate(base + route);
          if (savedTheme === undefined) savedTheme = await b.evaluate("localStorage.getItem('color-theme')");
          await b.evaluate(`document.querySelector('[data-item=${theme}]').click();document.fonts.ready`);
          const disclosures = await b.evaluate("[...document.querySelectorAll('.content details')].map(e=>e.open)");
          await b.evaluate(`(() => {
            window.__printOpener = document.querySelector('[data-search-open]');
            __printOpener.focus();__printOpener.click();
            const input = document.querySelector('.hextra-search-input');
            window.__originalPrintSearch = window.hextraSearch.search;
            ${state === 'pending' ? 'window.hextraSearch.search = query => new Promise(resolve => {window.__finishPrintQuery = () => __originalPrintSearch(query).then(resolve);});' : ''}
            input.value = 'MorphMimic';input.dispatchEvent(new Event('input', {bubbles:true}));
          })()`);
          if (state !== 'pending') await wait("document.querySelector('#hextra-search-results a[role=option]')!==null");
          if (state === 'closing') await b.evaluate("document.querySelector('[data-search-close]').click()");
          const wasOpen = await b.evaluate("document.getElementById('hextra-search-dialog').open");
          await b.evaluate(`window.__printObserved = null;
            window.__screenObserved = false;
            matchMedia('print').addEventListener('change', event => {
              if (event.matches) window.__printObserved = {
                open:document.getElementById('hextra-search-dialog').open,
                expanded:document.querySelector('.hextra-search-input').getAttribute('aria-expanded')
              };
              else window.__screenObserved = true;
            })`);
          await b.cdp('Emulation.setEmulatedMedia', {media: 'print'});
          // CDP returns before media-change events are dispatched. Observe the
          // actual event rather than assuming it ran before the next command.
          await wait('__printObserved!==null');
          assert.deepEqual(await b.evaluate('__printObserved'), {open:false, expanded:'false'});
          const printed = await b.evaluate(`(() => {
            const d=document.getElementById('hextra-search-dialog'),i=d.querySelector('input');
            return {open:d.open,modal:d.matches(':modal'),closing:d.dataset.state,
              expanded:i.getAttribute('aria-expanded'),query:i.value,focused:document.activeElement===__printOpener};
          })()`);
          assert(!printed.open && !printed.modal, JSON.stringify({route, width, theme, reduced, state, printed}));
          assert.equal(printed.closing, undefined);
          assert.equal(printed.expanded, 'false');
          assert.equal(printed.query, 'MorphMimic');
          if (width === 1440 && !reduced && state === 'ready') {
            const pdf = await b.cdp('Page.printToPDF', {printBackground:true, preferCSSPageSize:true,
              paperWidth:8.2677, paperHeight:11.6929, marginTop:.4, marginBottom:.4, marginLeft:.4, marginRight:.4});
            const name = (route === '/zh-cn/' ? 'zh-home' : 'supermimic') + '-' + theme + '.pdf';
            fs.writeFileSync(path.join(output, name), Buffer.from(pdf.data, 'base64'));
            report.pdfs.push(name);
          }
          await b.cdp('Emulation.setEmulatedMedia', {media: ''});
          await wait('__screenObserved');
          if (wasOpen) assert(await b.evaluate('document.activeElement===__printOpener'), 'Returning from print did not restore the search opener');
          assert.deepEqual(await b.evaluate("[...document.querySelectorAll('.content details')].map(e=>e.open)"), disclosures);
          if (state === 'pending') {
            await b.evaluate('__finishPrintQuery()');
            await wait("document.querySelector('#hextra-search-results a[role=option]')!==null");
            assert(await b.evaluate("!document.getElementById('hextra-search-dialog').open && document.activeElement===__printOpener"));
          }
          // A close animation/event from printing must not dismiss a new session.
          await b.evaluate('__printOpener.click()');
          await b.pause(450);
          assert(await b.evaluate(`document.getElementById('hextra-search-dialog').open
            && document.querySelector('.hextra-search-input').getAttribute('aria-expanded')==='true'
            && document.querySelector('.hextra-search-input').value==='MorphMimic'`));
          await b.key('Escape', 'Escape', 27);
          await wait("!document.getElementById('hextra-search-dialog').open");
          report.cases.push({route, width, theme, reduced, state, queryPreserved:true, focusRestored:wasOpen});
        }
      }
    }
    const openForPrint = async () => {
      await b.viewport(1440);
      await b.cdp('Emulation.setEmulatedMedia', {media:'', features:[]});
      await b.navigate(base + '/posts/ai/supermimic/');
      if (savedTheme === undefined) savedTheme = await b.evaluate("localStorage.getItem('color-theme')");
      await b.evaluate(`(() => {
        window.__printOpener = [...document.querySelectorAll('[data-search-open]')].find(e=>e.checkVisibility());
        __printOpener.focus();__printOpener.click();
        document.querySelector('.hextra-search-input').value = 'MorphMimic';
        window.__edgePrintMode = false;window.__edgePrintTrace = [];
        matchMedia('print').addEventListener('change', e => {window.__edgePrintMode=e.matches});
        for(const type of ['beforeprint','afterprint']) window.addEventListener(type, () => {
          __edgePrintTrace.push({type,print:matchMedia('print').matches,
            open:document.getElementById('hextra-search-dialog').open});
        });
      })();document.fonts.ready`);
    };
    await openForPrint();
    const nativePDF = await b.cdp('Page.printToPDF', {printBackground:true, preferCSSPageSize:true,
      paperWidth:8.2677, paperHeight:11.6929, marginTop:.4, marginBottom:.4, marginLeft:.4, marginRight:.4});
    fs.writeFileSync(path.join(output, 'supermimic-native-entry.pdf'), Buffer.from(nativePDF.data, 'base64'));
    report.pdfs.push('supermimic-native-entry.pdf');
    await wait("!document.getElementById('hextra-search-dialog').open && document.activeElement===__printOpener");
    report.nativePrintTrace = await b.evaluate('__edgePrintTrace');
    assert(report.nativePrintTrace.some(e=>e.type==='beforeprint'&&!e.open));

    await openForPrint();
    await b.cdp('Emulation.setEmulatedMedia', {media:'print'});await wait('__edgePrintMode');
    assert(await b.evaluate(`(() => {window.__readerFocus=document.querySelector('.content a[href]');
      __readerFocus.focus({preventScroll:true});return document.activeElement===__readerFocus;})()`));
    await b.cdp('Emulation.setEmulatedMedia', {media:''});await wait('!__edgePrintMode');
    assert(await b.evaluate('document.activeElement===__readerFocus'), 'Print restoration stole a later focus');
    report.laterFocusPreserved = true;

    await openForPrint();
    await b.cdp('Emulation.setEmulatedMedia', {media:'print'});await wait('__edgePrintMode');
    await b.evaluate("document.body.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true}))");
    await b.cdp('Emulation.setEmulatedMedia', {media:''});await wait('!__edgePrintMode');
    assert(await b.evaluate('document.activeElement===document.body'), 'Reader pointer input was overridden');
    report.readerPointerPreserved = true;

    await openForPrint();
    await b.cdp('Emulation.setEmulatedMedia', {media:'print'});await wait('__edgePrintMode');
    const other = await connect(port);
    try {
      assert(!await b.evaluate('document.hasFocus()'));
      await b.cdp('Emulation.setEmulatedMedia', {media:''});
      // Background media-change callbacks can wait for the next visible frame.
      // Deliver afterprint while still backgrounded, then inspect the delayed
      // media notification after returning to the original tab.
      assert(!await b.evaluate("matchMedia('print').matches"));
      await b.evaluate("window.dispatchEvent(new Event('afterprint'))");
      assert(await b.evaluate('document.activeElement===document.body'));
      assert(await other.evaluate('document.hasFocus()'), 'Background print restoration stole the active tab');
    } finally {await other.close();await b.cdp('Page.bringToFront');}
    await wait('!__edgePrintMode');
    assert(await b.evaluate('document.activeElement===document.body'));
    report.backgroundTabPreserved = true;

    // Same-task close/reopen explicitly exercises the queued native close event.
    await b.evaluate(`__printOpener.click();window.dispatchEvent(new Event('beforeprint'));__printOpener.click()`);
    await b.pause(450);
    assert(await b.evaluate("document.getElementById('hextra-search-dialog').open && document.querySelector('.hextra-search-input').getAttribute('aria-expanded')==='true'"));
    await b.evaluate("window.dispatchEvent(new Event('afterprint'))");
    report.synchronousReopen = true;
    assert.equal(b.exceptions.length, 0, JSON.stringify(b.exceptions));
    console.log(JSON.stringify({cases:report.cases.length, pdfs:report.pdfs.length, synchronousReopen:true,
      nativePrint:true, laterFocus:true, pointerInput:true, backgroundTab:true, exceptions:0}));
  } finally {
    await b.cdp('Emulation.setEmulatedMedia', {media:'', features:[]}).catch(() => {});
    if (savedTheme !== undefined) await b.evaluate(savedTheme === null ? "localStorage.removeItem('color-theme')"
      : 'localStorage.setItem("color-theme",' + JSON.stringify(savedTheme) + ')').catch(() => {});
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(report, null, 2) + '\n');
    await b.close();
  }
})().catch(error => {console.error(error);process.exitCode=1;});
