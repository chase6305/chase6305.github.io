// Real table arrows, exact async code copies and all new Muon chapter links.
// Serial private-CDP use only.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('../../../scripts/lib/browser.cjs');
const [base, port, out] = process.argv.slice(2);
const routes = ['/posts/ai/muon-optimizer/','/posts/ai/muon-learning-rate-batch-size/'];
(async () => {
  fs.mkdirSync(out, {recursive: true});
  const b = await connect(Number(port));
  const report = {tables: [], copies: [], chapters: [], exceptions: b.exceptions};
  try {
    for (const route of routes) {
    for (const scripts of [true, false]) {
      await b.cdp('Emulation.setScriptExecutionDisabled', {value: !scripts});
      await b.viewport(320);
      for (const size of [100, 200]) for (const header of (route.includes('batch-size') ? ['比较条件'] : ['想弄清的问题','方法','约定'])) {
        await b.navigate(base + route);
        await b.evaluate(`document.documentElement.style.fontSize='${size}%';document.fonts.ready`);
        await b.pause(180);
        const before = await b.evaluate(`(()=>{
          document.querySelectorAll('.content details:not(.blog-mobile-toc)').forEach(d=>d.open=true);
          window.__table=[...document.querySelectorAll('.content table')].find(t=>
            [...t.querySelectorAll('th')].some(h=>h.textContent.trim()===${JSON.stringify(header)}));
          __table.scrollIntoView({block:'center'});__table.scrollLeft=0;__table.focus();
          return {width:__table.clientWidth,scroll:__table.scrollWidth,focused:document.activeElement===__table,tabIndex:__table.tabIndex}})()`);
        if (before.scroll <= before.width + 2) {
          // A fitting two-column table needs no temporary tab stop after enhancement.
          assert(scripts ? before.tabIndex === -1 && !before.focused : before.tabIndex === 0 && before.focused,
            JSON.stringify({scripts, size, header, before}));
          report.tables.push({route,scripts, size, header, before, fits:true});
          continue;
        }
        assert(before.focused && before.tabIndex === 0, JSON.stringify(before));
        await b.key('ArrowRight', 'ArrowRight', 39); await b.pause(180);
        const after = await b.evaluate('({distance:__table.scrollLeft,focused:document.activeElement===__table,pageX:scrollX})');
        assert(after.focused && after.distance > 0 && after.pageX === 0, JSON.stringify(after));
        report.tables.push({route,scripts, size, header, before, after});
      }
    }
    await b.cdp('Emulation.setScriptExecutionDisabled', {value: false});
    const source = fs.readFileSync(path.join(__dirname, '../../../content'+route+'index.md'), 'utf8');
    const blocks = [...source.matchAll(/```(?:bash|python)\n([\s\S]*?)```/g)].map(m => m[1]);
    for (const marker of (route.includes('batch-size') ? ['python3 -B batch_math.py'] : ['import torch','python3 -B muon_math.py'])) {
      const snippet = blocks.find(s => s.includes(marker)); assert(snippet);
      const expected = snippet.replace(/\n$/, '');
      for (const width of [320, 1440]) for (const size of [100, 200]) {
        await b.viewport(width); await b.navigate(base + route);
        await b.evaluate(`document.documentElement.style.fontSize='${size}%';document.fonts.ready`);
        await b.pause(180);
        await b.evaluate(`(()=>{
          window.__copies=[];Object.defineProperty(navigator,'clipboard',{configurable:true,value:{
            writeText:text=>{__copies.push(text);return new Promise(resolve=>{window.__finishCopy=resolve})}}});
          const block=[...document.querySelectorAll('.hextra-code-block')].find(e=>e.querySelector('code')?.textContent===${JSON.stringify(expected)});
          window.__button=block.querySelector('.hextra-code-copy-btn');
          __button.scrollIntoView({block:'center'});__button.focus();})()`);
        await b.key('Enter', 'Enter', 13); await b.pause(70);
        const pending = await b.evaluate('({count:__copies.length,focused:document.activeElement===__button,disabled:__button.disabled,busy:__button.getAttribute("aria-busy")})');
        assert(pending.count === 1 && pending.focused && !pending.disabled && pending.busy === 'true', JSON.stringify(pending));
        assert.equal(await b.evaluate('__copies[0]'), expected);
        await b.key('Enter', 'Enter', 13); assert.equal(await b.evaluate('__copies.length'), 1);
        await b.key('Tab', 'Tab', 9);
        await b.evaluate('window.__later=document.activeElement;__finishCopy()'); await b.pause(70);
        const settled = await b.evaluate('({laterFocus:document.activeElement===__later,busy:__button.hasAttribute("aria-busy")})');
        assert(settled.laterFocus && !settled.busy);
        report.copies.push({route,marker, width, size, sha256:crypto.createHash('sha256').update(expected).digest('hex'), pending, exact:true, duplicateRejected:true, settled});
      }
    }
    for (const width of [320, 768]) for (const size of [100, 200]) {
      await b.viewport(width); await b.navigate(base + route);
      await b.evaluate(`document.documentElement.style.fontSize='${size}%';document.fonts.ready`);
      const count = await b.evaluate("document.querySelectorAll('.blog-mobile-toc a').length");
      for (let index = 0; index < count; index++) {
        const state = await b.evaluate(`(()=>{
          const toc=document.querySelector('.blog-mobile-toc');toc.open=true;
          const link=toc.querySelectorAll('a')[${index}],id=decodeURIComponent(link.hash.slice(1));
          const anchor=document.getElementById(id),heading=anchor.closest('h1,h2,h3,h4,h5,h6')||anchor;link.click();
          const r=heading.getBoundingClientRect(),nav=document.querySelector('.hextra-nav-container nav');
          return {id,hash:decodeURIComponent(location.hash.slice(1)),focused:document.activeElement===heading,closed:!toc.open,
            top:r.top,navBottom:nav?.getBoundingClientRect().bottom||0,height:innerHeight,
            overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1};})()`);
        assert(state.id === state.hash && state.focused && state.closed && !state.overflow && state.top >= state.navBottom - 1 && state.top < state.height - 24, JSON.stringify({width,size,index,...state}));
        report.chapters.push({route,width, size, index, ...state});
      }
    }
    }
    assert.deepEqual(b.exceptions, []);
    console.log(JSON.stringify({tableContexts:report.tables.length, tableArrows:report.tables.filter(r=>!r.fits).length, fittingTables:report.tables.filter(r=>r.fits).length, exactCopies:report.copies.length, chapterClicks:report.chapters.length, exceptions:0}));
  } finally {
    fs.writeFileSync(path.join(out, 'results.json'), JSON.stringify(report, null, 2) + '\n');
    await b.cdp('Emulation.setScriptExecutionDisabled', {value:false}).catch(()=>{});
    await b.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
