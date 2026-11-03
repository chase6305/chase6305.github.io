// Actual arrow keys on revised tables and exact asynchronous copying of the new example.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('../../../scripts/lib/browser.cjs');
const [base, portText, out] = process.argv.slice(2);
const tables = [
  ['/posts/ai/supermimic/', '更新的参数'],
  ['/posts/ai/supermimic/', '窗口回报'],
  ['/posts/ai/rlhf-rlvr-rlcd/', '形成的训练记录'],
  ['/posts/ai/rlhf-rlvr-rlcd/', '在对应训练阶段中怎样使用'],
];
(async () => {
  fs.mkdirSync(out, {recursive: true});
  const b = await connect(Number(portText)), report = {tables: [], copies: [], exceptions: b.exceptions};
  try {
    for (const scripts of [true, false]) {
      await b.cdp('Emulation.setScriptExecutionDisabled', {value: !scripts});
      await b.viewport(320);
      for (const [route, header] of tables) for (const size of [100, 200]) {
        await b.navigate(base + route);
        await b.evaluate(`document.documentElement.style.fontSize='${size}%';document.fonts.ready`);
        await b.pause(250);
        const before = await b.evaluate(`(()=>{
          window.__table=[...document.querySelectorAll('.content table')].find(t=>
            [...t.querySelectorAll('th')].some(h=>h.textContent.trim()===${JSON.stringify(header)}));
          if(!__table)return null;__table.scrollIntoView({block:'center'});__table.scrollLeft=0;__table.focus();
          return {width:__table.clientWidth,scroll:__table.scrollWidth,focused:document.activeElement===__table,tabIndex:__table.tabIndex}})()`);
        assert(before && before.focused && before.tabIndex === 0 && before.scroll > before.width + 2, JSON.stringify({route, header, size, before}));
        await b.key('ArrowRight', 'ArrowRight', 39); await b.pause(220);
        const after = await b.evaluate('({distance:__table.scrollLeft,focused:document.activeElement===__table,pageScrollX:scrollX})');
        assert(after.focused && after.distance > 0 && after.pageScrollX === 0, JSON.stringify(after));
        report.tables.push({route, header, size, scripts, before, after});
      }
    }
    await b.cdp('Emulation.setScriptExecutionDisabled', {value: false});
    const source = fs.readFileSync(path.join(__dirname, '../../../content/posts/ai/supermimic/index.md'), 'utf8');
    const expected = [...source.matchAll(/```python\n([\s\S]*?)```/g)].map(m=>m[1]).find(s=>s.includes('import torch'));
    assert(expected);
    report.publishedSnippetSha256 = crypto.createHash('sha256').update(expected).digest('hex');
    // Hugo omits the one newline separating source code from the closing fence.
    // All visible code characters, including indentation/internal blank lines, must match.
    const renderedExpected = expected.replace(/\n$/, '');
    report.sourceFenceTrailingNewlineExcluded = true;
    for (const width of [320, 1440]) for (const size of [100, 200]) {
      await b.viewport(width); await b.navigate(base + '/posts/ai/supermimic/');
      await b.evaluate(`document.documentElement.style.fontSize='${size}%';
        document.getElementById('frozen-model-autograd').open=true;document.fonts.ready`);
      await b.pause(250);
      await b.evaluate(`(()=>{
        window.__copies=[];Object.defineProperty(navigator,'clipboard',{configurable:true,value:{
          writeText:text=>{__copies.push(text);return new Promise(resolve=>{window.__finishCopy=resolve})}}});
        window.__button=document.querySelector('#frozen-model-autograd .hextra-code-copy-btn');
        __button.scrollIntoView({block:'center'});__button.focus();})()`);
      await b.key('Enter', 'Enter', 13); await b.pause(100);
      const pending = await b.evaluate('({count:__copies.length,focused:document.activeElement===__button,disabled:__button.disabled,busy:__button.getAttribute("aria-busy")})');
      assert(pending.count === 1 && pending.focused && !pending.disabled && pending.busy === 'true', JSON.stringify(pending));
      await b.key('Enter', 'Enter', 13);
      assert.equal(await b.evaluate('__copies.length'), 1);
      const copied = await b.evaluate('__copies[0]'); assert.equal(copied, renderedExpected);
      assert.equal(copied, await b.evaluate('document.querySelector("#frozen-model-autograd .hextra-code-block code").textContent'));
      await b.key('Tab', 'Tab', 9);
      await b.evaluate('window.__laterFocus=document.activeElement;__finishCopy()'); await b.pause(100);
      const restored = await b.evaluate('({laterFocus:document.activeElement===__laterFocus,busy:__button.hasAttribute("aria-busy")})');
      assert(restored.laterFocus && !restored.busy, JSON.stringify(restored));
      report.copies.push({width, size, pending, exactRenderedText: true, duplicateRejected: true, restored});
    }
    assert.deepEqual(b.exceptions, []);
    console.log(JSON.stringify({arrowCases: report.tables.length, exactCopies: report.copies.length, exceptions: 0}));
  } finally {
    fs.writeFileSync(path.join(out, 'results.json'), JSON.stringify(report, null, 2) + '\n');
    await b.cdp('Emulation.setScriptExecutionDisabled', {value: false}).catch(()=>{});
    await b.close();
  }
})().catch(e=>{console.error(e);process.exitCode=1});
