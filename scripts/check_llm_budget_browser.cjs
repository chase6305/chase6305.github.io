#!/usr/bin/env node
// Run after starting Hugo and a loopback Chrome CDP endpoint.
// Usage: node scripts/check_llm_budget_browser.cjs [baseURL] [CDP port] [screenshots]
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const base = process.argv[2] || 'http://127.0.0.1:13139';
const port = process.argv[3] || '9237';
const output = process.argv[4] || '/tmp/chase-llm-budget';
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
let socket;
let closePage;

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const tab = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {method: 'PUT'})).json();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  socket = ws;
  await new Promise((resolve, reject) => {ws.onopen = resolve; ws.onerror = reject;});
  let serial = 0;
  const pending = new Map();
  const exceptions = [];
  ws.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const request = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) request.reject(message.error);
      else request.resolve(message.result);
    } else if (message.method === 'Runtime.exceptionThrown') {
      exceptions.push(message.params.exceptionDetails);
    }
  };
  const cdp = (method, params = {}) => new Promise((resolve, reject) => {
    pending.set(++serial, {resolve, reject});
    ws.send(JSON.stringify({id: serial, method, params}));
  });
  closePage = () => cdp('Page.close');
  const evaluate = async expression => {
    const result = await cdp('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
    if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  await cdp('Page.enable');
  await cdp('Page.bringToFront');
  await cdp('Runtime.enable');
  await cdp('Network.enable');
  await cdp('Network.setCacheDisabled', {cacheDisabled: true});
  await cdp('Page.navigate', {url: base + '/posts/ai/llm-training-metrics/'});
  for (let attempt = 0; attempt < 100; attempt++) {
    await sleep(100);
    if (await evaluate("document.querySelector('[data-llm-budget]')?.dataset.ready === 'true'")) break;
    if (attempt === 99) throw Error('Calculator initialization timed out');
  }
  const values = () => evaluate("Object.fromEntries([...document.querySelectorAll('[data-value]')].map(x => [x.dataset.value, x.textContent]))");
  const set = (name, value) => evaluate(`(() => {
    const input = document.querySelector('.llm-budget [name=' + ${JSON.stringify(name)} + ']');
    input.value = ${JSON.stringify(value)};
    input.dispatchEvent(new Event('input', {bubbles: true}));
  })()`);
  assert.deepEqual(await values(), {weights: '13.04 GiB', kv: '0.50 GiB', total: '13.54 GiB'});
  await set('weightBits', '4');
  assert.equal((await values()).weights, '3.26 GiB');
  assert.equal((await values()).kv, '0.50 GiB');
  await set('batch', '8');
  assert.equal((await values()).kv, '4.00 GiB');
  await evaluate("document.querySelector('[data-preset=qwen]').click()");
  assert.deepEqual(await values(), {weights: '14.19 GiB', kv: '0.22 GiB', total: '14.40 GiB'});
  for (const value of ['', '0', '-1', '1.5', '10000001']) {
    await set('tokens', value);
    assert.equal((await values()).weights, '—');
    assert(await evaluate("!document.querySelector('.llm-budget__error').hidden"));
    assert(await evaluate("document.querySelector('[name=tokens]').getAttribute('aria-invalid') === 'true'"));
  }
  await set('tokens', '4096');
  assert.equal((await values()).kv, '0.22 GiB');
  await evaluate("document.querySelector('.llm-budget__advanced').open = true");
  assert(await evaluate("document.querySelector('[name=layers]').getBoundingClientRect().height > 0"));
  await set('layers', '0');
  assert.equal((await values()).weights, '—');
  await set('layers', '28');
  await evaluate("document.querySelector('.llm-budget__advanced').open = false");
  let layouts = 0;
  for (const [width, height] of [[320, 844], [390, 844], [768, 1000], [1440, 1000]]) {
    for (const theme of ['light', 'dark', 'warm']) {
      await cdp('Emulation.setDeviceMetricsOverride', {width, height, deviceScaleFactor: 1, mobile: width < 768});
      await evaluate(`document.documentElement.classList.remove('dark', 'warm');
        document.documentElement.classList.toggle(${JSON.stringify(theme)}, ${theme !== 'light'});
        document.querySelector('.llm-budget').scrollIntoView({block: 'start'});`);
      await sleep(100);
      const layout = await evaluate(`({
        overflow: document.documentElement.scrollWidth > innerWidth + 1,
        inputs: [...document.querySelectorAll('.llm-budget input, .llm-budget select')].map(x => ({
          label: !!document.querySelector('label[for=' + x.id + ']'),
          left: x.getBoundingClientRect().left, right: x.getBoundingClientRect().right
        })), mathErrors: document.querySelectorAll('.katex-error').length
      })`);
      assert(!layout.overflow);
      assert.equal(layout.mathErrors, 0);
      for (const input of layout.inputs) {
        assert(input.label);
        assert(input.left >= 0 && input.right <= width);
      }
      const {data} = await cdp('Page.captureScreenshot', {format: 'png'});
      fs.writeFileSync(path.join(output, `calculator-${width}-${theme}.png`), Buffer.from(data, 'base64'));
      layouts++;
    }
  }
  await evaluate("document.querySelector('[data-preset=example]').focus()");
  await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13, text: '\r'});
  await cdp('Input.dispatchKeyEvent', {type: 'keyUp', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13});
  assert.equal((await values()).weights, '13.04 GiB');
  await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9});
  await cdp('Input.dispatchKeyEvent', {type: 'keyUp', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9});
  assert.equal(await evaluate('document.activeElement.dataset.preset'), 'qwen');
  assert.deepEqual(exceptions, []);
  // Progressive enhancement: a disabled script must not leave stale results visible.
  await cdp('Emulation.setScriptExecutionDisabled', {value: true});
  await cdp('Page.reload', {ignoreCache: true});
  for (let attempt = 0; attempt < 150; attempt++) {
    await sleep(100);
    if (await evaluate("document.readyState === 'complete' && !!document.querySelector('.llm-budget noscript')")) break;
    if (attempt === 149) throw Error('No-JavaScript page reload timed out');
  }
  assert(await evaluate("document.querySelector('.llm-budget__interactive').hidden && document.querySelector('.llm-budget noscript').textContent.includes('JavaScript')"));
  await cdp('Emulation.setScriptExecutionDisabled', {value: false});
  await closePage();
  closePage = null;
  ws.close();
  console.log(JSON.stringify({layouts, keyboard: 'passed', invalidInputs: 'passed', noJavaScript: 'passed', exceptions}, null, 2));
})().catch(async error => {
  console.error(error);
  if (closePage) {try {await closePage();} catch (_) {}}
  if (socket) socket.close();
  process.exitCode = 1;
});
