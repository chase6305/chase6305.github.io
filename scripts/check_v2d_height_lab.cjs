#!/usr/bin/env node
// Run serially with other focus-dependent checks against loopback Chrome CDP.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const base = process.argv[2] || 'http://127.0.0.1:1395';
const port = process.argv[3] || '9258';
const output = process.argv[4] || '/tmp/v2d-height-lab-browser';
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const luminance = color => {
  const values = color.match(/[\d.]+/g).slice(0, 3).map(Number).map(n => n / 255)
    .map(n => n <= .04045 ? n / 12.92 : ((n + .055) / 1.055) ** 2.4);
  return values.reduce((sum, n, i) => sum + n * [.2126, .7152, .0722][i], 0);
};
const contrast = (one, two) => {
  const a = luminance(one), b = luminance(two);
  return (Math.max(a, b) + .05) / (Math.min(a, b) + .05);
};

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const endpoint = 'http://127.0.0.1:' + port;
  const tab = await (await fetch(endpoint + '/json/new?about:blank', {method: 'PUT'})).json();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  const pending = new Map(), exceptions = [], rows = [];
  let serial = 0;
  let restorePreference;
  try {
    await new Promise((resolve, reject) => {ws.onopen = resolve; ws.onerror = reject;});
    ws.onmessage = event => {
      const message = JSON.parse(event.data);
      if (message.id && pending.has(message.id)) {
        const task = pending.get(message.id);
        pending.delete(message.id); clearTimeout(task.timer);
        message.error ? task.reject(Error(JSON.stringify(message.error))) : task.resolve(message.result);
      } else if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
    };
    const cdp = (method, params = {}) => new Promise((resolve, reject) => {
      const id = ++serial;
      const timer = setTimeout(() => {pending.delete(id); reject(Error('CDP timeout: ' + method));}, 30000);
      pending.set(id, {resolve, reject, timer});
      ws.send(JSON.stringify({id, method, params}));
    });
    const evaluate = async expression => {
      const result = await cdp('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
      assert(!result.exceptionDetails, JSON.stringify(result.exceptionDetails));
      return result.result.value;
    };
    const waitFor = async expression => {
      const deadline = Date.now() + 15000;
      while (Date.now() < deadline) {if (await evaluate(expression)) return; await delay(100);}
      throw Error('Condition timed out: ' + expression);
    };
    const screenshot = async name => {
      const {data} = await cdp('Page.captureScreenshot', {format: 'png'});
      fs.writeFileSync(output + '/' + name + '.png', Buffer.from(data, 'base64'));
    };
    await cdp('Page.enable'); await cdp('Runtime.enable'); await cdp('Page.bringToFront');
    await cdp('Page.navigate', {url: base + '/labs/video-to-data/'});
    await waitFor("document.readyState==='complete' && document.querySelector('#status')?.dataset.state==='pass'");
    const savedPreference = await evaluate("localStorage.getItem('color-theme')");
    restorePreference = async () => {
      await evaluate(savedPreference === null ? "localStorage.removeItem('color-theme')"
        : "localStorage.setItem('color-theme'," + JSON.stringify(savedPreference) + ")");
    };
    for (const width of [320, 768, 1440]) {
      await cdp('Emulation.setDeviceMetricsOverride', {width, height: 1000, deviceScaleFactor: 1, mobile: width < 768});
      for (const theme of ['light', 'dark', 'warm']) {
        await evaluate("localStorage.setItem('color-theme'," + JSON.stringify(theme) + ")");
        await cdp('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: theme === 'dark' ? 'light' : 'dark'}]});
        await cdp('Page.navigate', {url: base + '/labs/video-to-data/'});
        await waitFor("document.readyState==='complete' && document.querySelector('#status')?.dataset.state==='pass'");
        for (const [name, expected] of [['stable', 'pass'], ['aloft', 'fail'], ['drop', 'fail']]) {
          await evaluate(`document.querySelector('[data-preset="${name}"]').click()`);
          assert.equal(await evaluate("document.querySelector('#status').dataset.state"), expected, name);
          assert.equal(await evaluate("document.querySelectorAll('#checks li').length"), 6);
          const payload = await evaluate("JSON.parse(decodeURIComponent(document.querySelector('#download-trace').href.split(',').slice(1).join(',')))");
          assert.equal(payload.fps, 20); assert.equal(payload.heights_m.length, 95);
          assert.equal(payload.illustrative, true);
          if (name === 'drop') {
            assert.match(await evaluate("document.querySelector('#checks').children[2].textContent"), /^失败/);
            assert.match(await evaluate("document.querySelector('#checks').children[4].textContent"), /^通过/);
          }
          await evaluate("document.querySelector('.result').scrollIntoView({block:'start'})");
          await screenshot(`${name}-${width}-${theme}`);
        }
        await evaluate("document.querySelector('[data-preset=stable]').click();{const e=document.querySelector('[name=jitter]');e.value='.003';e.dispatchEvent(new Event('input',{bubbles:true}));}");
        assert.equal(await evaluate("document.querySelector('#status').dataset.state"), 'fail');
        assert.match(await evaluate("document.querySelector('#checks').children[4].textContent"), /0.006 m/);
        await evaluate("document.querySelector('[data-preset=stable]').click();{const e=document.querySelector('[name=descentSteps]');e.value='11';e.dispatchEvent(new Event('input',{bubbles:true}));}");
        assert.equal(await evaluate("document.querySelector('#status').dataset.state"), 'fail');
        assert.equal(await evaluate("document.querySelector('#precision-warning').hidden"), false);
        assert.match(await evaluate("document.querySelector('#precision-warning').textContent"), /0\.0100000000000000\d+/);
        await evaluate("{const e=document.querySelector('[name=descentSteps]');e.value='1.5';e.dispatchEvent(new Event('input',{bubbles:true}));}");
        assert.equal(await evaluate("document.querySelector('#status').dataset.state"), 'invalid');
        assert.equal(await evaluate("document.querySelector('#details').hidden"), true);
        await evaluate("document.querySelector('[data-preset=stable]').click();{const e=document.querySelector('[name=finalHeight]');e.value='.2';e.dispatchEvent(new Event('input',{bubbles:true}));}");
        assert(await evaluate("[...document.querySelectorAll('svg text')].some(t=>t.textContent==='0.2')"));
        await evaluate("{for(const [name,value] of [['finalHeight','-.05'],['jitter','.03']])document.querySelector('[name='+name+']').value=value;document.querySelector('[name=jitter]').dispatchEvent(new Event('input',{bubbles:true}));}");
        assert(await evaluate("[...document.querySelectorAll('svg text')].some(t=>t.textContent==='-0.08')"));
        assert.equal(await evaluate("document.documentElement.scrollWidth>innerWidth+1"), false);
        await evaluate("document.querySelector('[data-preset=stable]').click();document.querySelector('[name=jitter]').focus()");
        await cdp('Input.dispatchKeyEvent', {type:'keyDown', key:'ArrowUp', code:'ArrowUp', windowsVirtualKeyCode:38});
        await cdp('Input.dispatchKeyEvent', {type:'keyUp', key:'ArrowUp', code:'ArrowUp', windowsVirtualKeyCode:38});
        assert.equal(await evaluate("document.querySelector('[name=jitter]').valueAsNumber"), .0005);
        assert.equal(await evaluate("document.querySelector('#status').dataset.state"), 'pass');
        const state = await evaluate(`({overflow:document.documentElement.scrollWidth>innerWidth+1,
          unlabeled:[...document.querySelectorAll('input')].filter(e=>!e.labels.length).length,
          scheme:getComputedStyle(document.body).backgroundColor,
          backLink:document.querySelector('main>a').pathname,
          focus:document.activeElement.name,
          chartWidth:document.querySelector('svg').getBoundingClientRect().width,
          checks:document.querySelectorAll('#checks li').length})`);
        assert(!state.overflow && !state.unlabeled, JSON.stringify(state));
        assert.equal(state.backLink, '/posts/ai/video-to-data/');
        assert.equal(state.focus, 'jitter');
        assert.equal(state.scheme, {light:'rgb(255, 255, 255)', dark:'rgb(18, 24, 32)', warm:'rgb(253, 250, 244)'}[theme]);
        const colors = await evaluate(`(() => {
          const styles = selector => getComputedStyle(document.querySelector(selector));
          return {background:styles('body').backgroundColor,
            inputBorder:styles('input').borderColor,
            buttonBorder:styles('button').borderColor,
            buttonBackground:styles('button').backgroundColor,
            chartBackground:styles('.result').backgroundColor,
            trace:styles('.height-trace').stroke};
        })()`);
        assert(contrast(colors.inputBorder, colors.background) >= 3, JSON.stringify(colors));
        assert(contrast(colors.buttonBorder, colors.buttonBackground) >= 3, JSON.stringify(colors));
        assert(contrast(colors.trace, colors.chartBackground) >= 3, JSON.stringify(colors));
        // The SVG title/description are replaced after input. Allow Chrome's
        // accessibility tree to incorporate the current rendered chart.
        await delay(150);
        const {nodes} = await cdp('Accessibility.getFullAXTree');
        assert(nodes.some(node => node.role?.value === 'image' && node.name?.value === '构造的 20 Hz 相对高度曲线'), 'Missing chart accessible name');
        await screenshot(`controls-${width}-${theme}`);
        rows.push({width, theme, ...state});
      }
    }
    const {identifier} = await cdp('Page.addScriptToEvaluateOnNewDocument', {
      source: "Object.defineProperty(window,'localStorage',{configurable:true,get(){throw new DOMException('Blocked','SecurityError')}})"
    });
    try {
      await cdp('Page.navigate', {url: base + '/labs/video-to-data/'});
      await waitFor("document.readyState==='complete' && document.querySelector('#status')?.dataset.state==='pass'");
      assert.equal(await evaluate("getComputedStyle(document.body).backgroundColor"), 'rgb(18, 24, 32)');
    } finally {
      await cdp('Page.removeScriptToEvaluateOnNewDocument', {identifier});
    }
    await cdp('Emulation.setScriptExecutionDisabled', {value: true});
    await cdp('Page.navigate', {url: base + '/labs/video-to-data/'});
    await waitFor("document.readyState==='complete' && !!document.querySelector('noscript p')");
    assert(await evaluate("document.querySelector('noscript p').getBoundingClientRect().height>0"));
    assert.equal(exceptions.length, 0, JSON.stringify(exceptions));
    fs.writeFileSync(output + '/results.json', JSON.stringify({rows, exceptions, nojs: true, blockedStorage: true}, null, 2));
    console.log(JSON.stringify({passed: true, cases: rows.length, nojs: true, blockedStorage: true, output}));
  } finally {
    if (restorePreference) await restorePreference();
    for (const task of pending.values()) clearTimeout(task.timer);
    ws.close();
    await fetch(endpoint + '/json/close/' + tab.id);
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
