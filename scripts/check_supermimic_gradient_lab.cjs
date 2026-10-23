// Use a dedicated tab on an existing loopback CDP browser; run browser checks serially.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const base = process.argv[2] || 'http://127.0.0.1:13143';
const port = process.argv[3] || '9226';
const output = process.argv[4] || '/tmp/supermimic-gradient-lab';
const route = '/labs/supermimic/';
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const endpoint = 'http://127.0.0.1:' + port;
  const tab = await (await fetch(endpoint + '/json/new?about:blank', {method: 'PUT'})).json();
  const socket = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {socket.onopen = resolve; socket.onerror = reject;});
  const pending = new Map(), exceptions = [], requests = [], rows = [];
  let serial = 0;
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const request = pending.get(message.id);
      pending.delete(message.id); clearTimeout(request.timer);
      message.error ? request.reject(Error(JSON.stringify(message.error))) : request.resolve(message.result);
    } else if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
    else if (message.method === 'Network.requestWillBeSent') requests.push(message.params.request.url);
  };
  const cdp = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++serial;
    const timer = setTimeout(() => {pending.delete(id); reject(Error('CDP timeout: ' + method));}, 20000);
    pending.set(id, {resolve, reject, timer});
    socket.send(JSON.stringify({id, method, params}));
  });
  const evaluate = async expression => {
    const result = await cdp('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
    assert(!result.exceptionDetails, JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const navigate = async () => {
    await cdp('Page.navigate', {url: base + route});
    for (let i = 0; i < 100; i++) {
      if (await evaluate(`location.pathname === ${JSON.stringify(route)} && document.readyState === 'complete'`)) return;
      await pause(100);
    }
    throw Error('Lab navigation timed out');
  };
  const key = async (name, code, value) => {
    await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: name, code, windowsVirtualKeyCode: value,
      ...(name === 'Enter' ? {text: '\r', unmodifiedText: '\r'} : name === ' ' ? {text: ' ', unmodifiedText: ' '} : {})});
    await cdp('Input.dispatchKeyEvent', {type: 'keyUp', key: name, code, windowsVirtualKeyCode: value});
  };
  const screenshot = async name => {
    const shot = await cdp('Page.captureScreenshot', {format: 'png'});
    fs.writeFileSync(path.join(output, name + '.png'), Buffer.from(shot.data, 'base64'));
  };
  const number = id => evaluate(`Number(document.getElementById(${JSON.stringify(id)}).textContent)`);
  let download;
  try {
    await cdp('Page.enable'); await cdp('Runtime.enable'); await cdp('Network.enable'); await cdp('Page.bringToFront');
    await navigate();
    for (const width of [320, 390, 768, 1440]) {
      await cdp('Emulation.setDeviceMetricsOverride', {width, height: 1000, deviceScaleFactor: 1, mobile: width < 768});
      for (const theme of ['light', 'dark', 'warm']) {
        await evaluate(`localStorage.setItem('color-theme', ${JSON.stringify(theme)})`);
        await navigate();
        assert(await evaluate(`document.documentElement.dataset.theme === ${JSON.stringify(theme)}`));
        assert(await evaluate(`!document.getElementById('interactive').hidden`));
        for (const [preset, gradient, referenceAfter] of [['full', 8, -.488], ['detached', 5.6, -.61472], ['opposite', -8, -1.768]]) {
          await evaluate(`document.querySelector('[data-preset=${preset}]').click()`);
          assert.equal(await number('gradient-selected'), gradient);
          assert.equal(await number('reference-after'), referenceAfter);
          assert.equal(await number('model-state-1'), .2);
          assert.equal(await number('model-state-2'), .4);
          assert.equal(await number('state-error'), 0);
        }
        const state = await evaluate(`({
          width:innerWidth, pageWidth:document.documentElement.scrollWidth,
          badAnchors:[...document.querySelectorAll('a[href^="#"]')].filter(a=>!document.getElementById(a.hash.slice(1))).map(a=>a.hash),
          controls:[...document.querySelectorAll('button,input[type=number],.switch,.lab-nav a')].map(e=>({width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height})),
          metrics:[...document.querySelectorAll('.metrics dd')].every(e=>e.scrollWidth<=e.clientWidth+1),
          modelAfter:Number(document.getElementById('model-after').textContent)
        })`);
        assert(state.pageWidth <= width + 1 && !state.badAnchors.length && state.metrics, JSON.stringify(state));
        assert(state.controls.every(c => c.width > 0 && c.height >= 44), JSON.stringify(state.controls));
        assert.equal(state.modelAfter, -.488);
        await evaluate(`document.getElementById('results').scrollIntoView({block:'start'})`);
        await screenshot(`${width}-${theme}-opposite`);
        if (width === 320) {
          await evaluate('window.scrollTo(0,0)'); await screenshot(`${width}-${theme}-inputs`);
        }
        rows.push({width, theme, ...state});
      }
    }
    // Enter activates the preset; Space toggles only the graph switch.
    await evaluate(`document.querySelector('[data-preset=detached]').focus()`);
    await key('Enter', 'Enter', 13); await pause(100);
    assert.equal(await number('gradient-selected'), 5.6);
    await evaluate(`document.querySelector('[name=detach]').focus()`);
    await key(' ', 'Space', 32); await pause(100);
    assert.equal(await number('gradient-selected'), 8);
    assert.equal(await number('model-state-2'), .4);
    await evaluate(`document.querySelector('a[href="#results"]').focus()`);
    await key('Enter', 'Enter', 13); await pause(100);
    assert.equal(await evaluate('document.activeElement.id'), 'results');
    await evaluate(`document.querySelector('.lab-return').focus()`);
    await key('Enter', 'Enter', 13); await pause(100);
    assert.equal(await evaluate('document.activeElement.id'), 'gradient-form');

    // Invalid input hides the previous numeric result and downloadable payload.
    await evaluate(`(() => {const input=document.querySelector('[name=theta]');input.value='';input.dispatchEvent(new Event('input',{bubbles:true}));})()`);
    assert(await evaluate(`document.getElementById('result-details').hidden && document.getElementById('download-case').hidden && !document.getElementById('download-case').hasAttribute('href')`));
    assert(await evaluate(`document.getElementById('status').textContent.includes('请完整填写')`));
    await evaluate(`document.querySelector('[data-preset=opposite]').click()`);
    download = JSON.parse(decodeURIComponent((await evaluate(`document.getElementById('download-case').href`)).split(',')[1]));
    assert.equal(download.schema, 'supermimic-scalar-gradient-v1'); assert(download.illustrative);
    fs.writeFileSync(path.join(output, 'downloaded-case.json'), JSON.stringify({...download, result: {tampered: true}}, null, 2));
    const recomputed = JSON.parse(execFileSync(process.execPath,
      [path.resolve(__dirname, '../static/labs/supermimic/gradient_lab.js'), path.join(output, 'downloaded-case.json')], {encoding: 'utf8'}));
    assert.deepEqual(recomputed, download.result);

    // Saved storage may be unavailable; CSS still follows the system appearance.
    const blocker = await cdp('Page.addScriptToEvaluateOnNewDocument', {source: `Object.defineProperty(window,'localStorage',{get(){throw new DOMException('blocked','SecurityError')}});`});
    await cdp('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: 'dark'}]});
    await navigate();
    assert(await evaluate(`getComputedStyle(document.body).backgroundColor === 'rgb(18, 24, 32)' && !document.getElementById('result-details').hidden`));
    await cdp('Page.removeScriptToEvaluateOnNewDocument', {identifier: blocker.identifier});
    await cdp('Emulation.setEmulatedMedia', {features: []});

    await cdp('Emulation.setScriptExecutionDisabled', {value: true});
    for (const width of [320, 1440]) {
      await cdp('Emulation.setDeviceMetricsOverride', {width, height: 1000, deviceScaleFactor: 1, mobile: width < 768});
      await navigate();
      assert(await evaluate(`document.getElementById('interactive').hidden && document.body.textContent.includes('完整梯度为 8') && document.documentElement.scrollWidth<=innerWidth+1`));
      await screenshot(`${width}-no-javascript`);
    }
    await cdp('Emulation.setScriptExecutionDisabled', {value: false});
    const external = requests.filter(url => /^https?:/.test(url) && new URL(url).origin !== new URL(base).origin);
    assert.equal(external.length, 0, JSON.stringify(external));
    assert.equal(exceptions.length, 0, JSON.stringify(exceptions));
    const result = {rows, presets: 3, keyboardPresetAndSwitch: true, keyboardChapterLinks: true,
      invalidInputClearsResult: true, downloadRecomputed: true, blockedStorage: true,
      noJavascriptWidths: [320, 1440], externalRequests: external.length, exceptions};
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify({...result, rows: rows.length}));
  } finally {
    for (const p of pending.values()) clearTimeout(p.timer);
    socket.close();
    await fetch(endpoint + '/json/close/' + tab.id);
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
