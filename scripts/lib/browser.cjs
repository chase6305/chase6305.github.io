// Small CDP harness for repository-owned browser checks. Use a dedicated tab,
// and run suites serially because keyboard focus is shared by the browser.
const fs = require('node:fs');
const pause = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

async function connect(port = 9226) {
  const endpoint = `http://127.0.0.1:${port}`;
  const tab = await (await fetch(`${endpoint}/json/new?about:blank`, {method: 'PUT'})).json();
  const socket = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onerror = reject;
  });
  const pending = new Map(), exceptions = [], requests = [];
  let serial = 0, closed = false;
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const request = pending.get(message.id);
      pending.delete(message.id);
      clearTimeout(request.timer);
      message.error ? request.reject(Error(request.method + ': ' + JSON.stringify(message.error))) : request.resolve(message.result);
    } else if (message.method === 'Runtime.exceptionThrown') {
      exceptions.push(message.params.exceptionDetails);
    } else if (message.method === 'Network.requestWillBeSent') {
      requests.push(message.params.request.url);
    }
  };
  const cdp = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++serial;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(Error('CDP timeout: ' + method));
    }, 30000);
    pending.set(id, {resolve, reject, timer, method});
    socket.send(JSON.stringify({id, method, params}));
  });
  const evaluate = async expression => {
    const result = await cdp('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
    if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const navigate = async url => {
    await cdp('Page.navigate', {url});
    const deadline = Date.now() + 30000;
    while (Date.now() < deadline) {
      try {
        if (await evaluate(`document.readyState === 'complete' && location.href === ${JSON.stringify(url)}`)) return;
      } catch (error) {
        if (!/context|attached|navigated or closed/i.test(error.message)) throw error;
      }
      await pause(100);
    }
    throw Error('Navigation timed out: ' + url);
  };
  const viewport = (width, height = 1000) => cdp('Emulation.setDeviceMetricsOverride', {
    width, height, deviceScaleFactor: 1, mobile: width < 768,
  });
  const screenshot = async file => {
    const shot = await cdp('Page.captureScreenshot', {format: 'png'});
    fs.writeFileSync(file, Buffer.from(shot.data, 'base64'));
  };
  const key = async (name, code, value) => {
    await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: name, code, windowsVirtualKeyCode: value,
      ...(name === 'Enter' ? {text: '\r', unmodifiedText: '\r'} : {})});
    await cdp('Input.dispatchKeyEvent', {type: 'keyUp', key: name, code, windowsVirtualKeyCode: value});
  };
  const close = async () => {
    if (closed) return;
    closed = true;
    for (const request of pending.values()) clearTimeout(request.timer);
    socket.close();
    // A browser exit must not replace an earlier assertion/navigation failure.
    await fetch(`${endpoint}/json/close/${tab.id}`).catch(() => {});
  };
  await cdp('Page.enable');
  await cdp('Runtime.enable');
  await cdp('Network.enable');
  await cdp('Page.bringToFront');
  return {cdp, evaluate, navigate, viewport, screenshot, key, close, exceptions, requests, pause};
}

module.exports = {connect, pause};
