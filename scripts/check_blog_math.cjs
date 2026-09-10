#!/usr/bin/env node
// Read-only browser checks. Start Hugo -D and Chrome's loopback CDP endpoint first.
// Run UI suites serially on a shared browser; keyboard focus is browser-wide.
// node scripts/check_blog_browser.cjs [baseURL] [CDP port] [screenshot directory]
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const base = process.argv[2] || "http://127.0.0.1:13139";
const port = process.argv[3] || "9229";
const output = process.argv[4] || "/tmp/chase-blog-browser";
const root = path.resolve(__dirname, "..");
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
let browserSocket;

(async () => {
  fs.mkdirSync(output, {recursive: true});
  // Own the tab: never navigate a user's or another check's existing page.
  const tab = await (await fetch("http://127.0.0.1:" + port + "/json/new?about:blank",
    {method: "PUT"})).json();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  browserSocket = ws;
  await new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve, {once: true});
    ws.addEventListener("error", reject, {once: true});
  });
  let serial = 0;
  const pending = new Map();
  const exceptions = [];
  ws.onmessage = event => {
    const response = JSON.parse(event.data);
    if (response.id) {
      const request = pending.get(response.id);
      pending.delete(response.id);
      if (response.error) request.reject(response.error);
      else request.resolve(response.result);
    } else if (response.method === "Runtime.exceptionThrown") {
      exceptions.push(response.params.exceptionDetails);
    }
  };
  const cdp = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++serial;
    pending.set(id, {resolve, reject});
    ws.send(JSON.stringify({id, method, params}));
  });
  const evaluate = async expression => {
    const response = await cdp("Runtime.evaluate",
      {expression, returnByValue: true, awaitPromise: true});
    if (response.exceptionDetails) throw Error(JSON.stringify(response.exceptionDetails));
    return response.result.value;
  };
  const viewport = async (width, height) => {
    await cdp("Emulation.setDeviceMetricsOverride",
      {width, height, deviceScaleFactor: 1, mobile: width < 768});
  };
  const navigate = async route => {
    await cdp("Page.navigate", {url: base + route});
    for (let tries = 0; tries < 100; tries++) {
      await sleep(100);
      if (await evaluate("location.pathname === " + JSON.stringify(route) +
        " && document.readyState === 'complete'")) break;
      if (tries === 99) throw Error("Navigation timeout: " + route);
    }
    await evaluate("Promise.all(Array.from(document.querySelectorAll('.content img')).map(img => { img.loading='eager'; return img.decode().catch(() => null); }))");
    await sleep(120);
    // KaTeX HTML/CSS version drift can break sizing without a parse error.
    const mathStyles = await evaluate("(() => { const base = document.querySelector('.katex-html > .base'); const sub = document.querySelector('.katex .sizing.reset-size6.size3'); return {base: !base || getComputedStyle(base).display === 'inline-block', sub: !sub || parseFloat(getComputedStyle(sub).fontSize) < 0.8 * parseFloat(getComputedStyle(sub.closest('.katex')).fontSize)}; })()");
    assert(mathStyles.base && mathStyles.sub, 'KaTeX stylesheet does not match rendered HTML: ' + JSON.stringify(mathStyles));
  };
  const screenshot = async name => {
    const {data} = await cdp("Page.captureScreenshot", {format: "png"});
    fs.writeFileSync(path.join(output, name + ".png"), Buffer.from(data, "base64"));
  };
  await cdp("Page.enable");
  await cdp("Page.navigate", {url: "about:blank"});
  await sleep(200);
  await cdp("Runtime.enable");
  await viewport(390, 844);
  const checks = [];
  await navigate('/posts/thesis/elmp/');
  for (const theme of ['light', 'dark']) {
    await viewport(390, 844);
    await evaluate(`document.querySelector('[data-item=${theme}]').click()`);
    await sleep(200);
    const chosen = await evaluate(`(() => {
      const formula = [...document.querySelectorAll('.katex-display')].find(el => el.scrollWidth > el.clientWidth + 20 && el.scrollWidth < 650);
      if (!formula) return false;
      window.__formula = formula;
      formula.scrollIntoView({block:'center'});
      formula.focus({preventScroll:true});
      formula.scrollLeft = 0;
      return document.activeElement === formula && formula.tabIndex === 0;
    })()`);
    assert(chosen, 'An overflowing equation must be keyboard focusable');
    await cdp('Input.dispatchKeyEvent', {type:'keyDown', key:'ArrowRight', code:'ArrowRight', windowsVirtualKeyCode:39});
    await cdp('Input.dispatchKeyEvent', {type:'keyUp', key:'ArrowRight', code:'ArrowRight', windowsVirtualKeyCode:39});
    await sleep(250);
    assert(await evaluate('window.__formula.scrollLeft > 0'), 'ArrowRight must scroll the actual formula');
    await screenshot('formula-390-' + theme);
    await viewport(1440, 1000);
    await sleep(200);
    assert(await evaluate(`(() => {
      const f=window.__formula;
      return f.scrollWidth <= f.clientWidth + 1 && !f.hasAttribute('tabindex') && !f.hasAttribute('aria-describedby') && f.previousElementSibling.hidden;
    })()`), 'Fitting equation must lose its tab stop and visible hint');
    await screenshot('formula-1440-' + theme);
    checks.push({theme, keyboardScroll:true, responsiveHint:true});
  }
  // Reparent a real equation into a closed details block, then open it.
  await viewport(390, 844);
  await evaluate(`(() => {
    const f=window.__formula;
    const d=document.createElement('details');
    const summary=document.createElement('summary'); summary.textContent='Test collapsed equation';
    d.append(summary); const hint=f.previousElementSibling; hint.before(d); d.append(hint, f); window.__details=d;
  })()`);
  await sleep(200);
  await evaluate('window.__details.open=true');
  await sleep(200);
  assert(await evaluate('window.__formula.tabIndex === 0 && !document.getElementById(window.__formula.getAttribute("aria-describedby")).hidden'));
  assert.equal(exceptions.length, 0);
  fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify({checks, collapsedEquation:true, exceptions}, null, 2));
  console.log(JSON.stringify({checks, collapsedEquation:true, exceptions}, null, 2));
  await cdp('Page.close'); ws.close();
})().catch(error => { console.error(error); browserSocket?.close(); process.exitCode=1; });
