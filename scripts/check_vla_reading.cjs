#!/usr/bin/env node
// Check initial chapter links after lazy images decode, with and without JS.
// Run serially against an existing loopback Chrome CDP endpoint.
// node scripts/check_vla_reading.cjs [baseURL] [CDP port] [result directory]
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const base = process.argv[2] || "http://127.0.0.1:1362";
const port = process.argv[3] || "9237";
const output = process.argv[4] || "/tmp/vla-reading-browser";
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function main() {
  fs.mkdirSync(output, {recursive: true});
  const endpoint = "http://127.0.0.1:" + port;
  const tab = await (await fetch(endpoint + "/json/new?about:blank", {method: "PUT"})).json();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  const pending = new Map(), exceptions = [], rows = [];
  let serial = 0;
  try {
    await new Promise((resolve, reject) => {
      ws.addEventListener("open", resolve, {once: true});
      ws.addEventListener("error", reject, {once: true});
    });
    ws.addEventListener("message", event => {
      const message = JSON.parse(event.data);
      if (message.id && pending.has(message.id)) {
        const task = pending.get(message.id);
        pending.delete(message.id); clearTimeout(task.timer);
        if (message.error) task.reject(Error(JSON.stringify(message.error)));
        else task.resolve(message.result);
      } else if (message.method === "Runtime.exceptionThrown") {
        exceptions.push(message.params.exceptionDetails);
      }
    });
    const cdp = (method, params = {}) => new Promise((resolve, reject) => {
      const id = ++serial;
      const timer = setTimeout(() => {
        pending.delete(id); reject(Error("CDP timeout: " + method));
      }, 30000);
      pending.set(id, {resolve, reject, timer});
      ws.send(JSON.stringify({id, method, params}));
    });
    const evaluate = async expression => {
      const result = await cdp("Runtime.evaluate", {expression, returnByValue: true, awaitPromise: true});
      assert(!result.exceptionDetails, JSON.stringify(result.exceptionDetails));
      return result.result.value;
    };
    const waitFor = async expression => {
      const deadline = Date.now() + 30000;
      while (Date.now() < deadline) {
        if (await evaluate(expression)) return;
        await delay(100);
      }
      throw Error("Condition timed out: " + expression);
    };
    await cdp("Page.enable"); await cdp("Runtime.enable"); await cdp("Page.bringToFront");
    const ids = ["one-decision", "flow", "flow-learning-lab", "clocks", "pi05", "pi07", "glossary", "checklist"];
    for (const scripts of [true, false]) {
      await cdp("Emulation.setScriptExecutionDisabled", {value: !scripts});
      for (const width of [320, 390, 1440]) {
        const height = width < 768 ? 844 : 1000;
        await cdp("Emulation.setDeviceMetricsOverride", {width, height, deviceScaleFactor: 1, mobile: width < 768});
        for (const id of ids) {
          // Force a fresh document so this does not merely test in-page scrolling.
          await cdp("Page.navigate", {url: "about:blank"});
          await waitFor("location.href === 'about:blank' && document.readyState === 'complete'");
          await cdp("Page.navigate", {url: base + "/posts/ai/vla-evolution/#" + id});
          await waitFor("location.pathname === '/posts/ai/vla-evolution/' && document.readyState === 'complete' && !!document.getElementById(" + JSON.stringify(id) + ")");
          await evaluate("document.fonts.ready");
          // Loading images above the target used to move it thousands of pixels.
          await evaluate("Promise.all([...document.querySelectorAll('.content img')].map(img => {img.loading='eager';return img.decode().catch(() => null)}))");
          await delay(200);
          const measured = await evaluate(`(() => {
            const heading = document.getElementById(${JSON.stringify(id)}).closest('h2,h3');
            const widgets = [...document.querySelectorAll('[data-vla-flow],[data-vla-clocks]')];
            return {
              top: heading.getBoundingClientRect().top,
              bottom: heading.getBoundingClientRect().bottom,
              hash: location.hash,
              widgetsReady: widgets.length === 2 && widgets.every(w => w.dataset.ready === 'true'),
              brokenImages: [...document.querySelectorAll('.content img')].filter(img => !img.complete || !img.naturalWidth).map(img => img.src),
              overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1
            };
          })()`);
          rows.push({id, width, scripts, ...measured});
          assert.equal(measured.hash, "#" + id);
          assert(measured.top >= 55 && measured.bottom <= height, JSON.stringify(rows.at(-1)));
          assert(!measured.overflow && !measured.brokenImages.length, JSON.stringify(rows.at(-1)));
          assert.equal(measured.widgetsReady, scripts, JSON.stringify(rows.at(-1)));
        }
      }
    }
    assert.equal(exceptions.length, 0, JSON.stringify(exceptions));
    console.log(JSON.stringify({cases: rows.length, exceptions: exceptions.length, result: path.join(output, "results.json")}));
  } finally {
    fs.writeFileSync(path.join(output, "results.json"), JSON.stringify({rows, exceptions}, null, 2));
    for (const task of pending.values()) clearTimeout(task.timer);
    ws.close();
    await fetch(endpoint + "/json/close/" + tab.id);
  }
}

main().catch(error => {console.error(error); process.exitCode = 1;});
