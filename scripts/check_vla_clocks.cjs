#!/usr/bin/env node
// Run serially against an existing loopback Chrome CDP endpoint.
// Compare the article widget to its independent standard-library Python example.
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {calculate, presets} = require("../assets/js/vla-clocks.js");
const base = process.argv[2] || "http://127.0.0.1:1362";
const port = process.argv[3] || "9237";
const output = process.argv[4] || "/tmp/vla-clocks-browser";
const root = path.resolve(__dirname, "..");
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function main() {
  fs.mkdirSync(output, {recursive: true});
  const oracle = spawnSync("python3", ["-B", "-c", `
import importlib.util,json,sys
from pathlib import Path
p=Path(sys.argv[1])/'content/posts/ai/vla-evolution/vla_lab.py'
spec=importlib.util.spec_from_file_location('vla_lab',p)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
rows=[]
for h in (1,10,50,100):
 for e in sorted(set((1,max(1,h//2),h))):
  for f in (1,20,29.97,50,200):
   for n in (1,5,10,100):
    rows.append(dict(values=dict(horizon=h,execute=e,frequency=f,sampling=n),
                     expected=mod.action_clocks(h,f,e,n)))
print(json.dumps(rows))
`, root], {encoding: "utf8"});
  assert.equal(oracle.status, 0, oracle.stderr);
  const cases = JSON.parse(oracle.stdout);
  function compare(actual, expected) {
    for (const [key, value] of Object.entries({period: expected.action_period_s * 1000,
      coverage: expected.chunk_coverage_s, prefix: expected.prefix_duration_s,
      replans: expected.ideal_replans_per_s, lastTarget: expected.last_target_offset_s,
      sampling: expected.flow_steps_per_chunk})) {
      assert(Math.abs(actual[key] - value) < 1e-10, JSON.stringify({key, actual, expected}));
    }
  }
  for (const test of cases) compare(calculate(test.values), test.expected);
  for (const key of Object.keys(presets.default)) {
    for (const value of [undefined, NaN, Infinity, -1, 0, true, "5", 201]) {
      assert.throws(() => calculate({...presets.default, [key]: value}), RangeError);
    }
    if (key !== "frequency") assert.throws(() => calculate({...presets.default, [key]: 1.5}), RangeError);
  }
  assert.throws(() => calculate({...presets.default, execute: 51}), RangeError);
  const tab = await (await fetch("http://127.0.0.1:" + port + "/json/new?about:blank", {method: "PUT"})).json();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  const pending = new Map();
  const exceptions = [];
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
      } else if (message.method === "Runtime.exceptionThrown") exceptions.push(message.params.exceptionDetails);
    });
    const cdp = (method, params = {}) => new Promise((resolve, reject) => {
      const id = ++serial;
      const timer = setTimeout(() => {pending.delete(id);reject(Error("CDP timeout: " + method));}, 20000);
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
    await cdp("Page.navigate", {url: base + "/posts/ai/vla-evolution/"});
    await waitFor("document.readyState === 'complete' && !!document.querySelector('[data-vla-clocks]')?.dataset.ready");
    const actual = await evaluate(`(() => {
      const widget = document.querySelector('[data-vla-clocks]'), form = widget.querySelector('form');
      return ${JSON.stringify(cases.map(test => test.values))}.map(values => {
        for (const [key, value] of Object.entries(values)) form.elements.namedItem(key).value = value;
        form.dispatchEvent(new Event('input', {bubbles:true}));
        return {result: JSON.parse(widget.dataset.clocksResult),
          samplingCells: widget.querySelector('[data-clocks-strip=sampling]').children.length,
          actionCells: widget.querySelector('[data-clocks-strip=actions]').children.length,
          executedCells: widget.querySelectorAll('[data-execute]').length};
      });
    })()`);
    cases.forEach((test, index) => {
      compare(actual[index].result, test.expected);
      assert.equal(actual[index].samplingCells, test.values.sampling);
      assert.equal(actual[index].actionCells, test.values.horizon);
      assert.equal(actual[index].executedCells, test.values.execute);
    });
    const setValues = async values => evaluate(`(() => {
      const form = document.querySelector('[data-vla-clocks] form');
      for (const [key,value] of Object.entries(${JSON.stringify(values)})) form.elements.namedItem(key).value = value;
      form.dispatchEvent(new Event('input', {bubbles:true}));
    })()`);
    for (const invalid of [{horizon: ""}, {sampling: "0"}, {execute: "51"}, {frequency: "201"}, {horizon: "1.5"}]) {
      await setValues({...presets.default, ...invalid});
      assert(await evaluate(`(() => {const w=document.querySelector('[data-vla-clocks]');return !w.dataset.clocksResult &&
        !w.querySelector('[data-clocks-error]').hidden && w.querySelector('[data-clocks-diagrams]').hidden &&
        [...w.querySelectorAll('[data-clocks-value]')].every(x=>x.textContent==='—') && !!w.querySelector('[aria-invalid=true]');})()`));
    }
    for (const [name, values] of Object.entries(presets)) {
      await evaluate("document.querySelector('[data-clocks-preset=" + name + "]').click()");
      assert.deepEqual(await evaluate(`(() => {const f=document.querySelector('[data-vla-clocks] form');return Object.fromEntries([...f.elements].filter(x=>x.name).map(x=>[x.name,Number(x.value)]));})()`), values);
    }
    await evaluate("document.querySelector('[data-clocks-preset=default]').click()");
    await evaluate("document.querySelector('[data-vla-clocks] input[name=sampling]').focus()");
    await cdp("Input.dispatchKeyEvent", {type: "keyDown", key: "ArrowUp", code: "ArrowUp", windowsVirtualKeyCode: 38});
    await cdp("Input.dispatchKeyEvent", {type: "keyUp", key: "ArrowUp", code: "ArrowUp", windowsVirtualKeyCode: 38});
    assert.equal(await evaluate("JSON.parse(document.querySelector('[data-vla-clocks]').dataset.clocksResult).sampling"), 11);
    const layouts = [];
    for (const width of [320, 390, 768, 1440]) {
      await cdp("Emulation.setDeviceMetricsOverride", {width, height: 1100, deviceScaleFactor: 1, mobile: width < 768});
      for (const theme of ["light", "dark", "warm"]) {
        await evaluate("document.querySelector('[data-item=" + theme + "]').click()");
        await delay(100);
        await evaluate("document.querySelector('[data-vla-clocks]').scrollIntoView({block:'start'})");
        const measured = await evaluate(`(() => {const w=document.querySelector('[data-vla-clocks]');return {
          overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1 || w.scrollWidth>w.clientWidth+1,
          clipped:[...w.querySelectorAll('output,label,button')].some(x=>x.scrollWidth>x.clientWidth+1),
          unlabeled:[...w.querySelectorAll('input')].some(x=>!x.labels.length),
          liveRegion:!!w.querySelector('[aria-live=polite]')};})()`);
        assert(!measured.overflow && !measured.clipped && !measured.unlabeled && measured.liveRegion, JSON.stringify({width, theme, measured}));
        layouts.push({width, theme, ...measured});
        if (width === 390 || width === 1440) {
          const {data} = await cdp("Page.captureScreenshot", {format: "png"});
          fs.writeFileSync(path.join(output, `clocks-${width}-${theme}.png`), Buffer.from(data, "base64"));
        }
      }
    }
    await evaluate("document.querySelector('[data-item=light]').click()");
    await cdp("Emulation.setScriptExecutionDisabled", {value: true});
    await cdp("Page.reload");
    await waitFor("document.readyState==='complete' && !!document.querySelector('[data-clocks-fallback]')");
    assert(await evaluate("!document.querySelector('[data-clocks-fallback]').hidden && document.querySelector('[data-clocks-interactive]').hidden"));
    await cdp("Emulation.setScriptExecutionDisabled", {value: false});
    assert.deepEqual(exceptions, []);
    const report = {arithmeticCases: cases.length, browserCases: cases.length, invalidInputs: 5,
      presets: 3, keyboard: "passed", noJavaScript: "static explanation retained", layouts};
    fs.writeFileSync(path.join(output, "results.json"), JSON.stringify(report, null, 2) + "\n");
    console.log(JSON.stringify(report));
  } finally {
    for (const task of pending.values()) clearTimeout(task.timer);
    ws.close();
    await fetch("http://127.0.0.1:" + port + "/json/close/" + tab.id).catch(() => {});
  }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
