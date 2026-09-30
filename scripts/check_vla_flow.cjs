#!/usr/bin/env node
// Compare the browser's analytic flow with the independently implemented Python example.
// Run browser suites serially; only this script's CDP tab is touched.
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {velocity, trajectory, snapshot} = require("../assets/js/vla-flow.js");
const root = path.resolve(__dirname, "..");
const base = process.argv[2] || "http://127.0.0.1:1362";
const port = process.argv[3] || "9237";
const output = process.argv[4] || "/tmp/vla-flow-browser";
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
for i in range(-40,41):
 noise=i/20;points=mod.bimodal_flow(noise)
 for progress in (0,1,10,25,50,75,90,99,100):
  j=progress*10;s,x=points[j]
  rows.append(dict(noise=noise,progress=progress,index=j,s=s,x=x,
    velocity=mod.bimodal_velocity(x,s),end=points[-1][1],
    next=points[j+1][1] if j<1000 else None))
print(json.dumps(rows))
`, root], {encoding: "utf8", maxBuffer: 2 * 1024 * 1024});
  assert.equal(oracle.status, 0, oracle.stderr);
  const cases = JSON.parse(oracle.stdout);
  function compare(actual, expected) {
    for (const key of ["index", "s", "x", "velocity", "end", "next"]) {
      if (expected[key] === null) assert.equal(actual[key], null);
      else assert(Math.abs(actual[key] - expected[key]) < 2e-11, JSON.stringify({key, actual, expected}));
    }
  }
  for (const test of cases) compare(snapshot(trajectory(test.noise), test.progress), test);
  for (const invalid of [NaN, Infinity, "0.2", true, -2.01, 2.01]) assert.throws(() => trajectory(invalid), RangeError);
  for (const invalid of [-1, 101, .5, NaN, "50"]) assert.throws(() => snapshot(trajectory(.2), invalid), RangeError);
  for (const invalid of [0, -1, 1.01, Infinity]) assert.throws(() => velocity(.2, invalid), RangeError);
  assert(trajectory(0).every(point => point.x === 0));
  assert(Math.abs(velocity(.2, .5) + .35989792451045) < 1e-12);
  const tab = await (await fetch("http://127.0.0.1:" + port + "/json/new?about:blank", {method: "PUT"})).json();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  const pending = new Map(), exceptions = [];
  let serial = 0;
  try {
    await new Promise((resolve, reject) => {
      ws.addEventListener("open", resolve, {once: true});
      ws.addEventListener("error", reject, {once: true});
    });
    ws.addEventListener("message", event => {
      const message = JSON.parse(event.data);
      if (message.id && pending.has(message.id)) {
        const task = pending.get(message.id); pending.delete(message.id); clearTimeout(task.timer);
        if (message.error) task.reject(Error(JSON.stringify(message.error)));
        else task.resolve(message.result);
      } else if (message.method === "Runtime.exceptionThrown") exceptions.push(message.params.exceptionDetails);
    });
    const cdp = (method, params = {}) => new Promise((resolve, reject) => {
      const id = ++serial;
      const timer = setTimeout(() => {pending.delete(id);reject(Error("CDP timeout: " + method));}, 30000);
      pending.set(id, {resolve, reject, timer}); ws.send(JSON.stringify({id, method, params}));
    });
    const evaluate = async expression => {
      const result = await cdp("Runtime.evaluate", {expression, returnByValue: true, awaitPromise: true});
      assert(!result.exceptionDetails, JSON.stringify(result.exceptionDetails)); return result.result.value;
    };
    const waitFor = async expression => {
      const deadline = Date.now() + 30000;
      while (Date.now() < deadline) {if (await evaluate(expression)) return; await delay(100);}
      throw Error("Condition timed out: " + expression);
    };
    await cdp("Page.enable"); await cdp("Runtime.enable"); await cdp("Page.bringToFront");
    await cdp("Page.navigate", {url: base + "/posts/ai/vla-evolution/"});
    await waitFor("document.readyState === 'complete' && !!document.querySelector('[data-vla-flow]')?.dataset.ready");
    const actual = await evaluate(`(() => {
      const root=document.querySelector('[data-vla-flow]');
      const noise=root.querySelector('[name=flow-noise]'), progress=root.querySelector('[name=flow-progress]');
      return ${JSON.stringify(cases.map(({noise, progress}) => ({noise, progress})))}.map(test => {
        noise.value=test.noise;progress.value=test.progress;noise.dispatchEvent(new Event('input',{bubbles:true}));
        const box=root.querySelector('[data-flow-path=full]').getBBox();
        return {result:JSON.parse(root.dataset.flowResult),
          curveInside:box.x>=45.99&&box.x+box.width<=378.01&&box.y>=21.99&&box.y+box.height<=190.01,
          current:[...root.querySelectorAll('[data-flow-current]')].map(x=>[Number(x.getAttribute('cx')),Number(x.getAttribute('cy'))])[0]};
      });
    })()`);
    cases.forEach((test, i) => {
      compare(actual[i].result, test); assert(actual[i].curveInside);
      assert(Math.abs(actual[i].current[0] - (46 + test.index / 1000 * 332)) < 1e-10);
      assert(Math.abs(actual[i].current[1] - (106 - test.x / 2.2 * 84)) < 1e-9);
    });
    for (const initial of [-.2, .2, 0]) {
      await evaluate(`document.querySelector('[data-flow-preset="${initial}"]').click()`);
      compare(await evaluate("JSON.parse(document.querySelector('[data-vla-flow]').dataset.flowResult)"), cases.find(test => test.noise === initial && test.progress === 100));
    }
    assert(await evaluate("document.querySelector('[data-flow-note]').textContent.includes('概率为零')"));
    await evaluate("document.querySelector('[data-flow-preset=\"0.2\"]').click();document.querySelector('[name=flow-noise]').focus()");
    await cdp("Input.dispatchKeyEvent", {type: "keyDown", key: "ArrowRight", code: "ArrowRight", windowsVirtualKeyCode: 39});
    await cdp("Input.dispatchKeyEvent", {type: "keyUp", key: "ArrowRight", code: "ArrowRight", windowsVirtualKeyCode: 39});
    assert.equal(await evaluate("Number(document.querySelector('[name=flow-noise]').value)"), .25);
    const layouts = [];
    for (const width of [320, 390, 768, 1440]) {
      await cdp("Emulation.setDeviceMetricsOverride", {width, height: 1200, deviceScaleFactor: 1, mobile: width < 768});
      for (const theme of ["light", "dark", "warm"]) {
        await evaluate("document.querySelector('[data-item=" + theme + "]').click()"); await delay(100);
        await evaluate("document.querySelector('[data-vla-flow]').scrollIntoView({block:'start'})");
        const measured = await evaluate(`(() => {const w=document.querySelector('[data-vla-flow]');return {
          overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1||w.scrollWidth>w.clientWidth+1,
          clipped:[...w.querySelectorAll('output,label,button')].some(x=>x.scrollWidth>x.clientWidth+1),
          unlabeled:[...w.querySelectorAll('input')].some(x=>!x.labels.length),
          described:w.querySelector('svg desc').textContent.includes('末端值'),
          liveRegion:!!w.querySelector('[aria-live=polite]')};})()`);
        assert(!measured.overflow&&!measured.clipped&&!measured.unlabeled&&measured.described&&measured.liveRegion,JSON.stringify({width,theme,measured}));
        layouts.push({width,theme,...measured});
        if (width === 390 || width === 1440) {
          const {data} = await cdp("Page.captureScreenshot", {format: "png"});
          fs.writeFileSync(path.join(output, `flow-${width}-${theme}.png`), Buffer.from(data,"base64"));
        }
      }
    }
    await evaluate("document.querySelector('[data-item=light]').click()");
    await cdp("Emulation.setScriptExecutionDisabled", {value: true}); await cdp("Page.reload");
    await waitFor("document.readyState==='complete'&&!!document.querySelector('[data-flow-fallback]')");
    assert(await evaluate("!document.querySelector('[data-flow-fallback]').hidden&&document.querySelector('[data-flow-interactive]').hidden"));
    await cdp("Emulation.setScriptExecutionDisabled", {value: false});
    assert.deepEqual(exceptions, []);
    const report={arithmeticCases:cases.length,browserCases:cases.length,presets:3,keyboard:"passed",noJavaScript:"static explanation retained",layouts,exceptions};
    fs.writeFileSync(path.join(output,"results.json"),JSON.stringify(report,null,2)+"\n");
    console.log(JSON.stringify({arithmeticCases:cases.length,browserCases:cases.length,layouts:layouts.length,exceptions:exceptions.length,output}));
  } finally {
    for (const task of pending.values()) clearTimeout(task.timer);
    ws.close(); await fetch("http://127.0.0.1:"+port+"/json/close/"+tab.id).catch(()=>{});
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
