#!/usr/bin/env node
// Serial browser exercise against a local Hugo preview and an existing CDP endpoint.
// Compares UI outputs with Python calculations, including score boundaries and no acceptance.
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const base = process.argv[2] || "http://127.0.0.1:1362";
const port = process.argv[3] || "9237";
const output = process.argv[4] || "/tmp/jev-routing-browser";
const root = path.resolve(__dirname, "..");
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function main() {
  fs.mkdirSync(output, {recursive:true});
  const reference = spawnSync("python3", ["-B", "-c", `
import importlib.util,json,sys
from pathlib import Path
p=Path(sys.argv[1])/'content/posts/ai/jev-decision-layer/routing_lab.py'
spec=importlib.util.spec_from_file_location('routing_lab',p); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
records=mod.fixture_records(); rows=[]
for group in mod.GROUPS:
 for index in range(101):
  for cost in (1,20,40):
   result=mod.route_metrics([r for r in records if r['group']==group],index/100,fallback_cost=cost)
   rows.append(dict(group=group,threshold=index/100,fallback_cost=cost,result=result))
print(json.dumps({'cases':rows,'selected_threshold':mod.build_results()['threshold_selection']['selected']['threshold']}))
`, root], {encoding:"utf8"});
  assert.equal(reference.status, 0, reference.stderr);
  const expected = JSON.parse(reference.stdout);
  const cases = expected.cases;
  const tab = await (await fetch("http://127.0.0.1:"+port+"/json/new?about:blank", {method:"PUT"})).json();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  const pending = new Map();
  const exceptions = [];
  let serial = 0;
  try {
    await new Promise((resolve,reject) => {
      ws.addEventListener("open", resolve, {once:true});
      ws.addEventListener("error", reject, {once:true});
    });
    ws.addEventListener("message", event => {
      const message = JSON.parse(event.data);
      if (message.id && pending.has(message.id)) {
        const request = pending.get(message.id);
        pending.delete(message.id); clearTimeout(request.timer);
        if (message.error) request.reject(Error(JSON.stringify(message.error)));
        else request.resolve(message.result);
      } else if (message.method === "Runtime.exceptionThrown") exceptions.push(message.params.exceptionDetails);
    });
    const cdp = (method,params={}) => new Promise((resolve,reject) => {
      const id = ++serial;
      const timer = setTimeout(() => {pending.delete(id);reject(Error("CDP timeout: "+method));},20000);
      pending.set(id,{resolve,reject,timer});
      ws.send(JSON.stringify({id,method,params}));
    });
    const evaluate = async expression => {
      const r = await cdp("Runtime.evaluate",{expression,returnByValue:true,awaitPromise:true});
      assert(!r.exceptionDetails,JSON.stringify(r.exceptionDetails));
      return r.result.value;
    };
    const waitFor = async expression => {
      // A cold visit also loads the site's existing KaTeX font CDN resources.
      // Preserve the complete-page check while allowing normal network variation.
      const deadline = Date.now() + 30000;
      while(Date.now() < deadline) {
        if(await evaluate(expression))return;
        await delay(100);
      }
      const state = await evaluate(`({url:location.href,readyState:document.readyState,
        fonts:document.fonts.status,widget:!!document.querySelector('[data-jev-routing-lab]'),
        status:document.querySelector('[data-routing-status]')?.textContent})`).catch(()=>null);
      throw Error("Condition timed out: "+expression+"; page="+JSON.stringify(state));
    };
    const navigate = async () => {
      await cdp("Page.navigate",{url:base+"/posts/ai/jev-decision-layer/"});
      await waitFor("document.readyState==='complete' && !!document.querySelector('[data-jev-routing-lab]')");
    };
    await cdp("Page.enable"); await cdp("Runtime.enable"); await cdp("Network.enable");
    await cdp("Page.bringToFront");
    await cdp("Network.setCacheDisabled",{cacheDisabled:true});
    await navigate();
    await waitFor("!!document.querySelector('[data-jev-routing-lab]').dataset.routingResult");
    assert.equal(await evaluate("Number(document.querySelector('[data-jev-routing-lab] input[name=threshold]').value)"),expected.selected_threshold);
    // Escaping a formula's opening dollar can leave readable raw TeX without a KaTeX error.
    const rawMath = await evaluate(`(() => {
      const article=document.querySelector('.content').cloneNode(true);
      article.querySelectorAll('.katex,pre,code,script,style').forEach(x=>x.remove());
      return article.textContent.match(/\\\\(?:frac|sqrt|log|sum|hat|Pr|mathrm)\\b/g) || [];
    })()`);
    assert.deepEqual(rawMath,[],"Unrendered inline TeX in article prose");
    const actual = await evaluate(`(() => {
      const root=document.querySelector('[data-jev-routing-lab]'),form=root.querySelector('form');
      return ${JSON.stringify(cases.map(({group,threshold,fallback_cost})=>({group,threshold,fallback_cost})))}.map(item=>{
        form.elements.group.value=item.group;
        form.elements.threshold.value=String(item.threshold);
        form.elements.cost.value=String(item.fallback_cost);
        form.dispatchEvent(new Event('input',{bubbles:true}));
        return JSON.parse(root.dataset.routingResult);
      });
    })()`);
    cases.forEach((item,i)=>{
      const a=actual[i],e=item.result;
      assert.deepEqual([a.total,a.accepted,a.errors,a.fallback],[e.records,e.accepted,e.accepted_errors,e.rejected],JSON.stringify(item));
      assert.equal(a.error,e.selective_error,JSON.stringify(item));
      assert(Math.abs(a.cost-e.mean_call_cost)<1e-10,JSON.stringify(item));
    });
    // Reset changes only threshold; it must not tune on the selected group or reset costs.
    await evaluate(`(() => {const root=document.querySelector('[data-jev-routing-lab]'),f=root.querySelector('form');f.elements.group.value='shifted';f.elements.cost.value='1';f.elements.threshold.value='.9';f.dispatchEvent(new Event('input',{bubbles:true}));root.querySelector('[data-routing-reset]').click();})()`);
    assert.deepEqual(await evaluate(`(() => {const f=document.querySelector('[data-jev-routing-lab] form');return [f.elements.group.value,Number(f.elements.threshold.value),f.elements.cost.value];})()`),["shifted",expected.selected_threshold,"1"]);
    // Real keyboard event exercises the native accessible slider, not only DOM mutations.
    await evaluate("document.querySelector('[data-jev-routing-lab] input[name=threshold]').focus()");
    await cdp("Input.dispatchKeyEvent",{type:"keyDown",key:"ArrowRight",code:"ArrowRight",windowsVirtualKeyCode:39});
    await cdp("Input.dispatchKeyEvent",{type:"keyUp",key:"ArrowRight",code:"ArrowRight",windowsVirtualKeyCode:39});
    assert.equal(await evaluate("Number(document.querySelector('[data-jev-routing-lab] input[name=threshold]').value)"),Math.min(1,expected.selected_threshold+.01));
    const layouts=[];
    for(const width of [320,390,768,1440]) {
      await cdp("Emulation.setDeviceMetricsOverride",{width,height:1000,deviceScaleFactor:1,mobile:width<768});
      for(const theme of ["light","dark","warm"]) {
        await evaluate("document.querySelector('[data-item="+theme+"]').click()");
        await delay(150);
        await evaluate("document.querySelector('[data-jev-routing-lab]').scrollIntoView({block:'center'})");
        const measured=await evaluate(`(() => {
          const root=document.querySelector('[data-jev-routing-lab]');
          return {documentOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1,
            widgetOverflow:root.scrollWidth>root.clientWidth+1,
            metricOverflow:[...root.querySelectorAll('output')].some(x=>x.scrollWidth>x.clientWidth+1),
            unlabeled:[...root.querySelectorAll('input,select')].filter(x=>!x.labels.length).length,
            liveRegion:!!root.querySelector('[aria-live=polite]')};
        })()`);
        assert(!measured.documentOverflow&&!measured.widgetOverflow&&!measured.metricOverflow&&!measured.unlabeled&&measured.liveRegion,JSON.stringify({width,theme,measured}));
        layouts.push({width,theme,...measured});
        if(width===390||width===1440) {
          const {data}=await cdp("Page.captureScreenshot",{format:"png"});
          fs.writeFileSync(path.join(output,`widget-${width}-${theme}.png`),Buffer.from(data,"base64"));
        }
      }
    }
    await evaluate("document.querySelector('[data-item=light]').click()");
    // Data failure keeps a useful explanation and the article's static material.
    await cdp("Network.setBlockedURLs",{urls:["*routing-fixtures*.json*"]});
    await navigate();
    await waitFor("document.querySelector('[data-routing-status]').textContent.includes('暂时无法读取')");
    assert(await evaluate("document.querySelector('[data-routing-interactive]').hidden && !!document.getElementById('fig-jev-routing-tradeoff')"));
    await cdp("Network.setBlockedURLs",{urls:[]});
    // Without JavaScript the form stays hidden and noscript instructions remain available.
    await cdp("Emulation.setScriptExecutionDisabled",{value:true});
    await navigate();
    assert(await evaluate("document.querySelector('[data-routing-interactive]').hidden && document.querySelector('[data-jev-routing-lab] noscript').textContent.includes('静态算例')"));
    await cdp("Emulation.setScriptExecutionDisabled",{value:false});
    assert.equal(exceptions.length,0,JSON.stringify(exceptions));
    const result={pythonComparisonCases:cases.length,layouts,keyboard:true,thresholdReset:true,dataFailureFallback:true,noScriptFallback:true,exceptions};
    fs.writeFileSync(path.join(output,"results.json"),JSON.stringify(result,null,2)+"\n");
    console.log(JSON.stringify({pythonComparisonCases:cases.length,layouts:layouts.length,exceptions:0,output}));
  } finally {
    for(const request of pending.values())clearTimeout(request.timer);
    ws.close();
    await fetch("http://127.0.0.1:"+port+"/json/close/"+tab.id);
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
