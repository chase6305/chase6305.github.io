#!/usr/bin/env node
// Run against a Hugo preview and a loopback Chrome CDP endpoint.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {selectArticles} = require("../assets/js/blog-filter.js");
const base = process.argv[2] || "http://127.0.0.1:13143";
const port = process.argv[3] || "9229";
const output = process.argv[4] || "/tmp/chase-blog-round5-discovery";
const drafts = process.argv.includes("--include-drafts");
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
let ws;

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const tab = await (await fetch("http://127.0.0.1:" + port + "/json/new?about:blank",
    {method: "PUT"})).json();
  ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise(resolve => ws.addEventListener("open", resolve, {once: true}));
  let serial = 0;
  const pending = new Map(), exceptions = [];
  ws.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const request = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) request.reject(Error(JSON.stringify(message.error)));
      else request.resolve(message.result);
    } else if (message.method === "Runtime.exceptionThrown") exceptions.push(message.params);
  };
  const cdp = (method, params={}) => new Promise((resolve, reject) => {
    const id = ++serial;
    pending.set(id, {resolve, reject});
    ws.send(JSON.stringify({id, method, params}));
  });
  const evaluate = async expression => {
    const result = await cdp("Runtime.evaluate", {expression, awaitPromise:true, returnByValue:true});
    if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const navigate = async route => {
    await cdp("Page.navigate", {url: base + route});
    for (let i=0; i<80; i++) {
      await sleep(100);
      if (await evaluate("location.pathname + location.search === " + JSON.stringify(route) +
        " && document.readyState === 'complete'")) return;
    }
    throw Error("Navigation timeout: " + route);
  };
  const viewport = (width, height=900) => cdp("Emulation.setDeviceMetricsOverride",
    {width, height, deviceScaleFactor:1, mobile:width<768});
  const screenshot = async name => {
    const {data} = await cdp("Page.captureScreenshot", {format:"png"});
    fs.writeFileSync(path.join(output,name+".png"), Buffer.from(data,"base64"));
  };
  const apply = async (q="", topic="", sort="newest") => {
    await evaluate("document.getElementById('blog-query').value=" + JSON.stringify(q) +
      ";document.getElementById('blog-topic').value=" + JSON.stringify(topic) +
      ";document.getElementById('blog-sort').value=" + JSON.stringify(sort) +
      ";document.querySelector('[data-blog-filter]').requestSubmit()");
    await sleep(40);
  };
  const resultUrls = () => evaluate("Array.from(document.querySelectorAll('#blog-filter-results h2 a')).map(a=>a.getAttribute('href'))");
  const checkState = async state => {
    await apply(state.q, state.topic, state.sort);
    assert.deepEqual(await resultUrls(), selectArticles(articles,state).map(a=>a.url));
  };
  await cdp("Page.enable");
  await cdp("Runtime.enable");
  await viewport(1440, 1000);
  await navigate("/posts/");
  const articles = await evaluate("JSON.parse(document.getElementById('blog-filter-data').textContent)");
  assert.equal(articles.length, drafts ? 66 : 63);
  assert.equal(articles.filter(a=>a.draft).length, drafts ? 3 : 0);
  assert(await evaluate("!document.querySelector('[data-blog-filter]').hidden"));
  const original = await evaluate("Array.from(document.querySelectorAll('[data-blog-original-list] h2 a')).map(a=>a.getAttribute('href'))");
  assert.equal(original.length, 10);
  const oldest = [...articles].sort((a,b)=>a.date.localeCompare(b.date))[0];
  assert(!original.includes(oldest.url), "Cross-page fixture is on the first page");
  await apply(oldest.title);
  assert((await resultUrls()).includes(oldest.url), "Filtering only searched the current page");
  const shared = await evaluate("location.pathname + location.search");
  await navigate(shared);
  assert((await resultUrls()).includes(oldest.url), "URL state did not survive reload");

  const states = [
    {q:"C++",topic:"cpp",sort:"newest"},
    {q:"Ｐｙｔｈｏｎ",topic:"",sort:"oldest"},
    {q:"",topic:"",sort:"shortest"},
    {q:"Jacobian Pinocchio",topic:"",sort:"newest"},
    ...Array.from(new Set(articles.map(a=>a.topic))).map(topic=>({q:"",topic,sort:"newest"}))
  ];
  for (const state of states) {
    await checkState(state);
    if (state.sort === "shortest") assert.equal(await evaluate("document.querySelectorAll('#blog-filter-results .blog-draft-label').length"), drafts ? 3 : 0);
  }
  await apply("C++", "cpp");
  assert(!(await evaluate("document.getElementById('blog-filter-results').textContent")).includes("&#43;"), "HTML entities leaked into visible card text");
  await evaluate("document.getElementById('blog-query').focus()");
  for (const next of ["blog-topic", "blog-sort", "reset"]) {
    await cdp("Input.dispatchKeyEvent", {type:"keyDown",key:"Tab",code:"Tab",windowsVirtualKeyCode:9});
    await cdp("Input.dispatchKeyEvent", {type:"keyUp",key:"Tab",code:"Tab",windowsVirtualKeyCode:9});
    assert(await evaluate(next === "reset" ? "document.activeElement.hasAttribute('data-blog-reset')" : "document.activeElement.id===" + JSON.stringify(next)));
  }
  await apply("zzxxyy987notfound");
  assert(await evaluate("!document.querySelector('[data-blog-empty]').hidden && document.querySelector('[data-blog-original-list]').hidden && document.querySelector('[data-blog-original-pager]').hidden"));
  await apply('<img src=x onerror="window.__filterXss=1">');
  assert(await evaluate("!window.__filterXss && !document.querySelector('#blog-filter-results img')"));
  await evaluate("document.querySelector('[data-blog-reset]').click()");
  assert(await evaluate("document.activeElement.id==='blog-query' && !document.querySelector('[data-blog-original-list]').hidden && !location.search"));

  // Do not filter in the middle of a Chinese IME composition.
  await evaluate("const q=document.getElementById('blog-query'); q.dispatchEvent(new CompositionEvent('compositionstart')); q.value='机器人'; q.dispatchEvent(new InputEvent('input',{bubbles:true,isComposing:true}))");
  await sleep(250);
  assert(await evaluate("!document.querySelector('[data-blog-original-list]').hidden"));
  await evaluate("document.getElementById('blog-query').dispatchEvent(new CompositionEvent('compositionend'))");
  await sleep(250);
  assert((await resultUrls()).length > 0);

  // Browser history restores state without forcing a new search request.
  await navigate("/posts/?q=Jacobian");
  await evaluate("history.pushState(null,'','?topic=cpp');dispatchEvent(new PopStateEvent('popstate'))");
  assert(await evaluate("document.getElementById('blog-topic').value==='cpp'"));
  await evaluate("history.back()");
  await sleep(250);
  assert(await evaluate("document.getElementById('blog-query').value==='Jacobian' && !document.getElementById('blog-topic').value"));
  await navigate("/posts/page/2/");
  await apply(articles[0].title);
  assert((await resultUrls()).includes(articles[0].url));
  await evaluate("document.querySelector('[data-blog-reset]').click()");
  assert(await evaluate("document.querySelector('[data-blog-filter-status]').textContent.includes('2 /')"));

  const layouts = [];
  for (const width of [320,390,768,1440]) {
    await viewport(width, width<768 ? 844 : 1000);
    await navigate("/posts/");
    await apply("", "cpp");
    for (const theme of ["light","dark","warm"]) {
      await evaluate("document.querySelector('[data-item=" + theme + "]').click()");
      const detail = await evaluate("({overflow:document.documentElement.scrollWidth>innerWidth+1, controls:Array.from(document.querySelectorAll('.blog-filter input,.blog-filter select,.blog-filter button')).map(e=>({width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height})), firstCardTop:document.querySelector('#blog-filter-results .blog-card').getBoundingClientRect().top})");
      assert(!detail.overflow && detail.controls.every(c=>c.width>0 && c.height>=44));
      layouts.push({width,theme,...detail});
      if (width===390 || width===1440) await screenshot("filter-"+width+"-"+theme);
    }
  }
  await navigate("/posts/?q=notfoundzz987");
  await screenshot("filter-empty-desktop");

  // Progressive fallback must be visible when all JavaScript is disabled.
  await cdp("Emulation.setScriptExecutionDisabled", {value:true});
  await navigate("/posts/page/2/");
  assert(await evaluate("document.querySelector('[data-blog-filter]').hidden && document.querySelectorAll('[data-blog-original-list] .blog-card').length===10 && !document.querySelector('[data-blog-original-pager]').hidden"));
  assert(await evaluate("document.querySelector('noscript').textContent.includes('完整归档')"));
  await screenshot("no-javascript-page-2");
  await cdp("Emulation.setScriptExecutionDisabled", {value:false});

  // Malformed metadata must fail open to native pagination, not a blank page.
  await cdp("Page.addScriptToEvaluateOnNewDocument", {source:
    "document.addEventListener('DOMContentLoaded',()=>{const data=document.getElementById('blog-filter-data');if(data)data.textContent='invalid JSON';},{once:true});"});
  await navigate("/posts/");
  assert(await evaluate("document.querySelector('[data-blog-filter]').hidden && !document.querySelector('[data-blog-original-list]').hidden"));
  // Use the current page's native links to cover taxonomy and archive views.
  const collections = ["/tags/","/categories/","/archives/","/learning-paths/", "/", "/zh-cn/", "/projects/", "/zh-cn/projects/"];
  const tagURL = articles.flatMap(a=>a.tags)[0].url;
  collections.push(tagURL);
  for (const route of collections) {
    await navigate(route);
    await viewport(390,844);
    assert(await evaluate("document.documentElement.scrollWidth<=innerWidth+1"), route);
    await evaluate("document.querySelector('.hextra-hamburger-menu').click()");
    assert(await evaluate("document.querySelector('.hextra-hamburger-menu').getAttribute('aria-expanded')==='true' && document.querySelector('.hextra-sidebar-container').getAttribute('aria-hidden')==='false'"), route);
    await cdp("Input.dispatchKeyEvent", {type:"keyDown",key:"Escape",code:"Escape",windowsVirtualKeyCode:27});
    await cdp("Input.dispatchKeyEvent", {type:"keyUp",key:"Escape",code:"Escape",windowsVirtualKeyCode:27});
    assert(await evaluate("document.querySelector('.hextra-hamburger-menu').getAttribute('aria-expanded')==='false'"), route);
    await viewport(1440,1000);
    assert(await evaluate("document.documentElement.scrollWidth<=innerWidth+1"), route);
  }
  if (drafts) {
    await navigate(articles.find(a=>a.draft).url);
    assert(await evaluate("!!document.querySelector('.blog-draft-notice')"));
    await screenshot("draft-preview-notice");
  }
  assert.equal(exceptions.length,0,JSON.stringify(exceptions));
  const report = {articles:articles.length,drafts:drafts?3:0,crossPageSearch:true,
    shareableURL:true,queryStates:states.length,ime:true,history:true,literalQuery:true,
    noJavaScript:true,malformedDataFallback:true,keyboardControls:true,collections,layouts,exceptions};
  fs.writeFileSync(path.join(output,"results.json"),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
  await cdp("Page.close");
  ws.close();
})().catch(error=>{console.error(error);ws?.close();process.exitCode=1;});
