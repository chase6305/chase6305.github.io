#!/usr/bin/env node
// Read-only browser checks. Start Hugo -D and Chrome's loopback CDP endpoint first.
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
  const tabs = await (await fetch("http://127.0.0.1:" + port + "/json/list")).json();
  const ws = new WebSocket(tabs.find(tab => tab.type === "page").webSocketDebuggerUrl);
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
  const review = JSON.parse(fs.readFileSync(path.join(root, "docs/blog-editorial-review.json")));
  const results = [];
  for (const post of review.posts) {
    const route = "/" + post.path.replace(/^content\//, "")
      .replace(/\/index\.md$/, "").replace(/\.md$/, "").toLowerCase() + "/";
    await navigate(route);
    const result = await evaluate("(() => ({title: document.title, lang: document.querySelector('article')?.lang, guide: !!document.querySelector('.blog-reading-guide'), related: document.querySelectorAll('.blog-related__card').length, overflow: document.documentElement.scrollWidth > innerWidth + 1, brokenImages: Array.from(document.querySelectorAll('.content img')).filter(i => !i.complete || !i.naturalWidth).map(i=>i.getAttribute('src')), mathErrors: Array.from(document.querySelectorAll('.katex-error')).map(e=>({text:e.textContent,error:e.title})), mathCount: document.querySelectorAll('.katex').length, responsiveImages: document.querySelectorAll('.content img[srcset]').length}))()");
    const editorial = await evaluate("(() => { const toc=document.querySelector('.blog-mobile-toc'); const node=document.querySelector('script[type=\"application/ld+json\"]'); const schema=node ? JSON.parse(node.textContent) : null; return {mobileToc: !!toc && getComputedStyle(toc).display !== 'none', selfCheck: !!document.getElementById('阅读自测与验收'), schemaValid: schema?.['@type'] === 'BlogPosting' && schema.inLanguage === 'zh-CN' && !!schema.author?.length && !!schema.dateModified}; })()");
    const topic = await evaluate("(() => {const nav=document.querySelector('.blog-topic-nav'); return {topicNavigation:!!nav, topicLinks:nav ? Array.from(nav.querySelectorAll('a')).map(a=>a.getAttribute('href')) : []};})()");
    await viewport(1440, 1000);
    await sleep(30);
    const desktop = await evaluate("({desktopOverflow:document.documentElement.scrollWidth > innerWidth + 1, desktopTocHidden:getComputedStyle(document.querySelector('.blog-mobile-toc')).display === 'none'})");
    await viewport(390, 844);
    results.push({route, draft: post.draft, ...result, ...editorial, ...topic, ...desktop});
    if (results.length % 15 === 0) console.log("checked pages:", results.length);
  }
  await navigate("/posts/robotics/kinematics/pinocchio/");
  await viewport(1440, 1000);
  await evaluate("document.querySelector('[data-item=light]').click()");
  await sleep(200);
  await screenshot("desktop-light");
  await evaluate("document.querySelector('.content img').scrollIntoView({block:'center'})");
  await sleep(200);
  await screenshot("desktop-illustration");
  await viewport(390, 844);
  await evaluate("document.querySelector('[data-item=dark]').click(); window.scrollTo(0,0)");
  await sleep(200);
  assert(await evaluate("document.documentElement.classList.contains('dark')"));
  await screenshot("mobile-dark");
  await evaluate("document.querySelector('.blog-mobile-toc summary').click(); document.querySelector('.blog-mobile-toc').scrollIntoView({block:'center'})");
  await screenshot("mobile-toc");
  await evaluate("document.querySelector('.blog-mobile-toc a').click()");
  await sleep(200);
  assert(await evaluate("!document.querySelector('.blog-mobile-toc').open && document.activeElement.contains(document.getElementById(decodeURIComponent(location.hash.slice(1))))"));
  const codeChecks = await evaluate("(() => {const pre=document.querySelector('.hextra-code-block pre'); pre.focus(); return {scrollable:pre.scrollHeight > pre.clientHeight, keyboard:document.activeElement === pre};})()");
  assert(codeChecks.scrollable && codeChecks.keyboard);
  assert(await evaluate("getComputedStyle(document.querySelector('.hextra-code-copy-btn-container')).opacity === '1'"));
  await evaluate("Object.defineProperty(navigator, 'clipboard', {configurable:true, value:{writeText:async text => {window.__blogCopied=text;}}}); document.querySelector('.hextra-code-copy-btn').click()");
  await sleep(100);
  assert(await evaluate("window.__blogCopied === document.querySelector('.hextra-code-block code').textContent"));
  await screenshot("mobile-code");
  assert(await evaluate("document.querySelector('.blog-code-copy-status').nextElementSibling === document.querySelector('.hextra-code-block')"));
  const copyFallbacks = [];
  for (const mode of ['denied', 'missing', 'throws']) {
    const clipboard = mode === 'missing' ? 'undefined' : mode === 'denied'
      ? '{writeText:async () => {throw new DOMException("Denied", "NotAllowedError");}}'
      : '{writeText:() => {throw new Error("Unavailable");}}';
    await evaluate("Object.defineProperty(navigator, 'clipboard', {configurable:true, value:" + clipboard + "}); document.querySelector('.hextra-code-copy-btn').click()");
    await sleep(80);
    const fallback = await evaluate("(() => {const btn=document.querySelector('.hextra-code-copy-btn'); const code=document.querySelector('.hextra-code-block code'); const range=window.getSelection().getRangeAt(0); const expected=document.createRange(); expected.selectNodeContents(code); return {enabled:!btn.disabled, success:btn.classList.contains('copied'), feedback:document.querySelector('.blog-code-copy-status').textContent, selected:range.compareBoundaryPoints(Range.START_TO_START,expected)===0 && range.compareBoundaryPoints(Range.END_TO_END,expected)===0, overflow:document.documentElement.scrollWidth>innerWidth+1};})()");
    assert(fallback.enabled && !fallback.success && fallback.selected && !fallback.overflow);
    assert(fallback.feedback.includes('手动复制'));
    copyFallbacks.push({mode, ...fallback});
  }
  await screenshot("mobile-copy-fallback-dark");
  await viewport(1440, 1000);
  await evaluate("document.querySelector('[data-item=light]').click(); document.querySelector('.hextra-code-block').scrollIntoView({block:'center'})");
  await screenshot("desktop-copy-fallback-light");
  // A denied copy must be retryable; success feedback must reset cleanly.
  await evaluate("Object.defineProperty(navigator, 'clipboard', {configurable:true, value:{writeText:async text => {window.__blogCopied=text;}}}); document.querySelector('.hextra-code-copy-btn').click()");
  await sleep(80);
  assert(await evaluate("document.querySelector('.hextra-code-copy-btn').classList.contains('copied')"));
  await sleep(1600);
  assert(await evaluate("!document.querySelector('.hextra-code-copy-btn').classList.contains('copied') && !document.querySelector('.blog-code-copy-status').textContent"));
  await evaluate("window.getSelection().removeAllRanges(); document.querySelector('[data-item=dark]').click()");
  await viewport(390, 844);
  await evaluate("document.querySelector('.content img').scrollIntoView({block:'center'})");
  await sleep(200);
  await screenshot("mobile-illustration");
  await evaluate("document.querySelector('.content img').click()");
  await sleep(400);
  const zoom = await evaluate("!!document.querySelector('.medium-zoom-image--opened')");
  await cdp("Input.dispatchKeyEvent", {type:"keyDown", key:"Escape", code:"Escape", windowsVirtualKeyCode:27});
  await cdp("Input.dispatchKeyEvent", {type:"keyUp", key:"Escape", code:"Escape", windowsVirtualKeyCode:27});
  await sleep(400);
  await evaluate("document.querySelector('.blog-related').scrollIntoView({block:'center'})");
  await screenshot("mobile-related");
  const mobileCards = await evaluate("Array.from(document.querySelectorAll('.blog-related__card')).map(e => {const b=e.getBoundingClientRect(); return {x:b.x, width:b.width, top:b.top};})");
  await evaluate("window.scrollTo(0,0);document.querySelector('.hextra-hamburger-menu').click()");
  const menu = await evaluate("document.querySelector('.hextra-hamburger-menu').getAttribute('aria-expanded')");
  await screenshot("mobile-menu");
  await evaluate("document.querySelector('.hextra-hamburger-menu').click()");
  await navigate("/learning-paths/");
  const learningPaths = await evaluate("({groups:document.querySelectorAll('.blog-topic-index section').length, articles:document.querySelectorAll('.blog-topic-index section ol a').length, jumpLinks:document.querySelectorAll('.blog-topic-index__jump a').length, overflow:document.documentElement.scrollWidth > innerWidth + 1})");
  assert.equal(learningPaths.groups, 10);
  assert.equal(learningPaths.articles, 66);
  assert.equal(learningPaths.jumpLinks, 10);
  assert(!learningPaths.overflow);
  await screenshot("mobile-learning-paths");
  await viewport(1440, 1000);
  await screenshot("desktop-learning-paths");
  await viewport(390, 844);
  await navigate("/posts/trajectory/toppra/");
  await evaluate("document.querySelector('.content img').scrollIntoView({block:'center'})");
  await screenshot("mobile-trajectory-comparison");
  await viewport(1440, 1000);
  await evaluate("document.querySelector('[data-item=light]').click(); document.querySelector('.content img').scrollIntoView({block:'center'})");
  await screenshot("desktop-trajectory-comparison");
  await viewport(390, 844);
  await navigate("/posts/");
  await evaluate("document.querySelector('.hextra-hamburger-menu').click()");
  assert.equal(await evaluate("document.querySelector('.hextra-hamburger-menu').getAttribute('aria-expanded')"), "true");
  await evaluate("document.querySelector('.hextra-hamburger-menu').click()");
  await screenshot("mobile-blog-list");
  await viewport(1440, 1000);
  await evaluate("document.querySelector('[data-item=light]').click()");
  await screenshot("desktop-blog-list");
  const searches = {};
  for (const query of ["逆运动学", "PPO", "CasADi", "SPSC", "zzzxxyy987654321notfound"]) {
    searches[query] = await evaluate("window.hextraSearch.search(" + JSON.stringify(query) + ")");
  }
  for (const query of ["逆运动学", "PPO", "CasADi", "SPSC"]) assert(searches[query].length > 0, query + " search failed");
  assert.equal(searches["zzzxxyy987654321notfound"].length, 0);
  await evaluate("document.querySelector('[data-search-open]').click(); const input=document.querySelector('.hextra-search-input'); input.value='逆运动学'; input.dispatchEvent(new Event('input', {bubbles:true}))");
  await sleep(600);
  assert(await evaluate("document.getElementById('hextra-search-dialog').open && document.querySelectorAll('.hextra-search-results a').length > 0"));
  await screenshot("desktop-search");
  const report = {pages: results, learningPaths, zoom, mobileMenu: menu === "true", mobileCards, codeChecks, copyFallbacks, searches, exceptions};
  fs.writeFileSync(path.join(output, "results.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({checked:results.length, zoom, mobileMenu:menu,
    exceptions, failures:results.filter(r=>r.overflow || r.brokenImages.length ||
      r.mathErrors.length || !r.guide || r.lang !== "zh-CN" || !r.mobileToc || !r.selfCheck || !r.schemaValid || !r.topicNavigation || r.desktopOverflow || !r.desktopTocHidden), screenshots:output}, null, 2));
  ws.close();
  assert(results.every(r => !r.overflow && !r.brokenImages.length && !r.mathErrors.length && r.guide && r.lang === 'zh-CN' && r.mobileToc && r.selfCheck && r.schemaValid && r.topicNavigation && !r.desktopOverflow && r.desktopTocHidden));
  assert(zoom && menu === 'true');
  assert.equal(exceptions.length, 0, 'Unexpected browser JavaScript exceptions');
})().catch(error => { console.error(error); browserSocket?.close(); process.exitCode = 1; });
