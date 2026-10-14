#!/usr/bin/env node
// Site navigation, theme interaction and same-origin math/search/zoom checks.
// Run serially against an existing loopback Chrome CDP endpoint.
// node scripts/check_site_accessibility.cjs [baseURL] [CDP port] [output] [--capture-only]
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const base = process.argv[2] || 'http://127.0.0.1:1385';
const endpoint = 'http://127.0.0.1:' + (process.argv[3] || '9248');
const output = process.argv[4] || '/tmp/chase-site-accessibility';
const captureOnly = process.argv.includes('--capture-only');
const interactionsOnly = process.argv.includes('--interactions-only');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function main() {
  fs.mkdirSync(output, {recursive: true});
  const tab = await (await fetch(endpoint + '/json/new?about:blank', {method: 'PUT'})).json();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  const pending = new Map(), exceptions = [], requests = [], layouts = [];
  let serial = 0, savedTheme, savedScrollRestoration;
  const report = {layouts, exceptions, requests};
  const cdp = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++serial;
    const timer = setTimeout(() => { pending.delete(id); reject(Error('CDP timeout: ' + method)); }, 30000);
    pending.set(id, {resolve, reject, timer});
    ws.send(JSON.stringify({id, method, params}));
  });
  const evaluate = async expression => {
    const result = await cdp('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
    assert(!result.exceptionDetails, JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const check = (passed, message) => { if (!captureOnly) assert(passed, message); };
  const navigate = async route => {
    await cdp('Page.navigate', {url: base + route});
    const deadline = Date.now() + 30000;
    while (Date.now() < deadline) {
      if (await evaluate('location.pathname === ' + JSON.stringify(route) +
        " && document.readyState === 'complete' && !!document.querySelector('#content')")) {
        await evaluate('document.fonts.ready');
        await delay(100);
        return;
      }
      await delay(100);
    }
    throw Error('Navigation timeout: ' + route);
  };
  const key = async (key, code, value) => {
    const params = {key, code, windowsVirtualKeyCode: value};
    const text = key === 'Enter' ? {text: '\r', unmodifiedText: '\r'} : {};
    await cdp('Input.dispatchKeyEvent', {type: 'keyDown', ...params, ...text});
    await cdp('Input.dispatchKeyEvent', {type: 'keyUp', ...params});
  };
  const screenshot = async name => {
    const result = await cdp('Page.captureScreenshot', {format: 'png'});
    fs.writeFileSync(path.join(output, name + '.png'), Buffer.from(result.data, 'base64'));
  };
  const contrast = (foreground, background) => {
    const luminance = color => color.match(/[\d.]+/g).slice(0, 3).map(Number).map(v => {
      v /= 255;
      return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;
    }).reduce((sum, v, i) => sum + v * [.2126, .7152, .0722][i], 0);
    const values = [luminance(foreground), luminance(background)].sort((a, b) => b - a);
    return (values[0] + .05) / (values[1] + .05);
  };
  try {
    await new Promise((resolve, reject) => {
      ws.addEventListener('open', resolve, {once: true});
      ws.addEventListener('error', reject, {once: true});
    });
    ws.onmessage = event => {
      const message = JSON.parse(event.data);
      if (message.id && pending.has(message.id)) {
        const task = pending.get(message.id);
        pending.delete(message.id); clearTimeout(task.timer);
        message.error ? task.reject(Error(JSON.stringify(message.error))) : task.resolve(message.result);
      } else if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
      else if (message.method === 'Network.requestWillBeSent') requests.push(message.params.request.url);
    };
    await cdp('Page.enable'); await cdp('Runtime.enable'); await cdp('Network.enable');
    await cdp('Network.setCacheDisabled', {cacheDisabled: true});
    await cdp('Page.bringToFront');
    await navigate('/');
    savedTheme = await evaluate("localStorage.getItem('color-theme')");
    savedScrollRestoration = await evaluate('history.scrollRestoration');
    await evaluate("history.scrollRestoration='manual'");
    const routes = interactionsOnly ? [] : ['/', '/zh-cn/', '/posts/', '/learning-paths/', '/archives/',
      '/tags/', '/categories/', '/tags/transformer/', '/posts/ai/behavior-prompting/'];
    for (const route of routes) {
      await navigate(route);
      for (const width of [320, 390, 768, 1024, 1440]) {
        await cdp('Emulation.setDeviceMetricsOverride', {width, height: 1000, deviceScaleFactor: 1, mobile: width < 768});
        for (const theme of ['light', 'dark', 'warm']) {
          await evaluate("document.querySelector('[data-item=" + theme + "]').click();window.scrollTo(0,0)");
          await delay(100);
          const measured = await evaluate(`(() => ({
            overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
            controls: [...document.querySelectorAll('.hextra-nav-container button,.hextra-nav-container nav > a[title]')]
              .filter(e => e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden')
              .map(e => ({label: e.getAttribute('aria-label') || e.title,
                width: e.getBoundingClientRect().width, height: e.getBoundingClientRect().height})),
            badges: [...document.querySelectorAll('.article-figure__number')].map(e => ({
              foreground: getComputedStyle(e).color, background: getComputedStyle(e).backgroundColor
            }))
          }))()`);
          measured.badges.forEach(badge => badge.contrast = contrast(badge.foreground, badge.background));
          const row = {route, width, theme, ...measured};
          layouts.push(row);
          check(!row.overflow, JSON.stringify(row));
          check(row.controls.every(c => c.width >= 39.5 && c.height >= 39.5), JSON.stringify(row));
          check(row.badges.every(b => b.contrast >= 4.5), JSON.stringify(row));
          if ((width === 390 && theme === 'dark') || (width === 1440 && theme === 'light')) {
            await screenshot(route.replace(/\//g, '-') + width + '-' + theme);
          }
        }
      }
    }
    if (captureOnly) return;

    await navigate('/archives/');
    report.yearLinks = await evaluate(`(() => [...document.querySelectorAll('.archive-year-nav a')].map(a => ({
      href: a.getAttribute('href'), count: Number(a.querySelector('span').textContent),
      actual: document.getElementById(a.hash.slice(1))?.closest('section').querySelectorAll('.archive-year__posts a').length
    })))()`);
    assert(report.yearLinks.length > 1 && report.yearLinks.every(y => y.count === y.actual));
    await evaluate("document.querySelector('.archive-year-nav a:last-child').click()");
    await delay(200);
    assert(await evaluate("document.querySelector(location.hash).getBoundingClientRect().top >= 55"));
    for (const route of ['/archives/', '/tags/', '/categories/', '/tags/transformer/']) {
      await navigate(route);
      assert(await evaluate("document.querySelector('.collection-nav a[href=\"/posts/\"]') && document.querySelector('.collection-nav a[href=\"/learning-paths/\"]')"));
    }

    await cdp('Emulation.setDeviceMetricsOverride', {width: 320, height: 1000, deviceScaleFactor: 1, mobile: true});
    await navigate('/posts/');
    assert(await evaluate("document.querySelector('.hextra-sidebar-container').inert && !document.querySelector('#content').inert"));
    await evaluate("document.querySelector('.hextra-hamburger-menu').focus()");
    await key('Tab', 'Tab', 9);
    assert(await evaluate("!document.activeElement.closest('.hextra-sidebar-container')"));
    await evaluate("document.querySelector('.hextra-hamburger-menu').focus()");
    await key('Enter', 'Enter', 13);
    const menuOpened = await evaluate("({sidebarInert:document.querySelector('.hextra-sidebar-container').inert,backgroundInert:document.querySelector('#content').inert,active:document.activeElement.outerHTML.slice(0,300),focused:!!document.activeElement.closest('.hextra-sidebar-container'),expanded:document.querySelector('.hextra-hamburger-menu').getAttribute('aria-expanded'),route:location.pathname})");
    assert(!menuOpened.sidebarInert && menuOpened.backgroundInert && menuOpened.focused, JSON.stringify(menuOpened));
    await evaluate("document.querySelector('.hextra-sidebar-container .hextra-theme-toggle').focus()");
    await key('ArrowDown', 'ArrowDown', 40);
    await key('Escape', 'Escape', 27);
    assert(await evaluate("document.querySelector('.hextra-hamburger-menu').getAttribute('aria-expanded') === 'true'"));
    await key('Escape', 'Escape', 27);
    assert(await evaluate("document.querySelector('.hextra-sidebar-container').inert && !document.querySelector('#content').inert && document.activeElement.classList.contains('hextra-hamburger-menu')"));
    await key('Enter', 'Enter', 13);
    await cdp('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
    await delay(100);
    assert(await evaluate("!document.querySelector('.hextra-sidebar-container').inert && !document.querySelector('#content').inert && document.querySelector('.hextra-hamburger-menu').getAttribute('aria-expanded') === 'false'"));
    report.mobileMenuFocus = true;

    await navigate('/posts/ai/behavior-prompting/');
    await evaluate("document.querySelector('.hextra-theme-toggle').focus()");
    await key('ArrowDown', 'ArrowDown', 40);
    assert(await evaluate("document.querySelector('.hextra-theme-toggle').getAttribute('aria-expanded') === 'true' && document.activeElement.dataset.item === 'light'"));
    await key('End', 'End', 35);
    assert(await evaluate("document.activeElement.dataset.item === 'system'"));
    await key('Home', 'Home', 36);
    assert(await evaluate("document.activeElement.dataset.item === 'light'"));
    await key('ArrowUp', 'ArrowUp', 38);
    assert(await evaluate("document.activeElement.dataset.item === 'system'"));
    await key('Escape', 'Escape', 27);
    assert(await evaluate("document.activeElement.classList.contains('hextra-theme-toggle') && document.activeElement.getAttribute('aria-expanded') === 'false'"));
    await key('ArrowUp', 'ArrowUp', 38);
    assert(await evaluate("document.activeElement.dataset.item === 'system'"));
    await key('ArrowUp', 'ArrowUp', 38);
    await key('Enter', 'Enter', 13);
    assert(await evaluate("document.documentElement.classList.contains('dark') && document.activeElement.classList.contains('hextra-theme-toggle') && document.activeElement.getAttribute('aria-expanded') === 'false'"));
    await key('ArrowDown', 'ArrowDown', 40);
    await key('Tab', 'Tab', 9);
    assert(await evaluate("document.querySelector('.hextra-theme-toggle').getAttribute('aria-expanded') === 'false' && !document.activeElement.closest('.hextra-theme-toggle-options')"));
    report.themeKeyboard = true;
    await cdp('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: 'dark'}]});
    await evaluate("document.querySelector('[data-item=system]').click()");
    assert(await evaluate("document.documentElement.classList.contains('dark')"));
    await cdp('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: 'light'}]});
    await delay(100);
    assert(await evaluate("document.documentElement.classList.contains('light')"));
    await evaluate("window.dispatchEvent(new StorageEvent('storage',{key:'color-theme',newValue:'warm'}))");
    assert(await evaluate("document.documentElement.classList.contains('warm') && [...document.querySelectorAll('[data-item=warm]')].every(e=>e.getAttribute('aria-checked')==='true')"));
    report.systemAndCrossTabTheme = true;

    const injection = await cdp('Page.addScriptToEvaluateOnNewDocument', {source:
      "Object.defineProperty(window,'localStorage',{get(){throw new DOMException('Storage blocked','SecurityError')}})"});
    await navigate('/posts/');
    for (const theme of ['dark', 'warm', 'light', 'system']) {
      await evaluate("document.querySelector('[data-item=" + theme + "]').click()");
      assert(await evaluate("document.querySelector('.theme-picker').dataset.theme === " + JSON.stringify(theme)));
    }
    assert.equal(exceptions.length, 0, JSON.stringify(exceptions));
    report.blockedStorage = true;
    await cdp('Page.removeScriptToEvaluateOnNewDocument', {identifier: injection.identifier});

    await cdp('Network.setCacheDisabled', {cacheDisabled: true});
    await cdp('Network.setBlockedURLs', {urls: ['https://*']});
    requests.length = 0;
    await navigate('/posts/ai/behavior-prompting/');
    await evaluate("Promise.all([...document.querySelectorAll('.content img')].map(i=>{i.loading='eager';return i.decode()}))");
    const fonts = await evaluate("[...document.fonts].filter(f=>f.family.startsWith('KaTeX') && f.status==='loaded').map(f=>f.family)");
    assert(fonts.length > 0, 'No loaded KaTeX fonts');
    assert(requests.every(url => !url.startsWith('https://')), JSON.stringify(requests));
    assert(await evaluate("document.querySelector('.katex-html > .base') && getComputedStyle(document.querySelector('.katex-html > .base')).display==='inline-block'"));
    assert((await evaluate("window.hextraSearch.search('逆运动学')")).length > 0);
    await evaluate("document.querySelector('.article-figure img').scrollIntoView({block:'center'});document.querySelector('.article-figure img').click()");
    await delay(450);
    assert(await evaluate("!!document.querySelector('.medium-zoom-image--opened')"));
    await key('Escape', 'Escape', 27);
    await delay(400);
    await evaluate("document.querySelector('.post-image-link').focus()");
    await key('Enter', 'Enter', 13);
    await delay(450);
    assert(await evaluate("!!document.querySelector('.medium-zoom-image--opened')"));
    await key('Escape', 'Escape', 27);
    await delay(400);
    assert(await evaluate("!document.querySelector('.medium-zoom-image--opened') && document.activeElement.classList.contains('post-image-link')"));
    await key('Enter', 'Enter', 13);
    await delay(450);
    await key('Tab', 'Tab', 9);
    await delay(400);
    assert(await evaluate("!document.querySelector('.medium-zoom-image--opened') && document.activeElement.classList.contains('post-image-link')"));
    await cdp('Emulation.setEmulatedMedia', {features: [{name: 'prefers-reduced-motion', value: 'reduce'}]});
    await key('Enter', 'Enter', 13);
    await delay(150);
    assert(await evaluate("!!document.querySelector('.medium-zoom-image--opened')"));
    await key('Escape', 'Escape', 27);
    await delay(150);
    assert(await evaluate("!document.querySelector('.medium-zoom-image--opened')"));
    report.keyboardZoom = true;
    report.sameOriginFeatures = {fonts, requests: requests.length, search: true, zoom: true};
    assert.equal(exceptions.length, 0, JSON.stringify(exceptions));
    console.log(JSON.stringify({layouts: layouts.length, yearLinks: report.yearLinks.length, mobileMenuFocus: report.mobileMenuFocus, themeKeyboard: report.themeKeyboard, systemAndCrossTabTheme: report.systemAndCrossTabTheme, blockedStorage: report.blockedStorage, keyboardZoom: report.keyboardZoom, sameOriginFeatures: report.sameOriginFeatures, exceptions: exceptions.length, output}, null, 2));
  } finally {
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(report, null, 2));
    if (savedTheme !== undefined) {
      await cdp('Network.setBlockedURLs', {urls: []}).catch(() => {});
      await cdp('Emulation.setEmulatedMedia', {features: []}).catch(() => {});
      await navigate('/').catch(() => {});
      await evaluate(savedTheme === null ? "localStorage.removeItem('color-theme')" :
        "localStorage.setItem('color-theme'," + JSON.stringify(savedTheme) + ')').catch(() => {});
      if (savedScrollRestoration) await evaluate('history.scrollRestoration=' + JSON.stringify(savedScrollRestoration)).catch(() => {});
    }
    for (const task of pending.values()) clearTimeout(task.timer);
    ws.close();
    await fetch(endpoint + '/json/close/' + tab.id);
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
