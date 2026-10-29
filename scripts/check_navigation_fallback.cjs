#!/usr/bin/env node
// Native site navigation and late menu enhancement; run browser suites serially.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const {connect} = require('./lib/browser.cjs');
const base = (process.argv[2] || 'http://127.0.0.1:13302').replace(/\/$/, '');
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-navigation-fallback';
const origin = new URL(base).origin;
assert(['127.0.0.1', 'localhost', '[::1]'].includes(new URL(base).hostname), 'Use a loopback preview');

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const b = await connect(port), rows = [], proxyRequests = [];
  let releaseCore, delayedCore, server;
  const waitFor = async expression => {
    const deadline = Date.now() + 15000;
    while (Date.now() < deadline) {
      if (await b.evaluate(expression)) return;
      await b.pause(50);
    }
    const state = await b.evaluate("({url:location.href,ready:document.readyState,body:document.body?.innerText.slice(0,300)})");
    throw Error('Timed out: ' + expression + '; ' + JSON.stringify(state));
  };
  try {
    await b.cdp('Network.setCacheDisabled', {cacheDisabled: true});
    const delayedOnly = process.argv.includes('--delayed-only');
    const shortOnly = process.argv.includes('--short-only');
    for (const mode of delayedOnly || shortOnly ? [] : ['scripts-off', 'blocked-core']) {
      await b.cdp('Emulation.setScriptExecutionDisabled', {value: mode === 'scripts-off'});
      await b.cdp('Network.setBlockedURLs', {urls: mode === 'blocked-core' ? ['*/js/main.js', '*/js/main.min.*.js'] : []});
      for (const route of ['/', '/zh-cn/', '/posts/ai/supermimic/']) {
        for (const width of [320, 768, 1440]) {
          await b.viewport(width, 844);
          for (const scale of [1, 2]) for (const theme of ['light', 'dark', 'warm']) {
            await b.navigate(base + route);
            await b.evaluate(`document.documentElement.classList.remove('dark','warm');
              document.documentElement.classList.add(${JSON.stringify(theme)});
              document.documentElement.style.fontSize=${JSON.stringify(scale * 100 + '%')};document.fonts.ready`);
            const state = await b.evaluate(`(() => {
              const native=document.querySelector('.native-navigation'), nav=native.closest('.hextra-nav-container');
              native.querySelector('summary').focus();
              return {ready:!!nav.dataset.navigationReady,visible:native.checkVisibility(),
                inactive:[...document.querySelectorAll('.hextra-hamburger-menu,.hextra-nav-menu-toggle,.hextra-theme-toggle,.hextra-language-switcher')].filter(e=>e.checkVisibility()).length};
            })()`);
            assert.deepEqual(state, {ready: false, visible: true, inactive: 0}, JSON.stringify({route,width,scale,theme,mode,state}));
            await b.key('Enter', 'Enter', 13);
            await waitFor("document.querySelector('.native-navigation').open");
            const links = await b.evaluate(`(() => {
              const native=document.querySelector('.native-navigation');
              return [...native.querySelectorAll('a')].map(a=>({text:a.textContent.trim(),href:a.href,pathname:a.pathname,language:a.hreflang}));
            })()`);
            assert(links.length >= 9, 'Expected the configured site and language destinations');
            for (const expected of ['/posts/', '/learning-paths/', '/categories/', '/tags/', '/archives/', '/zh-cn/']) {
              assert(links.some(link => link.pathname === expected), 'Missing native destination: ' + expected);
            }
            const zh = route === '/zh-cn/';
            assert.equal(links.find(link => link.language === 'en').pathname, zh ? '/' : route);
            assert.equal(links.find(link => link.language === 'zh-cn').pathname, '/zh-cn/');
            assert.equal(links[0].pathname, zh ? '/zh-cn/projects/' : '/projects/');
            assert.equal(links[0].text, zh ? '开源项目' : 'Projects');
            // Every link stays in DOM order and scrolls into view at large text.
            for (let i = 0; i < links.length; i++) {
              await b.key('Tab', 'Tab', 9);
              const focus = await b.evaluate(`(() => {
                const a=document.activeElement,p=document.querySelector('.native-navigation__links');
                const r=a.getBoundingClientRect(),s=p.getBoundingClientRect();
                return {href:a.href,inside:p.contains(a),visible:r.bottom>s.top&&r.top<s.bottom};
              })()`);
              assert.equal(focus.href, links[i].href);
              assert(focus.inside && focus.visible, JSON.stringify({mode,width,scale,theme,i,focus}));
            }
            if (width < 768) {
              await b.key('Tab', 'Tab', 9);
              assert(await b.evaluate("!document.querySelector('.hextra-sidebar-container').contains(document.activeElement)"),
                'Inactive offscreen sidebar must not receive keyboard focus');
            }
            const geometry = await b.evaluate(`(() => {
              const p=document.querySelector('.native-navigation__links').getBoundingClientRect();
              return {overflow:document.documentElement.scrollWidth>innerWidth+1,left:p.left,right:p.right,bottom:p.bottom};
            })()`);
            assert(!geometry.overflow && geometry.left >= 0 && geometry.right <= width + 1 && geometry.bottom <= 844,
              JSON.stringify({mode,width,scale,theme,geometry}));
            rows.push({mode,route,width,scale,theme,links:links.length,passed:true});
            if (width === 320 && scale === 1 && theme === 'light' && zh && mode === 'scripts-off') {
              await b.evaluate("document.querySelector('.native-navigation summary').focus()");
              await b.screenshot(path.join(output, 'native-mobile.png'));
              // Actual native keyboard link activation, with no click handler.
              await b.key('Tab', 'Tab', 9); await b.key('Enter', 'Enter', 13);
              await waitFor("document.readyState==='complete' && location.pathname==='/zh-cn/projects/'");
            }
          }
        }
      }
    }
    for (const mode of delayedOnly ? [] : ['scripts-off', 'blocked-core']) {
      await b.cdp('Emulation.setScriptExecutionDisabled', {value: mode === 'scripts-off'});
      await b.cdp('Network.setBlockedURLs', {urls: mode === 'blocked-core' ? ['*/js/main.js', '*/js/main.min.*.js'] : []});
      for (const [width, height] of [[320, 320], [568, 320], [320, 480], [768, 256]]) {
        await b.viewport(width, height);
        for (const scale of [1, 2]) for (const theme of ['light', 'dark', 'warm']) {
          await b.navigate(base + '/zh-cn/');
          await b.evaluate(`document.documentElement.classList.remove('dark','warm');
            document.documentElement.classList.add(${JSON.stringify(theme)});
            document.documentElement.style.fontSize=${JSON.stringify(scale * 100 + '%')};
            document.querySelector('.native-navigation summary').focus();document.fonts.ready`);
          await b.key('Enter', 'Enter', 13);
          const count = await b.evaluate("document.querySelectorAll('.native-navigation a').length");
          for (let i = 0; i < count; i++) await b.key('Tab', 'Tab', 9);
          const state = await b.evaluate(`(() => {
            const p=document.querySelector('.native-navigation__links'),r=p.getBoundingClientRect(),a=document.activeElement,s=a.getBoundingClientRect();
            return {last:a===p.querySelector('ul:last-child li:last-child a'),
              panelBottom:r.bottom,panelTop:r.top,linkTop:s.top,linkBottom:s.bottom,scrollTop:p.scrollTop,
              overflow:document.documentElement.scrollWidth>innerWidth+1};
          })()`);
          assert(state.last && !state.overflow && state.panelBottom <= height + 1 && state.panelTop >= 0,
            JSON.stringify({mode,width,height,scale,theme,state}));
          assert(state.linkTop >= state.panelTop - 1 && state.linkBottom <= state.panelBottom + 1,
            'The final language destination must scroll fully into view: ' + JSON.stringify(state));
          rows.push({mode,route:'/zh-cn/',width,height,scale,theme,shortScreen:true,passed:true});
          if (mode === 'scripts-off' && width === 320 && height === 320 && scale === 2 && theme === 'light') {
            await b.screenshot(path.join(output,'native-short-text-200.png'));
          }
          if (width === 320 && height === 320 && scale === 2 && theme === 'light') {
            const beforeY = await b.evaluate('scrollY');
            for (let i = 0; i < 3; i++) await b.key('ArrowUp', 'ArrowUp', 38);
            await b.pause(200);
            const scrolled = await b.evaluate("({top:document.querySelector('.native-navigation__links').scrollTop,pageY:scrollY})");
            assert(scrolled.top < state.scrollTop && scrolled.pageY === beforeY,
              'Arrow keys should scroll the focused native panel: ' + JSON.stringify({state,scrolled,beforeY}));
            rows.push({mode,width,height,scale,nativeArrowScroll:true,passed:true});
          }
        }
      }
    }
    if (shortOnly) {
      console.log(JSON.stringify({cases:rows.length,passed:true,output}));
      return;
    }
    await b.cdp('Emulation.setScriptExecutionDisabled', {value: false});
    await b.cdp('Network.setBlockedURLs', {urls: []});
    // Proxy only the given loopback origin, holding the core script until the
    // reader has acted. Other assets retain their exact bytes and integrity.
    server = http.createServer(async (request, response) => {
      try {
        const target = new URL(request.url, base);
        const record = {url:target.href};proxyRequests.push(record);
        if (target.origin !== origin) {response.writeHead(403).end(); return;}
        if (/\/js\/main(?:\.min(?:\.[a-f0-9]+)?)?\.js$/.test(target.pathname)) {
          delayedCore = true;
          await new Promise(resolve => { releaseCore = resolve; });
        }
        const upstream = await fetch(target);
        record.status = upstream.status;
        const body = Buffer.from(await upstream.arrayBuffer());
        response.writeHead(upstream.status, {'Content-Type': upstream.headers.get('content-type') || 'application/octet-stream'});
        response.end(body);
      } catch (error) {proxyRequests.push({error:String(error)});response.writeHead(502).end(String(error));}
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    const proxy = 'http://127.0.0.1:' + server.address().port;
    for (const width of [320, 1440]) for (const action of ['open', 'focused', 'untouched']) {
      releaseCore = null; delayedCore = false;
      await b.viewport(width);
      await b.cdp('Page.navigate', {url: proxy + '/zh-cn/'});
      await waitFor("!!document.querySelector('.native-navigation summary')");
      const deadline = Date.now() + 15000;
      while (!delayedCore && Date.now() < deadline) await b.pause(20);
      assert(delayedCore, 'Expected controlled delayed core request');
      if (action !== 'untouched') await b.evaluate("document.querySelector('.native-navigation summary').focus()");
      if (action === 'open') await b.key('Enter', 'Enter', 13);
      releaseCore(); releaseCore = null;
      await waitFor("document.readyState==='complete'");
      if (action === 'untouched') {
        await waitFor("document.querySelector('.hextra-nav-container').dataset.navigationReady==='true'");
        assert(await b.evaluate("!document.querySelector('.native-navigation').checkVisibility()"));
      } else {
        assert(await b.evaluate(`document.activeElement.matches('.native-navigation summary') &&
          document.querySelector('.native-navigation').checkVisibility() && !document.querySelector('.hextra-nav-container').dataset.navigationReady`));
        if (action === 'open') await b.key('Enter', 'Enter', 13);
        await b.pause(50);
        assert(await b.evaluate("document.activeElement.matches('.native-navigation summary') && document.querySelector('.native-navigation').checkVisibility()"));
        await b.key('Tab', 'Tab', 9);
        const focused = await b.evaluate("document.activeElement.outerHTML");
        await waitFor("document.querySelector('.hextra-nav-container').dataset.navigationReady==='true'");
        assert.equal(await b.evaluate("document.activeElement.outerHTML"), focused, 'Enhancement must retain later reader focus');
      }
      rows.push({mode:'delayed-core',width,action,passed:true});
    }
    assert.equal(b.exceptions.length, 0, JSON.stringify(b.exceptions));
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify({cases:rows,exceptions:b.exceptions},null,2)+'\n');
    console.log(JSON.stringify({cases:rows.length,passed:true,output}));
  } finally {
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify({cases:rows,exceptions:b.exceptions,proxyRequests},null,2)+'\n');
    releaseCore?.();
    await b.close();
    if (server) await new Promise(resolve => {server.close(resolve);server.closeAllConnections();});
  }
})().catch(error => {console.error(error);process.exitCode=1;});
