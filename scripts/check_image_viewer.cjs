#!/usr/bin/env node
// Exercise the real article/viewer with a deliberately delayed or failed image.
// Owns a temporary loopback server; run serially with other browser checks.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const {connect} = require('./lib/browser.cjs');
const {contrastAudit} = require('./lib/contrast.cjs');
const publicRoot = path.resolve(process.argv[2]);
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-image-viewer';
const mime = {'.html':'text/html; charset=utf-8','.js':'text/javascript','.css':'text/css',
  '.json':'application/json','.webp':'image/webp','.png':'image/png','.svg':'image/svg+xml','.woff2':'font/woff2'};
const hold = new Set(), fail = new Set(), pending = new Set(), requests = [];
let fixtureAssets;
const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost');
    const route = decodeURIComponent(url.pathname);
    if (route === '/image-viewer-fixture/' && fixtureAssets) {
      response.writeHead(200,{'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store'});
      response.end(`<!doctype html><html lang="en"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Viewer fixture</title>
        <link rel="stylesheet" href="${fixtureAssets.css}">
        ${url.searchParams.has('unsupported') ? '<script>window.HTMLDialogElement=undefined</script>' : ''}
        <script defer src="${fixtureAssets.script}"></script><body>
        <a class="post-image-link" href="/orca-avatar-64.webp" target="_blank" rel="noopener"><img data-zoomable src="/orca-avatar-64.webp" width="64" height="64" alt="Small avatar"></a>
        <a id="author-image" href="https://example.com/paper"><img data-zoomable src="/orca-avatar-64.webp" width="64" height="64" alt="Author link"></a>`);
      return;
    }
    let file = path.resolve(publicRoot, '.' + route);
    if (!file.startsWith(publicRoot + path.sep) && file !== publicRoot) {response.writeHead(403).end();return;}
    if (route.endsWith('/')) file = path.join(file, 'index.html');
    requests.push(route);
    const send = async () => {
      if (response.destroyed) return;
      if (fail.has(route)) {response.writeHead(503).end('Unavailable');return;}
      const body = await fs.promises.readFile(file);
      response.writeHead(200, {'Content-Type': mime[path.extname(file)] || 'application/octet-stream', 'Cache-Control':'no-store'});
      response.end(body);
    };
    if (hold.has(route)) {
      const item = {route, send};pending.add(item);
      response.on('close', () => pending.delete(item));
    } else await send();
  } catch (_) {if (!response.destroyed) response.writeHead(404).end();}
});

(async () => {
  fs.mkdirSync(output,{recursive:true});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = 'http://127.0.0.1:' + server.address().port;
  const browser = await connect(port), report = {layouts:[],exceptions:browser.exceptions};
  const wait = async expression => {
    const until=Date.now()+10000;
    while (Date.now()<until) {if(await browser.evaluate(expression))return;await browser.pause(40);}
    throw Error('Condition timed out: '+expression);
  };
  const load = async () => {
    await browser.navigate(base + '/posts/ai/supermimic/');
    await browser.evaluate('document.fonts.ready');
    return browser.evaluate(`(async()=>{
      const image=document.querySelector('.post-image-link img');image.loading='eager';
      await image.decode();image.scrollIntoView({block:'center'});image.closest('a').focus();
      return {original:new URL(image.dataset.zoomSrc,location.href).pathname,preview:new URL(image.currentSrc).pathname};
    })()`);
  };
  const open = async () => {
    await browser.evaluate(`document.querySelector('.post-image-link').click()`);
    assert(await browser.evaluate(`document.querySelector('[data-image-viewer]').open`));
  };
  const closed = async () => {
    await wait(`!document.querySelector('[data-image-viewer]').open`);
    assert(await browser.evaluate(`document.activeElement.classList.contains('post-image-link')`));
    assert.equal(await browser.evaluate('document.documentElement.style.overflow'),'');
  };
  const release = async route => {
    hold.delete(route);
    await Promise.all([...pending].filter(item=>item.route===route).map(item=>item.send()));
  };
  try {
    await browser.cdp('Network.setCacheDisabled',{cacheDisabled:true});
    await browser.viewport(1440);
    let source = await load();
    fixtureAssets=await browser.evaluate(`({css:document.querySelector('link[rel=stylesheet][href*="/css/compiled/main"]').href,
      script:document.querySelector('script[src*="/js/image-viewer"]').src})`);
    assert.notEqual(source.preview,source.original);
    assert(!requests.includes(source.original),'Original was requested before opening the viewer');
    hold.add(source.original);
    await open();
    await wait(`document.querySelector('[data-image-viewer]').dataset.previewReady==='true'`);
    assert.equal(await browser.evaluate(`document.querySelector('[data-image-viewer]').dataset.state`),'loading');
    assert(await browser.evaluate(`document.querySelector('[data-image-viewer] img').currentSrc.endsWith(${JSON.stringify(source.preview)})`));
    // A missing original cannot keep Escape or focus restoration waiting.
    await browser.key('Escape','Escape',27);await closed();report.closeDuringDownload=true;
    await release(source.original);
    await browser.pause(80);
    assert(!await browser.evaluate(`document.querySelector('[data-image-viewer]').open`));

    source=await load();hold.add(source.original);await open();
    await wait(`document.querySelector('[data-image-viewer]').dataset.previewReady==='true'`);
    const before=await browser.evaluate(`(()=>{const r=document.querySelector('[data-image-viewer] img').getBoundingClientRect();return {width:r.width,height:r.height}})()`);
    await release(source.original);
    await wait(`document.querySelector('[data-image-viewer]').dataset.state==='ready'`);
    const after=await browser.evaluate(`(()=>{const i=document.querySelector('[data-image-viewer] img'),r=i.getBoundingClientRect();return {width:r.width,height:r.height,source:new URL(i.currentSrc).pathname}})()`);
    assert.equal(after.source,source.original);
    assert(Math.abs(before.width-after.width)<1&&Math.abs(before.height-after.height)<1);
    report.upgrade={before,after};
    await browser.evaluate(`document.querySelector('[data-image-viewer-close]').click()`);await closed();

    source=await load();fail.add(source.original);await open();
    await wait(`document.querySelector('[data-image-viewer]').dataset.state==='error'`);
    assert(await browser.evaluate(`document.querySelector('[data-image-viewer] img').naturalWidth>0`));
    assert(await browser.evaluate(`document.querySelector('[data-image-viewer-original]').getAttribute('target')==='_blank'`));
    await browser.key('Escape','Escape',27);await closed();fail.clear();
    await open();await wait(`document.querySelector('[data-image-viewer]').dataset.state==='ready'`);
    await browser.key('Escape','Escape',27);await closed();report.failureAndRetry=true;

    // Queued close/load events from an earlier session must not alter a newer one.
    await load();
    const switching=await browser.evaluate(`(async()=>{
      const images=[...document.querySelectorAll('.post-image-link img[data-zoom-src]')];
      await Promise.all(images.map(i=>{i.loading='eager';return i.decode()}));
      window.__switchSources=images.filter(i=>new URL(i.currentSrc).href!==new URL(i.dataset.zoomSrc,document.baseURI).href).slice(0,2);
      return window.__switchSources.map(i=>({original:new URL(i.dataset.zoomSrc,document.baseURI).pathname,preview:new URL(i.currentSrc).pathname}));
    })()`);
    assert.equal(switching.length,2);
    switching.forEach(source=>hold.add(source.original));
    await browser.evaluate(`(()=>{for(let i=0;i<20;i++)window.__switchSources[i%2].closest('a').click()})()`);
    await wait(`document.querySelector('[data-image-viewer]')?.dataset.previewReady==='true'`);
    assert(await browser.evaluate(`document.querySelector('[data-image-viewer]').open&&document.querySelector('[data-image-viewer]').dataset.state==='loading'
      &&new URL(document.querySelector('[data-image-viewer] img').currentSrc).pathname===${JSON.stringify(switching[1].preview)}`));
    await release(switching[0].original);await browser.pause(80);
    assert(await browser.evaluate(`document.querySelector('[data-image-viewer]').open&&document.querySelector('[data-image-viewer]').dataset.state==='loading'`));
    await release(switching[1].original);await wait(`document.querySelector('[data-image-viewer]').dataset.state==='ready'`);
    assert(await browser.evaluate(`new URL(document.querySelector('[data-image-viewer] img').currentSrc).pathname===${JSON.stringify(switching[1].original)}`));
    await browser.key('Escape','Escape',27);await closed();
    assert(await browser.evaluate(`document.activeElement===window.__switchSources[1].closest('a')`));
    report.rapidSessionSwitches=20;

    const modifier = await browser.evaluate(`(()=>{
      const link=document.querySelector('.post-image-link');let cancelled;
      document.addEventListener('click',e=>{cancelled=e.defaultPrevented;e.preventDefault()},{once:true});
      link.dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true,ctrlKey:true,button:0}));
      return {cancelled,open:document.querySelector('[data-image-viewer]').open};
    })()`);
    assert.deepEqual(modifier,{cancelled:false,open:false});report.modifierClick=true;

    for (const width of [320,390,768,1024,1440]) for (const theme of ['light','dark','warm']) {
      await browser.viewport(width);await load();
      await browser.evaluate(`document.querySelector('[data-item=${theme}]').click();document.querySelector('.post-image-link').focus()`);
      await browser.key('Enter','Enter',13);
      await wait(`document.querySelector('[data-image-viewer]').open`);
      await wait(`getComputedStyle(document.querySelector('[data-image-viewer]')).opacity==='1'`);
      const state=await browser.evaluate(`(()=>{const d=document.querySelector('[data-image-viewer]'),r=d.getBoundingClientRect();return {
        modal:d.matches(':modal'),focused:d.contains(document.activeElement),left:r.left,right:r.right,top:r.top,bottom:r.bottom,
        viewport:document.documentElement.clientWidth,height:innerHeight,overflow:d.scrollWidth>d.clientWidth+1
      }})()`);
      assert(state.modal&&state.focused&&!state.overflow&&state.left>=-1&&state.right<=state.viewport+1&&state.top>=-1&&state.bottom<=state.height+1,JSON.stringify({width,theme,...state}));
      for(let i=0;i<4;i++){await browser.key('Tab','Tab',9);assert(await browser.evaluate(`document.querySelector('[data-image-viewer]').contains(document.activeElement)`));}
      const contrast=await browser.evaluate('('+contrastAudit.toString()+')('+JSON.stringify('[data-image-viewer]')+')');
      assert(contrast.checked>0);
      assert.equal(contrast.issues.length,0,JSON.stringify(contrast.issues));
      report.layouts.push({width,theme,...state,contrastSamples:contrast.checked});
      if(width===390&&theme==='dark')await browser.screenshot(path.join(output,'mobile-dark.png'));
      await browser.key('Escape','Escape',27);await closed();
    }
    await browser.viewport(320);await load();
    await browser.evaluate(`document.documentElement.style.fontSize='200%'`);await open();
    assert(await browser.evaluate(`(()=>{const d=document.querySelector('[data-image-viewer]'),r=d.getBoundingClientRect();return d.scrollWidth<=d.clientWidth+1&&r.right<=innerWidth+1})()`));
    await browser.key('Escape','Escape',27);await closed();report.text200=true;
    await browser.cdp('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});
    await open();assert.equal(await browser.evaluate(`getComputedStyle(document.querySelector('[data-image-viewer]')).animationName`),'none');
    await browser.key('Escape','Escape',27);await closed();report.reducedMotion=true;
    await browser.cdp('Emulation.setEmulatedMedia',{features:[]});

    await browser.viewport(1440);await browser.navigate(base+'/image-viewer-fixture/');
    await open();await wait(`document.querySelector('[data-image-viewer]').dataset.state==='ready'`);
    const small=await browser.evaluate(`(()=>{const d=document.querySelector('[data-image-viewer]'),r=d.querySelector('img').getBoundingClientRect();return {
      title:d.querySelector('h2').textContent,width:r.width,height:r.height,source:d.querySelector('img').currentSrc};})()`);
    assert.equal(small.title,'Image viewer');assert(small.width<=64.5&&small.height<=64.5);
    await browser.cdp('Input.dispatchMouseEvent',{type:'mousePressed',x:5,y:5,button:'left',clickCount:1});
    await browser.cdp('Input.dispatchMouseEvent',{type:'mouseReleased',x:5,y:5,button:'left',clickCount:1});
    await closed();report.smallImageAndBackdrop=small;
    assert(await browser.evaluate(`(()=>{let cancelled;document.addEventListener('click',e=>{cancelled=e.defaultPrevented;e.preventDefault()},{once:true});
      document.querySelector('#author-image').click();return !cancelled&&!document.querySelector('[data-image-viewer]').open})()`));
    report.authorLink=true;
    await browser.navigate(base+'/image-viewer-fixture/?unsupported=1');
    assert(await browser.evaluate(`!document.querySelector('[data-image-viewer]')&&document.querySelector('.post-image-link').href.endsWith('/orca-avatar-64.webp')`));
    report.unsupportedFallback=true;

    await browser.viewport(320,320);source=await load();fail.add(source.original);
    await browser.evaluate(`document.documentElement.style.fontSize='200%'`);await open();
    await wait(`document.querySelector('[data-image-viewer]').dataset.state==='error'`);
    const short=await browser.evaluate(`(()=>{const d=document.querySelector('[data-image-viewer]'),s=d.querySelector('[role=status]');d.scrollTop=d.scrollHeight;
      return {height:d.clientHeight,scroll:d.scrollHeight,overflow:getComputedStyle(d).overflowY,
        imageHeight:d.querySelector('[data-image-viewer-stage]').clientHeight,
        statusBottom:s.getBoundingClientRect().bottom,bottom:d.getBoundingClientRect().bottom};})()`);
    assert(short.scroll<=short.height+1||['auto','scroll'].includes(short.overflow));
    assert(short.imageHeight>0);
    assert(short.statusBottom<=short.bottom+1,'Status cannot be reached in short landscape: '+JSON.stringify(short));
    await browser.key('Escape','Escape',27);await closed();fail.clear();report.shortLandscape=short;

    await browser.viewport(716);source=await load();hold.add(source.original);await open();
    await browser.cdp('Emulation.setEmulatedMedia',{media:'print'});
    await wait(`!document.querySelector('[data-image-viewer]').open`);
    assert(await browser.evaluate(`getComputedStyle(document.querySelector('.post-image-link img')).visibility==='visible'`));
    const pdf=await browser.cdp('Page.printToPDF',{printBackground:true,paperWidth:8.27,paperHeight:11.69,marginTop:.4,marginBottom:.4,marginLeft:.4,marginRight:.4});
    fs.writeFileSync(path.join(output,'print-while-loading.pdf'),Buffer.from(pdf.data,'base64'));
    await browser.cdp('Emulation.setEmulatedMedia',{media:''});report.printDuringDownload=true;
    assert.equal(browser.exceptions.length,0,JSON.stringify(browser.exceptions));
    console.log(JSON.stringify({layouts:report.layouts.length,closeDuringDownload:report.closeDuringDownload,failureAndRetry:report.failureAndRetry,
      modifierClick:report.modifierClick,rapidSessionSwitches:report.rapidSessionSwitches,text200:report.text200,reducedMotion:report.reducedMotion,printDuringDownload:report.printDuringDownload,exceptions:browser.exceptions.length},null,2));
  } finally {
    fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(report,null,2)+'\n');
    await browser.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
  }
})().catch(error=>{console.error(error);process.exitCode=1});
