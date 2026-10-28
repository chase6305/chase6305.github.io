#!/usr/bin/env node
// Search progressively enhances a native link, including when its UI cannot run.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('./lib/browser.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13143';
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-search-fallback';

(async () => {
  fs.mkdirSync(output, {recursive:true});
  const b = await connect(port);
  const report = {cases:[], modifiedClicks:[], exceptions:b.exceptions};
  let savedTheme, interceptor;
  const wait = async expression => {
    const deadline = Date.now() + 20000;
    while (Date.now() < deadline) {
      if (await b.evaluate(expression)) return;
      await b.pause(50);
    }
    throw Error('Condition timed out: '+expression);
  };
  try {
    await b.navigate(base+'/');
    savedTheme = await b.evaluate("localStorage.getItem('color-theme')");
    for (const route of ['/','/zh-cn/']) for (const width of [320,1440])
      for (const theme of ['light','dark','warm']) for (const mode of ['native','no-scripts','no-dialog','ui-blocked','engine-blocked']) {
        await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});
        await b.cdp('Network.setBlockedURLs',{urls:[]});
        if (interceptor) {
          await b.cdp('Page.removeScriptToEvaluateOnNewDocument',{identifier:interceptor});
          interceptor = undefined;
        }
        await b.evaluate(`localStorage.setItem('color-theme',${JSON.stringify(theme)})`);
        await b.viewport(width);
        if (mode==='no-scripts') await b.cdp('Emulation.setScriptExecutionDisabled',{value:true});
        if (mode==='no-dialog') ({identifier:interceptor}=await b.cdp('Page.addScriptToEvaluateOnNewDocument',{
          source:'HTMLDialogElement.prototype.showModal=undefined;'}));
        if (mode==='ui-blocked') await b.cdp('Network.setBlockedURLs',{urls:['*search-dialog*.js*']});
        if (mode==='engine-blocked') await b.cdp('Network.setBlockedURLs',{urls:['*.search.min.*.js*']});
        await b.navigate(base+route);
        const state = await b.evaluate(`(()=>{
          const a=document.querySelector('[data-search-open]'),r=a.getBoundingClientRect(),k=a.querySelector('kbd');
          return {tag:a.tagName,href:a.getAttribute('href'),popup:a.getAttribute('aria-haspopup'),
            controls:a.getAttribute('aria-controls'),shortcuts:a.getAttribute('aria-keyshortcuts'),
            name:a.getAttribute('aria-label'),hintPresent:!!k,
            hintVisible:!!k&&getComputedStyle(k).display!=='none',width:r.width,height:r.height,
            dialogVisible:getComputedStyle(document.querySelector('#hextra-search-dialog')).display!=='none',
            pageWidth:document.documentElement.scrollWidth};})()`);
        assert.equal(state.tag,'A');assert.equal(state.href,'/posts/');
        assert(state.name,'Search entry needs an accessible name');
        assert(state.width>=40 && state.height>=40);assert(state.pageWidth<=width+1);
        assert.equal(state.dialogVisible,false);
        assert.equal(state.popup,mode==='native'?'dialog':null);
        assert.equal(state.controls,mode==='native'?'hextra-search-dialog':null);
        assert.equal(state.shortcuts,mode==='native'?'Control+k Meta+k':null);
        assert.equal(state.hintVisible,state.hintPresent&&mode==='native'&&width>1024);
        await b.evaluate(`document.querySelector('[data-search-open]').focus()`);
        await b.key('Enter','Enter',13);
        if (mode==='native') {
          await wait(`document.querySelector('#hextra-search-dialog').open`);
          assert.equal(await b.evaluate('location.pathname'),route);
          await b.key('Escape','Escape',27);
          await wait(`!document.querySelector('#hextra-search-dialog').open`);
          assert(await b.evaluate(`document.activeElement===document.querySelector('[data-search-open]')`));
          await b.cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'k',code:'KeyK',windowsVirtualKeyCode:75,modifiers:2});
          await b.cdp('Input.dispatchKeyEvent',{type:'keyUp',key:'k',code:'KeyK',windowsVirtualKeyCode:75});
          await wait(`document.querySelector('#hextra-search-dialog').open`);
          await b.key('Escape','Escape',27);
          await wait(`!document.querySelector('#hextra-search-dialog').open`);
          await b.navigate(base+'/posts/');
        } else await wait(`location.pathname==='/posts/'&&document.readyState==='complete'`);
        const index = await b.evaluate(`({cards:document.querySelectorAll('[data-blog-original-list] .blog-card').length,
          fullHelpVisible:!document.querySelector('[data-search-full-help]').hidden,
          filterHidden:document.querySelector('[data-blog-filter]').hidden})`);
        assert(index.cards>0);assert.equal(index.fullHelpVisible,mode==='native');
        assert.equal(index.filterHidden,mode==='no-scripts');
        report.cases.push({route,width,theme,mode,...state,...index});
        if(route==='/zh-cn/'&&width===320&&theme==='light'&&mode==='no-scripts')
          await b.screenshot(path.join(output,'native-index-mobile.png'));
      }
    await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});
    await b.cdp('Network.setBlockedURLs',{urls:[]});
    if(interceptor) {
      await b.cdp('Page.removeScriptToEvaluateOnNewDocument',{identifier:interceptor});interceptor=undefined;
    }
    await b.navigate(base+'/');
    for (const options of [{ctrlKey:true},{metaKey:true},{shiftKey:true},{altKey:true},{button:1}]) {
      const state = await b.evaluate(`(()=>{
        const a=document.querySelector('[data-search-open]');let intercepted;
        document.addEventListener('click',event=>{intercepted=event.defaultPrevented;event.preventDefault()},{once:true});
        a.dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true,...${JSON.stringify(options)}}));
        return {intercepted,open:document.querySelector('#hextra-search-dialog').open};})()`);
      assert.equal(state.intercepted,false);assert.equal(state.open,false);
      report.modifiedClicks.push({options,...state});
    }
    assert.equal(b.exceptions.length,0,JSON.stringify(b.exceptions));
    report.summary={cases:report.cases.length,modifiedClicks:report.modifiedClicks.length,exceptions:b.exceptions.length};
    console.log(JSON.stringify(report.summary));
  } finally {
    await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});
    await b.cdp('Network.setBlockedURLs',{urls:[]}).catch(()=>{});
    if(interceptor) await b.cdp('Page.removeScriptToEvaluateOnNewDocument',{identifier:interceptor}).catch(()=>{});
    if(savedTheme!==undefined) await b.evaluate(savedTheme===null?"localStorage.removeItem('color-theme')":
      `localStorage.setItem('color-theme',${JSON.stringify(savedTheme)})`).catch(()=>{});
    fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(report,null,2)+'\n');
    await b.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1});
