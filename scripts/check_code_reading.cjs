#!/usr/bin/env node
// Dynamic text/disclosure sizing and asynchronous keyboard copying.
// Run serially with other CDP browser checks.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('./lib/browser.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13163';
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-code-reading';
(async()=>{
  fs.mkdirSync(output,{recursive:true});
  const browser=await connect(port),report={layouts:[],exceptions:browser.exceptions};
  const inspect=()=>browser.evaluate(`([...document.querySelectorAll('.content pre')].filter(p=>p.clientWidth>0).map((p,index)=>({
    index,horizontal:p.scrollWidth>p.clientWidth+1,vertical:p.scrollHeight>p.clientHeight+1,tabIndex:p.getAttribute('tabindex')})))`);
  try{
    await browser.viewport(1440);
    for(const route of ['/posts/python/tqdm/','/posts/ai/supermimic/','/posts/cpp/gcc/']){
      await browser.navigate(base+route);await browser.evaluate('document.fonts.ready');
      const before=await inspect();
      for(const size of ['200%','100%']){
        await browser.evaluate(`document.documentElement.style.fontSize=${JSON.stringify(size)}`);await browser.pause(200);
        const blocks=await inspect();
        report.layouts.push({route,size,blocks});
        assert(blocks.every(p=>!(p.horizontal||p.vertical)||p.tabIndex==='0'),JSON.stringify({route,size,blocks}));
      }
      await browser.evaluate(`document.querySelectorAll('.content details').forEach(d=>d.open=true)`);await browser.pause(200);
      const blocks=await inspect();report.layouts.push({route,disclosures:'open',blocks});
      assert(blocks.every(p=>!(p.horizontal||p.vertical)||p.tabIndex==='0'),JSON.stringify({route,blocks}));
      report.layouts.at(-1).initialBlocks=before.length;
    }
    for(const [route,width] of [['/posts/cuda/warp/',768],['/posts/cuda/warp/',1440],['/posts/open3d/introduction/',1440]]){
      await browser.viewport(width);await browser.navigate(base+route);await browser.evaluate('document.fonts.ready');
      await browser.evaluate(`const spacing=document.createElement('style');spacing.textContent=
        'body :is(p,li,dt,dd,th,td,summary,h1,h2,h3,h4,h5,h6,a,button,label,input,select){line-height:1.5!important;letter-spacing:0.12em!important;word-spacing:0.16em!important} main p{margin-bottom:2em!important}';document.head.append(spacing)`);
      await browser.pause(200);const blocks=await inspect();report.layouts.push({route,width,textSpacing:true,blocks});
      assert(blocks.every(p=>!(p.horizontal||p.vertical)||p.tabIndex==='0'),JSON.stringify({route,width,blocks}));
    }
    await browser.viewport(1440);
    await browser.navigate(base+'/posts/ai/supermimic/');
    await browser.evaluate(`(()=>{
      const pre=document.querySelector('.content pre'),block=pre.closest('.hextra-code-block');
      const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent='Collapsed example';
      block.before(details);details.append(summary,block);window.__codeDisclosure=details;window.__disclosedPre=pre;
    })()`);await browser.pause(200);
    assert(await browser.evaluate(`(()=>{window.__disclosedPre.focus();return !window.__disclosedPre.checkVisibility()&&document.activeElement!==window.__disclosedPre})()`));
    await browser.evaluate(`document.documentElement.style.fontSize='200%'`);
    await browser.evaluate('window.__codeDisclosure.open=true');await browser.pause(200);
    assert(await browser.evaluate(`window.__disclosedPre.clientWidth>0&&window.__disclosedPre.tabIndex===0`));
    report.controlledDisclosure=true;
    await browser.viewport(390);await browser.navigate(base+'/posts/python/tqdm/');
    await browser.evaluate(`(()=>{
      window.__copyCalls=[];
      Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:text=>{
        window.__copyCalls.push(text);return new Promise(resolve=>{window.__finishCopy=resolve});
      }}});
      const button=document.querySelector('.hextra-code-copy-btn');button.scrollIntoView({block:'center'});button.focus();
    })()`);
    await browser.key('Enter','Enter',13);await browser.pause(150);
    const pending=await browser.evaluate(`({focused:document.activeElement===document.querySelector('.hextra-code-copy-btn'),
      calls:window.__copyCalls.length,disabled:document.querySelector('.hextra-code-copy-btn').disabled,
      busy:document.querySelector('.hextra-code-copy-btn').getAttribute('aria-busy'),
      unavailable:document.querySelector('.hextra-code-copy-btn').getAttribute('aria-disabled')})`);
    report.pendingCopy=pending;
    assert(pending.focused&&pending.calls===1&&!pending.disabled&&pending.busy==='true'&&pending.unavailable==='true',JSON.stringify(pending));
    await browser.key('Enter','Enter',13);assert.equal(await browser.evaluate('window.__copyCalls.length'),1);
    await browser.evaluate('window.__finishCopy()');await browser.pause(50);
    assert(await browser.evaluate(`document.activeElement===document.querySelector('.hextra-code-copy-btn')
      && !document.querySelector('.hextra-code-copy-btn').hasAttribute('aria-busy')
      && !document.querySelector('.hextra-code-copy-btn').hasAttribute('aria-disabled')`));
    assert(await browser.evaluate(`window.__copyCalls[0]===document.querySelector('.hextra-code-block code').textContent`));
    report.keyboardCopy=true;
    // Finishing an asynchronous copy must not steal focus after the reader tabs away.
    await browser.key('Enter','Enter',13);await browser.key('Tab','Tab',9);
    await browser.evaluate('window.__focusAfterTab=document.activeElement;window.__finishCopy()');await browser.pause(50);
    assert(await browser.evaluate('document.activeElement===window.__focusAfterTab'));
    report.pendingTab=true;

    // A late permission rejection must not pull the reader back to the code
    // or replace a selection made after leaving the pending copy button.
    await browser.navigate(base+'/posts/python/tqdm/');
    await browser.evaluate(`(()=>{
      Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:()=>
        new Promise((resolve,reject)=>{window.__rejectCopy=reject})}});
      const button=document.querySelector('.hextra-code-copy-btn');
      button.scrollIntoView({block:'center'});button.focus();
    })()`);
    await browser.key('Enter','Enter',13);await browser.key('Tab','Tab',9);
    await browser.evaluate(`(()=>{
      window.__focusBeforeRejection=document.activeElement;
      const paragraph=document.querySelector('.content p'),range=document.createRange();
      range.selectNodeContents(paragraph);const selection=getSelection();
      selection.removeAllRanges();selection.addRange(range);
      window.__selectionBeforeRejection=selection.toString();
      window.__rejectCopy(new DOMException('Permission denied','NotAllowedError'));
    })()`);await browser.pause(100);
    const rejection=await browser.evaluate(`({
      focusPreserved:document.activeElement===window.__focusBeforeRejection,
      selectionPreserved:getSelection().toString()===window.__selectionBeforeRejection,
      buttonEnabled:!document.querySelector('.hextra-code-copy-btn').disabled,
      status:document.querySelector('.blog-code-copy-status').textContent
    })`);
    report.rejectionAfterTab=rejection;
    assert(rejection.focusPreserved&&rejection.selectionPreserved&&rejection.buttonEnabled,JSON.stringify(rejection));
    assert(!rejection.status.includes('已选中'));

    // If the reader stays, permission denial still offers exact manual copying.
    await browser.evaluate(`getSelection().removeAllRanges();document.querySelector('.hextra-code-copy-btn').focus()`);
    await browser.key('Enter','Enter',13);
    await browser.evaluate(`window.__rejectCopy(new DOMException('Permission denied','NotAllowedError'))`);
    await browser.pause(100);
    const manual=await browser.evaluate(`(()=>{
      const pre=document.querySelector('.content pre'),code=pre.querySelector('code'),button=document.querySelector('.hextra-code-copy-btn');
      return {focused:document.activeElement===pre,exact:getSelection().toString()===code.textContent,
        busy:button.hasAttribute('aria-busy'),unavailable:button.hasAttribute('aria-disabled'),
        status:document.querySelector('.blog-code-copy-status').textContent};
    })()`);
    report.permissionFallback=manual;
    assert(manual.focused&&manual.exact&&!manual.busy&&!manual.unavailable&&manual.status.includes('已选中'),JSON.stringify(manual));

    // Reading or assistive selection can change without moving DOM focus.
    await browser.evaluate(`getSelection().removeAllRanges();document.querySelector('.hextra-code-copy-btn').focus()`);
    await browser.key('Enter','Enter',13);
    await browser.evaluate(`(()=>{const range=document.createRange();range.selectNodeContents(document.querySelector('.content p'));
      getSelection().removeAllRanges();getSelection().addRange(range);window.__newSelection=getSelection().toString();
      window.__rejectCopy(new DOMException('Permission denied','NotAllowedError'));})()`);
    await browser.pause(100);
    assert(await browser.evaluate(`document.activeElement===document.querySelector('.hextra-code-copy-btn')&&getSelection().toString()===window.__newSelection`));
    report.changedSelectionPreserved=true;

    // A permission failure in a background tab must not select hidden content.
    await browser.evaluate(`getSelection().removeAllRanges();document.querySelector('.hextra-code-copy-btn').focus()`);
    await browser.key('Enter','Enter',13);
    const otherTab=await connect(port);
    try{
      assert(await browser.evaluate('!document.hasFocus()'));
      await browser.evaluate(`window.__rejectCopy(new DOMException('Permission denied','NotAllowedError'))`);
      await browser.pause(100);
      assert(await browser.evaluate(`getSelection().rangeCount===0&&document.activeElement===document.querySelector('.hextra-code-copy-btn')`));
      report.backgroundFailurePreserved=true;
    }finally{await otherTab.close();await browser.cdp('Page.bringToFront')}

    // The unavailable-API fallback also works for a short, non-scrolling block.
    await browser.viewport(1440);await browser.navigate(base+'/posts/process/pid/');
    await browser.evaluate(`(()=>{Object.defineProperty(navigator,'clipboard',{configurable:true,value:undefined});
      getSelection().removeAllRanges();document.querySelector('.hextra-code-copy-btn').focus();})()`);
    const shortBlock=await browser.evaluate(`(()=>{const p=document.querySelector('.content pre');return p.scrollWidth<=p.clientWidth&&p.scrollHeight<=p.clientHeight})()`);
    assert(shortBlock);
    await browser.key('Enter','Enter',13);await browser.pause(200);
    const unavailable=await browser.evaluate(`(()=>{const p=document.querySelector('.content pre');return {
      focused:document.activeElement===p,exact:getSelection().toString()===p.querySelector('code').textContent};})()`);
    assert(unavailable.focused&&unavailable.exact,JSON.stringify(unavailable));report.unavailableClipboard=unavailable;
    await browser.key('Tab','Tab',9);await browser.pause(150);
    assert(await browser.evaluate(`!document.querySelector('.content pre').hasAttribute('tabindex')`));
    report.shortBlockFocusCleanup=true;
    assert.equal(browser.exceptions.length,0,JSON.stringify(browser.exceptions));
    console.log(JSON.stringify({layouts:report.layouts.length,keyboardCopy:true,pendingTab:true,controlledDisclosure:true,exceptions:0},null,2));
  }finally{
    fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(report,null,2)+'\n');await browser.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1});
