#!/usr/bin/env node
// Controlled data failures and recovery in article labs; run browser checks serially.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const {connect} = require('./lib/browser.cjs');
const {contrastAudit} = require('./lib/contrast.cjs');
const base = (process.argv[2] || 'http://127.0.0.1:13311').replace(/\/$/, '');
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-lab-loading';
assert(['127.0.0.1', 'localhost', '[::1]'].includes(new URL(base).hostname), 'Use a loopback preview');
const labs = [
  {name:'filter',route:'/posts/robotics/control/robot-filters/',root:'[data-robot-filter-lab]',
    interactive:'[data-filter-interactive]',token:'filter-signal',field:'method'},
  {name:'routing',route:'/posts/ai/jev-decision-layer/',root:'[data-jev-routing-lab]',
    interactive:'[data-routing-interactive]',token:'routing-fixtures',field:'group'}
];

(async () => {
  fs.mkdirSync(output, {recursive:true});
  const b = await connect(port), rows = [], baselines = {};
  let fixture, server;
  const release = () => fixture?.gates.splice(0).forEach(resolve => resolve());
  const waitFor = async (expression, milliseconds = 15000) => {
    const deadline = Date.now()+milliseconds;
    while (Date.now()<deadline) {if (await b.evaluate(expression)) return;await b.pause(40);}
    throw Error('Timed out: '+expression);
  };
  const waitState = (lab, state, timeout) => waitFor(`document.querySelector(${JSON.stringify(lab.root)})?.dataset.loadState===${JSON.stringify(state)}`, timeout);
  const focusRetry = async lab => {
    await b.key('Tab','Tab',9);
    await b.evaluate(`(() => {const r=document.querySelector(${JSON.stringify(lab.root)}).querySelector('[data-lab-retry]');r.focus({preventScroll:true});r.scrollIntoView({block:'center'});})()`);
  };
  const snapshot = lab => b.evaluate(`(() => {
    const r=document.querySelector(${JSON.stringify(lab.root)});
    return {metrics:[...r.querySelectorAll('[data-metric],[data-routing-metric]')].map(e=>e.textContent),
      fields:[...r.querySelector('form').elements].filter(e=>e.name).map(e=>[e.name,e.value]),
      listeners:window.__labListeners[${JSON.stringify(lab.name)}]||[]};
  })()`);
  try {
    await b.cdp('Network.setCacheDisabled', {cacheDisabled:true});
    await b.cdp('Page.addScriptToEvaluateOnNewDocument', {source:`
      window.__labListeners={};const original=EventTarget.prototype.addEventListener;
      EventTarget.prototype.addEventListener=function(type,...args){
        if(this instanceof HTMLFormElement){const name=this.closest('[data-robot-filter-lab]')?'filter':this.closest('[data-jev-routing-lab]')?'routing':null;
          if(name)(window.__labListeners[name]??=[]).push(type);}
        return original.call(this,type,...args);
      };`});
    server = http.createServer(async (request,response) => {
      try {
        const target = new URL(request.url,base);
        if(target.origin!==new URL(base).origin){response.writeHead(403).end();return;}
        const current = fixture;
        if(current && target.pathname.endsWith('.json') && target.pathname.includes(current.lab.token)) {
          const index=current.requests++;
          const action=current.responses[index]||'valid';
          if(action==='hold')await new Promise(resolve=>current.gates.push(resolve));
          if(response.destroyed)return;
          if(action==='http'){response.writeHead(503,{'Content-Type':'application/json'}).end('{"error":"fixture unavailable"}');return;}
          if(action==='json'){response.writeHead(200,{'Content-Type':'application/json'}).end('{broken');return;}
          if(action==='schema'){response.writeHead(200,{'Content-Type':'application/json'}).end('{"records":[]}');return;}
        }
        if(response.destroyed)return;
        const upstream=await fetch(target),body=Buffer.from(await upstream.arrayBuffer());
        if(!response.destroyed)response.writeHead(upstream.status,{'Content-Type':upstream.headers.get('content-type')||'application/octet-stream'}).end(body);
      }catch(error){if(!response.destroyed)response.writeHead(502).end(String(error));}
    });
    await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
    const proxy='http://127.0.0.1:'+server.address().port;
    const load = async (lab,responses,{width=320,scale=1,theme='light'}={}) => {
      release();fixture={lab,responses,requests:0,gates:[]};
      await b.viewport(width);
      await b.navigate('about:blank');await b.navigate(proxy+lab.route);
      await b.evaluate(`document.documentElement.classList.remove('dark','warm');document.documentElement.classList.add(${JSON.stringify(theme)});
        document.documentElement.style.fontSize=${JSON.stringify(scale*100+'%')};document.fonts.ready`);
      return fixture;
    };
    for(const lab of labs){await load(lab,['valid']);await waitState(lab,'ready');baselines[lab.name]=await snapshot(lab);rows.push({lab:lab.name,mode:'initial',passed:true});}
    for(const lab of labs)for(const width of [320,1440])for(const scale of [1,2])for(const theme of ['light','dark','warm'])for(const failure of ['http','json','schema']){
      const plan=await load(lab,[failure,'valid'],{width,scale,theme});await waitState(lab,'error');await focusRetry(lab);
      const state=await b.evaluate(`(() => {
        const r=document.querySelector(${JSON.stringify(lab.root)}),button=r.querySelector('[data-lab-retry]'),s=getComputedStyle(button),box=button.getBoundingClientRect();
        const canvas=document.createElement('canvas');canvas.width=canvas.height=1;const ctx=canvas.getContext('2d');
        const rgb=color=>{ctx.clearRect(0,0,1,1);ctx.fillStyle=color;ctx.fillRect(0,0,1,1);return [...ctx.getImageData(0,0,1,1).data].slice(0,3).map(x=>x/255)};
        const lum=c=>c.map(x=>x<=.04045?x/12.92:((x+.055)/1.055)**2.4).reduce((sum,x,i)=>sum+x*[.2126,.7152,.0722][i],0);
        const contrast=color=>{const a=lum(rgb(color)),z=lum(rgb(getComputedStyle(r).backgroundColor));return(Math.max(a,z)+.05)/(Math.min(a,z)+.05)};
        return {visible:button.checkVisibility(),disabled:button.disabled,ariaDisabled:button.getAttribute('aria-disabled'),
          hidden:r.querySelector(${JSON.stringify(lab.interactive)}).hidden,height:box.height,outline:s.outlineStyle,
          focused:document.activeElement===button,focusVisible:button.matches(':focus-visible'),outlineContrast:contrast(s.outlineColor),borderContrast:contrast(s.borderColor),
          overflow:document.documentElement.scrollWidth>innerWidth+1,described:!!document.getElementById(button.getAttribute('aria-describedby'))};
      })()`);
      assert(state.visible&&!state.disabled&&state.ariaDisabled==='false'&&state.hidden&&state.height>=44&&state.focused&&state.focusVisible&&!state.overflow&&state.described,JSON.stringify({lab:lab.name,width,scale,theme,failure,state}));
      assert(state.outline!=='none'&&state.outlineContrast>=3&&state.borderContrast>=3,JSON.stringify(state));
      const audit=await b.evaluate('('+contrastAudit.toString()+')('+JSON.stringify(lab.root)+')');assert.equal(audit.issues.length,0,JSON.stringify(audit.issues));
      if(width===320&&scale===1&&theme==='dark'&&failure==='http')await b.screenshot(path.join(output,lab.name+'-retry-dark.png'));
      await b.key('Enter','Enter',13);await waitState(lab,'ready');
      assert(await b.evaluate(`document.activeElement===document.querySelector(${JSON.stringify(lab.root)}).querySelector('[data-lab-retry]')`));
      await b.key('Enter','Enter',13);await b.pause(30);assert.equal(plan.requests,2,'Ready controls must not reload data');
      assert.deepEqual(await snapshot(lab),baselines[lab.name]);
      await b.key('Tab','Tab',9);assert.equal(await b.evaluate('document.activeElement.name'),lab.field);
      assert(await b.evaluate(`document.querySelector(${JSON.stringify(lab.root)}).querySelector('[data-lab-retry]').hidden`));
      rows.push({lab:lab.name,width,scale,theme,failure,requests:plan.requests,passed:true,...state});
    }
    for(const lab of labs)for(const action of ['retain','leave','background','chapter']){
      const plan=await load(lab,['http','hold']);await waitState(lab,'error');await focusRetry(lab);
      await b.key('Enter','Enter',13);await waitState(lab,'loading');
      while(plan.requests<2)await b.pause(20);
      const pending=await b.evaluate(`(() => {const r=document.querySelector(${JSON.stringify(lab.root)}),x=r.querySelector('[data-lab-retry]');return{focused:document.activeElement===x,disabled:x.disabled,aria:x.getAttribute('aria-disabled'),busy:r.getAttribute('aria-busy')}})()`);
      assert.deepEqual(pending,{focused:true,disabled:false,aria:'true',busy:'true'});
      await b.key('Enter','Enter',13);await b.key(' ','Space',32);await b.pause(30);assert.equal(plan.requests,2,'Ignore duplicate pending activation');
      let other,focusHTML,chapter;
      if(action==='leave'){
        focusHTML=await b.evaluate("(()=>{const a=document.querySelector('.hextra-nav-container a');a.focus({preventScroll:true});return a.outerHTML})()");
      }else if(action==='background'){
        other=await connect(port);assert.equal(await b.evaluate('document.hasFocus()'),false);
      }else if(action==='chapter'){
        chapter=await b.evaluate(`(()=>{const r=document.querySelector(${JSON.stringify(lab.root)}),h=[...document.querySelectorAll('.content h2,.content h3')].find(h=>Boolean(r.compareDocumentPosition(h)&Node.DOCUMENT_POSITION_FOLLOWING));h.tabIndex=-1;h.focus({preventScroll:true});h.scrollIntoView({block:'start'});window.__loadingChapter=h;return{html:h.outerHTML,top:h.getBoundingClientRect().top}})()`);
      }
      release();await waitState(lab,'ready');await b.pause(120);
      assert.deepEqual(await snapshot(lab),baselines[lab.name]);
      if(action==='retain')assert(await b.evaluate(`document.activeElement===document.querySelector(${JSON.stringify(lab.root)}).querySelector('[data-lab-retry]')`));
      if(action==='leave')assert.equal(await b.evaluate('document.activeElement.outerHTML'),focusHTML);
      if(action==='background'){assert.equal(await b.evaluate('document.hasFocus()'),false);await other.close();await b.cdp('Page.bringToFront');}
      if(action==='chapter'){
        const actual=await b.evaluate("({html:document.activeElement.outerHTML,top:window.__loadingChapter.getBoundingClientRect().top})");
        assert.equal(actual.html,chapter.html);assert(Math.abs(actual.top-chapter.top)<=2,JSON.stringify({lab:lab.name,chapter,actual}));
      }
      rows.push({lab:lab.name,mode:'pending',action,requests:plan.requests,passed:true});
    }
    for(const lab of labs){
      const plan=await load(lab,['hold','valid']);await waitState(lab,'error',21000);await focusRetry(lab);await b.key('Enter','Enter',13);await waitState(lab,'ready');release();await b.pause(100);assert.equal(plan.requests,2);assert.deepEqual(await snapshot(lab),baselines[lab.name]);rows.push({lab:lab.name,mode:'timeout-and-retry',passed:true});
    }
    for(const lab of labs){
      const plan=await load(lab,['http','json','valid']);await waitState(lab,'error');await focusRetry(lab);await b.key('Enter','Enter',13);await waitState(lab,'error');await b.key('Enter','Enter',13);await waitState(lab,'ready');assert.equal(plan.requests,3);assert.deepEqual(await snapshot(lab),baselines[lab.name]);rows.push({lab:lab.name,mode:'repeated-error',passed:true});
    }
    await b.cdp('Emulation.setScriptExecutionDisabled',{value:true});
    for(const lab of labs){await load(lab,['valid']);assert(await b.evaluate(`(()=>{const r=document.querySelector(${JSON.stringify(lab.root)});return r.querySelector('[data-lab-retry]').hidden&&r.querySelector(${JSON.stringify(lab.interactive)}).hidden&&r.querySelector('noscript').textContent.includes('静态')})()`));rows.push({lab:lab.name,mode:'scripts-off',passed:true});}
    await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});
    // A retry can obtain valid data on a browser lacking a setup dependency.
    // Keep that failure's focused control until blur rather than hiding it.
    const noResize=await b.cdp('Page.addScriptToEvaluateOnNewDocument',{source:'window.ResizeObserver=undefined'});
    const filter=labs[0],failedSetup=await load(filter,['http','valid']);
    await waitState(filter,'error');await focusRetry(filter);await b.key('Enter','Enter',13);
    while(failedSetup.requests<2)await b.pause(20);
    await waitState(filter,'error');
    assert(await b.evaluate(`(() => {const r=document.querySelector(${JSON.stringify(filter.root)}),x=r.querySelector('[data-lab-retry]');return document.activeElement===x&&!x.hidden&&!x.disabled&&x.getAttribute('aria-disabled')==='true'&&r.querySelector(${JSON.stringify(filter.interactive)}).hidden})()`));
    await b.key('Enter','Enter',13);assert.equal(failedSetup.requests,2,'Do not repeat partial activation');
    await b.key('Tab','Tab',9);assert(await b.evaluate(`document.querySelector(${JSON.stringify(filter.root)}).querySelector('[data-lab-retry]').hidden`));
    await b.cdp('Page.removeScriptToEvaluateOnNewDocument',{identifier:noResize.identifier});
    rows.push({lab:filter.name,mode:'activation-failure-keeps-focus',passed:true});
    assert.equal(b.exceptions.length,0,JSON.stringify(b.exceptions));
    console.log(JSON.stringify({cases:rows.length,passed:true,output}));
  }finally{
    release();fs.writeFileSync(path.join(output,'results.json'),JSON.stringify({cases:rows,exceptions:b.exceptions},null,2)+'\n');
    await b.close();if(server)await new Promise(resolve=>{server.close(resolve);server.closeAllConnections();});
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
