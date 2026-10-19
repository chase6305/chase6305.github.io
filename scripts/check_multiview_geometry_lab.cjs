// Read-only UI checks; use a dedicated tab on an existing loopback CDP browser.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const base = process.argv[2] || 'http://127.0.0.1:1397';
const port = process.argv[3] || '9260';
const out = process.argv[4] || '/tmp/multiview-geometry-lab-review';
const route = '/labs/multiview-geometry/';
const pause = ms => new Promise(resolve => setTimeout(resolve,ms));
(async()=>{
  fs.mkdirSync(out,{recursive:true});
  const endpoint='http://127.0.0.1:'+port;
  const tab=await (await fetch(endpoint+'/json/new?about:blank',{method:'PUT'})).json();
  const ws=new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
  const pending=new Map(),exceptions=[],rows=[];let serial=0;
  ws.onmessage=event=>{
    const m=JSON.parse(event.data);
    if(m.id&&pending.has(m.id)){
      const p=pending.get(m.id);pending.delete(m.id);clearTimeout(p.timer);
      m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);
    }else if(m.method==='Runtime.exceptionThrown')exceptions.push(m.params.exceptionDetails);
  };
  const cdp=(method,params={})=>new Promise((resolve,reject)=>{
    const id=++serial,timer=setTimeout(()=>reject(Error('CDP timeout '+method)),20000);
    pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));
  });
  const ev=async expression=>{
    const r=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
    assert(!r.exceptionDetails,JSON.stringify(r.exceptionDetails));return r.result.value;
  };
  const waitFor=async expression=>{
    for(let i=0;i<100;i++){if(await ev(expression))return;await pause(100);}
    throw Error('Timeout '+expression);
  };
  const navigate=async target=>{
    await cdp('Page.navigate',{url:base+target});
    await waitFor('document.readyState === "complete" && location.pathname === '+JSON.stringify(target));
  };
  const preset=async name=>ev('document.querySelector("[data-preset='+name+']").click()');
  const metric=id=>ev('document.getElementById('+JSON.stringify('value-'+id)+').textContent');
  const download=async()=>JSON.parse(decodeURIComponent((await ev('document.getElementById("download-case").href')).split(',')[1]));
  const shot=async name=>{
    const {data}=await cdp('Page.captureScreenshot',{format:'png'});
    fs.writeFileSync(path.join(out,name+'.png'),Buffer.from(data,'base64'));
  };
  try{
    await cdp('Page.enable');await cdp('Runtime.enable');await cdp('Page.bringToFront');
    await navigate(route);await waitFor('!document.getElementById("details").hidden');
    for(const width of [320,390,768,1440])for(const theme of ['light','dark','warm']){
      await cdp('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<768});
      await ev('localStorage.setItem("color-theme",'+JSON.stringify(theme)+')');
      await navigate(route);await waitFor('!document.getElementById("details").hidden');
      assert(await ev('document.documentElement.dataset.theme === '+JSON.stringify(theme)));
      for(const [name,error] of Object.entries({correct:0,reversed:96,updated:0,stale:24,scaled:0,baseline:48,mismatch:0})){
        await preset(name);
        assert.equal(parseFloat(await metric('reprojection')),error,name);
        assert.equal(parseFloat(await metric('epipolar')),0,name);
        if(name==='scaled'){
          assert.equal(parseFloat(await metric('depthError')),100);
          assert((await ev('document.getElementById("status").textContent')).includes('尺度不一致'));
        }
        if(name==='mismatch'){
          assert.equal(parseFloat(await metric('matchError')),100);
          assert(await ev('!document.getElementById("match-warning").hidden'));
        }
        const payload=await download();
        assert.equal(payload.reprojection_error_px,error);
        assert(payload.illustrative&&payload.input&&payload.reference);
      }
      const data=await ev(`(() => ({overflow:document.documentElement.scrollWidth>innerWidth+1,
        controlSizes:[...document.querySelectorAll('button,input[type=number],.switches label')].map(e=>({width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height})),
        svgWidth:document.querySelector('svg').getBoundingClientRect().width,
        badAnchors:[...document.querySelectorAll('a[href^="#"]')].filter(a=>!document.getElementById(a.hash.slice(1))).map(a=>a.hash),
        invalidSVG:[...document.querySelectorAll('svg *')].some(e=>[...e.attributes].some(a=>/NaN|Infinity/.test(a.value)))
      }))()`);
      assert(!data.overflow&&!data.badAnchors.length&&!data.invalidSVG,JSON.stringify(data));
      assert(data.controlSizes.every(c=>c.width>0&&c.height>=44),JSON.stringify(data));
      await ev('document.getElementById("results").scrollIntoView({block:"start"})');
      await shot('results-'+width+'-'+theme);
      await preset('scaled');await ev('document.getElementById("results").scrollIntoView({block:"start"})');
      await shot('scale-'+width+'-'+theme);
      rows.push({width,theme,...data});
    }
    await preset('correct');
    assert(await ev('document.querySelector("[name=stale_intrinsics]").disabled'));
    await ev('document.querySelector("[name=crop_resize]").click()');
    assert(await ev('!document.querySelector("[name=stale_intrinsics]").disabled'));
    await ev('document.querySelector("[name=stale_intrinsics]").click()');
    assert.equal(parseFloat(await metric('reprojection')),24);
    await ev('document.querySelector("[name=crop_resize]").click()');
    assert(await ev('!document.querySelector("[name=stale_intrinsics]").checked'));
    assert.equal(parseFloat(await metric('reprojection')),0);
    for(const invalid of ['', '-1', '0.001']){
      await ev('(() => {const input=document.querySelector("[name=baseline_m]");input.value='+JSON.stringify(invalid)+';input.dispatchEvent(new Event("input",{bubbles:true}));})()');
      assert(await ev('document.getElementById("details").hidden && !document.getElementById("download-case").hasAttribute("href")'));
      await preset('correct');
    }
    await ev('document.querySelector("[name=baseline_m]").focus()');
    for(const type of ['keyDown','keyUp'])await cdp('Input.dispatchKeyEvent',{type,key:'a',code:'KeyA',windowsVirtualKeyCode:65,modifiers:2});
    await cdp('Input.insertText',{text:'0.13'});
    assert.equal(parseFloat(await metric('reprojection')),4);
    await ev('(() => {const b=document.querySelector("[name=baseline_m]"),o=document.querySelector("[name=match_offset_px]");b.value=.1234567;o.value=.125;b.dispatchEvent(new Event("input",{bubbles:true}));})()');
    assert(await ev('document.getElementById("geometry-form").checkValidity() && !document.getElementById("details").hidden'));
    await ev('document.querySelector(".lab-nav a[href=\\"#results\\"]").click()');
    assert(await ev('document.activeElement.id === "results"'));
    await ev('document.querySelector(".lab-return").click()');
    assert(await ev('document.activeElement.id === "geometry-form"'));
    await ev('const b=document.querySelector("[name=baseline_m]"),z=document.querySelector("[name=depth_z_m]");b.value=.5;z.value=.1;b.dispatchEvent(new Event("input",{bubbles:true}))');
    assert(await ev('!document.getElementById("bounds-warning").hidden'));
    await preset('scaled');
    const exported=await download();
    const companion=path.resolve(__dirname,'../content/posts/ai/multiview-geometry/geometry_sanity.py');
    const replay=JSON.parse(execFileSync('python3',['-B',companion,'--input','-'],{input:JSON.stringify(exported),encoding:'utf8',timeout:10000}));
    assert.equal(replay.reprojection_error_px,exported.reprojection_error_px);
    assert.equal(replay.depth_scale,2);
    await preset('reversed');
    await ev('(() => {const o=document.querySelector("[name=match_offset_px]");o.value=96;o.dispatchEvent(new Event("input",{bubbles:true}));})()');
    assert.equal(parseFloat(await metric('matchError')),0);
    assert.equal(parseFloat(await metric('reprojection')),96);
    const injection=await cdp('Page.addScriptToEvaluateOnNewDocument',{source:'Object.defineProperty(window,"localStorage",{get(){throw new Error("blocked storage")}})'});
    await navigate(route);await waitFor('!document.getElementById("details").hidden');
    assert.equal(parseFloat(await metric('reprojection')),0);
    await cdp('Page.removeScriptToEvaluateOnNewDocument',{identifier:injection.identifier});
    await cdp('Emulation.setScriptExecutionDisabled',{value:true});
    await navigate(route);
    assert(await ev('document.getElementById("interactive").hidden && document.querySelector("noscript").textContent.includes("Python") && document.documentElement.scrollWidth<=innerWidth+1'));
    await cdp('Emulation.setScriptExecutionDisabled',{value:false});
    assert.equal(exceptions.length,0,JSON.stringify(exceptions));
    fs.writeFileSync(path.join(out,'results.json'),JSON.stringify({rows,downloadReplay:true,blockedStorage:true,noJavaScript:true,keyboard:true,exceptions},null,2));
    console.log(JSON.stringify({passed:true,scenarios:rows.length,presets:7,downloadReplay:true,blockedStorage:true,noJavaScript:true,keyboard:true,output:out}));
  }finally{
    await cdp('Emulation.setScriptExecutionDisabled',{value:false});
    for(const p of pending.values())clearTimeout(p.timer);
    ws.close();await fetch(endpoint+'/json/close/'+tab.id);
  }
})().catch(e=>{console.error(e);process.exit(1);});
