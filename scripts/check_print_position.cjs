#!/usr/bin/env node
// Native long-article printing and cancellation of delayed position restoration.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {connect}=require('./lib/browser.cjs');
const base=process.argv[2]||'http://127.0.0.1:13143',port=Number(process.argv[3]||9226);
const output=process.argv[4]||'/tmp/chase-print-position',edgesOnly=process.argv.includes('--edges-only');

(async()=>{
  fs.mkdirSync(output,{recursive:true});const b=await connect(port),report={native:[],edges:[],exceptions:b.exceptions};let savedTheme;
  const load=async(route,width,theme)=>{
    await b.viewport(width);await b.navigate(base+route);
    if(savedTheme===undefined)savedTheme=await b.evaluate("localStorage.getItem('color-theme')");
    await b.evaluate(`document.querySelector('[data-item=${theme}]').click();document.fonts.ready`);
    return b.evaluate(`(()=>{window.__readingHeading=[...document.querySelectorAll('.content h2')].at(-1);
      __readingHeading.tabIndex=-1;__readingHeading.scrollIntoView({block:'start'});__readingHeading.focus({preventScroll:true});
      return{x:scrollX,y:scrollY,top:__readingHeading.getBoundingClientRect().top,
        details:[...document.querySelectorAll('.content details:not(.blog-mobile-toc)')].map(e=>e.open)}})()`);
  };
  const screenFrames=async()=>{
    await b.evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(()=>requestAnimationFrame(()=>requestAnimationFrame(r)))))');
    await b.pause(100);
  };
  const holdFrames=async()=>b.evaluate(`(()=>{
    window.__realRequest=requestAnimationFrame;window.__realCancel=cancelAnimationFrame;
    window.__frames=new Map();window.__frameId=0;
    window.requestAnimationFrame=callback=>{const id=--__frameId;__frames.set(id,callback);return id};
    window.cancelAnimationFrame=id=>{if(id<0)__frames.delete(id);else __realCancel(id)};
    window.__flushFrame=()=>{const callbacks=[...__frames.values()];__frames.clear();callbacks.forEach(fn=>fn(performance.now()))};
    window.__releaseFrames=()=>{for(let i=0;i<5&&__frames.size;i++)__flushFrame();window.requestAnimationFrame=__realRequest;window.cancelAnimationFrame=__realCancel};
    document.body.tabIndex=-1;document.body.focus({preventScroll:true});
    window.dispatchEvent(new Event('beforeprint'));window.scrollTo({top:0,behavior:'instant'});
  })()`);
  try{
    report.browser=await b.cdp('Browser.getVersion');
    for(const route of edgesOnly?[]:['/posts/pid/','/posts/robotics/control/robot-filters/','/posts/ai/cosmos3/'])
      for(const width of[320,1440])for(const theme of['light','dark','warm'])for(let repetition=0;repetition<2;repetition++){
        const before=await load(route,width,theme);
        const pdf=await b.cdp('Page.printToPDF',{printBackground:true,generateTaggedPDF:true,
          paperWidth:8.2677,paperHeight:11.6929,marginTop:.4,marginBottom:.4,marginLeft:.4,marginRight:.4});
        const data=Buffer.from(pdf.data,'base64');assert(/\/StructTreeRoot\b/.test(data.toString('latin1')));
        if(width===320&&theme==='dark'&&repetition===1)fs.writeFileSync(path.join(output,route.split('/').filter(Boolean).at(-1)+'.pdf'),data);
        await screenFrames();
        const after=await b.evaluate(`({x:scrollX,y:scrollY,top:__readingHeading.getBoundingClientRect().top,
          focused:document.activeElement===__readingHeading,
          details:[...document.querySelectorAll('.content details:not(.blog-mobile-toc)')].map(e=>e.open)})`);
        assert(Math.abs(after.x-before.x)<=1&&Math.abs(after.y-before.y)<=1,JSON.stringify({route,width,theme,repetition,before,after}));
        assert(after.focused);assert.deepEqual(after.details,before.details);
        report.native.push({route,width,theme,repetition,before,after,tagged:true});
      }
    const mediaBefore=await load('/posts/pid/',320,'dark');
    await b.evaluate("window.dispatchEvent(new Event('beforeprint'))");
    await b.cdp('Emulation.setEmulatedMedia',{media:'print'});await b.pause(100);
    await b.cdp('Page.printToPDF',{printBackground:true,paperWidth:8.2677,paperHeight:11.6929});
    await b.pause(100);await b.cdp('Emulation.setEmulatedMedia',{media:''});await screenFrames();
    const mediaAfter=await b.evaluate('scrollY');
    assert(Math.abs(mediaAfter-mediaBefore.y)<=1,JSON.stringify({entry:'native-during-print-media',before:mediaBefore.y,after:mediaAfter}));
    report.printMediaRoundTrip=true;
    for(const action of['keyboard','pointer','wheel','touch','focus-before-frame','focus-after-frame','chapter','resize','font-size']){
      await load('/posts/pid/',320,'dark');await holdFrames();
      // Separate tasks let the beforeprint microtask finish before afterprint,
      // matching native browser event ordering instead of masking focus races.
      await b.evaluate("window.dispatchEvent(new Event('afterprint'))");
      if(action==='keyboard')await b.key('ArrowDown','ArrowDown',40);
      else if(action==='pointer'){
        await b.cdp('Input.dispatchMouseEvent',{type:'mousePressed',x:10,y:500,button:'left',clickCount:1});
        await b.cdp('Input.dispatchMouseEvent',{type:'mouseReleased',x:10,y:500,button:'left',clickCount:1});
      }else if(action==='wheel')await b.cdp('Input.dispatchMouseEvent',{type:'mouseWheel',x:10,y:500,deltaX:0,deltaY:80});
      else if(action==='touch'){
        await b.cdp('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:10,y:500}]});
        await b.cdp('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
      }else if(action.startsWith('focus-')){
        if(action==='focus-after-frame')await b.evaluate('__flushFrame()');
        await b.evaluate("window.__readerTarget=document.querySelector('h1');__readerTarget.tabIndex=-1;__readerTarget.focus({preventScroll:true})");
      }else if(action==='chapter')await b.evaluate("history.replaceState(null,'','#reader-changed')");
      else if(action==='resize')await b.viewport(768);
      else await b.evaluate("document.documentElement.style.fontSize='200%'");
      await b.pause(120);await b.evaluate('__releaseFrames()');await b.pause(100);
      const after=await b.evaluate(`({y:scrollY,focused:document.activeElement===window.__readerTarget,href:location.href})`);
      assert(after.y<500,JSON.stringify({action,after}));if(action.startsWith('focus-'))assert(after.focused);
      report.edges.push({action,readerPositionPreserved:true,...after});
    }
    const before=await load('/posts/pid/',320,'dark');await holdFrames();
    await b.evaluate("window.dispatchEvent(new Event('afterprint'));window.dispatchEvent(new Event('beforeprint'));window.dispatchEvent(new Event('afterprint'))");
    await b.evaluate('__releaseFrames()');await b.pause(100);
    assert(Math.abs(await b.evaluate('scrollY')-before.y)<=1);report.immediateReprint=true;

    await load('/posts/pid/',320,'dark');await holdFrames();await b.evaluate("window.dispatchEvent(new Event('afterprint'))");
    const other=await connect(port);
    try{await b.evaluate('__releaseFrames()');assert(await other.evaluate('document.hasFocus()'));report.backgroundTabPreserved=true;}
    finally{await other.close();await b.cdp('Page.bringToFront');}
    assert.equal(b.exceptions.length,0,JSON.stringify(b.exceptions));
    report.summary={nativeCases:report.native.length,printMediaRoundTrip:true,readerActions:report.edges.length,immediateReprint:true,backgroundTab:true,exceptions:0};console.log(JSON.stringify(report.summary));
  }finally{
    if(savedTheme!==undefined)await b.evaluate(savedTheme===null?"localStorage.removeItem('color-theme')":`localStorage.setItem('color-theme',${JSON.stringify(savedTheme)})`).catch(()=>{});
    fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(report,null,2)+'\n');await b.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1});
