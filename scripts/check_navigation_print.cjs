#!/usr/bin/env node
// Native PDF tagging and focus when navigation/search/image overlays are open.
// Run serially: the CDP keyboard and active tab are shared with other checks.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('./lib/browser.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13143';
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-navigation-print';

(async () => {
  fs.mkdirSync(output,{recursive:true});
  const b = await connect(port), report = {menus:[],stacked:[],pdfs:[],exceptions:b.exceptions};
  let savedTheme;
  const wait = async expression => {
    const deadline=Date.now()+15000;
    while(Date.now()<deadline) {if(await b.evaluate(expression))return;await b.pause(40);}
    throw Error('Condition timed out: '+expression);
  };
  const load = async (route,width,theme) => {
    await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]});
    await b.viewport(width);await b.navigate(base+route);
    if(savedTheme===undefined) savedTheme=await b.evaluate("localStorage.getItem('color-theme')");
    await b.evaluate(`document.querySelector('[data-item=${theme}]').click();document.fonts.ready`);
    await b.evaluate(`window.__printMode=false;matchMedia('print').addEventListener('change',e=>{__printMode=e.matches})`);
  };
  const search = async () => {
    await b.cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'k',code:'KeyK',windowsVirtualKeyCode:75,modifiers:2});
    await b.cdp('Input.dispatchKeyEvent',{type:'keyUp',key:'k',code:'KeyK',windowsVirtualKeyCode:75});
    await wait("document.getElementById('hextra-search-dialog').open");
  };
  const enterPrint = async (entry,name) => {
    if(entry==='media') {
      await b.cdp('Emulation.setEmulatedMedia',{media:'print'});await wait('__printMode');
    } else {
      const pdf=await b.cdp('Page.printToPDF',{printBackground:true,generateTaggedPDF:true,
        paperWidth:8.2677,paperHeight:11.6929,marginTop:.4,marginBottom:.4,marginLeft:.4,marginRight:.4});
      const data=Buffer.from(pdf.data,'base64');
      assert(/\/StructTreeRoot\b/.test(data.toString('latin1')),'Native PDF lost its tagged structure: '+name);
      fs.writeFileSync(path.join(output,name+'.pdf'),data);report.pdfs.push(name+'.pdf');
    }
    await wait("!document.querySelector('#hextra-search-dialog').open && !document.querySelector('[data-image-viewer]')?.open");
    const state=await b.evaluate(`({inert:document.getElementById('content').inert,
      expanded:document.querySelector('.hextra-hamburger-menu').getAttribute('aria-expanded'),
      locked:document.body.classList.contains('hx:overflow-hidden'),
      rootOverflow:document.documentElement.style.overflow})`);
    assert(!state.inert && state.expanded==='false' && !state.locked && state.rootOverflow==='');
    const tree=await b.cdp('Accessibility.getFullAXTree');
    state.headings=tree.nodes.filter(n=>!n.ignored&&n.role?.value==='heading').length;
    assert(state.headings>0,'The printed article is absent from the accessibility tree');
    return state;
  };
  const leavePrint = async entry => {
    if(entry==='media') {await b.cdp('Emulation.setEmulatedMedia',{media:''});await wait('!__printMode');}
  };
  const openMenu = async scenario => {
    await b.evaluate("document.querySelector('.hextra-hamburger-menu').focus()");
    if(scenario==='pointer') {
      const point=await b.evaluate(`(()=>{const r=document.querySelector('.hextra-hamburger-menu').getBoundingClientRect();return{x:r.x+r.width/2,y:r.y+r.height/2}})()`);
      await b.cdp('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...point});
      await b.cdp('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...point});
    } else await b.key('Enter','Enter',13);
    assert(await b.evaluate("document.getElementById('content').inert"));
    if(scenario==='search-header') {
      await b.evaluate("document.querySelector('[data-search-open]').focus()");await b.key('Enter','Enter',13);
      await wait("document.getElementById('hextra-search-dialog').open");
    } else if(scenario==='search-sidebar') await search();
  };
  const openStack = async () => {
    await b.evaluate(`(async()=>{window.__imageOpener=document.querySelector('.post-image-link');
      const image=__imageOpener.querySelector('img');image.loading='eager';await image.decode();
      __imageOpener.scrollIntoView({block:'center'});__imageOpener.focus()})()`);
    await b.key('Enter','Enter',13);await wait("document.querySelector('[data-image-viewer]').open");
    await search();
    assert.equal(await b.evaluate("document.querySelectorAll('dialog[open]').length"),2);
  };
  try {
    for(const route of ['/posts/ai/supermimic/','/zh-cn/']) for(const width of [320,767])
      for(const theme of ['light','dark','warm']) for(const scenario of ['keyboard','pointer','search-header','search-sidebar']) {
        await load(route,width,theme);await openMenu(scenario);
        const state=await enterPrint('media');await leavePrint('media');
        const selector=scenario==='search-header'?'[data-search-open]':'.hextra-hamburger-menu';
        await wait(`document.activeElement===document.querySelector(${JSON.stringify(selector)})`);
        report.menus.push({route,width,theme,scenario,entry:'media',focusRestored:true,...state});
      }
    for(const route of ['/posts/ai/supermimic/','/zh-cn/']) for(const theme of ['light','dark','warm'])
      for(const scenario of ['keyboard','search-sidebar']) {
        await load(route,320,theme);await openMenu(scenario);
        const name='menu-'+(route==='/zh-cn/'?'zh':'article')+'-'+theme+'-'+scenario;
        const state=await enterPrint('native',name);await leavePrint('native');
        await wait("document.activeElement===document.querySelector('.hextra-hamburger-menu')");
        report.menus.push({route,width:320,theme,scenario,entry:'native',focusRestored:true,...state});
      }
    for(const width of [320,1440]) for(const theme of ['light','dark','warm']) for(const entry of ['media','native']) {
      await load('/posts/ai/supermimic/',width,theme);await openStack();
      const state=await enterPrint(entry,'stack-'+width+'-'+theme);await leavePrint(entry);
      await wait('document.activeElement===__imageOpener');
      report.stacked.push({width,theme,entry,focusRestored:true,...state});
    }
    // Additional reader-input and background cases are deliberately separate
    // from the matrix so a focus regression does not hide behind layout passes.
    await load('/posts/ai/supermimic/',1440,'light');await openStack();
    await enterPrint('media');
    await b.evaluate(`window.__readerFocus=document.querySelector('.content a[href]:not(.post-image-link)');__readerFocus.focus()`);
    await leavePrint('media');
    assert(await b.evaluate('document.activeElement===__readerFocus'));report.laterFocusPreserved=true;

    await load('/posts/ai/supermimic/',1440,'light');await openStack();
    await enterPrint('media');
    await b.evaluate("document.body.focus();document.body.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true}))");
    await b.evaluate('window.__pointerFocus=document.activeElement');
    await leavePrint('media');assert(await b.evaluate('document.activeElement===__pointerFocus'));
    report.pointerFocusPreserved=true;

    await load('/posts/ai/supermimic/',1440,'light');await openStack();await enterPrint('media');
    const other=await connect(port);
    try {
      await b.cdp('Emulation.setEmulatedMedia',{media:''});
      await b.evaluate("window.dispatchEvent(new Event('afterprint'))");
      assert(await other.evaluate('document.hasFocus()'));report.backgroundTabPreserved=true;
    } finally {await other.close();await b.cdp('Page.bringToFront');}
    await wait('!__printMode');
    assert(await b.evaluate('document.activeElement===document.body'));

    await load('/posts/ai/supermimic/',320,'light');await openMenu('keyboard');await enterPrint('media');
    await b.evaluate(`window.__readerFocus=document.querySelector('.content a[href]:not(.post-image-link)');__readerFocus.focus()`);
    await leavePrint('media');assert(await b.evaluate('document.activeElement===__readerFocus'));
    report.menuLaterFocusPreserved=true;

    await load('/posts/ai/supermimic/',320,'light');await openMenu('keyboard');await enterPrint('media');
    await b.evaluate("document.body.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true}));window.__pointerFocus=document.activeElement");
    await leavePrint('media');assert(await b.evaluate('document.activeElement===__pointerFocus'));
    report.menuPointerFocusPreserved=true;

    await load('/posts/ai/supermimic/',320,'light');await openMenu('keyboard');await enterPrint('media');
    const menuBackground=await connect(port);
    try {
      await b.cdp('Emulation.setEmulatedMedia',{media:''});await b.evaluate("window.dispatchEvent(new Event('afterprint'))");
      assert(await menuBackground.evaluate('document.hasFocus()'));report.menuBackgroundTabPreserved=true;
    } finally {await menuBackground.close();await b.cdp('Page.bringToFront');}
    await wait('!__printMode');assert(await b.evaluate('document.activeElement===document.body'));

    await load('/posts/ai/supermimic/',320,'light');await openMenu('keyboard');
    await b.evaluate(`window.dispatchEvent(new Event('beforeprint'));
      document.querySelector('.hextra-hamburger-menu').click();window.dispatchEvent(new Event('afterprint'))`);
    assert(await b.evaluate(`document.querySelector('.hextra-hamburger-menu').getAttribute('aria-expanded')==='true'
      &&document.querySelector('.hextra-sidebar-container').contains(document.activeElement)
      &&document.getElementById('content').inert`));
    report.menuImmediateReopenPreserved=true;

    await load('/posts/ai/supermimic/',1440,'light');await openStack();
    await b.evaluate(`window.dispatchEvent(new Event('beforeprint'));__imageOpener.click();
      window.dispatchEvent(new Event('afterprint'))`);
    await b.pause(100);
    assert(await b.evaluate(`document.querySelector('[data-image-viewer]').open
      &&document.querySelector('[data-image-viewer]').contains(document.activeElement)
      &&!document.getElementById('hextra-search-dialog').open`));
    report.viewerImmediateReopenPreserved=true;

    assert.equal(b.exceptions.length,0,JSON.stringify(b.exceptions));
    report.summary={menuCases:report.menus.length,stackCases:report.stacked.length,
      nativePDFs:report.pdfs.length,exceptions:b.exceptions.length};
    console.log(JSON.stringify(report.summary));
  } finally {
    await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]}).catch(()=>{});
    if(savedTheme!==undefined) await b.evaluate(savedTheme===null?"localStorage.removeItem('color-theme')":
      `localStorage.setItem('color-theme',${JSON.stringify(savedTheme)})`).catch(()=>{});
    fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(report,null,2)+'\n');
    await b.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1});
