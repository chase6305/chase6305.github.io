#!/usr/bin/env node
// Keyboard/pointer language switching, nested navigation and modal Escape order.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {connect}=require('./lib/browser.cjs');
const base=process.argv[2]||'http://127.0.0.1:13143',port=Number(process.argv[3]||9226);
const output=process.argv[4]||'/tmp/chase-language-menu';

(async()=>{
  fs.mkdirSync(output,{recursive:true});const b=await connect(port),report={cases:[],nested:[],exceptions:b.exceptions};let savedTheme;
  const wait=async expression=>{const end=Date.now()+15000;while(Date.now()<end){if(await b.evaluate(expression))return;await b.pause(40)}throw Error('Condition timed out: '+expression)};
  const load=async(route,width,theme)=>{
    await b.viewport(width);await b.navigate(base+route);
    if(savedTheme===undefined)savedTheme=await b.evaluate("localStorage.getItem('color-theme')");
    await b.evaluate(`document.querySelector('[data-item=${theme}]').click();document.fonts.ready`);
  };
  const select=async selector=>b.evaluate(`window.__language=document.querySelector(${JSON.stringify(selector)});
    window.__options=__language.nextElementSibling;__language.focus()`);
  const closed=async()=>assert(await b.evaluate("__language.getAttribute('aria-expanded')==='false'&&__options.classList.contains('hx:hidden')"));
  try{
    for(const route of ['/','/zh-cn/','/posts/ai/supermimic/'])for(const width of[320,768,1440])
      for(const theme of['light','dark','warm'])for(const size of['100%','200%'])for(const activation of['Enter','Space']){
        await load(route,width,theme);await b.evaluate(`document.documentElement.style.fontSize=${JSON.stringify(size)}`);
        await select('.hextra-nav-container .hextra-language-switcher');
        await b.key(activation==='Enter'?'Enter':' ',activation,activation==='Enter'?13:32);
        assert(await b.evaluate("__language.getAttribute('aria-expanded')==='true'&&document.activeElement===__options.querySelector('[role=menuitem]')"),'Keyboard opening must focus the first item');
        await b.key('End','End',35);assert(await b.evaluate("document.activeElement===[...__options.querySelectorAll('[role=menuitem]')].at(-1)"));
        await b.key('Home','Home',36);assert(await b.evaluate("document.activeElement===__options.querySelector('[role=menuitem]')"));
        await b.key('Escape','Escape',27);await closed();assert(await b.evaluate('document.activeElement===__language'));
        assert(await b.evaluate('document.documentElement.scrollWidth<=innerWidth+1'));
        report.cases.push({route,width,theme,size,activation,firstItem:true,homeEnd:true,escape:true});
      }
    for(const width of[320,1440]){
      await load('/',width,'light');await select('.hextra-nav-container .hextra-language-switcher');
      const point=await b.evaluate('(()=>{const r=__language.getBoundingClientRect();return{x:r.x+r.width/2,y:r.y+r.height/2}})()');
      await b.cdp('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...point});
      await b.cdp('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...point});
      assert(await b.evaluate("__language.getAttribute('aria-expanded')==='true'&&document.activeElement===__language"));
      await b.key('Escape','Escape',27);await closed();report.nested.push({width,state:'pointer-escape',passed:true});

      await b.key('ArrowDown','ArrowDown',40);
      await b.evaluate("window.__laterFocus=document.querySelector('.hextra-nav-container a');__laterFocus.focus()");
      await b.key('Escape','Escape',27);await closed();assert(await b.evaluate('document.activeElement===__laterFocus'));
      report.nested.push({width,state:'later-focus',passed:true});

      await select('.hextra-nav-container .hextra-language-switcher');await b.key('Enter','Enter',13);
      await b.cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'k',code:'KeyK',windowsVirtualKeyCode:75,modifiers:2});
      await b.cdp('Input.dispatchKeyEvent',{type:'keyUp',key:'k',code:'KeyK',windowsVirtualKeyCode:75});
      await wait("document.querySelector('#hextra-search-dialog').open");
      await b.key('Escape','Escape',27);await wait("!document.querySelector('#hextra-search-dialog').open");
      assert(await b.evaluate("__language.getAttribute('aria-expanded')==='true'"),'Escape dismissed the underlying language menu');
      await b.key('Escape','Escape',27);await closed();report.nested.push({width,state:'search-escape-order',passed:true});
    }
    for(const route of['/','/zh-cn/'])for(const theme of['light','dark','warm']){
      await load(route,320,theme);
      await b.evaluate("document.querySelector('.hextra-hamburger-menu').focus()");await b.key('Enter','Enter',13);
      await select('.hextra-sidebar-container .hextra-language-switcher');await b.key('Enter','Enter',13);
      await b.key('Escape','Escape',27);await closed();
      assert(await b.evaluate("document.querySelector('.hextra-hamburger-menu').getAttribute('aria-expanded')==='true'&&document.querySelector('#content').inert"));
      await b.key('Escape','Escape',27);
      assert(await b.evaluate("document.querySelector('.hextra-hamburger-menu').getAttribute('aria-expanded')==='false'&&!document.querySelector('#content').inert&&document.activeElement===document.querySelector('.hextra-hamburger-menu')"));
      report.nested.push({route,theme,state:'drawer-escape-order',passed:true});
    }
    await load('/',1440,'light');
    await b.evaluate(`const controls=[...document.querySelectorAll('.hextra-language-switcher')].filter(e=>e.checkVisibility());
      controls[0].click();controls.at(-1).click()`);
    assert.equal(await b.evaluate("document.querySelectorAll('.hextra-language-switcher[aria-expanded=true]').length"),1);
    report.singleOpenMenu=true;
    assert.equal(b.exceptions.length,0,JSON.stringify(b.exceptions));report.summary={cases:report.cases.length,nestedCases:report.nested.length,exceptions:0};console.log(JSON.stringify(report.summary));
  }finally{
    if(savedTheme!==undefined)await b.evaluate(savedTheme===null?"localStorage.removeItem('color-theme')":`localStorage.setItem('color-theme',${JSON.stringify(savedTheme)})`).catch(()=>{});
    fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(report,null,2)+'\n');await b.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1});
