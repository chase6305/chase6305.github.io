// All article tag footers must fit their reading sidebar at large text sizes.
// Run serially with other CDP checks against a draft-inclusive preview.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const{connect}=require('../../../scripts/lib/browser.cjs');
const[base,portText,out]=process.argv.slice(2),root=path.resolve(__dirname,'../../..');
const posts=JSON.parse(fs.readFileSync(path.join(root,'docs/blog-editorial-review.json'))).posts;
const spacing="body :is(p,li,dt,dd,th,td,summary,h1,h2,h3,h4,h5,h6,a,button,label,input,select){line-height:1.5!important;letter-spacing:.12em!important;word-spacing:.16em!important}.katex{letter-spacing:normal!important;word-spacing:normal!important}";
(async()=>{fs.mkdirSync(out,{recursive:true});const b=await connect(Number(portText)),report={cases:[],keyboard:[],exceptions:b.exceptions};try{
 for(const scripts of[true,false])for(const width of[1280,1440])for(const mode of['base','text-200','text-200-spacing'])for(const post of posts){
  const route='/'+post.path.replace(/^content\//,'').replace(/\/index\.md$|\.md$/,'').toLowerCase()+'/';
  await b.viewport(width);await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.navigate(base+route);
  const theme=['light','dark','warm'][post.id%3];await b.evaluate(`(()=>{document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='${mode==='base'?100:200}%';${mode==='text-200-spacing'?"const s=document.createElement('style');s.textContent="+JSON.stringify(spacing)+";document.head.append(s);":""}return document.fonts.ready})()`);await b.pause(100);
  const state=await b.evaluate(`(()=>{const nav=document.querySelector('.hextra-toc'),box=nav?.querySelector('.hextra-scrollbar');if(!nav||!nav.checkVisibility())return{visible:false};const tags=[...nav.querySelectorAll('a')].filter(a=>a.classList.contains('hx:whitespace-nowrap')),issues=[];for(const a of tags){const r=a.getBoundingClientRect(),footer=a.closest('div[class*="hx:bottom-0"]'),f=footer.getBoundingClientRect(),text=document.createRange();text.selectNodeContents(a);if(r.left<f.left-1||r.right>f.right+1)issues.push({text:a.textContent,reason:'tag exceeds footer',tag:r.toJSON(),footer:f.toJSON()});for(const t of text.getClientRects())if(t.left<r.left-1||t.right>r.right+1)issues.push({text:a.textContent,reason:'horizontal text exceeds link',textBox:t.toJSON(),tag:r.toJSON()});}window.__sidebarTags=tags;return{visible:true,tags:tags.map(a=>({text:a.textContent,href:a.getAttribute('href'),whiteSpace:getComputedStyle(a).whiteSpace})),issues,bodyOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1,scroller:{client:box.clientWidth,scroll:box.scrollWidth}}})()`);
  assert(state.visible,'Expected desktop reading sidebar: '+route);assert(!state.bodyOverflow&&state.issues.length===0,JSON.stringify({route,scripts,width,mode,state}));
  if(state.tags.length){
   await b.evaluate('__sidebarTags.at(-1).focus()');const focus=await b.evaluate(`(()=>{const a=__sidebarTags.at(-1),r=a.getBoundingClientRect(),nav=document.querySelector('.hextra-nav-container nav').getBoundingClientRect();return{focused:document.activeElement===a,top:r.top,bottom:r.bottom,navBottom:nav.bottom,height:innerHeight}})()`);assert(focus.focused&&focus.top>=focus.navBottom-1&&focus.bottom<=focus.height+1,JSON.stringify({route,scripts,width,mode,focus}));state.lastTagFocus=focus;
   if(post.id===95){
    await b.evaluate('__sidebarTags[0].focus()');for(let i=1;i<state.tags.length;i++){await b.key('Tab','Tab',9);assert(await b.evaluate(`document.activeElement===__sidebarTags[${i}]`));}report.keyboard.push({route,scripts,width,mode,tags:state.tags.length,tabTransitions:state.tags.length-1});
    if(mode==='text-200')await b.screenshot(path.join(out,`pion-tags-${width}-${scripts}.png`));
   }
  }
  report.cases.push({route,scripts,width,mode,theme,...state});
 }
 assert.deepEqual(b.exceptions,[]);console.log(JSON.stringify({cases:report.cases.length,tagLinks:report.cases.reduce((n,c)=>n+c.tags.length,0),keyboardContexts:report.keyboard.length,exceptions:0}));
}finally{fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});await b.close()}})().catch(e=>{console.error(e);process.exitCode=1});
