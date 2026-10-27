#!/usr/bin/env node
// Chapter navigation keeps fragment history and respects subsequent input.
// Run serially with the other loopback CDP checks.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const{connect}=require('./lib/browser.cjs');
const base=process.argv[2]||'http://127.0.0.1:13185',port=Number(process.argv[3]||9226),output=process.argv[4]||'/tmp/chase-chapter-focus';
(async()=>{fs.mkdirSync(output,{recursive:true});const b=await connect(port),report={layouts:[],exceptions:b.exceptions};let savedTheme;
const holdClick=`(index,legacy=false)=>{
 const toc=document.querySelector('.blog-mobile-toc');toc.open=true;const link=toc.querySelectorAll('a')[index];
 let anchor=document.getElementById(decodeURIComponent(link.hash.slice(1)));const heading=anchor.closest('h1,h2,h3,h4,h5,h6')||anchor;
 if(legacy&&anchor===heading){const span=document.createElement('span');span.id=heading.id;heading.removeAttribute('id');span.style.cssText='position:absolute;margin-top:-80px';heading.append(span);anchor=span;}
 window.__chapterTarget=heading;window.__chapterFrames=[];const raf=requestAnimationFrame;
 window.requestAnimationFrame=fn=>{__chapterFrames.push(fn);return 123456+__chapterFrames.length};
 try{link.click()}finally{window.requestAnimationFrame=raf}
 return{targetId:anchor.id,hash:location.hash,focused:document.activeElement===heading,closed:!toc.open,heldFrames:__chapterFrames.length};
}`;
try{
for(const width of [320,768])for(const route of ['/posts/python/tqdm/','/posts/ai/supermimic/','/posts/open3d/introduction/'])for(const theme of ['light','dark','warm'])for(const size of ['100%','200%']){
 await b.viewport(width);await b.navigate(base+route);if(savedTheme===undefined)savedTheme=await b.evaluate("localStorage.getItem('color-theme')");
 await b.evaluate(`document.querySelector('[data-item=${theme}]').click();document.documentElement.style.fontSize='${size}';document.fonts.ready`);
 const state=await b.evaluate('('+holdClick+')(0)');assert(state.focused&&state.closed,JSON.stringify({route,width,theme,size,state}));
 await b.key('Tab','Tab',9);await b.evaluate('window.__afterChapterTab=document.activeElement;window.__afterChapterScroll=scrollY;__chapterFrames.forEach(fn=>fn(performance.now()))');
 const retained=await b.evaluate('document.activeElement===__afterChapterTab&&Math.abs(scrollY-__afterChapterScroll)<=1');assert(retained);
 report.layouts.push({route,width,theme,size,...state,laterTabPreserved:retained,passed:true});
}
await b.viewport(390);await b.navigate(base+'/posts/python/tqdm/');
// Ctrl/Meta/Shift/Alt clicks and auxiliary buttons must retain native behavior.
report.modifiers=[];
for(const flags of [{ctrlKey:true},{metaKey:true},{shiftKey:true},{altKey:true},{button:1}]){
 const state=await b.evaluate(`(()=>{const toc=document.querySelector('.blog-mobile-toc');toc.open=true;const link=toc.querySelector('a'),before=location.hash,focus=document.activeElement;
 document.addEventListener('click',e=>e.preventDefault(),{once:true});
 link.dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true,button:0,...${JSON.stringify(flags)}}));
 return{open:toc.open,hashPreserved:location.hash===before,focusPreserved:document.activeElement===focus};})()`);
 assert(state.open&&state.hashPreserved&&state.focusPreserved,JSON.stringify({flags,state}));report.modifiers.push({flags,...state});
}
// A legacy span anchor inside a heading remains supported.
const legacy=await b.evaluate('('+holdClick+')(0,true)');assert(legacy.focused&&legacy.closed,JSON.stringify(legacy));
await b.evaluate('__chapterFrames.forEach(fn=>fn(performance.now()))');assert(await b.evaluate('document.activeElement===__chapterTarget'));
report.legacyAnchor=legacy;
// Native hash assignment must preserve Back/Forward between chapters.
const first=await b.evaluate('location.hash');const second=await b.evaluate('('+holdClick+')(1)');assert(second.focused&&second.closed);
await b.evaluate('history.back()');for(let i=0;i<100;i++){if(await b.evaluate('location.hash==='+JSON.stringify(first)))break;await b.pause(30)}assert(await b.evaluate('location.hash==='+JSON.stringify(first)));
await b.evaluate('history.forward()');for(let i=0;i<100;i++){if(await b.evaluate('location.hash==='+JSON.stringify(second.hash)))break;await b.pause(30)}assert(await b.evaluate('location.hash==='+JSON.stringify(second.hash)));report.fragmentHistory=true;
assert.equal(b.exceptions.length,0,JSON.stringify(b.exceptions));console.log(JSON.stringify({layouts:report.layouts.length,modifiers:report.modifiers.length,legacyAnchor:true,fragmentHistory:true,exceptions:0},null,2));
}finally{if(savedTheme!==undefined)await b.evaluate(savedTheme===null?"localStorage.removeItem('color-theme')":'localStorage.setItem("color-theme",'+JSON.stringify(savedTheme)+')').catch(()=>{});fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(report,null,2)+'\n');await b.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
