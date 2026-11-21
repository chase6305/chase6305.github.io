#!/usr/bin/env node
// Scoped integration checks for this newly published article.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {connect}=require('../../../scripts/lib/browser.cjs');
const base=process.argv[2].replace(/\/$/,''),port=Number(process.argv[3]),out=process.argv[4];
const route='/posts/ai/pear/';fs.mkdirSync(out,{recursive:true});
(async()=>{
 const b=await connect(port),rows=[];
 try{
  for(const scripts of [false,true])for(const width of [320,1440])for(const entrance of ['/','/zh-cn/','/posts/','/tags/pear/']){
   await b.viewport(width,1000);
   await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]});
   await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});
   await b.navigate(base+entrance);await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});
   await b.evaluate(`document.documentElement.style.fontSize='${width===320?200:100}%';document.fonts.ready`);
   const state=await b.evaluate(`(()=>{const links=[...document.querySelectorAll('main a[href="${route}"]')];return {links:links.map(a=>({text:a.textContent.trim(),visible:!!a.getClientRects().length,lang:a.closest('[lang]')?.lang})),overflow:document.documentElement.scrollWidth>innerWidth+2}})()`);
   assert(state.links.some(a=>a.visible&&a.text.includes('PEAR')),JSON.stringify({entrance,...state}));
   assert(!state.overflow,JSON.stringify({entrance,width,scripts,...state}));
   rows.push({entrance,width,scripts,...state});
   if(entrance==='/zh-cn/'&&width===320&&scripts)await b.screenshot(path.join(out,'chinese-home-large.png'));
  }
  await b.navigate(base+'/posts/?q=PEAR');await b.pause(250);
  const selected=await b.evaluate(`({urls:[...document.querySelectorAll('#blog-filter-results h2 a')].map(a=>a.getAttribute('href')),query:document.querySelector('[data-blog-filter] input[name="q"]').value,status:document.querySelector('[data-blog-filter-status]').textContent})`);
  assert(selected.urls.includes(route),JSON.stringify(selected));assert.equal(selected.query,'PEAR');
  rows.push({case:'filter URL and native article card',...selected});
  await b.navigate(base+route);
  const metadata=await b.evaluate(`({canonical:document.querySelector('link[rel="canonical"]')?.href,title:document.title,language:document.querySelector('main [lang],main')?.lang||document.documentElement.lang,description:document.querySelector('meta[name="description"]')?.content,image:document.querySelector('meta[property="og:image"]')?.content,robots:document.querySelector('meta[name="robots"]')?.content||'',related:[...document.querySelectorAll('a[href*="pyrua-lean"]')].map(a=>a.getAttribute('href'))})`);
  assert.equal(new URL(metadata.canonical).pathname,route);assert(metadata.title.includes('PEAR'));
  assert(metadata.description.includes('30 次'));assert(metadata.image.includes('robot-use-loop'));assert(!metadata.robots.includes('noindex'));
  assert(metadata.related.length);rows.push({case:'article discovery metadata',...metadata});
  assert.equal(b.exceptions.length,0,JSON.stringify(b.exceptions));
  fs.writeFileSync(path.join(out,'results.json'),JSON.stringify({rows,exceptions:b.exceptions},null,2)+'\n');
  console.log(JSON.stringify({contexts:rows.length,exceptions:b.exceptions.length}));
 }finally{await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
