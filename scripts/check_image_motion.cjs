#!/usr/bin/env node
// Native GIF preferences and viewer transitions; run serially on a loopback CDP endpoint.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {connect}=require('./lib/browser.cjs');
const base=process.argv[2]||'http://127.0.0.1:13221',port=Number(process.argv[3]||9226);
const output=process.argv[4]||'/tmp/chase-image-motion';
const routes=['/posts/calibration/kinematics/','/posts/calibration/zero/','/posts/ai/gymnasium/1/'];
(async()=>{
 fs.mkdirSync(output,{recursive:true});const b=await connect(port),report={cases:[],transitions:[],exceptions:b.exceptions};let server;
 const motion=value=>b.cdp('Emulation.setEmulatedMedia',{media:'',features:[{name:'prefers-reduced-motion',value}]});
 const until=async(expression)=>{for(let i=0;i<100;i++){if(await b.evaluate(expression))return;await b.pause(40)}throw Error('Timed out: '+expression)};
 const source=async()=>{await b.evaluate(`window.__motion=[...document.querySelectorAll('.content img')].find(i=>/\.gif$/i.test(i.src));window.__motion.scrollIntoView({block:'center'})`);await until(`window.__motion.complete&&window.__motion.naturalWidth>0&&window.__motion.currentSrc===((matchMedia('print').matches||matchMedia('(prefers-reduced-motion: reduce)').matches)?new URL(window.__motion.closest('picture').querySelector('source').srcset,document.baseURI).href:window.__motion.src)`);};
 const viewer=()=>b.evaluate(`(()=>{const d=document.querySelector('[data-image-viewer]');return{open:d.open,state:d.dataset.state,src:d.querySelector('img')?.currentSrc,status:d.querySelector('[role=status]').textContent,original:d.querySelector('[data-image-viewer-original]').href}})()`);
 const open=async()=>{await b.evaluate("window.__motion.closest('a').click()");await until("document.querySelector('[data-image-viewer]').dataset.state==='ready'");};
 try{
 await b.cdp('Network.setCacheDisabled',{cacheDisabled:true});
 for(const scripts of[true,false]){
  await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});
  for(const preference of['no-preference','reduce'])for(const route of routes)for(const width of[320,1440])for(const theme of['light','dark','warm']){
   await motion(preference);const requestStart=b.requests.length;await b.viewport(width);await b.navigate(base+route);
   await b.evaluate(`document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add(${theme==='warm'?"'light','warm'":JSON.stringify(theme)});document.fonts.ready`);await source();
   const state=await b.evaluate(`({src:window.__motion.currentSrc,original:window.__motion.src,poster:new URL(window.__motion.closest('picture').querySelector('source').srcset,document.baseURI).href,href:window.__motion.closest('a').href,width:window.__motion.getAttribute('width'),height:window.__motion.getAttribute('height'),overflow:document.documentElement.scrollWidth>innerWidth+1,y:scrollY})`);
   assert.equal(state.src,preference==='reduce'?state.poster:state.original);assert.equal(state.href,state.original);assert(!state.overflow&&Number(state.width)>0&&Number(state.height)>0);
   const gifRequests=b.requests.slice(requestStart).filter(url=>/\.gif(?:\?|$)/i.test(url));if(preference==='reduce')assert.equal(gifRequests.length,0,'Reduced motion fetched animation');
   if(scripts){await open();const v=await viewer();assert.equal(v.src,state.src);assert.equal(v.original,state.original);if(preference==='reduce')assert(v.status.includes('静态预览'));await b.key('Escape','Escape',27);assert(await b.evaluate("document.activeElement===window.__motion.closest('a')"));assert(Math.abs(await b.evaluate('scrollY')-state.y)<=1);}
   else assert(await b.evaluate("!document.querySelector('[data-image-viewer]')"));
   await b.cdp('Emulation.setEmulatedMedia',{media:'print',features:[{name:'prefers-reduced-motion',value:preference}]});await source();assert.equal(await b.evaluate('window.__motion.currentSrc'),state.poster);
   await motion(preference);await source();assert.equal(await b.evaluate('window.__motion.currentSrc'),state.src);
   if(scripts&&preference==='reduce'&&theme==='light'&&width===320)await b.screenshot(path.join(output,route.split('/').filter(Boolean).slice(-1)[0]+'-static-mobile.png'));
   report.cases.push({route,scripts,preference,width,theme,gifRequests:gifRequests.length,...state});
  }
 }
 await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});await b.viewport(320);
 for(const route of routes)for(const control of['[data-image-viewer-close]','[data-image-viewer-original]']){
  await motion('no-preference');await b.navigate(base+route);await source();await open();await b.evaluate(`document.querySelector(${JSON.stringify(control)}).focus()`);
  const expected=await b.evaluate("({original:window.__motion.src,poster:new URL(window.__motion.dataset.zoomStill,document.baseURI).href})");
  for(const preference of['reduce','no-preference','reduce']){await motion(preference);const target=preference==='reduce'?expected.poster:expected.original;await until(`document.querySelector('[data-image-viewer]').dataset.state==='ready'&&document.querySelector('[data-image-viewer] img').currentSrc===${JSON.stringify(target)}`);assert.equal((await viewer()).src,target);assert(await b.evaluate(`document.activeElement===document.querySelector(${JSON.stringify(control)})`));}
  for(let i=0;i<6;i++)await motion(i%2?'no-preference':'reduce');await motion('reduce');await until(`document.querySelector('[data-image-viewer]').dataset.state==='ready'&&document.querySelector('[data-image-viewer] img').currentSrc===${JSON.stringify(expected.poster)}`);await b.pause(150);assert.equal((await viewer()).src,expected.poster);assert(await b.evaluate(`document.activeElement===document.querySelector(${JSON.stringify(control)})`));
  await b.key('Escape','Escape',27);report.transitions.push({route,control,rapidChanges:7,focusPreserved:true});
 }
 await motion('reduce');await b.navigate(base+routes[0]);await source();await b.evaluate("window.__motion.closest('[lang]').lang='en'");await open();assert((await viewer()).status.includes('Static preview'));await b.key('Escape','Escape',27);report.englishStatus=true;
 let received=0,completed=0;const gif=fs.readFileSync(path.join(__dirname,'../content/posts/calibration/zero/calibrate_tcp1.gif'));
 server=http.createServer((req,res)=>{received++;setTimeout(()=>{if(res.destroyed)return;completed++;res.writeHead(200,{'Content-Type':'image/gif','Cache-Control':'no-store'});res.end(gif)},800)});
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 await motion('no-preference');await b.navigate(base+routes[0]);await source();await b.evaluate(`window.__motion.dataset.zoomSrc='http://127.0.0.1:${server.address().port}/delayed.gif';window.__motion.closest('a').click();document.querySelector('[data-image-viewer-original]').focus()`);
 for(let i=0;i<100&&!received;i++)await b.pause(20);assert(received>0);
 await motion('reduce');await until("document.querySelector('[data-image-viewer]').dataset.state==='ready'");const still=(await viewer()).src;assert(still.endsWith('.png'));await b.pause(1100);assert.equal((await viewer()).src,still);assert((await viewer()).status.includes('静态预览'));assert(await b.evaluate("document.activeElement===document.querySelector('[data-image-viewer-original]')"));await b.key('Escape','Escape',27);report.delayedOriginal={received,completed,staticPreserved:true};
 for(const route of routes){await motion('no-preference');await b.navigate(base+route);await source();await open();const pdf=await b.cdp('Page.printToPDF',{printBackground:false,paperWidth:8.27,paperHeight:11.69});fs.writeFileSync(path.join(output,route.split('/').filter(Boolean).slice(-1)[0]+'.pdf'),Buffer.from(pdf.data,'base64'));await b.evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(()=>requestAnimationFrame(r))))');assert(!(await viewer()).open);assert(await b.evaluate("document.activeElement===window.__motion.closest('a')"));}
 report.nativePrints=routes.length;report.summary={cases:report.cases.length,transitions:report.transitions.length,exceptions:b.exceptions.length};console.log(JSON.stringify(report.summary,null,2));assert.equal(b.exceptions.length,0);
 }finally{await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]}).catch(()=>{});await b.cdp('Network.setCacheDisabled',{cacheDisabled:false}).catch(()=>{});if(server){server.closeAllConnections();await new Promise(resolve=>server.close(resolve));}fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(report,null,2)+'\n');await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
