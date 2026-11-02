// Native Tab must expose desktop chapter links, including with page scripts off.
// A sticky-footer control demonstrates the occlusion that this check detects.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const{connect}=require('../../../scripts/lib/browser.cjs');
const[base,portText,out]=process.argv.slice(2),root=path.resolve(__dirname,'../../..');
const posts=JSON.parse(fs.readFileSync(path.join(root,'docs/blog-editorial-review.json'))).posts;
const spacing="body :is(p,li,dt,dd,th,td,summary,h1,h2,h3,h4,h5,h6,a,button,label,input,select){line-height:1.5!important;letter-spacing:.12em!important;word-spacing:.16em!important}.katex{letter-spacing:normal!important;word-spacing:normal!important}";
const read=`(()=>{const a=document.activeElement,r=a.getBoundingClientRect(),scroller=a.closest('.hextra-scrollbar'),box=scroller.getBoundingClientRect(),nav=document.querySelector('.hextra-nav-container nav').getBoundingClientRect();const top=Math.max(r.top,box.top,nav.bottom),bottom=Math.min(r.bottom,box.bottom,innerHeight),x=(r.left+r.right)/2,y=(top+bottom)/2,hit=bottom>top?document.elementFromPoint(x,y):null;return{text:a.textContent,href:a.getAttribute('href'),rect:r.toJSON(),viewport:{top:Math.max(box.top,nav.bottom),bottom:Math.min(box.bottom,innerHeight)},centerExposed:!!hit&&(hit===a||a.contains(hit)),hit:hit?.tagName,fullyVisibleWhenItFits:r.height>Math.min(box.bottom,innerHeight)-Math.max(box.top,nav.bottom)-2||(r.top>=Math.max(box.top,nav.bottom)-1&&r.bottom<=Math.min(box.bottom,innerHeight)+1)}})()`;
(async()=>{fs.mkdirSync(out,{recursive:true});const b=await connect(Number(portText)),report={cases:[],control:null,exceptions:b.exceptions};try{
 await b.cdp('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});
 await b.viewport(1440);await b.navigate(base+'/posts/ai/pion-muon-vla-rlvr/');await b.evaluate(`(()=>{document.documentElement.style.fontSize='200%';const s=document.createElement('style');s.textContent='.hextra-toc .hextra-scrollbar > div:last-child{position:sticky}';document.head.append(s);return document.fonts.ready})()`);await b.pause(100);
 const control=await b.evaluate(`(()=>{const links=[...document.querySelectorAll('.hextra-toc ul a')],bad=[];for(const a of links){a.focus();const r=a.getBoundingClientRect(),box=a.closest('.hextra-scrollbar').getBoundingClientRect(),hit=document.elementFromPoint((r.left+r.right)/2,(Math.max(r.top,box.top)+Math.min(r.bottom,box.bottom))/2);if(!hit||hit!==a&&!a.contains(hit))bad.push(a.textContent)}return{links:links.length,occluded:bad.length,texts:bad,scope:'Deliberately restore sticky positioning only; normal-flow production styling is checked below.'}})()`);assert(control.occluded>0);report.control=control;
 for(const scripts of[true,false])for(const width of[1280,1440])for(const mode of['base','text-200','text-200-spacing'])for(const post of posts){
  const route='/'+post.path.replace(/^content\//,'').replace(/\/index\.md$|\.md$/,'').toLowerCase()+'/';await b.viewport(width);await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.navigate(base+route);
  const theme=['light','dark','warm'][post.id%3];await b.evaluate(`(()=>{document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='${mode==='base'?100:200}%';${mode==='text-200-spacing'?"const s=document.createElement('style');s.textContent="+JSON.stringify(spacing)+";document.head.append(s);":""}return document.fonts.ready})()`);await b.pause(80);
  const count=await b.evaluate(`window.__tocLinks=[...document.querySelectorAll('.hextra-toc ul a')];__tocLinks.length`),entry={route,scripts,width,mode,theme,passed:false,links:[]};report.cases.push(entry);
  if(count)await b.evaluate('__tocLinks[0].focus()');
  for(let i=0;i<count;i++){
   if(i)await b.key('Tab','Tab',9);await b.pause(35);const correct=await b.evaluate(`document.activeElement===__tocLinks[${i}]`);assert(correct,JSON.stringify({route,scripts,width,mode,index:i,reason:'Native tab order changed'}));
   const state=await b.evaluate(read);entry.links.push(state);assert(state.centerExposed&&state.fullyVisibleWhenItFits,JSON.stringify({route,scripts,width,mode,index:i,state}));
  }
  entry.passed=true;if(report.cases.length%100===0)console.log(JSON.stringify({contexts:report.cases.length,links:report.cases.reduce((n,c)=>n+c.links.length,0)}));
 }
 assert.deepEqual(b.exceptions,[]);console.log(JSON.stringify({contexts:report.cases.length,links:report.cases.reduce((n,c)=>n+c.links.length,0),controlOccluded:report.control.occluded,exceptions:0}));
}finally{fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});await b.close()}})().catch(e=>{console.error(e);process.exitCode=1});
