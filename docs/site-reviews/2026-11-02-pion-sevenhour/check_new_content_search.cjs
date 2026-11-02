// Current-source body queries must find Pion from both language entrances.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const{connect}=require('../../../scripts/lib/browser.cjs');
const[base,portText,out]=process.argv.slice(2),article='/posts/ai/pion-muon-vla-rlvr/';
(async()=>{fs.mkdirSync(out,{recursive:true});const b=await connect(Number(portText)),report={cases:[],exceptions:b.exceptions};try{
for(const entrance of['/','/zh-cn/'])for(const width of[320,1440]){
 await b.viewport(width);await b.navigate(base+entrance);
 const result=await b.evaluate(`(async()=>{const queries=['交叉协方差','相关观测','标量输出','动量秩','组内优势','GRPO 提议','复制 梯度','标量输出 GRPO','条件残差','残差占主导','四维卷积','展平 缩放'];const out={};for(const query of queries)out[query]=await window.hextraSearch.search(query);return out})()`);
 for(const[query,groups]of Object.entries(result)){
  const group=groups.find(g=>g.route===article);assert(group,'Missing Pion body query: '+query);
  const targets=group.matches.map(m=>m.route);const response=await fetch(base+article);assert(response.ok);const html=await response.text();
  const ids=await b.evaluate(`(()=>{const d=new DOMParser().parseFromString(${JSON.stringify(html)},'text/html');return${JSON.stringify(targets)}.map(route=>{const u=new URL(route,location.href);return{route,fragment:u.hash,exists:!u.hash||!!d.getElementById(decodeURIComponent(u.hash.slice(1)))}})})()`);
  assert(ids.every(i=>i.exists),JSON.stringify({query,ids}));report.cases.push({entrance,width,query,rank:groups.indexOf(group)+1,matches:ids});
 }
}
assert.deepEqual(b.exceptions,[]);console.log(JSON.stringify({cases:report.cases.length,currentOnlyQueries:['交叉协方差','标量输出 GRPO'],exceptions:0}));
}finally{fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');await b.close()}})().catch(e=>{console.error(e);process.exitCode=1});
