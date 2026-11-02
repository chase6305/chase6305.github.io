// Reproduce a warm browser HTTP cache and verify fresh default CDP checks.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const{connect}=require('../../../scripts/lib/browser.cjs');
const[portText,out]=process.argv.slice(2);let revision='revision-one';const requests={};
const server=http.createServer((req,res)=>{requests[req.url]=(requests[req.url]||0)+1;if(req.url==='/'){res.writeHead(200,{'Content-Type':'text/html','Cache-Control':'no-store'});res.end('<!doctype html><title>Cache check</title><p>Cache fixture</p>');}else{res.writeHead(200,{'Content-Type':'text/plain','Cache-Control':'public, max-age=3600','Last-Modified':'Thu, 01 Jan 2026 00:00:00 GMT'});res.end(revision)}});
(async()=>{fs.mkdirSync(out,{recursive:true});await new Promise(r=>server.listen(0,'127.0.0.1',r));const base='http://127.0.0.1:'+server.address().port;let b;const report={};
 try{
  b=await connect(Number(portText));await b.navigate(base+'/');await b.cdp('Network.setCacheDisabled',{cacheDisabled:false});
  const first=await b.evaluate("fetch('/payload.txt').then(r=>r.text())");revision='revision-two';const warm=await b.evaluate("fetch('/payload.txt').then(r=>r.text())");assert.equal(first,'revision-one');assert.equal(warm,'revision-one');assert.equal(requests['/payload.txt'],1);report.cacheEnabledControl={first,warm,requests:requests['/payload.txt']};await b.close();
  b=await connect(Number(portText));await b.navigate(base+'/');const fresh=await b.evaluate("fetch('/payload.txt').then(r=>r.text())");assert.equal(fresh,'revision-two');assert.equal(requests['/payload.txt'],2);report.newDefault={fresh,requests:requests['/payload.txt'],bypassesPreviouslyWarmResponse:true};assert.deepEqual(b.exceptions,[]);console.log(JSON.stringify(report));
 }finally{if(b)await b.close();await new Promise(r=>server.close(r));fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');}
})().catch(e=>{console.error(e);process.exitCode=1});
