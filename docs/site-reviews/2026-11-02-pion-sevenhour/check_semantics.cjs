// Independently audit initial/expanded Pion content with pinned test-only axe.
const assert=require('node:assert/strict'),crypto=require('node:crypto'),fs=require('node:fs'),path=require('node:path');
const {connect}=require('../../../scripts/lib/browser.cjs');
const [base,portText,out,axePath]=process.argv.slice(2),axe=fs.readFileSync(axePath,'utf8');
const hash='20c09fe157a8a34a30e241aaa1fcdade657734f08ab379ecfbeb7d45cc46e878';assert.equal(crypto.createHash('sha256').update(axe).digest('hex'),hash);
(async()=>{fs.mkdirSync(out,{recursive:true});const b=await connect(Number(portText)),report={engine:'axe-core4.14.0',sha256:hash,cases:[],exceptions:b.exceptions};
 try{
  for(const scripts of[true,false])for(const width of[320,1440])for(const size of[100,200])for(const expanded of[false,true]){
   await b.viewport(width);await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.navigate(base+'/posts/ai/pion-muon-vla-rlvr/');await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});
   await b.evaluate(`document.documentElement.style.fontSize='${size}%';document.querySelectorAll('.content details:not(.blog-mobile-toc)').forEach(d=>d.open=${expanded});document.fonts.ready`);await b.pause(120);
   await b.evaluate(axe+'\n;void 0');const result=await b.evaluate(`axe.run(document,{rules:{'color-contrast':{enabled:false}},resultTypes:['violations','incomplete']}).then(r=>({version:r.testEngine.version,violations:r.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>({target:n.target,html:n.html,reason:n.failureSummary}))})),incomplete:r.incomplete.map(v=>({id:v.id,nodes:v.nodes.map(n=>({target:n.target,html:n.html,reason:n.failureSummary}))}))}))`);
   const references=await b.evaluate(`(()=>{const names=['aria-labelledby','aria-describedby','aria-controls','aria-activedescendant'],missing=[];let count=0;for(const e of document.querySelectorAll(names.map(n=>'['+n+']').join(',')))for(const name of names)for(const id of(e.getAttribute(name)||'').split(/\\s+/).filter(Boolean)){count++;if(!document.getElementById(id))missing.push({name,id,tag:e.tagName})}return{count,missing}})()`);
   report.cases.push({scripts,width,size,expanded,...result,references});assert.equal(result.version,'4.14.0');assert.deepEqual(result.violations,[]);assert.deepEqual(references.missing,[]);
  }
  assert.deepEqual(b.exceptions,[]);console.log(JSON.stringify({contexts:report.cases.length,violations:0,missingReferences:0,incompleteRecords:report.cases.reduce((n,c)=>n+c.incomplete.length,0),exceptions:0}));
 }finally{fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
