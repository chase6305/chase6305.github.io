// Inspect current SNR prose and the final table columns at 200% root text.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const{connect}=require('../../../scripts/lib/browser.cjs');const[base,portText,out]=process.argv.slice(2);
(async()=>{fs.mkdirSync(out,{recursive:true});const b=await connect(Number(portText)),rows=[];try{
for(const width of[320,1440])for(const theme of['light','dark']){
 await b.viewport(width);await b.navigate(base+'/posts/ai/pion-muon-vla-rlvr/');
 await b.evaluate(`document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add('${theme}');document.documentElement.style.fontSize='200%';document.getElementById('snr-estimation').open=true;document.fonts.ready`);await b.pause(200);
 await b.evaluate(`document.getElementById('snr-estimation').scrollIntoView({block:'start'})`);await b.screenshot(path.join(out,`snr-${width}-${theme}-summary.png`));
 await b.evaluate(`window.__snrTable=document.querySelector('#snr-estimation table');__snrTable.scrollIntoView({block:'center'});__snrTable.scrollLeft=__snrTable.scrollWidth;__snrTable.focus()`);await b.pause(120);await b.screenshot(path.join(out,`snr-${width}-${theme}-table-end.png`));
 const state=await b.evaluate(`({bodyOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1,tableWidth:__snrTable.clientWidth,tableScroll:__snrTable.scrollWidth,lastColumnReachable:__snrTable.scrollLeft>=__snrTable.scrollWidth-__snrTable.clientWidth-1,focus:document.activeElement===__snrTable})`);assert(!state.bodyOverflow&&state.lastColumnReachable);rows.push({width,theme,fontSize:200,...state});
}
assert.deepEqual(b.exceptions,[]);console.log(JSON.stringify({contexts:rows.length,screenshots:rows.length*2,exceptions:0}));
}finally{fs.writeFileSync(path.join(out,'results.json'),JSON.stringify({cases:rows,exceptions:b.exceptions},null,2)+'\n');await b.close()}})().catch(e=>{console.error(e);process.exitCode=1});
