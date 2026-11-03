// Native Tab order and painted outlines. Color calculation follows check_site_focus.cjs.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {connect}=require('../../../scripts/lib/browser.cjs');
const [base,portText,out]=process.argv.slice(2);
function inspect(index){
  const element=document.querySelectorAll('.muon-batch button,.muon-batch input')[index],style=getComputedStyle(element);
  const canvas=document.createElement('canvas');canvas.width=canvas.height=1;
  const ctx=canvas.getContext('2d',{willReadFrequently:true});
  const rgba=color=>{ctx.clearRect(0,0,1,1);ctx.fillStyle=color;ctx.fillRect(0,0,1,1);return [...ctx.getImageData(0,0,1,1).data].map(v=>v/255)};
  const over=(a,b)=>[0,1,2].map(i=>a[i]*a[3]+b[i]*(1-a[3]));
  const background=node=>node?over(rgba(getComputedStyle(node).backgroundColor),background(node.parentElement)):[1,1,1];
  const bg=background(element.parentElement),fg=over(rgba(style.outlineColor),bg);
  const lum=color=>color.map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((n,v,i)=>n+v*[.2126,.7152,.0722][i],0);
  const values=[lum(fg),lum(bg)].sort((a,b)=>b-a),r=element.getBoundingClientRect();
  return {index,name:element.name||element.dataset.preset||element.type,
    focused:document.activeElement===element,visible:element.matches(':focus-visible'),
    outline:style.outlineColor,outlineStyle:style.outlineStyle,outlineWidth:parseFloat(style.outlineWidth),
    ratio:(values[0]+.05)/(values[1]+.05),inside:r.left>=-1&&r.right<=document.documentElement.clientWidth+1};
}
(async()=>{
  fs.mkdirSync(out,{recursive:true});const b=await connect(Number(portText)),report={outlines:[],leave:[],exceptions:b.exceptions};
  const run=(fn,...args)=>b.evaluate('('+fn.toString()+')(...'+JSON.stringify(args)+')');
  try{
    for(const width of [320,1440])for(const size of [100,200])for(const theme of ['light','dark','warm'])for(const mode of ['normal','forced-light','forced-dark']){
      const features=mode==='normal'?[]:[{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:mode==='forced-light'?'light':'dark'}];
      await b.cdp('Emulation.setEmulatedMedia',{media:'',features});await b.viewport(width,1000);
      await b.navigate('about:blank');await b.navigate(base+'/posts/ai/muon-learning-rate-batch-size/#batch-calculator');
      await run((size,theme)=>{
        document.documentElement.style.fontSize=size+'%';document.documentElement.classList.remove('dark','warm');
        if(theme!=='light')document.documentElement.classList.add(theme);
        document.documentElement.style.colorScheme=theme==='dark'?'dark':'light';return document.fonts.ready;
      },size,theme);
      await b.key('Tab','Tab',9);
      await run(()=>document.querySelector('.muon-batch button').focus());
      const count=await b.evaluate("document.querySelectorAll('.muon-batch button,.muon-batch input').length");
      assert.equal(count,10);
      for(let index=0;index<count;index++){
        if(index)await b.key('Tab','Tab',9);
        const row=await run(inspect,index);
        assert(row.focused&&row.visible&&row.outlineWidth>=2&&row.outlineStyle!=='none'&&row.ratio>=3&&row.inside,JSON.stringify({width,size,theme,mode,row}));
        report.outlines.push({width,size,theme,mode,...row});
      }
      await b.key('Tab','Tab',9);
      assert(await b.evaluate("!document.querySelector('.muon-batch').contains(document.activeElement)"));
      for(const type of ['keyDown','keyUp'])await b.cdp('Input.dispatchKeyEvent',{type,key:'Tab',code:'Tab',windowsVirtualKeyCode:9,modifiers:8});
      assert(await b.evaluate("document.activeElement===document.querySelector('.muon-batch__submit')"));
      report.leave.push({width,size,theme,mode,forwardAndReverse:true});
      if(width===320&&size===100&&mode==='normal'){
        await run(()=>{const i=document.querySelector('[name=batchNew]');i.value='0';i.focus();i.dispatchEvent(new Event('input',{bubbles:true}));});
        await b.key('Enter','Enter',13);
        const row=await run(inspect,4);
        assert(row.focused&&row.visible&&row.ratio>=3);
        await b.screenshot(path.join(out,'invalid-'+theme+'.png'));
      }
    }
    assert.deepEqual(b.exceptions,[]);
    console.log(JSON.stringify({outlines:report.outlines.length,minimumContrast:Math.min(...report.outlines.map(r=>r.ratio)),tabExitCases:report.leave.length,exceptions:0}));
  }finally{
    fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');
    await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]}).catch(()=>{});await b.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1});
