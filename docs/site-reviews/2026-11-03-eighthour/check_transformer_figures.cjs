#!/usr/bin/env node
// Article-scoped serial check: node check-transformer-figures.cjs <baseURL> <CDP-port> <out>
const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const {connect} = require('../../../scripts/lib/browser.cjs');
const base=(process.argv[2]||'http://127.0.0.1:1313').replace(/\/$/,''), port=Number(process.argv[3]||9240);
const out=process.argv[4]||'/tmp/chase-site-eight-hour-20261103/transformer-figures-browser';
const route='/posts/ai/transformer-attention/';
const targets=[['softmax-row-calculation.svg',5],['causal-mask-calculation.svg',6],['multi-head-shapes.svg',7],
  ['moe-token-routing.svg',17],['attention-scoring-pipeline.svg',24]].map(([file,number])=>({file,number}));
const formulaTargets=[{name:'FFN',needle:'\\operatorname{FFN}(x)='},{name:'SwiGLU',needle:'\\operatorname{SwiGLU}(x)'}];
const source=fs.readFileSync(path.join(__dirname, '../../../content/posts/ai/transformer-attention/index.md'),'utf8');
for(const target of targets){
  const blocks=[...source.matchAll(/<figure\b[^>]*>[\s\S]*?<\/figure>/g)].map(m=>m[0]).filter(s=>s.includes('src="assets/'+target.file+'"'));
  assert.equal(blocks.length,1,target.file);
  const text=blocks[0].match(/class="article-figure__text"[^>]*>([\s\S]*?)<\/span>/)?.[1];
  assert(text&&!/<[^>]+>/.test(text),'Review caption markup: '+target.file);
  target.caption=text.replace(/\s+/g,' ').trim();
}
function parseSVG(text,url){
  const root=text.match(/<svg\b([^>]*)>/);assert(root,'Not SVG: '+url);
  const attr=name=>root[1].match(new RegExp('\\b'+name+'="([^"]+)"'))?.[1];
  const width=Number(attr('width')),height=Number(attr('height')),viewBox=attr('viewBox')?.trim().split(/[ ,]+/).map(Number);
  assert(width>0&&height>0&&viewBox?.length===4&&viewBox.every(Number.isFinite)&&viewBox[2]>0&&viewBox[3]>0,url);
  assert.equal(width/height,viewBox[2]/viewBox[3],url);return{width,height,viewBox};
}
function readFigures(targets){
  const lang=e=>e.closest('[lang]')?.getAttribute('lang')||document.documentElement.lang;
  return window.__tfImages.map((image,index)=>{
    const figure=image.closest('.article-figure'),caption=figure.querySelector('.article-figure__text'),link=image.closest('a');
    const r=image.getBoundingClientRect(),cr=caption.getBoundingClientRect(),range=document.createRange();range.selectNodeContents(caption);
    const style=getComputedStyle(caption),inside=[...range.getClientRects()].every(x=>x.left>=cr.left-1&&x.right<=cr.right+1&&x.top>=cr.top-1&&x.bottom<=cr.bottom+1);
    return{file:targets[index].file,label:figure.querySelector('.article-figure__number').textContent.trim(),source:image.currentSrc,
      original:link.href,target:link.target,rel:link.rel,nativeName:link.getAttribute('aria-label'),alt:image.alt,
      imageLanguage:lang(image),captionLanguage:lang(caption),captionText:caption.textContent.replace(/\s+/g,' ').trim(),
      captionVisible:style.display!=='none'&&style.visibility==='visible'&&cr.width>0&&cr.height>0,
      captionClipped:caption.scrollHeight>caption.clientHeight+1||caption.scrollWidth>caption.clientWidth+1,captionRectsInside:inside,
      width:Number(image.getAttribute('width')),height:Number(image.getAttribute('height')),naturalWidth:image.naturalWidth,naturalHeight:image.naturalHeight,
      renderWidth:r.width,renderHeight:r.height,top:r.top+scrollY,pageOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1};
  });
}
(async()=>{
  fs.mkdirSync(out,{recursive:true});const b=await connect(port);
  const report={route,cases:[],zoom:[],SVGs:{},exceptions:b.exceptions,
    scope:'24 layouts/120 figure inspections/48 formula regions; 10 keyboard viewer sessions; 6 screenshots.',
    fullResolution:'Ready checks original SVG intrinsic dimensions. Existing viewer has no separate 100%-zoom button.'};
  const page=(fn,...args)=>b.evaluate('('+fn.toString()+')('+args.map(x=>JSON.stringify(x)).join(',')+')');
  const write=()=>fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');
  const wait=async(fn,args=[],message='Condition did not settle',limit=5000)=>{
    const end=Date.now()+limit;while(Date.now()<end){if(await page(fn,...args))return;await b.pause(30)}throw Error(message);
  };
  const shiftTab=async()=>{const e={key:'Tab',code:'Tab',windowsVirtualKeyCode:9,modifiers:8};
    await b.cdp('Input.dispatchKeyEvent',{type:'keyDown',...e});await b.cdp('Input.dispatchKeyEvent',{type:'keyUp',...e});};
  try{
    for(const scripts of [true,false])for(const width of [320,768,1440])for(const size of [100,200])for(const theme of ['light','dark']){
      const context={scripts,width,size,theme};report.activeCase=context;
      await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});await b.viewport(width);await b.navigate(base+route);
      await page(async(size,theme)=>{
        document.documentElement.style.fontSize=size+'%';document.documentElement.classList.remove('light','dark','warm');
        document.documentElement.classList.add(theme);document.documentElement.dataset.theme=theme;
        document.documentElement.getBoundingClientRect();await document.fonts.ready;window.scrollTo(0,0);
      },size,theme);await b.pause(160);
      const reservation=await page(async targets=>{
        const all=[...document.querySelectorAll('.content img')];
        window.__tfImages=targets.map(target=>{const matches=all.filter(i=>new URL(i.src).pathname.endsWith('/assets/'+target.file));
          if(matches.length!==1)throw Error('Expected one rendered image: '+target.file);return matches[0]});
        const before=window.__tfImages.map(image=>{const r=image.getBoundingClientRect();return{top:r.top+scrollY,width:r.width,height:r.height}});
        // Decode once per navigation, including intervening images that could shift later figures.
        await Promise.all(all.map(image=>{image.loading='eager';return image.decode()}));return before;
      },targets);await b.pause(160);
      const figures=await page(readFigures,targets);
      for(let index=0;index<targets.length;index++){
        const target=targets[index],state=figures[index];report.activeFigure=state;
        if(!report.SVGs[state.original]){const response=await fetch(state.original);assert.equal(response.status,200,state.original);
          report.SVGs[state.original]=parseSVG(await response.text(),state.original)}
        const svg=report.SVGs[state.original],e=JSON.stringify({context,state,svg,reservation:reservation[index]});
        assert.equal(state.label,'图 '+target.number,e);assert.equal(state.captionText,target.caption,e);
        assert(state.captionVisible&&!state.captionClipped&&state.captionRectsInside,e);assert(!state.pageOverflow,e);
        assert.equal(state.width,svg.width,e);assert.equal(state.height,svg.height,e);
        assert.equal(state.naturalWidth,svg.width,e);assert.equal(state.naturalHeight,svg.height,e);
        assert(state.renderWidth>0&&state.renderHeight>0,e);
        assert(Math.abs(state.renderWidth-state.renderHeight*svg.width/svg.height)<=1,e);
        assert(Math.abs(state.top-reservation[index].top)<=2,e);assert(Math.abs(state.renderHeight-reservation[index].height)<=1,e);
        assert.equal(state.target,'_blank',e);assert(state.rel.split(/\s+/).includes('noopener'),e);
        assert.equal(new URL(state.original).pathname.split('/').at(-1),target.file,e);
        assert(state.alt.length>8&&state.nativeName==='查看大图：'+state.alt,e);
        assert.equal(state.imageLanguage,'zh-CN',e);assert.equal(state.captionLanguage,'zh-CN',e);
      }
      await page(formulaTargets=>{
        const all=[...document.querySelectorAll('.content .katex-display')];
        window.__tfRegions=formulaTargets.map(target=>{
          const matches=all.filter(r=>r.querySelector('annotation[encoding="application/x-tex"]')?.textContent.includes(target.needle));
          if(matches.length!==1)throw Error('Expected one displayed equation: '+target.name);return matches[0];
        });
      },formulaTargets);
      if(scripts)await wait(()=>window.__tfRegions.every(r=>r.scrollWidth>r.clientWidth+1?r.tabIndex===0:!r.hasAttribute('tabindex')),
        [],'Changed equation tab stops did not match actual overflow');
      const formulas=[];
      for(let index=0;index<formulaTargets.length;index++){
        const initial=await page(index=>{const r=window.__tfRegions[index];r.scrollIntoView({block:'center'});r.scrollLeft=0;
          return{width:r.clientWidth,scrollWidth:r.scrollWidth,tabIndex:r.tabIndex,hasTabIndex:r.hasAttribute('tabindex'),
            overflowX:getComputedStyle(r).overflowX,language:r.closest('[lang]')?.getAttribute('lang')};},index);
        const overflowing=initial.scrollWidth>initial.width+1,e=JSON.stringify({context,formula:formulaTargets[index].name,initial});
        assert(initial.width>0,e);assert.equal(initial.language,'zh-CN',e);
        if(!scripts||overflowing)assert.equal(initial.tabIndex,0,e);else assert(!initial.hasTabIndex,e);
        let keyboard=null;
        if(!scripts||overflowing){
          await page(index=>window.__tfRegions[index].focus(),index);await shiftTab();await b.key('Tab','Tab',9);
          assert(await page(index=>document.activeElement===window.__tfRegions[index],index),'Equation not reachable by real Tab: '+e);
          if(overflowing){
            assert(['auto','scroll'].includes(initial.overflowX),e);await page(index=>{window.__tfRegions[index].scrollLeft=0},index);
            for(let n=0;n<3;n++)await b.key('ArrowRight','ArrowRight',39);await b.pause(180);
            keyboard=await page(index=>({focused:document.activeElement===window.__tfRegions[index],distance:window.__tfRegions[index].scrollLeft}),index);
            assert(keyboard.focused&&keyboard.distance>0,'Equation cannot scroll with ArrowRight: '+e);
          }
          await b.key('Tab','Tab',9);
        }
        formulas.push({name:formulaTargets[index].name,...initial,overflowing,keyboard});
        if(scripts&&width===320&&size===200&&theme==='light'&&index===1){
          await page(index=>{const r=window.__tfRegions[index];r.scrollLeft=0;r.scrollIntoView({block:'center'})},index);
          await b.screenshot(path.join(out,'SwiGLU-320-200-light.png'));
        }
      }
      const zoomCase=scripts&&((width===320&&size===200&&theme==='light')||(width===1440&&size===100&&theme==='dark'));
      if(zoomCase)for(let index=0;index<targets.length;index++){
        const before=await page(index=>{const link=window.__tfImages[index].closest('a');link.scrollIntoView({block:'center'});link.focus({preventScroll:true});
          return{scroll:scrollY,overflow:document.documentElement.style.overflow}},index);
        await b.key('Enter','Enter',13);
        await wait(()=>document.querySelector('[data-image-viewer][open]')?.dataset.state==='ready',[],
          'Original SVG did not become Ready: '+targets[index].file,10000);
        const viewer=await page(()=>{
          const d=document.querySelector('[data-image-viewer][open]'),image=d.querySelector('img'),r=d.getBoundingClientRect(),original=d.querySelector('[data-image-viewer-original]');
          return{source:image.currentSrc,naturalWidth:image.naturalWidth,naturalHeight:image.naturalHeight,alt:image.alt,language:d.lang,
            modal:d.matches(':modal'),focusInside:d.contains(document.activeElement),status:d.querySelector('[role="status"]').textContent,
            original:original.href,target:original.target,fits:r.left>=-1&&r.top>=-1&&r.right<=innerWidth+1&&r.bottom<=innerHeight+1,
            horizontalOverflow:d.scrollWidth>d.clientWidth+1};
        });
        const image=figures[index],svg=report.SVGs[image.original],e=JSON.stringify({context,file:targets[index].file,viewer});
        assert.equal(viewer.source,image.original,e);assert.equal(viewer.original,image.original,e);assert.equal(viewer.target,'_blank',e);
        assert.equal(viewer.language,'zh-CN',e);assert.equal(viewer.status,'清晰原图已显示。',e);assert.equal(viewer.alt,image.alt,e);
        assert.equal(viewer.naturalWidth,svg.width,e);assert.equal(viewer.naturalHeight,svg.height,e);
        assert(viewer.modal&&viewer.focusInside&&viewer.fits&&!viewer.horizontalOverflow,e);
        await b.key('Escape','Escape',27);
        await wait(index=>!document.querySelector('[data-image-viewer][open]')&&document.activeElement===window.__tfImages[index].closest('a'),
          [index],'Viewer opener focus not restored');
        const after=await page(()=>({scroll:scrollY,overflow:document.documentElement.style.overflow}));
        assert.equal(after.overflow,before.overflow,e);assert(Math.abs(after.scroll-before.scroll)<=2,e);
        report.zoom.push({...context,file:targets[index].file,...viewer,focusRestored:true,scrollRestored:true});
      }
      if(!scripts)assert(await page(()=>!document.querySelector('[data-image-viewer]')),'Scripts-off page enhanced its viewer');
      if(scripts&&width===320&&size===200&&theme==='light')for(let index=0;index<targets.length;index++){
        await page(index=>{window.__tfImages[index].scrollIntoView({block:'start'});
          window.scrollBy(0,-document.querySelector('.hextra-nav-container').getBoundingClientRect().height-16)},index);
        await b.screenshot(path.join(out,'figure-'+targets[index].number+'-320-200-light.png'));
      }
      report.cases.push({...context,figures,formulas});delete report.activeFigure;write();
      console.log(report.cases.length,scripts,width,size,theme,'5 SVGs/captions/native links and 2 equations passed');
    }
    assert.equal(report.cases.length,24);assert.equal(report.zoom.length,10);assert.equal(b.exceptions.length,0,JSON.stringify(b.exceptions));
    report.summary={layouts:report.cases.length,figureInspections:report.cases.reduce((n,c)=>n+c.figures.length,0),
      equationInspections:report.cases.reduce((n,c)=>n+c.formulas.length,0),keyboardOverflowCases:report.cases.reduce((n,c)=>n+c.formulas.filter(f=>f.keyboard).length,0),
      zooms:report.zoom.length,runtimeExceptions:b.exceptions.length};delete report.activeCase;console.log(JSON.stringify(report.summary));
  }catch(error){report.failure={message:error.message,stack:error.stack};throw error}
  finally{write();await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});await b.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
