// Content-scoped reading checks. Run serially on an existing loopback CDP.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const {connect, pause} = require('../../../scripts/lib/browser.cjs');
const base = process.argv[2];
const port = Number(process.argv[3]);
const output = process.argv[4];
const selectedPost = process.argv[5]?.startsWith('--') ? null : process.argv[5];
const axeIndex = process.argv.indexOf('--axe');
const axeSource = axeIndex < 0 ? null : fs.readFileSync(process.argv[axeIndex + 1], 'utf8');
if (axeSource) assert.equal(crypto.createHash('sha256').update(axeSource).digest('hex'),
  '20c09fe157a8a34a30e241aaa1fcdade657734f08ab379ecfbeb7d45cc46e878');
const downloads = {
  card: 'contact_sanity.py', tampagent: 'belief_contract_sanity.py',
  cuarms: 'coordination_sanity.py', 'weight-space': 'weight_space_sanity.py',
};
const posts = [
  ['card', '/posts/ai/card-retargeting/'],
  ['tampagent', '/posts/planner/tampagent/'],
  ['cuarms', '/posts/robotics/cuarms/'],
  ['weight-space', '/posts/ai/weight-space-programming/'],
].filter(([name]) => !selectedPost || name === selectedPost);
assert(posts.length, 'Unknown article filter');

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const b = await connect(port);
  const report = {layouts: [], tables: [], formulas: [], zoom: [], print: [], downloads: [], semantics: []};
  try {
    for (const scripts of [true, false]) {
      await b.cdp('Emulation.setScriptExecutionDisabled', {value: !scripts});
      for (const [name, route] of posts) {
        for (const width of [320, 768, 1440]) {
          await b.viewport(width, 1000);
          await b.navigate(base + route);
          await b.evaluate(`Promise.all([document.fonts.ready,...[...document.querySelectorAll('article img')].map(i=>{i.loading='eager';return i.decode()})])`);
          for (const theme of ['light', 'dark', 'warm']) {
            await b.evaluate(`document.documentElement.classList.remove('light','dark','warm');document.documentElement.classList.add('${theme}')`);
            for (const [size, spacing] of [[100, false], [200, false], [200, true]]) {
              await b.evaluate(`document.documentElement.style.fontSize='${size}%';document.querySelector('#article-check-spacing')?.remove();${spacing ? `(()=>{const s=document.createElement('style');s.id='article-check-spacing';s.textContent='.content p,.content li,.content td,.content th{line-height:1.5!important;letter-spacing:.12em!important;word-spacing:.16em!important}.content p{margin-bottom:2em!important}';document.head.append(s)})()` : ''}`);
              await pause(100);
              const state = await b.evaluate(`(()=>{const a=document.querySelector('article');return {
                title:document.title,
                overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1,
                images:[...a.querySelectorAll('img')].map(i=>({alt:i.alt,ok:i.complete&&i.naturalWidth>0})),
                captions:[...a.querySelectorAll('figcaption')].map(c=>({text:c.innerText,clipped:[...c.querySelectorAll('span')].some(s=>s.scrollWidth>s.clientWidth+2)})),
                mathErrors:a.querySelectorAll('.katex-error').length,
                formulas:a.querySelectorAll('.katex').length,
                missingAnchors:[...a.querySelectorAll('a[href^="#"]')].filter(e=>!document.getElementById(decodeURIComponent(e.hash.slice(1)))).map(e=>e.hash),
                related:a.querySelectorAll('.blog-related__card').length
              }})()`);
              assert(!state.overflow, JSON.stringify({name,width,theme,size,spacing,scripts,state}));
              assert.equal(state.images.length, 2);
              assert(state.images.every(i=>i.ok&&i.alt));
              assert(state.captions.length===2&&state.captions.every(c=>!c.clipped));
              assert(state.formulas>5&&!state.mathErrors);
              assert(!state.missingAnchors.length);
              report.layouts.push({name,scripts,width,theme,size,spacing,...state});
            }
          }
          await b.evaluate(`document.querySelector('#article-check-spacing')?.remove();document.documentElement.style.fontSize='';document.documentElement.classList.remove('warm');document.documentElement.classList.add('dark')`);
          if (width===320) {
            for (const size of [100,200]) {
              await b.evaluate(`document.documentElement.style.fontSize='${size}%'`);
              await pause(100);
              for (const [kind, selector] of [['table','article table'],['formula','article .katex-display']]) {
                const regions = await b.evaluate(`[...document.querySelectorAll('${selector}')].map((t,index)=>({index,overflow:t.scrollWidth>t.clientWidth+2})).filter(t=>t.overflow)`);
                for (const {index} of regions) {
                  const focus = await b.evaluate(`(()=>{const t=document.querySelectorAll('${selector}')[${index}];t.scrollIntoView({block:'center'});t.scrollLeft=0;t.focus();return {active:document.activeElement===t,tab:t.tabIndex}})()`);
                  assert(focus.active&&focus.tab===0,JSON.stringify({name,scripts,size,kind,index,focus}));
                  await b.key('ArrowRight','ArrowRight',39);
                  await pause(180);
                  assert(await b.evaluate(`document.querySelectorAll('${selector}')[${index}].scrollLeft>0`));
                  await b.evaluate(`document.querySelectorAll('${selector}')[${index}].blur()`);
                  report[kind==='table'?'tables':'formulas'].push({name,scripts,size,index,passed:true});
                }
              }
            }
            await b.evaluate(`document.documentElement.style.fontSize=''`);
            for (const number of [1,2]) {
              await b.evaluate(`document.querySelector('#figure-${number}').scrollIntoView({block:'center'})`);
              if (scripts) {
                await b.screenshot(path.join(output,`${name}-figure-${number}-mobile-dark.png`));
                await b.evaluate(`document.querySelector('#figure-${number} img').click()`);
                await pause(180);
                assert(await b.evaluate(`!!document.querySelector('dialog[open]')`));
                await b.key('Escape','Escape',27);
                await pause(100);
                assert(await b.evaluate(`!document.querySelector('dialog[open]')`));
                report.zoom.push({name,number,passed:true});
              } else {
                assert(await b.evaluate(`!!document.querySelector('#figure-${number} a[href]')`));
              }
            }
          }
          if (width===1440&&scripts) {
            await b.evaluate('scrollTo(0,0)');
            await b.screenshot(path.join(output,`${name}-desktop.png`));
          }
        }
        await b.viewport(1440,1000);
        await b.navigate(base+route);
        await b.evaluate(`document.documentElement.classList.add('dark');Promise.all([document.fonts.ready,...[...document.querySelectorAll('article img')].map(i=>{i.loading='eager';return i.decode()})])`);
        await b.cdp('Emulation.setEmulatedMedia',{media:'print'});
        const bounds = await b.evaluate(`(()=>{const a=document.querySelector('article'),r=a.getBoundingClientRect();return {width:r.width,bad:[...a.querySelectorAll('figure,table,.katex-display')].filter(e=>{const t=e.getBoundingClientRect();return t.left<r.left-2||t.right>r.right+2}).map(e=>({tag:e.tagName,class:e.className,width:e.getBoundingClientRect().width})),mathErrors:a.querySelectorAll('.katex-error').length}})()`);
        assert(!bounds.bad.length&&!bounds.mathErrors,JSON.stringify({name,scripts,bounds}));
        const pdf=await b.cdp('Page.printToPDF',{printBackground:true,preferCSSPageSize:true,generateTaggedPDF:true});
        const file=path.join(output,`${name}-${scripts?'scripts':'native'}.pdf`);
        fs.writeFileSync(file,Buffer.from(pdf.data,'base64'));
        report.print.push({name,scripts,file,bounds});
        await b.cdp('Emulation.setEmulatedMedia',{media:'screen'});
        const filename = downloads[name];
        const source = await b.evaluate(`fetch(${JSON.stringify(new URL(route+filename,base).href)}).then(async r=>({status:r.status,text:await r.text()}))`);
        assert.equal(source.status,200);
        const digest = crypto.createHash('sha256').update(source.text).digest('hex');
        const local = path.resolve(__dirname,'../../../content',route.slice(1),filename);
        assert.equal(digest,crypto.createHash('sha256').update(fs.readFileSync(local)).digest('hex'));
        report.downloads.push({name,scripts,filename,sha256:digest});
        if (axeSource) {
          await b.cdp('Emulation.setScriptExecutionDisabled',{value:false});
          await b.evaluate(axeSource);
          const semantics = await b.evaluate(`axe.run(document,{rules:{'color-contrast':{enabled:false}}}).then(r=>({version:r.testEngine.version,violations:r.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),incomplete:r.incomplete.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)}))}))`);
          assert.equal(semantics.version,'4.14.0');
          assert(!semantics.violations.length,JSON.stringify({name,scripts,semantics}));
          report.semantics.push({name,scripts,...semantics});
          await b.cdp('Emulation.setScriptExecutionDisabled',{value:!scripts});
        }
        console.log(JSON.stringify({name,scripts,checked:true}));
      }
    }
    assert(!b.exceptions.length,JSON.stringify(b.exceptions));
    console.log(JSON.stringify({layouts:report.layouts.length,tables:report.tables.length,formulas:report.formulas.length,zoom:report.zoom.length,pdfs:report.print.length,downloads:report.downloads.length,semantics:report.semantics.length,exceptions:0}));
  } finally {
    fs.writeFileSync(path.join(output,'articles.json'),JSON.stringify(report,null,2)+'\n');
    await b.close();
  }
})().catch(e=>{console.error(e);process.exitCode=1});
