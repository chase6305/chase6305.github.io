// Capture the three new figures with their complete captions; private CDP only.
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('../../../scripts/lib/browser.cjs');
const [base, port, out] = process.argv.slice(2);
const ids = ['fig-jev-onejev-deployment', 'fig-onejev-probability-readout', 'fig-onejev-state-forking'];
(async () => {
  fs.mkdirSync(out, {recursive:true});
  const b = await connect(Number(port));
  const report = {captures:[], exceptions:b.exceptions};
  try {
    for (const [width, size, theme] of [[320,100,'light'],[320,100,'dark'],[320,200,'dark'],[1440,100,'light']]) {
      await b.viewport(width);
      await b.navigate(base + '/posts/ai/jev-decision-layer/');
      await b.evaluate(`document.documentElement.classList.remove('dark','warm');
        ${theme==='dark' ? "document.documentElement.classList.add('dark');" : ''}
        document.documentElement.style.fontSize='${size}%';
        Promise.all([document.fonts.ready,...[...document.querySelectorAll('.content img')].map(i=>{i.loading='eager';return i.decode()})])`);
      for (const id of ids) {
        const capture = await b.evaluate(`(()=>{const f=document.getElementById('${id}');
          f.scrollIntoView({block:'start'});const r=f.getBoundingClientRect();
          return {x:r.left+scrollX,y:r.top+scrollY,width:r.width,height:r.height,
            caption:f.querySelector('figcaption').textContent.trim(),image:f.querySelector('img').currentSrc}})()`);
        const file = path.join(out,`${id}-${width}-${size}-${theme}.png`);
        const shot = await b.cdp('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,
          clip:{x:capture.x,y:capture.y,width:capture.width,height:capture.height,scale:1}});
        fs.writeFileSync(file,Buffer.from(shot.data,'base64'));
        report.captures.push({id,viewportWidth:width,size,theme,file,...capture});
      }
    }
  } finally {
    fs.writeFileSync(path.join(out,'captures.json'),JSON.stringify(report,null,2)+'\n');
    await b.close();
  }
})().catch(e=>{console.error(e);process.exitCode=1});
