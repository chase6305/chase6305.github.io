// Native, tagged PDFs for the revised article bodies; use a private CDP endpoint.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const {connect} = require('../../../scripts/lib/browser.cjs');
const [base, portText, out] = process.argv.slice(2);
const articles = [
 {slug:'muon-optimizer',focus:'.content a[href="#implementation"]',tokens:['geometry','implementation','qk-clip'],phrases:['Newton–Schulz','4096','12.8','muon_math.py','QK-Clip','toy_training.py','matrix_update.py','SHA-256']},
 {slug:'muon-learning-rate-batch-size',focus:'.content a[href="../muon-optimizer/"]',tokens:['mean-field','spectrum','weight-decay'],phrases:['BatchSize','0.02199','batch_math.py','Hessian','EMA','0.68269','mean_field_probe.py','保持累计收缩的衰减系数']},
];
(async () => {
  fs.mkdirSync(out, {recursive: true});
  const b = await connect(Number(portText)), report = {pdfs: [], exceptions: b.exceptions};
  try {
    for (const article of articles) {
      const route = '/posts/ai/' + article.slug + '/';
      const response = await fetch(base + route); assert(response.ok);
      const html = await response.text();
      assert(article.tokens.every(token=>html.includes('id=' + token) || html.includes('id="' + token + '"')));
      const htmlSha256 = crypto.createHash('sha256').update(html).digest('hex');
      for (const [scripts, size, theme, paper] of [
        [true, 100, 'light', 'a4'], [true, 200, 'dark', 'a4'],
        [false, 100, 'warm', 'a4'], [false, 200, 'warm', 'letter'],
      ]) {
        await b.viewport(1440);
        await b.cdp('Emulation.setScriptExecutionDisabled', {value: !scripts});
        await b.navigate(base + route);
        await b.evaluate(`document.documentElement.classList.remove('dark','warm');
          ${theme === 'light' ? '' : `document.documentElement.classList.add('${theme}');`}
          document.documentElement.style.colorScheme='${theme === 'dark' ? 'dark' : 'light'}';
          document.documentElement.style.fontSize='${size}%';
          Promise.all([document.fonts.ready,...[...document.querySelectorAll('.content img')].map(i=>{i.loading='eager';return i.decode()})])`);
        const sourceCheck = await b.evaluate(`(()=>{
          const expected=new DOMParser().parseFromString(${JSON.stringify(html)},'text/html');
          const pick=d=>({headings:[...d.querySelectorAll('.content h2,.content h3,.content h4')].map(e=>[e.id,e.textContent.trim()]),
            math:[...d.querySelectorAll('.content annotation[encoding="application/x-tex"]')].map(e=>e.textContent),
            code:[...d.querySelectorAll('.hextra-code-block code')].map(e=>e.textContent),
            captions:[...d.querySelectorAll('.article-figure figcaption')].map(e=>e.textContent.trim())});
          const a=pick(expected),c=pick(document);return Object.fromEntries(Object.keys(a).map(k=>[k,{same:JSON.stringify(a[k])===JSON.stringify(c[k]),count:a[k].length}]))})()`);
        assert(Object.values(sourceCheck).every(r=>r.same), JSON.stringify(sourceCheck));
        await b.evaluate(`(()=>{const e=document.querySelector(${JSON.stringify(article.focus)});e.focus();
          window.__printBefore={scroll:scrollY,open:[...document.querySelectorAll('.content details')].map(d=>d.open),active:document.activeElement};
          return document.activeElement===e})()`);
        await b.pause(120);
        const result = await b.cdp('Page.printToPDF', {printBackground: true,
          paperWidth: paper === 'letter' ? 8.5 : 8.27, paperHeight: paper === 'letter' ? 11 : 11.69,
          marginTop: .4, marginBottom: .4, marginLeft: .4, marginRight: .4, generateTaggedPDF: true});
        const file = path.join(out, `${article.slug}-${scripts}-${size}-${theme}-${paper}.pdf`);
        fs.writeFileSync(file, Buffer.from(result.data, 'base64'));
        await b.pause(200);
        const restored = await b.evaluate(`({scrollError:Math.abs(scrollY-__printBefore.scroll),
          focus:document.activeElement===__printBefore.active,
          disclosures:JSON.stringify([...document.querySelectorAll('.content details')].map(d=>d.open))===JSON.stringify(__printBefore.open)})`);
        assert(restored.focus && restored.disclosures && restored.scrollError <= 1, JSON.stringify(restored));
        const info = execFileSync('pdfinfo', [file], {encoding: 'utf8'});
        assert(/Tagged:\s+yes/.test(info));
        const text = execFileSync('pdftotext', ['-layout', file, '-'], {encoding: 'utf8', maxBuffer: 8 * 1024 * 1024});
        const compact = text.replace(/[\s\u200b]/g, '');
        for (const phrase of article.phrases) assert(compact.includes(phrase), 'Missing current PDF text: ' + phrase);
        fs.writeFileSync(file.replace(/\.pdf$/, '.txt'), text);
        report.pdfs.push({route, scripts, size, theme, paper, htmlSha256, sourceCheck, file,
          pages: Number(info.match(/Pages:\s+(\d+)/)[1]), tagged: true, restored,
          requiredPhrases: article.phrases, scriptExecutionDisabledEntireCase: !scripts});
      }
    }
    assert.deepEqual(b.exceptions, []);
    console.log(JSON.stringify({exports: report.pdfs.length, pages: report.pdfs.map(p=>p.pages), exceptions: 0}));
  } finally {
    fs.writeFileSync(path.join(out, 'exports.json'), JSON.stringify(report, null, 2) + '\n');
    await b.cdp('Emulation.setScriptExecutionDisabled', {value: false}).catch(()=>{});
    await b.close();
  }
})().catch(e=>{console.error(e);process.exitCode=1});
