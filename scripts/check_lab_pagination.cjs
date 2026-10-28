#!/usr/bin/env node
// Check native PDF pagination of compact definition cards, using Poppler text bounds.
// Run serially with other browser suites; requires python3 and pdftotext.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const {connect} = require('./lib/browser.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13219';
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-lab-pagination';
const captureOnly = process.argv.includes('--capture-only');
const inspect = String.raw`
import json, sys, xml.etree.ElementTree as ET
root = ET.fromstring(sys.stdin.buffer.read())
ns = {'x': 'http://www.w3.org/1999/xhtml'}
cards = json.loads(sys.argv[1])
compact = lambda s: ''.join(s.split())
pages = []
for page in root.findall('.//x:page', ns):
    words = [dict(text=w.text or '', **{k: float(v) for k,v in w.attrib.items()})
             for w in page.findall('.//x:word', ns)]
    pages.append(words)
def matches(words, text):
    text = compact(text)
    def walk(group, remaining):
        if not remaining:
            yield {key: reducer(w[key] for w in group) for key,reducer in
                   [('xMin', min), ('yMin', min), ('xMax', max), ('yMax', max)]}
            return
        if len(group) >= 12:
            return
        previous = group[-1] if group else None
        for word in words:
            part = compact(word['text'])
            if not part or not remaining.startswith(part) or word in group:
                continue
            if previous:
                center = lambda w: (w['yMin']+w['yMax'])/2
                height = max(word['yMax']-word['yMin'], previous['yMax']-previous['yMin'])
                dy = center(word)-center(previous)
                same_line = abs(dy) < height*.3 and -1 <= word['xMin']-previous['xMax'] < height*.55
                next_line = height*.3 <= dy < height*1.7 and abs(word['xMin']-group[0]['xMin']) < 3
                if not (same_line or next_line):
                    continue
            yield from walk(group+[word], remaining[len(part):])
    yield from walk([], text)
rows = []
for card in cards:
    locations = []
    for number, words in enumerate(pages, 1):
        labels = list(matches(words, card['label']))
        values = list(matches(words, card['value']))
        for label in labels:
            for value in values:
                if abs(value['xMin']-label['xMin']) < 3 and value['yMin'] > label['yMin']+2 and value['yMin'] < label['yMax']+60:
                    locations.append({'page': number, 'label': label, 'value': value})
    rows.append(dict(card, passed=bool(locations), locations=locations))
print(json.dumps({'pages': len(pages), 'cards': rows, 'passed': all(r['passed'] for r in rows)}, ensure_ascii=False))
`;
(async () => {
  fs.mkdirSync(output, {recursive:true});
  const b = await connect(port), report = {cases:[], exceptions:b.exceptions};
  let savedTheme;
  try {
    await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]});
    await b.navigate(base+'/labs/supermimic/');
    savedTheme = await b.evaluate("localStorage.getItem('color-theme')");
    for (const route of ['/labs/supermimic/', '/labs/multiview-geometry/']) {
      for (const theme of ['light','dark','warm']) for (const size of ['100%','200%']) {
        await b.viewport(700,1000); await b.navigate(base+route);
        await b.evaluate(`document.documentElement.dataset.theme=${JSON.stringify(theme)};
          document.documentElement.style.fontSize=${JSON.stringify(size)};document.fonts.ready`);
        const cards = await b.evaluate(`Array.from(document.querySelectorAll('dl > div')).map(e=>({
          label:e.querySelector('dt').textContent.trim(),value:e.querySelector('dd').textContent.trim()}))`);
        assert(cards.length >= 10, 'Missing default result cards: '+route);
        const before = await b.evaluate("JSON.stringify([...document.querySelectorAll('input')].map(e=>[e.value,e.checked]))");
        for (const paper of [{name:'a4',width:8.27,height:11.69},{name:'letter',width:8.5,height:11}]) {
          const file=path.join(output,`${route.split('/')[2]}-${theme}-${size.slice(0,-1)}-${paper.name}.pdf`);
          const pdf=await b.cdp('Page.printToPDF',{printBackground:false,paperWidth:paper.width,paperHeight:paper.height,
            marginTop:.55,marginBottom:.55,marginLeft:.55,marginRight:.55});
          fs.writeFileSync(file,Buffer.from(pdf.data,'base64'));
          const xml=execFileSync('pdftotext',['-bbox',file,'-'],{encoding:'utf8',timeout:10000});
          const result=JSON.parse(execFileSync('python3',['-B','-c',inspect,JSON.stringify(cards)],
            {input:xml,encoding:'utf8',timeout:10000}));
          report.cases.push({route,theme,size,paper:paper.name,file,...result});
        }
        const after = await b.evaluate("JSON.stringify([...document.querySelectorAll('input')].map(e=>[e.value,e.checked]))");
        assert.equal(after,before,'Printing changed experiment inputs');
      }
    }
    report.summary={cases:report.cases.length,cardPairs:report.cases.reduce((n,c)=>n+c.cards.length,0),
      failures:report.cases.filter(c=>!c.passed).length,exceptions:b.exceptions.length};
    console.log(JSON.stringify(report.summary,null,2));
    if(!captureOnly){assert.equal(report.summary.failures,0,'Definition labels/values split in native PDFs; see results.json');assert.equal(b.exceptions.length,0);}
  } finally {
    await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]}).catch(()=>{});
    if(savedTheme!==undefined)await b.evaluate(savedTheme===null?"localStorage.removeItem('color-theme')":
      'localStorage.setItem("color-theme",'+JSON.stringify(savedTheme)+')').catch(()=>{});
    fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(report,null,2)+'\n');await b.close();
  }
})().catch(e=>{console.error(e);process.exitCode=1;});
