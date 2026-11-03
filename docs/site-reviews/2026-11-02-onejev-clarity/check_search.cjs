// Current article-body queries through both language entrances.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('../../../scripts/lib/browser.cjs');
const [base, portText, out] = process.argv.slice(2);
const articles = [{route:'/posts/ai/jev-decision-layer/',queries:['OneJev','OneJev logits','OneJev 本地','OneJev 训练','Flide','温度 校准']}];
(async () => {
  fs.mkdirSync(out, {recursive: true});
  const b = await connect(Number(portText)), report = {cases: [], exceptions: b.exceptions};
  try {
    for (const entrance of ['/', '/zh-cn/']) for (const width of [320, 1440]) {
      await b.viewport(width); await b.navigate(base + entrance);
      for (const {route, queries} of articles) {
        const response = await fetch(base + route); assert(response.ok); const html = await response.text();
        for (const query of queries) {
          const groups = await b.evaluate(`window.hextraSearch.search(${JSON.stringify(query)})`);
          const group = groups.find(g=>g.route === route); assert(group, 'Missing article: ' + query);
          const ids = await b.evaluate(`(()=>{const d=new DOMParser().parseFromString(${JSON.stringify(html)},'text/html');
            return ${JSON.stringify(group.matches.map(m=>m.route))}.map(route=>{const u=new URL(route,location.href);
              return {route,exists:!u.hash||!!d.getElementById(decodeURIComponent(u.hash.slice(1)))}})})()`);
          assert(ids.length && ids.every(i=>i.exists), JSON.stringify({query, ids}));
          report.cases.push({entrance, width, article: route, query, rank: groups.indexOf(group)+1, matches: ids});
        }
      }
    }
    assert.deepEqual(b.exceptions, []);
    console.log(JSON.stringify({cases: report.cases.length, exceptions: 0}));
  } finally {
    fs.writeFileSync(path.join(out, 'results.json'), JSON.stringify(report, null, 2) + '\n');
    await b.close();
  }
})().catch(e=>{console.error(e);process.exitCode=1});
