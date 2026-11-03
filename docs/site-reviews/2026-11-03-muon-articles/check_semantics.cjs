// Independent axe rules on the two Muon articles, including opened worked examples.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('../../../scripts/lib/browser.cjs');
const [base, portText, out, axePath] = process.argv.slice(2);
const source = fs.readFileSync(axePath, 'utf8');
const axeSha256 = '20c09fe157a8a34a30e241aaa1fcdade657734f08ab379ecfbeb7d45cc46e878';
assert.equal(crypto.createHash('sha256').update(source).digest('hex'), axeSha256);
(async () => {
  fs.mkdirSync(out, {recursive: true});
  const b = await connect(Number(portText)), report = {axeSha256, excludedRules: ['color-contrast'], cases: [], exceptions: b.exceptions};
  try {
    for (const scriptsDuringLoad of [true, false]) for (const width of [320, 1440]) {
      await b.viewport(width);
      for (const route of ['/posts/ai/muon-optimizer/','/posts/ai/muon-learning-rate-batch-size/']) {
        await b.cdp('Emulation.setScriptExecutionDisabled', {value: !scriptsDuringLoad});
        await b.navigate(base + route);
        // Re-enable execution for the injected engine; disabled page-load scripts
        // are not retroactively executed. Native keyboard behavior has its own check.
        await b.cdp('Emulation.setScriptExecutionDisabled', {value: false});
        await b.evaluate(source + '\n;void 0');
        for (const disclosures of ['initial', 'open']) {
          if (disclosures === 'open') await b.evaluate(`document.querySelectorAll('.content details:not(.blog-mobile-toc)').forEach(d=>d.open=true)`);
          const result = await b.evaluate(`axe.run(document,{rules:{'color-contrast':{enabled:false}},resultTypes:['violations','incomplete']})
            .then(r=>({version:r.testEngine.version,violations:r.violations.map(v=>({id:v.id,help:v.help,nodes:v.nodes.map(n=>({target:n.target,html:n.html,failureSummary:n.failureSummary}))})),
              incomplete:r.incomplete.map(v=>({id:v.id,help:v.help,nodes:v.nodes.map(n=>({target:n.target,html:n.html,failureSummary:n.failureSummary}))}))}))`);
          const missingReferences = await b.evaluate(`(()=>{const missing=[];
            for(const name of ['aria-labelledby','aria-describedby','aria-controls','aria-activedescendant'])
              document.querySelectorAll('['+name+']').forEach(e=>{if(!e.checkVisibility())return;
                for(const id of e.getAttribute(name).split(/\\s+/).filter(Boolean))if(!document.getElementById(id))missing.push({name,id})});
            return missing})()`);
          report.cases.push({route, width, scriptsDuringLoad, executionEnabledForAuditOnly: !scriptsDuringLoad,
            disclosures, ...result, missingReferences});
          assert.equal(result.version, '4.14.0');
          assert.deepEqual(result.violations, [], JSON.stringify(report.cases.at(-1)));
          assert.deepEqual(missingReferences, []);
        }
      }
    }
    assert.deepEqual(b.exceptions, []);
    console.log(JSON.stringify({cases: report.cases.length, violations: 0,
      incompleteRules: [...new Set(report.cases.flatMap(c=>c.incomplete.map(r=>r.id)))], exceptions: 0}));
  } finally {
    fs.writeFileSync(path.join(out, 'results.json'), JSON.stringify(report, null, 2) + '\n');
    await b.cdp('Emulation.setScriptExecutionDisabled', {value: false}).catch(()=>{});
    await b.close();
  }
})().catch(e=>{console.error(e);process.exitCode=1});
