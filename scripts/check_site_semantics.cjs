#!/usr/bin/env node
// Independent semantic rules; run serially with the other loopback CDP checks.
// Supply an external axe-core 4.14.0 file with --axe; no runtime site dependency.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('./lib/browser.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13191';
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-site-semantics';
const option = name => {const index = process.argv.indexOf(name);return index < 0 ? null : process.argv[index + 1];};
const axePath = option('--axe');
assert(axePath, 'Supply --axe /path/to/axe-core/axe.min.js (version 4.14.0)');
const source = fs.readFileSync(axePath, 'utf8');
const expectedHash = '20c09fe157a8a34a30e241aaa1fcdade657734f08ab379ecfbeb7d45cc46e878';
assert.equal(crypto.createHash('sha256').update(source).digest('hex'), expectedHash, 'Use the pinned axe-core source');
const scripts = !process.argv.includes('--scripts-off');
const captureOnly = process.argv.includes('--capture-only');
const interactive = process.argv.includes('--interactive');
assert(scripts || !interactive, '--interactive requires page scripts');
const nativeScroll = process.argv.includes('--native-scroll');
assert(!nativeScroll || !scripts, '--native-scroll requires --scripts-off');
const publicRoot = option('--all-pages');
function* htmlFiles(directory) {
  for (const entry of fs.readdirSync(directory, {withFileTypes: true})) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) yield* htmlFiles(file);
    else if (entry.isFile() && file.endsWith('.html')) yield file;
  }
}
const routes = publicRoot ? [...htmlFiles(publicRoot)]
  .filter(file => !/http-equiv\s*=\s*["']?refresh\b/i.test(fs.readFileSync(file, 'utf8')))
  .map(file => '/' + path.relative(publicRoot, file).split(path.sep).map(encodeURIComponent).join('/').replace(/index\.html$/, ''))
  .sort() : ['/', '/zh-cn/', '/posts/', '/posts/page/2/', '/tags/', '/categories/', '/archives/', '/learning-paths/',
    '/projects/', '/zh-cn/projects/', '/404.html', '/zh-cn/404.html', '/posts/ai/supermimic/', '/posts/ai/cosmos3/',
    '/posts/ai/morphmimic/', '/posts/cpp/std/', '/posts/open3d/introduction/', '/posts/robotics/control/robot-filters/',
    '/posts/rl/', '/labs/supermimic/', '/labs/video-to-data/', '/labs/multiview-geometry/', '/labs/rrsi/'];

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const b = await connect(port), report = {axeVersion: '4.14.0', axeSha256: expectedHash,
    excludedRules: ['color-contrast'], scripts, routes: routes.length, cases: [], exceptions: b.exceptions};
  let savedTheme;
  const save = () => fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(report, null, 2)+'\n');
  try {
    report.browser = await b.cdp('Browser.getVersion');
    const load = async (route, width) => {
      await b.viewport(width);
      await b.cdp('Emulation.setScriptExecutionDisabled', {value: !scripts});
      await b.navigate(new URL(route, base).href);
      await b.cdp('Emulation.setScriptExecutionDisabled', {value: false});
      if (savedTheme === undefined) savedTheme = await b.evaluate("localStorage.getItem('color-theme')");
      await b.evaluate(`document.querySelector('[data-item=light]')?.click();
        document.documentElement.classList.remove('dark','warm');document.documentElement.dataset.theme='light';document.fonts.ready`);
    };
    const audit = async context => {
      await b.evaluate(source + '\n;void 0');
      const auditTarget = context.scope ? JSON.stringify(context.scope) : 'document';
      const result = await b.evaluate(`axe.run(${auditTarget},{rules:{'color-contrast':{enabled:false}},
        resultTypes:['violations','incomplete']}).then(r=>({version:r.testEngine.version,
        violations:r.violations.map(v=>({id:v.id,impact:v.impact,help:v.help,helpUrl:v.helpUrl,
          nodes:v.nodes.map(n=>({target:n.target,html:n.html,failureSummary:n.failureSummary}))})),
        incomplete:r.incomplete.map(v=>({id:v.id,count:v.nodes.length,help:v.help,
          nodes:v.nodes.map(n=>({target:n.target,html:n.html,failureSummary:n.failureSummary}))}))}))`);
      const idReferences = await b.evaluate(`(() => {
        const names=['aria-labelledby','aria-describedby','aria-controls','aria-activedescendant'];
        const rows=[];
        document.querySelectorAll(names.map(name=>'['+name+']').join(',')).forEach(e=>{
          if(!e.checkVisibility())return;
          for(const name of names)for(const id of (e.getAttribute(name)||'').trim().split(/\\s+/).filter(Boolean))
            rows.push({attribute:name,id,resolved:!!document.getElementById(id)});
        });
        return{checked:rows.length,missing:rows.filter(row=>!row.resolved)};
      })()`);
      assert.equal(result.version, '4.14.0');
      report.cases.push({...context, ...result, idReferences});save();
      console.log(`${report.cases.length} ${context.width} ${context.route} ${context.state}: ${result.violations.map(v=>v.id).join(',') || 'no violations'}`);
    };
    const waitFor = async expression => {
      const deadline=Date.now()+15000;
      while(Date.now()<deadline) {if(await b.evaluate(expression))return;await b.pause(50);}
      throw Error('Condition timed out: '+expression);
    };
    for (const width of [320, 1440]) for (const route of routes) {
      await load(route,width);await audit({route,width,state:'initial'});
    }
    if(nativeScroll) {
      report.nativeScroll=[];
      for(const [route,selector] of [['/posts/cpp/std/','.content table'],
        ['/posts/ai/supermimic/','.content .katex-display'],['/posts/ai/cosmos3/','.content .katex-display']]) {
        await load(route,320);await b.cdp('Page.bringToFront');
        const focused=await b.evaluate(`(()=>{window.__nativeRegion=[...document.querySelectorAll('${selector}')]
          .find(e=>e.clientWidth>0&&e.scrollWidth>e.clientWidth+2);
          if(!__nativeRegion)return false;__nativeRegion.scrollIntoView({block:'center'});__nativeRegion.scrollLeft=0;
          __nativeRegion.focus();return document.activeElement===__nativeRegion;})()`);assert(focused,route);
        await b.key('ArrowRight','ArrowRight',39);await b.pause(220);
        const distance=await b.evaluate('__nativeRegion.scrollLeft');assert(distance>0,route);
        await b.viewport(1440);await b.pause(100);
        assert(await b.evaluate('document.activeElement===__nativeRegion'),route+' lost native focus');
        report.nativeScroll.push({route,selector,distance,resizeFocus:true});
      }
      report.nativeTasks=[];
      for(const [route,count] of [['/posts/ai/diffusion-models/',6],['/posts/ai/transformer-attention/',16],
        ['/posts/robotics/control/impedance-control/',16],['/posts/robotics/dynamics/parameter-identification/',8]]) {
        await load(route,320);
        const tree=await b.cdp('Accessibility.getFullAXTree');
        const checks=tree.nodes.filter(n=>!n.ignored&&n.role?.value==='checkbox');
        assert.equal(checks.length,count,route);assert(checks.every(n=>n.name?.value.trim()),route+' has an unnamed checklist item');
        report.nativeTasks.push({route,count,names:checks.map(n=>n.name.value)});
      }
    }
    if(interactive) {
      for(const width of [320,1440])for(const state of ['theme','language','search','filter','viewer','round-history','sidebar']) {
        if(state==='sidebar'&&width!==320)continue;
        const route=state==='filter'?'/posts/':state==='viewer'?'/posts/ai/supermimic/':state==='round-history'?'/labs/rrsi/':'/';
        await load(route,width);
        if(state==='theme'||state==='language') {
          const selector=state==='theme'?'.hextra-theme-toggle':'.hextra-language-switcher';
          await b.evaluate(`(()=>{const button=[...document.querySelectorAll('${selector}')].find(e=>e.checkVisibility());
            button.scrollIntoView({block:'center'});button.click();})()`);
          await waitFor(`document.querySelector('${selector}[aria-expanded="true"]')!==null`);
        } else if(state==='search') {
          await b.evaluate(`document.querySelector('[data-search-open]').click();
            const input=document.querySelector('.hextra-search-input');input.value='tqdm';input.dispatchEvent(new Event('input',{bubbles:true}));`);
          await waitFor(`document.querySelector('.hextra-search-results a[role="option"]')!==null`);
        } else if(state==='filter') {
          await b.evaluate(`document.getElementById('blog-query').value='C++';document.querySelector('[data-blog-filter]').requestSubmit()`);
          assert(await b.evaluate("document.querySelectorAll('#blog-filter-results .blog-card').length>0"));
        } else if(state==='viewer') {
          await b.evaluate(`Promise.all([...document.querySelectorAll('.content img')].map(e=>{e.loading='eager';return e.decode().catch(()=>null)}))`);
          await b.evaluate(`document.querySelector('.post-image-link').click()`);
          await waitFor(`document.querySelector('.post-image-viewer[open]')!==null`);
        } else if(state==='round-history') {
          await b.evaluate(`for(const name of ['A','B']){document.querySelector('[data-example='+name+']').click();
            document.getElementById('add-candidate').click();document.getElementById('apply-round').click()}
            document.querySelector('[data-example=C]').click();document.getElementById('add-candidate').click()`);
        } else await b.evaluate(`document.querySelector('.hextra-hamburger-menu').click()`);
        // A mobile navigation overlay intentionally makes the page main/H1
        // inert. Inspect its active subtree; initial-state audits cover the page.
        await audit({route,width,state,scope:state==='sidebar'?'.hextra-sidebar-container':undefined});
      }
      for(const platform of ['Win32','MacIntel'])for(const width of [320,1440]) {
        const {identifier}=await b.cdp('Page.addScriptToEvaluateOnNewDocument',{
          source:`Object.defineProperty(navigator,'platform',{configurable:true,get:()=>${JSON.stringify(platform)}});
            Object.defineProperty(navigator,'userAgent',{configurable:true,get:()=>${JSON.stringify(platform==='MacIntel'?'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0)':'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')}})`});
        try {
          await load('/',width);
          const label=await b.evaluate(`(()=>{const button=document.querySelector('.hextra-search-trigger');return{
            label:button.getAttribute('aria-label'),text:button.querySelector('.hextra-search-trigger-label').textContent,
            hint:button.querySelector('kbd').textContent,shortcuts:button.getAttribute('aria-keyshortcuts')};})()`);
          assert.equal(label.label,label.text+' '+label.hint);assert.equal(label.hint,platform==='MacIntel'?'⌘K':'CTRL K');
          assert.equal(label.shortcuts,'Control+k Meta+k');
          await b.cdp('Page.bringToFront');
          assert.equal(await b.evaluate(`document.querySelector('.hextra-theme-toggle').tagName`),'BUTTON');
          await b.evaluate(`document.querySelector('.hextra-theme-toggle').focus()`);
          const modifiers=platform==='MacIntel'?4:2;
          await b.cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'k',code:'KeyK',windowsVirtualKeyCode:75,modifiers});
          await b.cdp('Input.dispatchKeyEvent',{type:'keyUp',key:'k',code:'KeyK',windowsVirtualKeyCode:75});
          await waitFor(`document.querySelector('#hextra-search-dialog').open`);
          await b.key('Escape','Escape',27);
          await waitFor(`!document.querySelector('#hextra-search-dialog').open`);
          await b.evaluate(`document.querySelector('.hextra-theme-toggle').focus()`);
          await b.key('/','Slash',191);
          await waitFor(`document.querySelector('#hextra-search-dialog').open`);
          await b.key('Escape','Escape',27);
          await waitFor(`!document.querySelector('#hextra-search-dialog').open`);
          const editablePreserved=await b.evaluate(`(() => {
            for(const tag of ['input','select','textarea','div']) {
              const field=document.createElement(tag);if(tag==='div')field.contentEditable='true';
              if(tag==='select')field.append(new Option('Example','example'));
              document.body.append(field);field.focus();
              const event=new KeyboardEvent('keydown',{key:'k',ctrlKey:${platform==='Win32'},
                metaKey:${platform==='MacIntel'},bubbles:true,cancelable:true});
              field.dispatchEvent(event);const preserved=!event.defaultPrevented&&!document.querySelector('#hextra-search-dialog').open;
              field.remove();if(!preserved)return false;
            }
            return true;
          })()`);assert(editablePreserved);
          await audit({route:'/',width,state:'platform-'+platform,label,shortcutFromButton:true,slashFromButton:true,editablePreserved});
        } finally {await b.cdp('Page.removeScriptToEvaluateOnNewDocument',{identifier});}
      }
    }
    const failures = report.cases.filter(row => row.violations.length || row.idReferences.missing.length);
    const incomplete = report.cases.reduce((sum, row) => sum + row.incomplete.reduce((n, rule) => n + rule.count, 0), 0);
    report.summary = {cases: report.cases.length, failingCases: failures.length, incompleteNodes: incomplete,
      exceptions: b.exceptions.length};console.log(JSON.stringify(report.summary, null, 2));
    if (!captureOnly) {assert.equal(failures.length, 0, JSON.stringify(failures.slice(0, 3)));assert.equal(b.exceptions.length, 0);}
  } finally {
    await b.cdp('Emulation.setScriptExecutionDisabled', {value: false}).catch(()=>{});
    if (savedTheme !== undefined) await b.evaluate(savedTheme === null ? "localStorage.removeItem('color-theme')"
      : 'localStorage.setItem("color-theme",'+JSON.stringify(savedTheme)+')').catch(()=>{});
    save();await b.close();
  }
})().catch(error => {console.error(error);process.exitCode = 1;});
