#!/usr/bin/env node
// Larger text and user text spacing, including native navigation without JS.
// Run serially with the other browser suites against a loopback CDP endpoint.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('./lib/browser.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13145';
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-site-reflow';
const full = process.argv.includes('--all-articles');
const styles = {
  'text-200': ':root {font-size:200%!important}',
  spacing: `body :is(p,li,dt,dd,th,td,summary,h1,h2,h3,h4,h5,h6,a,button,label,input,select) {
    line-height:1.5!important;letter-spacing:0.12em!important;word-spacing:0.16em!important
  } main p {margin-bottom:2em!important}
  .katex {letter-spacing:normal!important;word-spacing:normal!important}`,
};
styles['text-200-spacing'] = styles['text-200'] + styles.spacing;

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const browser = await connect(port);
  const report = {layouts: [], exceptions: browser.exceptions};
  const stress = async mode => {
    await browser.evaluate(`(() => {const style=document.createElement('style');
      style.textContent=${JSON.stringify(styles[mode])};document.head.append(style);})()`);
    await browser.evaluate('document.fonts.ready');
    await browser.pause(150);
  };
  try {
    const routes = new Set(['/', '/zh-cn/', '/posts/', '/posts/page/2/', '/projects/', '/learning-paths/',
      '/archives/', '/tags/', '/categories/', '/404.html', '/zh-cn/404.html',
      '/posts/ai/supermimic/', '/posts/robotics/kinematics/six-dof-kinematics/']);
    if (full) {
      const posts = JSON.parse(fs.readFileSync(path.join(__dirname, '../docs/blog-editorial-review.json'))).posts;
      for (const post of posts) routes.add('/' + post.path.replace(/^content\//, '').replace(/\/index\.md$|\.md$/, '').toLowerCase() + '/');
    }
    for (const scripts of [true, false]) {
      await browser.cdp('Emulation.setScriptExecutionDisabled', {value: !scripts});
      for (const width of [320, 768, 1440]) for (const route of routes) for (const mode of Object.keys(styles)) {
        await browser.viewport(width);
        await browser.navigate(base + route);
        await stress(mode);
        const state = await browser.evaluate(`(() => {
          const viewport=document.documentElement.clientWidth,nav=document.querySelector('.hextra-nav-container nav');
          const visible=e=>e.getBoundingClientRect().width>0&&getComputedStyle(e).visibility!=='hidden';
          const bounds=e=>{const r=e.getBoundingClientRect();return {left:r.left,right:r.right,width:r.width,height:r.height}};
          const controls=[...nav.querySelectorAll('button, a[title]')].filter(visible).map(bounds);
          return {viewport,scrollWidth:document.documentElement.scrollWidth,nav:bounds(nav),
            offset:parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--hextra-navbar-height')),
            controls,overflow:document.documentElement.scrollWidth>viewport+1,
            tableFocusIssues:[...document.querySelectorAll('.content table')].filter(t=>t.clientWidth>0&&t.scrollWidth>t.clientWidth+1&&t.tabIndex!==0).length,
            codeFocusIssues:[...document.querySelectorAll('.hextra-code-block pre,.highlight pre')].filter(p=>p.clientWidth>0&&(p.scrollWidth>p.clientWidth+1||p.scrollHeight>p.clientHeight+1)&&p.tabIndex!==0).length};
        })()`);
        const passed = !state.overflow && state.controls.every(c => c.left >= -1 && c.right <= state.viewport + 1 && c.width >= 39.5 && c.height >= 39.5)
          && (!scripts || (Math.abs(state.nav.height - state.offset) <= 1 && state.tableFocusIssues === 0 && state.codeFocusIssues === 0));
        report.layouts.push({route, width, scripts, mode, passed, ...state});
        if (route === '/' && width === 320 && scripts && mode === 'text-200') await browser.screenshot(path.join(output, 'home-text-200.png'));
      }
    }

    await browser.cdp('Emulation.setScriptExecutionDisabled', {value: false});
    report.tableKeyboard = [];
    await browser.viewport(320);
    for (const route of ['/posts/ai/cosmos3/', '/posts/open3d/introduction/', '/posts/rl/']) {
      await browser.navigate(base + route);
      await stress('text-200');
      assert(await browser.evaluate(`(() => {
        const table=[...document.querySelectorAll('.content table')].find(t=>t.clientWidth
          && !t.closest('details:not([open])') && t.scrollWidth>t.clientWidth+2);
        if(!table)return false;
        window.__reflowTable=table;table.scrollLeft=0;table.scrollIntoView({block:'center'});table.focus();
        return document.activeElement===table;
      })()`), 'No focusable overflowing table: ' + route);
      for (let i=0;i<4;i++) await browser.key('ArrowRight', 'ArrowRight', 39);
      await browser.pause(250);
      const table = await browser.evaluate(`({focused:document.activeElement===window.__reflowTable,
        scrollLeft:window.__reflowTable.scrollLeft,clientWidth:window.__reflowTable.clientWidth,
        scrollWidth:window.__reflowTable.scrollWidth})`);
      assert(table.focused && table.scrollLeft>0, JSON.stringify({route,...table}));
      report.tableKeyboard.push({route,...table});
    }
    for (const route of ['/404.html', '/zh-cn/404.html']) {
      await browser.viewport(320);
      await browser.navigate(base + route);
      await stress('text-200');
      await browser.evaluate(`document.querySelector('.hextra-hamburger-menu').click()`);
      assert(await browser.evaluate(`document.querySelector('.hextra-hamburger-menu').getAttribute('aria-expanded')==='true' && !document.querySelector('.hextra-sidebar-container').inert`));
      await browser.key('Escape', 'Escape', 27);
      assert(await browser.evaluate(`document.querySelector('.hextra-hamburger-menu').getAttribute('aria-expanded')==='false' && document.activeElement===document.querySelector('.hextra-hamburger-menu')`));
    }
    report.notFoundMenu = true;

    for (const width of [320, 768]) {
      await browser.viewport(width);
      await browser.navigate(base + '/posts/');
      await stress('text-200');
      await browser.evaluate(`document.querySelector('[data-search-open]').click()`);
      await browser.pause(400);
      const bounds = await browser.evaluate(`(() => {
        const dialog=document.getElementById('hextra-search-dialog'),input=dialog.querySelector('input');
        const d=dialog.getBoundingClientRect(),i=input.getBoundingClientRect();
        return {dialog:{left:d.left,right:d.right},input:{left:i.left,right:i.right},focused:document.activeElement===input};
      })()`);
      assert(bounds.focused && bounds.input.left >= bounds.dialog.left && bounds.input.right <= bounds.dialog.right);
      await browser.key('Escape', 'Escape', 27);
    }
    report.searchInput = true;
    const failures = report.layouts.filter(row => !row.passed);
    report.summary = {layouts: report.layouts.length, failures: failures.length, runtimeExceptions: browser.exceptions.length};
    console.log(JSON.stringify({...report.summary, failedLayouts: failures.map(({route, width, scripts, mode, scrollWidth}) => ({route, width, scripts, mode, scrollWidth}))}, null, 2));
    assert.equal(browser.exceptions.length, 0, JSON.stringify(browser.exceptions));
    assert.equal(failures.length, 0, 'Reflow failures; see results.json');
  } finally {
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(report, null, 2) + '\n');
    await browser.cdp('Emulation.setScriptExecutionDisabled', {value: false}).catch(() => {});
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
