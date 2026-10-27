#!/usr/bin/env node
// Standalone lab print palettes, clipping, disclosures and screen restoration.
// Run serially with the other loopback CDP checks. --capture-only records a baseline.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('./lib/browser.cjs');
const {contrastAudit} = require('./lib/contrast.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13189';
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-lab-print';
const captureOnly = process.argv.includes('--capture-only');
const contrastOnWhite = color => {
  const channels = color.match(/[\d.]+/g)?.slice(0, 3).map(Number);
  if (!channels || channels.length !== 3) return 0;
  const linear = channels.map(c => c / 255).map(c => c <= .04045 ? c / 12.92 : ((c + .055) / 1.055) ** 2.4);
  return 1.05 / (linear.reduce((sum, c, i) => sum + c * [.2126, .7152, .0722][i], 0) + .05);
};
const routes = ['/labs/supermimic/', '/labs/video-to-data/', '/labs/multiview-geometry/', '/labs/rrsi/'];
const modes = [
  {name: 'normal', forced: 'none', scheme: 'light'},
  {name: 'forced-light', forced: 'active', scheme: 'light'},
  {name: 'forced-dark', forced: 'active', scheme: 'dark'},
];
(async () => {
  fs.mkdirSync(output, {recursive: true});
  const b = await connect(port), report = {cases: [], exceptions: b.exceptions};
  let savedTheme;
  const state = () => b.evaluate(`({theme:document.documentElement.dataset.theme||null,
    inputs:[...document.querySelectorAll('input,select')].map(e=>({name:e.name,value:e.value,checked:e.checked})),
    details:[...document.querySelectorAll('details')].map(e=>e.open),
    status:document.querySelector('#status')?.textContent,
    history:document.querySelector('#round-history')?.textContent,
    pool:document.querySelector('#pool-rows')?.textContent})`);
  try {
    await b.viewport(700, 1000);await b.navigate(base + routes[0]);
    savedTheme = await b.evaluate("localStorage.getItem('color-theme')");
    for (const scripts of [true, false]) for (const route of routes) for (const theme of ['light', 'dark', 'warm']) for (const mode of modes) {
      await b.cdp('Emulation.setScriptExecutionDisabled', {value: !scripts});
      const features = [{name: 'forced-colors', value: mode.forced}, {name: 'prefers-color-scheme', value: mode.scheme}];
      await b.cdp('Emulation.setEmulatedMedia', {media: 'screen', features});
      await b.navigate(base + route);
      await b.evaluate(`document.documentElement.dataset.theme=${JSON.stringify(theme)};document.fonts.ready`);
      if (scripts && route === '/labs/rrsi/') await b.evaluate(`(() => {
        for(const name of ['A','B']) {
          document.querySelector('[data-example='+name+']').click();
          document.getElementById('add-candidate').click();document.getElementById('apply-round').click();
        }
        const sections=document.querySelectorAll('#round-history details');sections[0].open=false;sections[1].open=true;
        document.querySelector('[data-example=C]').click();document.getElementById('add-candidate').click();
      })()`);
      const original = await state();
      const screen = await b.evaluate("({background:getComputedStyle(document.body).backgroundColor,color:getComputedStyle(document.body).color})");
      await b.cdp('Emulation.setEmulatedMedia', {media: 'print', features});await b.pause(180);
      const printed = await b.evaluate(`(() => {
        const visible=e=>e.getBoundingClientRect().width>0&&e.getBoundingClientRect().height>0&&getComputedStyle(e).visibility==='visible';
        const bounds=e=>{const r=e.getBoundingClientRect();return{left:r.left,right:r.right,width:r.width}};
        const root=document.documentElement,body=getComputedStyle(document.body);
        return {background:body.backgroundColor,color:body.color,scheme:getComputedStyle(root).colorScheme,
          viewport:root.clientWidth,scrollWidth:root.scrollWidth,
          code:[...document.querySelectorAll('pre')].filter(visible).map(e=>({wrap:getComputedStyle(e).whiteSpace,
            scroll:e.scrollWidth,client:e.clientWidth,overflow:getComputedStyle(e).overflow,...bounds(e)})),
          tables:[...document.querySelectorAll('table')].filter(visible).map(e=>({scroll:e.scrollWidth,client:e.clientWidth,...bounds(e)})),
          svg:[...document.querySelectorAll('svg')].filter(visible).map(e=>({...bounds(e),
            labels:[...e.querySelectorAll('text')].map(t=>({text:t.textContent,fill:getComputedStyle(t).fill}))})),
          disclosures:[...document.querySelectorAll('details')].map(e=>({open:e.open,
            contentVisibility:getComputedStyle(e,'::details-content').contentVisibility})),
          navigation:[...document.querySelectorAll('.lab-nav,.lab-return,button')].filter(visible).map(e=>e.tagName)};
      })()`);
      const contrast = await b.evaluate('(' + contrastAudit.toString() + ')()');
      const failures = [];
      if (printed.background !== 'rgb(255, 255, 255)' || printed.scheme !== 'light') failures.push('paper palette');
      if (printed.scrollWidth > printed.viewport + 1) failures.push('page overflow');
      if (printed.code.some(e => e.scroll > e.client + 1 || !['pre-wrap', 'break-spaces'].includes(e.wrap))) failures.push('code clipping');
      if (printed.tables.some(e => e.scroll > e.client + 1 || e.right > printed.viewport + 1)) failures.push('table clipping');
      if (printed.svg.some(e => e.right > printed.viewport + 1 || e.left < -1)) failures.push('chart bounds');
      if (printed.svg.some(e => e.labels.some(label => contrastOnWhite(label.fill) < 4.5))) failures.push('SVG label contrast on paper');
      if (printed.disclosures.some(e => !e.open && e.contentVisibility !== 'visible')) failures.push('closed explanations');
      if (printed.navigation.length) failures.push('printed action controls');
      if (contrast.issues.length) failures.push('text contrast');
      if (scripts && theme === 'dark' && ['normal', 'forced-dark'].includes(mode.name)) {
        const file = path.join(output, route.split('/')[2] + '-' + mode.name + '.pdf');
        const pdf = await b.cdp('Page.printToPDF', {printBackground: false, paperWidth: 8.27, paperHeight: 11.69,
          marginTop: .55, marginBottom: .55, marginLeft: .55, marginRight: .55});
        fs.writeFileSync(file, Buffer.from(pdf.data, 'base64'));
      }
      await b.cdp('Emulation.setEmulatedMedia', {media: 'screen', features});await b.pause(100);
      const restored = await state(), screenAfter = await b.evaluate("({background:getComputedStyle(document.body).backgroundColor,color:getComputedStyle(document.body).color})");
      if (JSON.stringify(restored) !== JSON.stringify(original) || JSON.stringify(screenAfter) !== JSON.stringify(screen)) failures.push('screen state restoration');
      report.cases.push({route, theme, scripts, mode: mode.name, printed, contrast, failures, passed: !failures.length});
    }
    const failures = report.cases.filter(row => !row.passed);
    console.log(JSON.stringify({cases: report.cases.length, failures: failures.length, exceptions: b.exceptions.length,
      reasons: [...new Set(failures.flatMap(row => row.failures))]}, null, 2));
    if (!captureOnly) {assert.equal(failures.length, 0, JSON.stringify(failures.slice(0, 3)));assert.equal(b.exceptions.length, 0);}
  } finally {
    await b.cdp('Emulation.setScriptExecutionDisabled', {value: false}).catch(()=>{});
    await b.cdp('Emulation.setEmulatedMedia', {media: 'screen', features: []}).catch(()=>{});
    if (savedTheme !== undefined) await b.evaluate(savedTheme === null ? "localStorage.removeItem('color-theme')"
      : 'localStorage.setItem("color-theme",'+JSON.stringify(savedTheme)+')').catch(()=>{});
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(report, null, 2)+'\n');await b.close();
  }
})().catch(error => {console.error(error);process.exitCode = 1;});
