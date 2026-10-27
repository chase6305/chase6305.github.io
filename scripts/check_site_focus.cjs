#!/usr/bin/env node
// Painted keyboard outlines on repository-owned cards and navigation links.
// Run serially against the loopback CDP endpoint used by other browser checks.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('./lib/browser.cjs');
const base = process.argv[2] || 'http://127.0.0.1:13180';
const port = Number(process.argv[3] || 9226);
const output = process.argv[4] || '/tmp/chase-site-focus';
const captureOnly = process.argv.includes('--capture-only');
const modes = process.argv.includes('--forced-colors') ? ['normal', 'forced-light', 'forced-dark'] : ['normal'];
const selectors = ['.latest-card h3 a', '.latest-card__link', '.chase-button', '.project-link',
  '.blog-card h2 a', '.blog-card__read', '.taxonomy-card', '.archive-year__posts a',
  '.collection-nav a', '.blog-topic-nav a', '.blog-topic-index a'];
const routes = ['/', '/zh-cn/', '/posts/', '/projects/', '/zh-cn/projects/', '/archives/',
  '/tags/', '/categories/', '/learning-paths/', '/posts/ai/morphmimic/'];

// An outside outline sits on its parent's painted background. Resolve alpha
// through ancestors; gradients/images still require a separate visual review.
function inspectOutline(selector) {
  const element = document.querySelector(selector), style = getComputedStyle(element);
  const canvas = document.createElement('canvas');canvas.width = canvas.height = 1;
  const context = canvas.getContext('2d', {willReadFrequently: true});
  const rgba = color => {
    context.clearRect(0, 0, 1, 1);context.fillStyle = color;context.fillRect(0, 0, 1, 1);
    return [...context.getImageData(0, 0, 1, 1).data].map(value => value / 255);
  };
  const over = (a, b) => [0, 1, 2].map(index => a[index] * a[3] + b[index] * (1-a[3]));
  const gradients = [];
  const background = node => {
    if (!node) return [1, 1, 1];
    const css = getComputedStyle(node);
    if (css.backgroundImage !== 'none') gradients.push(css.backgroundImage);
    return over(rgba(css.backgroundColor), background(node.parentElement));
  };
  const bg = background(element.parentElement), foreground = over(rgba(style.outlineColor), bg);
  const luminance = color => color.map(value => value <= .04045 ? value / 12.92 : ((value + .055) / 1.055) ** 2.4)
    .reduce((total, value, index) => total + value * [.2126, .7152, .0722][index], 0);
  const values = [luminance(foreground), luminance(bg)].sort((a, b) => b-a);
  const bounds = element.getBoundingClientRect();
  return {focused: document.activeElement === element, visible: element.matches(':focus-visible'),
    outline: style.outlineColor, outlineStyle: style.outlineStyle, outlineWidth: parseFloat(style.outlineWidth),
    ratio: (values[0]+.05)/(values[1]+.05), background: bg, gradients,
    insideViewport: bounds.left >= -1 && bounds.right <= document.documentElement.clientWidth+1};
}

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const browser = await connect(port), report = {outlines: [], exceptions: browser.exceptions};
  let savedTheme;
  try {
    for (const mode of modes) {
      await browser.cdp('Emulation.setEmulatedMedia', {features: mode === 'normal' ? [] : [
        {name: 'forced-colors', value: 'active'},
        {name: 'prefers-color-scheme', value: mode === 'forced-dark' ? 'dark' : 'light'}
      ]});
      for (const width of [320, 768, 1440]) {
        await browser.viewport(width);
        for (const route of routes) {
          await browser.navigate(base+route);
          if (savedTheme === undefined) savedTheme = await browser.evaluate("localStorage.getItem('color-theme')");
          for (const theme of ['light', 'dark', 'warm']) {
            await browser.evaluate(`document.querySelector('[data-item=${theme}]').click()`);
            await browser.key('Tab', 'Tab', 9);
            for (const selector of selectors) {
              const available = await browser.evaluate(`document.querySelector(${JSON.stringify(selector)})?.checkVisibility() || false`);
              if (!available) continue;
              await browser.evaluate(`document.querySelector(${JSON.stringify(selector)}).focus()`);
              const state = await browser.evaluate('('+inspectOutline.toString()+')('+JSON.stringify(selector)+')');
              const passed = state.focused && state.visible && state.outlineWidth > 0 && state.outlineStyle !== 'none'
                && state.ratio >= 3 && state.insideViewport;
              report.outlines.push({route, width, theme, mode, selector, passed, ...state});
            }
          }
        }
      }
    }
    report.summary = {outlines: report.outlines.length, failures: report.outlines.filter(row => !row.passed).length,
      gradientCasesForVisualReview: report.outlines.filter(row => row.gradients.length).length, exceptions: browser.exceptions.length};
    console.log(JSON.stringify({...report.summary, failedOutlines: report.outlines.filter(row => !row.passed)
      .map(({route, width, theme, mode, selector, ratio, outlineWidth, focused, visible}) => ({route, width, theme, mode, selector, ratio, outlineWidth, focused, visible}))}, null, 2));
    if (!captureOnly) {
      assert.equal(report.summary.failures, 0, 'Low-contrast or missing keyboard outlines; see results.json');
      assert.equal(browser.exceptions.length, 0, JSON.stringify(browser.exceptions));
    }
  } finally {
    if (savedTheme !== undefined) await browser.evaluate(savedTheme === null ? "localStorage.removeItem('color-theme')" :
      'localStorage.setItem("color-theme",'+JSON.stringify(savedTheme)+')').catch(() => {});
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(report, null, 2)+'\n');
    await browser.close();
  }
})().catch(error => {console.error(error);process.exitCode = 1;});
