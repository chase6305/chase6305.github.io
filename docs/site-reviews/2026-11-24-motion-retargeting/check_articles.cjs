#!/usr/bin/env node
// Article-scoped CDP checks. Use a private loopback Chrome instance and run serially.
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const {createHash} = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('../../../scripts/lib/browser.cjs');
const {contrastAudit} = require('../../../scripts/lib/contrast.cjs');

const [baseInput, portInput, out, axePath] = process.argv.slice(2);
assert(baseInput && portInput && out, 'Usage: check_articles.cjs <baseURL> <CDP port> <output> [axe.min.js]');
const base = baseInput.replace(/\/$/, '');
const prefix = new URL(base).pathname.replace(/\/$/, '');
const routes = ['/posts/robotics/motion-retargeting/', '/posts/robotics/contact-preserving-retargeting/'];
const figureIDs = [['fig-box-height', 'fig-source-time'], ['fig-window-overlap', 'fig-contact-distance']];
const axe = axePath ? fs.readFileSync(axePath, 'utf8') : null;
if (axe) {
  assert.equal(createHash('sha256').update(axe).digest('hex'), '20c09fe157a8a34a30e241aaa1fcdade657734f08ab379ecfbeb7d45cc46e878');
}
const report = {reading: [], discovery: [], chapters: [], behavior: [], print: [], semantics: []};

function periodicReport(value) {
  fs.mkdirSync(out, {recursive: true});
  fs.writeFileSync(path.join(out, 'results.json'), JSON.stringify(value, null, 2) + '\n');
}

async function waitFor(browser, expression) {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (await browser.evaluate(expression)) return;
    await browser.pause(70);
  }
  throw Error('Expected browser state: ' + expression);
}

let sequence = 0;
async function load(browser, route, profile, hash = '') {
  const {width, theme, size = 100, scripts = true, spaced = false, forced = false} = profile;
  await browser.viewport(width, width < 768 ? 900 : 1080);
  const color = theme === 'dark' ? 'dark' : 'light';
  const media = [{name: 'prefers-color-scheme', value: color}];
  if (forced) media.push({name: 'forced-colors', value: 'active'});
  await browser.cdp('Emulation.setEmulatedMedia', {media: '', features: media});
  await browser.cdp('Emulation.setScriptExecutionDisabled', {value: !scripts});
  await browser.navigate(`${base}${route}?check=${++sequence}${hash ? '#' + hash : ''}`);
  await browser.cdp('Emulation.setScriptExecutionDisabled', {value: false});
  await browser.evaluate(`(() => {
    const root = document.documentElement;
    root.classList.remove('light', 'dark', 'warm');
    root.classList.add(${JSON.stringify(theme)});
    root.style.colorScheme = ${JSON.stringify(color)};
    root.style.fontSize = '${size}%';
    if (${spaced}) { root.style.letterSpacing = '.12em'; root.style.wordSpacing = '.16em'; }
    return Promise.all([document.fonts.ready, ...[...document.querySelectorAll('.article-figure img')].map(image => {
      image.loading = 'eager';
      return image.decode();
    })]);
  })()`);
  await browser.pause(110);
}

async function reading(browser) {
  const profiles = [];
  for (const width of [320, 768, 1440]) {
    for (const theme of ['light', 'dark', 'warm']) {
      profiles.push({width, theme});
      profiles.push({width, theme, size: 200, spaced: true});
    }
  }
  profiles.push({width: 320, theme: 'light', scripts: false}, {width: 320, theme: 'dark', size: 200, scripts: false});
  profiles.push({width: 320, theme: 'dark', forced: true, size: 200}, {width: 1440, theme: 'light', forced: true});
  for (const route of routes) {
    for (const profile of profiles) {
      await load(browser, route, profile);
      const state = await browser.evaluate(`(() => {
        const root = document.documentElement;
        const article = document.querySelector('article');
        const body = article.querySelector('.content');
        const bounds = body.getBoundingClientRect();
        const regions = [...body.querySelectorAll('.katex-display, table')].filter(el => el.scrollWidth > el.clientWidth + 2);
        return {
          pageOverflow: root.scrollWidth > root.clientWidth + 2,
          images: [...body.querySelectorAll('.article-figure img')].map(i => ({loaded: i.complete && !!i.naturalWidth, alt: !!i.alt, width: i.clientWidth})),
          inlineOverflow: [...body.querySelectorAll('.katex')].filter(el => !el.closest('.katex-display, table, code')).flatMap(el => {
            const rect = el.getBoundingClientRect();
            return rect.left < bounds.left - 2 || rect.right > bounds.right + 2 ? [el.querySelector('annotation')?.textContent] : [];
          }),
          unfocusable: regions.filter(el => el.tabIndex !== 0).map(el => ({tag: el.tagName, classes: el.className})),
          invalidMath: body.querySelectorAll('.katex-error').length,
          language: article.lang,
          related: [...article.querySelectorAll('.content a')].some(a => ${JSON.stringify(prefix + routes.find(other => other !== route))} === new URL(a.href).pathname),
          calculatorReady: document.querySelector('[data-contact-window]')?.dataset.ready === 'true',
          crampedLabels: [...body.querySelectorAll('.contact-window label')].filter(el => {
            if (!el.offsetWidth) return false;
            return el.getBoundingClientRect().height > 3.2 * parseFloat(getComputedStyle(el).lineHeight);
          }).map(el => el.textContent),
        };
      })()`);
      const row = {route, ...profile, ...state};
      report.reading.push(row);
      assert(!state.pageOverflow && state.related && state.language === 'zh-CN', JSON.stringify(row));
      assert(state.images.length === figureIDs[routes.indexOf(route)].length &&
        state.images.every(image => image.loaded && image.alt && image.width > 0), JSON.stringify(row));
      assert.deepEqual(state.unfocusable, [], JSON.stringify(row));
      assert.deepEqual(state.inlineOverflow, [], JSON.stringify(row));
      assert.deepEqual(state.crampedLabels, [], JSON.stringify(row));
      assert.equal(state.invalidMath, 0);
      assert.equal(state.calculatorReady, route === routes[1] && profile.scripts !== false);
      if (route === routes[1] && profile.width === 320 && !profile.forced) {
        const colors = await browser.evaluate(`(${contrastAudit})('.contact-window')`);
        assert(colors.checked >= 9, JSON.stringify({profile, colors}));
        assert.deepEqual(colors.issues, [], JSON.stringify({profile, colors}));
        row.calculatorContrastText = colors.checked;
      }
    }
    periodicReport(report);
  }
}

async function discovery(browser) {
  for (const entrance of ['/', '/zh-cn/', '/posts/', '/tags/motion-retargeting/']) {
    await load(browser, entrance, {width: 320, theme: 'light', size: 200, scripts: false});
    const companionRoutes = entrance.startsWith('/tags/') ? ['/posts/ai/card-retargeting/', '/posts/ai/video-to-data/'] : [];
    const links = await browser.evaluate(`(() => {
      const hrefs = ${JSON.stringify([...routes, ...companionRoutes].map(route => prefix + route))};
      return [...document.querySelectorAll('main a[href]')].filter(link => hrefs.includes(link.pathname) && link.getClientRects().length)
        .map(link => ({route: link.pathname, text: link.textContent.trim()}));
    })()`);
    for (const route of routes) assert(links.some(link => link.route === prefix + route && link.text.includes('重定向')), JSON.stringify({entrance, links}));
    for (const route of companionRoutes) assert(links.some(link => link.route === prefix + route && link.text.length), JSON.stringify({entrance, links}));
    report.discovery.push({entrance, mode: 'native', links});
  }

  for (const entrance of ['/', '/zh-cn/']) {
    await load(browser, entrance, {width: 1440, theme: 'light'});
    await waitFor(browser, 'typeof window.hextraSearch?.search === "function"');
    const matches = await browser.evaluate(`(async () => {
      const queries = ['GMR UMR', 'OmniRetarget HOI-Retarget', 'PriMoRetarget', 'ReContact'];
      return Promise.all(queries.map(async query => (await window.hextraSearch.search(query)).map(group => group.route)));
    })()`);
    for (let i = 0; i < matches.length; i++) assert(matches[i].includes(prefix + routes[i % 2]), JSON.stringify({entrance, matches}));
    report.discovery.push({entrance, mode: 'search', matches});
  }
}

async function chapters(browser) {
  for (const route of routes) {
    const ids = [];
    await load(browser, route, {width: 320, theme: 'light', size: 200});
    if (route === routes[0]) await checkSemantics(browser, 'overview');
    const count = await browser.evaluate(`document.querySelectorAll('.blog-mobile-toc a').length`);
    assert(count >= 6, route);
    for (let i = 0; i < count; i++) {
      const id = await browser.evaluate(`(() => {
        const toc = document.querySelector('.blog-mobile-toc');
        toc.open = true;
        const link = toc.querySelectorAll('a')[${i}];
        link.focus();
        return decodeURIComponent(link.hash.slice(1));
      })()`);
      await browser.key('Enter', 'Enter', 13);
      const state = await browser.evaluate(`(() => {
        const heading = document.getElementById(${JSON.stringify(id)});
        const navBottom = document.querySelector('.hextra-nav-container nav').getBoundingClientRect().bottom;
        return {focused: document.activeElement === heading, closed: !document.querySelector('.blog-mobile-toc').open,
          id: decodeURIComponent(location.hash.slice(1)), top: heading?.getBoundingClientRect().top, navBottom};
      })()`);
      assert(state.focused && state.closed && state.id === id && state.top >= state.navBottom - 1 && state.top < 900,
        JSON.stringify({route, ...state}));
      ids.push(id);
    }
    assert.equal(new Set(ids).size, count);
    report.chapters.push({route, ids});
  }
}

async function articleLinks(browser) {
  const entries = [
    ['gmr-bvh', 'umr-character', 'ot-solver-callback'],
    ['released-data', 'released-pair', 'contact-metrics'],
  ];
  for (let index = 0; index < routes.length; index++) {
    const route = routes[index];
    for (const id of entries[index]) {
      await load(browser, route, {width: 320, theme: 'light', size: 200, scripts: false});
      const link = `.content :is(p, td) a[href="#${id}"]`;
      assert(await browser.evaluate(`(() => {
        const anchor = document.querySelector(${JSON.stringify(link)});
        anchor?.focus();
        return document.activeElement === anchor;
      })()`), `${route} ${link}`);
      await browser.key('Enter', 'Enter', 13);
      const state = await browser.evaluate(`(() => {
        const heading = document.getElementById(${JSON.stringify(id)});
        const top = heading?.getBoundingClientRect().top;
        return {hash: location.hash, top, navBottom: document.querySelector('.hextra-nav-container nav').getBoundingClientRect().bottom};
      })()`);
      assert(state.hash === `#${id}` && state.top >= state.navBottom - 1 && state.top < 900,
        JSON.stringify({route, id, ...state}));
      report.behavior.push({case: 'article link reaches visible heading without scripts', route, id});
    }
  }
}

async function keyboardScroll(browser) {
  for (const route of routes) {
    for (const scripts of [false, true]) {
      await load(browser, route, {width: 320, theme: 'light', size: 200, scripts});
      const region = await browser.evaluate(`(() => {
        const element = [...document.querySelectorAll('.content table, .content .katex-display')]
          .find(value => value.scrollWidth > value.clientWidth + 12);
        element?.focus();
        return element ? {tag:element.tagName, width:element.clientWidth, scrollWidth:element.scrollWidth, focused:document.activeElement === element} : null;
      })()`);
      assert(region?.focused, JSON.stringify({route, scripts, region}));
      await browser.key('ArrowRight', 'ArrowRight', 39);
      await waitFor(browser, 'document.activeElement.scrollLeft >= 4');
      report.behavior.push({case:'overflowing region scrolls with keyboard', route, scripts, ...region});
    }
  }
}

async function figures(browser) {
  const profile = {width: 390, theme: 'dark'};
  for (let index = 0; index < routes.length; index++) {
    for (const figure of figureIDs[index]) {
      await load(browser, routes[index], profile, figure);
      const selector = '#' + figure + ' a.post-image-link';
      await browser.evaluate(`document.querySelector(${JSON.stringify(selector)}).focus()`);
      await browser.key('Enter', 'Enter', 13);
      await waitFor(browser, `!!document.querySelector('dialog[open]')`);
      await browser.key('Escape', 'Escape', 27);
      await waitFor(browser, `!document.querySelector('dialog[open]')`);
      assert(await browser.evaluate(`document.activeElement === document.querySelector(${JSON.stringify(selector)})`));
      await browser.screenshot(path.join(out, figure + '-mobile-dark.png'));
      report.behavior.push({case: 'figure opens from keyboard and returns focus', figure});
    }
  }
}

const readOutputs = browser => browser.evaluate(`Object.fromEntries([...document.querySelectorAll('[data-contact-window] output')].map(e => [e.dataset.value, e.textContent]))`);
const defaultOutputs = {late_path: '0 → 0 → 0 → 4', late_gap: '8', window_path: '0 → 4 → 8 → 12', window_gap: '0'};
const profile = {width: 320, theme: 'dark', size: 200, spaced: true};

async function inject(browser, source) {
  await browser.evaluate(`new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = ${JSON.stringify(source)};
    script.onload = resolve;
    script.onerror = reject;
    document.body.append(script);
  })`);
  await waitFor(browser, `document.querySelector('[data-contact-window]').dataset.ready === 'true'`);
}

async function checkSemantics(browser, label) {
  if (!axe) return;
  await browser.evaluate(axe);
  const page = await browser.evaluate(`axe.run(document.querySelector('article'), {rules:{'color-contrast':{enabled:false}}}).then(result => ({
    violations: result.violations.map(v => ({id:v.id, nodes:v.nodes.map(n => n.target)})),
    incomplete: result.incomplete.map(v => ({id:v.id, nodes:v.nodes.map(n => n.target)})),
  }))`);
  report.semantics.push({label, ...page});
  assert.deepEqual(page.violations, [], JSON.stringify(page));
}

async function checkCalculator(browser) {
  await load(browser, routes[1], profile, 'window-experiment');
  assert.deepEqual(await readOutputs(browser), defaultOutputs);
  const {root} = await browser.cdp('DOM.getDocument');
  for (const [selector, role, live] of [['[data-status]', 'status', 'polite'], ['[data-error]', 'alert', 'assertive']]) {
    const {nodeId} = await browser.cdp('DOM.querySelector', {nodeId: root.nodeId, selector: '.contact-window ' + selector});
    const {nodes: [node]} = await browser.cdp('Accessibility.getPartialAXTree', {nodeId, fetchRelatives: false});
    assert(!node.ignored && node.role.value === role && node.properties.some(p => p.name === 'live' && p.value.value === live),
      JSON.stringify({selector, node}));
  }
  await checkSemantics(browser, 'enhanced');
  await browser.evaluate(`document.querySelector('input[name="distance"]').focus()`);
  for (const name of ['step', 'intervals', 'pinned']) {
    await browser.key('Tab', 'Tab', 9);
    assert.equal(await browser.evaluate('document.activeElement.name'), name);
  }
  await browser.key('ArrowUp', 'ArrowUp', 38);
  assert.equal(await browser.evaluate('document.activeElement.value'), '1');
  assert(Object.values(await readOutputs(browser)).every(value => value === '—'));
  await browser.key('Enter', 'Enter', 13);
  assert.deepEqual(await readOutputs(browser), {...defaultOutputs, window_path: '0 → 0 → 4 → 8', window_gap: '4'});
  assert.equal(await browser.evaluate('document.activeElement.name'), 'pinned');
  await browser.evaluate(`document.activeElement.value = '4'; document.activeElement.dispatchEvent(new Event('input', {bubbles:true}))`);
  await browser.key('Enter', 'Enter', 13);
  assert(Object.values(await readOutputs(browser)).every(value => value === '—'));
  assert(await browser.evaluate(`(() => {
    const e = document.activeElement;
    const error = document.getElementById(e.getAttribute('aria-describedby'));
    return e.name === 'pinned' && e.getAttribute('aria-invalid') === 'true' && error?.textContent.includes('固定间隔数最多');
  })()`));
  await checkSemantics(browser, 'invalid input');
  await browser.key('ArrowDown', 'ArrowDown', 40);
  await browser.key('Enter', 'Enter', 13);
  assert.deepEqual(await readOutputs(browser), {late_path: '0 → 0 → 0 → 0', late_gap: '12', window_path: '0 → 0 → 0 → 0', window_gap: '12'});
  assert.equal(await browser.evaluate('document.activeElement.name'), 'pinned');
  report.behavior.push({case: 'Tab, number input arrows, stale clearing, Enter, invalid input and recovery'});
  await browser.screenshot(path.join(out, 'window-mobile-dark-200.png'));

  await load(browser, routes[1], {...profile, scripts: false});
  assert.deepEqual(await readOutputs(browser), defaultOutputs);
  assert(await browser.evaluate(`document.querySelector('[data-contact-window] form').hidden`));
  await checkSemantics(browser, 'native fallback');

  await browser.cdp('Network.setBlockedURLs', {urls: ['*contact-window*.js*']});
  await load(browser, routes[1], profile);
  assert.deepEqual(await readOutputs(browser), defaultOutputs);
  const script = await browser.evaluate(`document.querySelector('script[src*="contact-window"]').src`);
  await browser.evaluate(`window.target = document.getElementById('benchmark'); target.tabIndex=-1; target.focus(); target.scrollIntoView({block:'start'}); window.oldTop = target.getBoundingClientRect().top;`);
  await browser.cdp('Network.setBlockedURLs', {urls: []});
  await inject(browser, script);
  const late = await browser.evaluate(`({focused: document.activeElement === target, delta: Math.abs(target.getBoundingClientRect().top - oldTop)})`);
  assert(late.focused && late.delta < 3, JSON.stringify(late));
  report.behavior.push({case: 'native defaults before late script, preserve reader focus/position', ...late});

  await browser.cdp('Network.setBlockedURLs', {urls: ['*contact-window*.js*']});
  await load(browser, routes[1], profile);
  await browser.evaluate(`document.querySelector('input[name="pinned"]').value = '4'`);
  await browser.cdp('Network.setBlockedURLs', {urls: []});
  await inject(browser, script);
  assert(Object.values(await readOutputs(browser)).every(value => value === '—'));
  await browser.evaluate(`document.querySelector('input[name="pinned"]').focus()`);
  await browser.key('ArrowDown', 'ArrowDown', 40);
  await browser.key('Enter', 'Enter', 13);
  assert.equal((await readOutputs(browser)).window_gap, '12');
  report.behavior.push({case: 'late enhancement keeps restored invalid values correctable'});

  for (const file of ['contact_window.py', 'probe_contact_sources.py', 'contact_audit.py', 'inspect_hoi_release.py']) {
    const download = await fetch(base + routes[1] + file);
    assert(download.ok && (await download.text()).startsWith('#!/usr/bin/env python3'), file);
  }
  for (const file of ['probe_motion_sources.py', 'audit_motion_time.py']) {
    const download = await fetch(base + routes[0] + file);
    assert(download.ok && (await download.text()).startsWith('#!/usr/bin/env python3'), file);
  }
  const config = await fetch(base + routes[0] + 'umr-character-cpu.json');
  assert(config.ok);
  assert.deepEqual((await config.json()).motion, {start: 12, stride: 2, max_frames: 8});
}

const squeeze = text => text.replace(/\s+/g, '');

async function printArticles(browser) {
  for (const [label, routeIndex, paper, view] of [
    ['overview-a4', 0, 'A4', {width: 1440, theme: 'light'}],
    ['overview-letter-forced', 0, 'Letter', {width: 390, theme: 'dark', size: 200, forced: true}],
    ['contacts-a4-live', 1, 'A4', profile],
    ['contacts-a4-standard', 1, 'A4', {width: 1440, theme: 'light'}],
    ['contacts-letter-native', 1, 'Letter', {width: 390, theme: 'light', size: 200, scripts: false, forced: true}],
  ]) {
    await load(browser, routes[routeIndex], view);
    if (routeIndex === 1 && view.scripts !== false) {
      await browser.evaluate(`document.querySelector('input[name="pinned"]').focus()`);
      await browser.key('ArrowUp', 'ArrowUp', 38);
      await browser.key('Enter', 'Enter', 13);
    }
    const snapshot = `({width: innerWidth, scroll: scrollY, focus: document.activeElement.id, outputs: [...document.querySelectorAll('[data-contact-window] output')].map(e => e.textContent)})`;
    const before = await browser.evaluate(snapshot);
    const pairs = await browser.evaluate(`[...document.querySelectorAll('.contact-window__results > div')].map(e => [e.querySelector('dt').textContent, e.querySelector('dd').textContent])`);
    const fields = await browser.evaluate(`[...document.querySelectorAll('.contact-window__field')].filter(e => !e.closest('[hidden]')).map(e => [e.querySelector('label').textContent, e.querySelector('input').value])`);
    const figures = await browser.evaluate(`[...document.querySelectorAll('.article-figure')].map(figure => {
      const image = figure.querySelector('img');
      return {aspect: image.naturalWidth / image.naturalHeight,
        caption: figure.querySelector('.article-figure__text').textContent};
    })`);
    const pdf = await browser.cdp('Page.printToPDF', {printBackground: true, generateTaggedPDF: true,
      paperWidth: paper === 'A4' ? 8.27 : 8.5, paperHeight: paper === 'A4' ? 11.69 : 11});
    const file = path.join(out, label + '.pdf');
    fs.writeFileSync(file, Buffer.from(pdf.data, 'base64'));
    await browser.pause(140);
    const after = await browser.evaluate(snapshot);
    assert.deepEqual({...after, scroll: before.scroll}, before);
    assert(Math.abs(after.scroll - before.scroll) <= 2, JSON.stringify({label, before, after}));
    const info = execFileSync('pdfinfo', [file], {encoding:'utf8'});
    assert.match(info, /^Tagged:\s+yes/m);
    const pages = execFileSync('pdftotext', ['-layout', file, '-'], {encoding:'utf8'}).split('\f').map(squeeze);
    assert(pages.some(page => page.includes(routeIndex === 0 ? 'OTRetarget' : 'HOI-Retarget')));
    const newSections = routeIndex === 0 ? ['对齐源时间再比较接触', 'PriMoRetarget'] : ['用公开单行数据复算掌点误差', 'ReContact'];
    for (const heading of newSections) assert(pages.some(page => page.includes(heading)), JSON.stringify({file, heading}));
    const pdfImages = execFileSync('pdfimages', ['-list', file], {encoding:'utf8'}).trim().split('\n').slice(2)
      .map(line => line.trim().split(/\s+/)).filter(columns => columns[2] === 'image' && Number(columns[6]) >= 3);
    for (const {aspect, caption} of figures) {
      const images = pdfImages.filter(columns => Math.abs(Number(columns[3]) / Number(columns[4]) - aspect) < 0.015);
      assert(images.length === 1 && pages[Number(images[0][0]) - 1]?.includes(squeeze(caption).slice(0, 18)),
        JSON.stringify({file, aspect, caption, images}));
    }
    for (const [label, value] of pairs) {
      assert(pages.some(page => page.includes(squeeze(label + value))), JSON.stringify({label, value, file}));
    }
    if (fields.length) {
      const flow = execFileSync('pdftotext', ['-raw', file, '-'], {encoding:'utf8'}).split('\f').map(squeeze);
      for (const [label, value] of fields) {
        assert(flow.some(page => page.includes(squeeze(label + value))), JSON.stringify({label, value, file}));
      }
      assert(flow.some(page => page.includes(squeeze('算一算握持前要留多少步' + fields[0].join('')))), file);
    }
    report.print.push({file, paper, pairs: pairs.length, figuresWithCaptions: figures.length, tagged: true, screenPositionRestored: true});
  }
}

async function main() {
  const browser = await connect(Number(portInput));
  try {
    await reading(browser);
    await discovery(browser);
    await chapters(browser);
    await articleLinks(browser);
    await keyboardScroll(browser);
    await figures(browser);
    await checkCalculator(browser);
    await printArticles(browser);
    assert.deepEqual(browser.exceptions, []);
  } finally {
    periodicReport(report);
    await browser.close();
  }
  console.log(JSON.stringify(Object.fromEntries(Object.entries(report).map(([name, items]) => [name, items.length]))));
}

main().catch(error => { console.error(error); process.exitCode = 1; });
