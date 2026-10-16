#!/usr/bin/env node
// Run serially with other focus-dependent checks against loopback Chrome CDP.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const base = process.argv[2] || 'http://127.0.0.1:1395';
const port = process.argv[3] || '9258';
const output = process.argv[4] || '/tmp/rrsi-lab-browser';
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const luminance = color => {
  const values = color.match(/[\d.]+/g).slice(0, 3).map(Number).map(n => n / 255)
    .map(n => n <= .04045 ? n / 12.92 : ((n + .055) / 1.055) ** 2.4);
  return values.reduce((sum, n, i) => sum + n * [.2126, .7152, .0722][i], 0);
};
const contrast = (one, two) => {
  const a = luminance(one), b = luminance(two);
  return (Math.max(a, b) + .05) / (Math.min(a, b) + .05);
};

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const endpoint = 'http://127.0.0.1:' + port;
  const tab = await (await fetch(endpoint + '/json/new?about:blank', {method: 'PUT'})).json();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  const pending = new Map(), exceptions = [], rows = [];
  let serial = 0;
  let restorePreference;
  try {
    await new Promise((resolve, reject) => {ws.onopen = resolve; ws.onerror = reject;});
    ws.onmessage = event => {
      const message = JSON.parse(event.data);
      if (message.id && pending.has(message.id)) {
        const task = pending.get(message.id);
        pending.delete(message.id); clearTimeout(task.timer);
        message.error ? task.reject(Error(JSON.stringify(message.error))) : task.resolve(message.result);
      } else if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
    };
    const cdp = (method, params = {}) => new Promise((resolve, reject) => {
      const id = ++serial;
      const timer = setTimeout(() => {pending.delete(id); reject(Error('CDP timeout: ' + method));}, 30000);
      pending.set(id, {resolve, reject, timer});
      ws.send(JSON.stringify({id, method, params}));
    });
    const evaluate = async expression => {
      const result = await cdp('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
      assert(!result.exceptionDetails, JSON.stringify(result.exceptionDetails));
      return result.result.value;
    };
    const waitFor = async expression => {
      const deadline = Date.now() + 15000;
      while (Date.now() < deadline) {if (await evaluate(expression)) return; await delay(100);}
      throw Error('Condition timed out: ' + expression);
    };
    const screenshot = async name => {
      const {data} = await cdp('Page.captureScreenshot', {format: 'png'});
      fs.writeFileSync(output + '/' + name + '.png', Buffer.from(data, 'base64'));
    };
    await cdp('Page.enable'); await cdp('Runtime.enable'); await cdp('Page.bringToFront');
    await cdp('Page.navigate', {url: base + '/labs/rrsi/'});
    await waitFor("document.readyState==='complete' && document.querySelector('#status')?.dataset.state==='pass'");
    const savedPreference = await evaluate("localStorage.getItem('color-theme')");
    restorePreference = async () => {
      await evaluate(savedPreference === null ? "localStorage.removeItem('color-theme')"
        : "localStorage.setItem('color-theme'," + JSON.stringify(savedPreference) + ")");
    };
    for (const width of [320, 768, 1440]) {
      await cdp('Emulation.setDeviceMetricsOverride', {width, height: 1000, deviceScaleFactor: 1, mobile: width < 768});
      for (const theme of ['light', 'dark', 'warm']) {
        await evaluate("localStorage.setItem('color-theme'," + JSON.stringify(theme) + ")");
        await cdp('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: theme === 'dark' ? 'light' : 'dark'}]});
        await cdp('Page.navigate', {url: base + '/labs/rrsi/'});
        await waitFor("document.readyState==='complete' && document.querySelector('#status')?.dataset.state==='pass'");
        for (const [name, expected] of [['A', 'pass'], ['B', 'fail'], ['C', 'pass'], ['D', 'fail'], ['E', 'pass'], ['F', 'fail']]) {
          await evaluate(`document.querySelector('[data-example="${name}"]').click()`);
          assert.equal(await evaluate("document.querySelector('#status').dataset.state"), expected, name);
          if (name === 'D') assert.match(await evaluate("document.querySelector('#cost-check').textContent"), /^未执行/);
          if (name === 'F') assert.doesNotMatch(await evaluate("document.querySelector('#guard-check').textContent"), /^未执行/);
        }
        // Boundary examples deliberately replace the whole experiment, including
        // accepted-type history, rather than only changing the candidate.
        await evaluate("document.querySelector('#reset-lab').click();document.querySelector('[data-example=E]').click();document.querySelector('#add-candidate').click();document.querySelector('#apply-round').click();document.querySelector('[data-example=A]').click();document.querySelector('#add-candidate').click()");
        assert.equal(await evaluate("document.querySelectorAll('#round-history>li').length"), 1);
        await evaluate("document.querySelector('[data-boundary=within]').click()");
        assert.equal(await evaluate("document.querySelector('#status').dataset.state"), 'pass');
        assert.match(await evaluate("document.querySelector('#cost-check').textContent"), /ΔS ≤ δ.*2\.88/);
        assert(await evaluate("document.querySelector('[name=domain]').value==='engineering' && document.querySelector('[name=delta]').valueAsNumber===.02 && document.querySelector('[name=score]').valueAsNumber===.81 && document.querySelector('[name=best]').valueAsNumber===.82 && document.querySelector('[name=cost]').valueAsNumber===1000000 && document.querySelector('[name=candidateCost]').valueAsNumber===2000000"));
        assert(await evaluate("!document.querySelector('#pool-rows').children.length && !document.querySelector('#round-history').children.length && document.querySelector('#download-history').hidden && !document.querySelector('input[name=accepted]:checked') && !document.querySelector('input[name=touched]:checked')"));
        await evaluate("document.querySelector('#engineering-boundary').scrollIntoView({block:'start'})");
        await screenshot(`boundary-within-${width}-${theme}`);
        await evaluate("document.querySelector('[data-boundary=above]').focus()");
        await cdp('Input.dispatchKeyEvent', {type:'keyDown', key:'Enter', code:'Enter', windowsVirtualKeyCode:13,
          text:'\r', unmodifiedText:'\r'});
        await cdp('Input.dispatchKeyEvent', {type:'keyUp', key:'Enter', code:'Enter', windowsVirtualKeyCode:13});
        assert.equal(await evaluate("document.querySelector('#status').dataset.state"), 'fail');
        assert.match(await evaluate("document.querySelector('#cost-check').textContent"), /ΔS > δ.*66\.24%/);
        await evaluate("document.querySelector('#engineering-boundary').scrollIntoView({block:'start'})");
        await screenshot(`boundary-above-${width}-${theme}`);
        await evaluate("document.querySelector('#reset-lab').click()");
        await evaluate("document.querySelector('[data-example=E]').click();document.querySelector('input[name=accepted][value=skill]').click()");
        assert.equal(await evaluate("document.querySelector('#status').dataset.state"), 'fail');
        assert.match(await evaluate("document.querySelector('#deltas').textContent"), /ν=0/);
        await evaluate("document.querySelector('[data-example=C]').click();document.querySelector('input[name=cost]').value='0';document.querySelector('input[name=cost]').dispatchEvent(new Event('input',{bubbles:true}))");
        assert.equal(await evaluate("document.querySelector('#status').dataset.state"), 'invalid');
        assert.equal(await evaluate("document.querySelector('#details').hidden"), true);
        assert(await evaluate("document.querySelector('#add-candidate').disabled && document.querySelector('#apply-round').disabled"));
        await evaluate("document.querySelector('#reset-lab').click();document.querySelector('[data-example=A]').click()");
        // Keyboard changes the checkbox and recomputes the result without pointer input.
        await evaluate("document.querySelector('input[name=guardOK]').focus()");
        await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: ' ', code: 'Space', windowsVirtualKeyCode: 32});
        await cdp('Input.dispatchKeyEvent', {type: 'keyUp', key: ' ', code: 'Space', windowsVirtualKeyCode: 32});
        assert.equal(await evaluate("document.querySelector('#status').dataset.state"), 'fail');
        const state = await evaluate(`({overflow:document.documentElement.scrollWidth>innerWidth+1,
          unlabeled:[...document.querySelectorAll('input,select')].filter(e=>!e.labels.length).length,
          scheme:getComputedStyle(document.body).backgroundColor,
          backLink:document.querySelector('main>a').pathname,
          focus:document.activeElement.name})`);
        assert(!state.overflow && !state.unlabeled, JSON.stringify(state));
        assert.equal(state.backLink, '/posts/ai/rrsi/');
        assert.equal(state.focus, 'guardOK');
        assert.equal(state.scheme, {light:'rgb(255, 255, 255)', dark:'rgb(18, 24, 32)', warm:'rgb(253, 250, 244)'}[theme]);
        const colors = await evaluate(`(() => {
          const styles = selector => getComputedStyle(document.querySelector(selector));
          return {background:styles('body').backgroundColor,
            inputBorder:styles('input[type=number]').borderColor,
            buttonBorder:styles('button').borderColor,
            buttonBackground:styles('button').backgroundColor};
        })()`);
        assert(contrast(colors.inputBorder, colors.background) >= 3, JSON.stringify(colors));
        assert(contrast(colors.buttonBorder, colors.buttonBackground) >= 3, JSON.stringify(colors));
        const {nodes} = await cdp('Accessibility.getFullAXTree');
        for (const name of ['本次候选触及', '过去已有获胜编辑触及']) {
          assert(nodes.some(node => node.role?.value === 'group' && node.name?.value === name),
            'Missing structural history group: ' + name);
        }
        // Candidate presets preserve path state; only the explicit reset starts over.
        await evaluate("document.querySelector('#reset-lab').click();document.querySelector('[data-example=E]').click();document.querySelector('#add-candidate').click();document.querySelector('#apply-round').click()");
        assert.deepEqual(await evaluate("({score:document.querySelector('[name=score]').valueAsNumber,best:document.querySelector('[name=best]').valueAsNumber,skill:document.querySelector('[name=accepted][value=skill]').checked})"),
          {score: .809, best: .82, skill: true});
        assert.match(await evaluate("document.querySelector('#round-history').textContent"), /ACCEPTED/);
        const exported = await evaluate("JSON.parse(decodeURIComponent(document.querySelector('#download-history').href.split(',').slice(1).join(',')))");
        assert.equal(exported.kind, 'constructed_rrsi_selection_rounds');
        assert.equal(exported.rounds.length, 1);
        assert.equal(exported.rounds[0].domain, 'coding');
        assert.equal(exported.rounds[0].t, 0);
        assert.equal(exported.rounds[0].config.delta, .017);
        assert.equal(Object.hasOwn(exported.rounds[0].base, 'candidate'), false);
        assert.equal(exported.rounds[0].base.score, .81);
        assert.equal(exported.rounds[0].next.score, .809);
        assert.deepEqual(exported.rounds[0].next.accepted, ['skill']);
        await evaluate("document.querySelector('[data-example=E]').click()");
        assert.equal(await evaluate("document.querySelector('[name=score]').valueAsNumber"), .809);
        assert.equal(await evaluate("document.querySelector('#status').dataset.state"), 'fail');
        await evaluate("document.querySelector('#add-candidate').click();document.querySelector('#apply-round').click()");
        assert.equal(await evaluate("document.querySelector('[name=best]').valueAsNumber"), .82);
        assert.match(await evaluate("document.querySelector('#round-history').lastElementChild.textContent"), /REJECTED/);
        assert.equal(await evaluate("document.querySelector('#round-history').children.length"), 2);
        // Every candidate is judged against the same base. Losing structural edits consume no novelty.
        await evaluate("document.querySelector('#reset-lab').click();for(const name of ['A','C','E']){document.querySelector('[data-example='+name+']').click();document.querySelector('#add-candidate').click()}");
        assert.equal(await evaluate("document.querySelectorAll('#pool-rows tr').length"), 3);
        assert.match(await evaluate("document.querySelector('#pool-summary').textContent"), /赢家是候选 1/);
        await evaluate("document.querySelector('#pool-title').scrollIntoView({block:'start'})");
        await screenshot(`pool-${width}-${theme}`);
        if (width === 320) {
          assert(await evaluate("document.querySelector('#pool-scroll').tabIndex===0"));
          await evaluate("document.querySelector('#pool-scroll').focus()");
          const before = await evaluate("document.querySelector('#pool-scroll').scrollLeft");
          await cdp('Input.dispatchKeyEvent', {type:'keyDown', key:'ArrowRight', code:'ArrowRight', windowsVirtualKeyCode:39});
          await cdp('Input.dispatchKeyEvent', {type:'keyUp', key:'ArrowRight', code:'ArrowRight', windowsVirtualKeyCode:39});
          await delay(200);
          assert(await evaluate("document.querySelector('#pool-scroll').scrollLeft") > before);
        }
        await evaluate("document.querySelector('#pool-rows button').click()");
        assert.match(await evaluate("document.querySelector('#pool-summary').textContent"), /赢家是候选 2/);
        await evaluate("document.querySelector('#apply-round').click()");
        assert.equal(await evaluate("document.querySelector('[name=score]').valueAsNumber"), .81);
        assert.equal(await evaluate("document.querySelector('[name=cost]').valueAsNumber"), 950000);
        assert.equal(await evaluate("document.querySelector('[name=accepted][value=skill]').checked"), false);
        assert.match(await evaluate("document.querySelector('#round-history').textContent"), /候选 3：LOST/);
        assert.equal(await evaluate("document.activeElement.id"), 'add-candidate');
        await evaluate("document.querySelector('#round-history').scrollIntoView({block:'center'})");
        await screenshot(`history-${width}-${theme}`);
        // Editing only a candidate keeps recorded rounds; changing the path starts a new trace.
        await evaluate("document.querySelector('[data-example=A]').click();document.querySelector('#add-candidate').click()");
        assert.equal(await evaluate("document.querySelector('#round-history').children.length"), 1);
        await evaluate("{const e=document.querySelector('[name=best]');e.value='.84';e.dispatchEvent(new Event('input',{bubbles:true}));}");
        assert.equal(await evaluate("document.querySelector('#round-history').children.length"), 0);
        assert.equal(await evaluate("document.querySelector('#download-history').hidden"), true);
        assert.equal(await evaluate("document.querySelector('#download-history').hasAttribute('href')"), false);
        assert.equal(await evaluate("document.querySelectorAll('#pool-rows tr').length"), 1);
        await evaluate("{const e=document.querySelector('[name=candidateCost]');e.value='0';e.dispatchEvent(new Event('input',{bubbles:true}));}");
        assert(await evaluate("document.querySelector('#add-candidate').disabled && document.querySelector('#apply-round').disabled && document.querySelector('#pool-content').hidden"));
        await evaluate("document.querySelector('[data-example=C]').click()");
        assert.equal(await evaluate("document.querySelectorAll('#pool-rows tr').length"), 1);
        assert.match(await evaluate("document.querySelector('#pool-rows').textContent"), /0.835/);
        await evaluate("document.querySelector('#reset-lab').click()");
        assert.equal(await evaluate("document.querySelectorAll('#pool-rows tr').length"), 0);
        assert.equal(await evaluate("document.querySelector('#round-history').children.length"), 0);
        assert(await evaluate("document.querySelector('#apply-round').disabled && document.querySelector('#clear-pool').disabled"));
        // A user-specified band, rather than the domain default, is recorded.
        await evaluate("{const e=document.querySelector('[name=delta]');e.value='.03';e.dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('#add-candidate').click();document.querySelector('#apply-round').click();}");
        const customBand = await evaluate("JSON.parse(decodeURIComponent(document.querySelector('#download-history').href.split(',').slice(1).join(',')))");
        assert.equal(customBand.rounds[0].base.delta, .03);
        assert.equal(customBand.rounds[0].config.delta, .03);
        assert.equal(customBand.rounds[0].winnerIndex, null);
        assert.equal(customBand.rounds[0].next.score, .81);
        await evaluate("document.querySelector('#reset-lab').click()");
        // Increasing the score across delta can select a stricter cost branch.
        await evaluate("{const e=document.querySelector('[name=domain]');e.value='engineering';e.dispatchEvent(new Event('change',{bubbles:true}));for(const [name,value] of [['candidate','.830'],['candidateCost','2000000']])document.querySelector('[name='+name+']').value=value;document.querySelector('[name=candidate]').dispatchEvent(new Event('input',{bubbles:true}));}");
        assert.equal(await evaluate("document.querySelector('#status').dataset.state"), 'pass');
        assert.equal(await evaluate("document.querySelector('#noise-boundary').hidden"), false);
        assert.match(await evaluate("document.querySelector('#cost-check').textContent"), /ΔS ≤ δ/);
        await evaluate("{const e=document.querySelector('[name=candidate]');e.value='.831';e.dispatchEvent(new Event('input',{bubbles:true}));}");
        assert.equal(await evaluate("document.querySelector('#status').dataset.state"), 'fail');
        assert.match(await evaluate("document.querySelector('#cost-check').textContent"), /66.24%/);
        // A rounded tie can actually lie above delta; preserve source comparison.
        await evaluate("{for(const [name,value] of [['score','.82'],['best','.82'],['candidate','.84']])document.querySelector('[name='+name+']').value=value;document.querySelector('[name=candidate]').dispatchEvent(new Event('input',{bubbles:true}));}");
        assert.equal(await evaluate("document.querySelector('#noise-boundary').hidden"), false);
        assert.match(await evaluate("document.querySelector('#noise-boundary').textContent"), /明确提分/);
        await evaluate("document.querySelector('#reset-lab').click()");
        assert.equal(await evaluate("document.documentElement.scrollWidth>innerWidth+1"), false);
        await screenshot(`controls-${width}-${theme}`);
        await evaluate("document.querySelector('.result').scrollIntoView({block:'center'})");
        await screenshot(`result-${width}-${theme}`);
        rows.push({width, theme, ...state});
      }
    }
    const {identifier} = await cdp('Page.addScriptToEvaluateOnNewDocument', {
      source: "Object.defineProperty(window,'localStorage',{configurable:true,get(){throw new DOMException('Blocked','SecurityError')}})"
    });
    try {
      await cdp('Page.navigate', {url: base + '/labs/rrsi/'});
      await waitFor("document.readyState==='complete' && document.querySelector('#status')?.dataset.state==='pass'");
      assert.equal(await evaluate("getComputedStyle(document.body).backgroundColor"), 'rgb(18, 24, 32)');
    } finally {
      await cdp('Page.removeScriptToEvaluateOnNewDocument', {identifier});
    }
    await cdp('Emulation.setScriptExecutionDisabled', {value: true});
    await cdp('Emulation.setDeviceMetricsOverride', {width:320, height:1000, deviceScaleFactor:1, mobile:true});
    await cdp('Page.navigate', {url: base + '/labs/rrsi/'});
    await waitFor("document.readyState==='complete' && !!document.querySelector('noscript p')");
    assert(await evaluate("document.querySelector('noscript p').getBoundingClientRect().height>0"));
    assert(await evaluate("[...document.querySelectorAll('[data-boundary]')].every(button=>button.disabled)"));
    assert.equal(await evaluate("document.documentElement.scrollWidth>innerWidth+1"), false);
    await evaluate("document.querySelector('#engineering-boundary').scrollIntoView({block:'start'})");
    await screenshot('boundary-nojs-320');
    assert.equal(exceptions.length, 0, JSON.stringify(exceptions));
    fs.writeFileSync(output + '/results.json', JSON.stringify({rows, exceptions, nojs: true, blockedStorage: true}, null, 2));
    console.log(JSON.stringify({passed: true, cases: rows.length, nojs: true, blockedStorage: true, output}));
  } finally {
    if (restorePreference) await restorePreference();
    for (const task of pending.values()) clearTimeout(task.timer);
    ws.close();
    await fetch(endpoint + '/json/close/' + tab.id);
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
