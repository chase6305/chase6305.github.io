/* Constructed height traces; no robot, model, or network execution. */
(function (root) {
  'use strict';
  const rules = Object.freeze({lift: .10, hold: .05, holdSamples: 20,
    placement: .03, placementSamples: 20, finalFloor: -.01,
    descentStep: .01, terminalStep: .005, terminalRange: .01});
  function evaluate(heights) {
    if (!Array.isArray(heights) || !heights.length ||
        heights.some(x => typeof x !== 'number' || !Number.isFinite(x))) {
      throw new Error('高度须为非空的有限数值数组，单位为米。');
    }
    const lifts = heights.slice(1).map(x => x - heights[0]);
    let peak = 0, run = 0, longest = 0, previous = 0, descent = 0;
    for (const value of lifts) {
      peak = Math.max(peak, value);
      if (peak >= rules.hold) descent = Math.max(descent, previous - value);
      run = value >= rules.hold ? run + 1 : 0;
      longest = Math.max(longest, run); previous = value;
    }
    const terminal = lifts.slice(-rules.placementSamples);
    const enough = lifts.length >= rules.placementSamples;
    const terminalMax = terminal.length ? Math.max(...terminal) : 0;
    const span = terminal.length ? terminalMax - Math.min(...terminal) : 0;
    const step = terminal.reduce((value, x, i) => i ? Math.max(value, Math.abs(x - terminal[i - 1])) : value, 0);
    const final = lifts.length ? lifts[lifts.length - 1] : 0;
    const checks = [peak >= rules.lift, longest >= rules.holdSamples,
      descent <= rules.descentStep, enough && terminalMax <= rules.placement,
      enough && step <= rules.terminalStep && span <= rules.terminalRange,
      final >= rules.finalFloor];
    return {success: checks.every(Boolean), checks, sampleCount: lifts.length,
      peak, longest, descent, enough, terminalMax, terminalStep: step,
      terminalRange: span, final};
  }
  function makeTrace({descentSteps = 22, finalHeight = 0, jitter = 0, terminalSamples = 25} = {}) {
    for (const [name, value] of Object.entries({descentSteps, finalHeight, jitter, terminalSamples})) {
      if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error('请填写完整的有限数值。');
      if (['descentSteps', 'terminalSamples'].includes(name) &&
          (!Number.isInteger(value) || value < 1 || value > 200)) throw new Error('步数须为 1～200 的整数。');
    }
    if (finalHeight < -.05 || finalHeight > .2) throw new Error('末段中心高度须在 −0.05～0.20 m。');
    if (jitter < 0 || jitter > .03) throw new Error('波动幅度须在 0～0.03 m。');
    const trace = [0];
    for (let i = 1; i <= 22; i++) trace.push(.11 * i / 22);
    trace.push(...Array(25).fill(.11));
    // Use the article companion's interpolation order: exact preset arrays
    // matter because a one-bit difference at 0.05 can change a hold count.
    for (let i = 1; i <= descentSteps; i++) {
      const t = i / descentSteps;
      trace.push(finalHeight === .11 ? .11 : .11 * (1 - t) + finalHeight * t);
    }
    for (let i = 0; i < terminalSamples; i++) trace.push(finalHeight + (i % 2 ? -jitter : jitter));
    return trace;
  }
  const presets = Object.freeze({stable: {descentSteps: 22, finalHeight: 0, jitter: 0, terminalSamples: 25},
    aloft: {descentSteps: 22, finalHeight: .11, jitter: 0, terminalSamples: 25},
    drop: {descentSteps: 1, finalHeight: 0, jitter: 0, terminalSamples: 46}});
  if (typeof module !== 'undefined' && module.exports) module.exports = {rules, evaluate, makeTrace, presets};
  if (!root.document) return;
  const doc = root.document, form = doc.getElementById('height-form');
  if (!form) return;
  const field = name => form.elements.namedItem(name);
  const fixed = n => Number(n.toPrecision(6)).toString();
  const svgNS = 'http://www.w3.org/2000/svg';
  function shape(name, attributes, text) {
    const node = doc.createElementNS(svgNS, name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  }
  let currentTrace = null;
  function plot(trace) {
    const svg = doc.getElementById('height-chart');
    const width = Math.max(270, Math.floor(svg.parentElement.clientWidth)), height = 320;
    const left = 52, right = width - 18, top = 30, bottom = 273;
    const low = Math.min(-.02, ...trace) - .005, high = Math.max(.13, ...trace) + .005;
    const x = i => left + (right - left) * i / (trace.length - 1);
    const y = v => bottom - (bottom - top) * (v - low) / (high - low);
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const title = shape('title', {id: 'chart-title'}, '构造的 20 Hz 相对高度曲线');
    const description = shape('desc', {id: 'chart-description'},
      `共 ${trace.length} 个样本；首样本建立基准，其余 ${trace.length - 1} 次更新参与判定。紫色区是最后 20 次更新，详细条件列在图后。`);
    svg.replaceChildren(title, description);
    svg.append(shape('rect', {x: x(Math.max(1, trace.length - 20)), y: top,
      width: right - x(Math.max(1, trace.length - 20)), height: bottom - top, class: 'terminal-window'}));
    for (const [value, cls] of [[.10, 'lift-line'], [.05, 'hold-line'], [.03, 'place-line'], [0, 'zero-line']]) {
      svg.append(shape('line', {x1: left, x2: right, y1: y(value), y2: y(value), class: cls}));
      svg.append(shape('text', {x: left - 8, y: y(value) + 4, 'text-anchor': 'end'}, fixed(value)));
    }
    // Keep the extent readable when a custom trace leaves the preset range.
    for (const value of [Math.min(...trace), Math.max(...trace)]) {
      if (value >= -.02 && value <= .13) continue;
      svg.append(shape('line', {x1: left - 4, x2: left, y1: y(value), y2: y(value), class: 'axis'}));
      svg.append(shape('text', {x: left - 8, y: y(value) + 4, 'text-anchor': 'end'}, fixed(value)));
    }
    svg.append(shape('line', {x1: left, x2: left, y1: top, y2: bottom, class: 'axis'}));
    svg.append(shape('line', {x1: left, x2: right, y1: bottom, y2: bottom, class: 'axis'}));
    for (const fraction of [0, .5, 1]) {
      const i = fraction * (trace.length - 1);
      svg.append(shape('text', {x: x(i), y: bottom + 21, 'text-anchor': 'middle'}, fixed(i / 20)));
    }
    svg.append(shape('text', {x: left, y: 17}, '相对高度 / m'));
    svg.append(shape('text', {x: (left + right) / 2, y: height - 5, 'text-anchor': 'middle'}, '时间 / s（含基准首样本）'));
    svg.append(shape('polyline', {points: trace.map((v, i) => `${x(i)},${y(v)}`).join(' '), class: 'height-trace'}));
  }
  function render() {
    const status = doc.getElementById('status'), details = doc.getElementById('details');
    try {
      const input = Object.fromEntries(Object.keys(presets.stable).map(name => [name, field(name).valueAsNumber]));
      const trace = makeTrace(input), result = evaluate(trace);
      details.hidden = false; currentTrace = trace;
      const message = result.success ? '通过：满足当前高度评估器的全部条件' : '未通过：至少一项高度条件不满足';
      if (status.textContent !== message) status.textContent = message;
      status.dataset.state = result.success ? 'pass' : 'fail';
      doc.getElementById('sample-count').textContent = `${trace.length} 个样本 − 1 个基准 = ${result.sampleCount} 次更新；首末时间戳相差 ${fixed((trace.length - 1) / 20)} 秒。`;
      const descriptions = [
        `峰值 ${fixed(result.peak)} m；要求 ≥ 0.10 m`,
        `最长连续 ${result.longest} 次更新不低于 0.05 m；要求 ≥ 20 次`,
        `达到保持高度后，最大单步下降 ${fixed(result.descent)} m；要求 ≤ 0.01 m`,
        `最后 20 次更新最高 ${fixed(result.terminalMax)} m；要求完整窗口且均 ≤ 0.03 m`,
        `末窗最大相邻变化 ${fixed(result.terminalStep)} m、极差 ${fixed(result.terminalRange)} m；要求分别 ≤ 0.005 m、≤ 0.01 m`,
        `最后高度 ${fixed(result.final)} m；要求 ≥ −0.01 m`
      ];
      const checks = doc.getElementById('checks'); checks.replaceChildren();
      descriptions.forEach((description, i) => {
        const li = doc.createElement('li'), label = doc.createElement('strong');
        label.textContent = result.checks[i] ? '通过' : '失败';
        label.className = result.checks[i] ? 'pass' : 'fail';
        li.append(label, '：' + description); checks.append(li);
      });
      const nearFailures = [
        ['峰值', result.peak, rules.lift, 'min'],
        ['最大单步下降', result.descent, rules.descentStep, 'max'],
        ['末窗最高高度', result.terminalMax, rules.placement, 'max'],
        ['末窗相邻变化', result.terminalStep, rules.terminalStep, 'max'],
        ['末窗极差', result.terminalRange, rules.terminalRange, 'max'],
        ['最后高度', result.final, rules.finalFloor, 'min']
      ].filter(([, value, limit, direction]) => fixed(value) === fixed(limit) &&
        (direction === 'min' ? value < limit : value > limit));
      const precision = doc.getElementById('precision-warning');
      precision.hidden = nearFailures.length === 0;
      precision.textContent = nearFailures.map(([name, value, limit]) =>
        `${name}的未舍入值是 ${value.toString()} m，严格比较后未满足 ${limit} m 阈值。`).join('');
      const download = doc.getElementById('download-trace');
      download.href = 'data:application/json;charset=utf-8,' + encodeURIComponent(JSON.stringify({illustrative: true, fps: 20, heights_m: trace}, null, 2));
      plot(trace);
    } catch (error) {
      status.textContent = error.message; status.dataset.state = 'invalid';
      details.hidden = true; currentTrace = null;
    }
  }
  doc.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => {
    for (const [name, value] of Object.entries(presets[button.dataset.preset])) field(name).value = value;
    render();
  }));
  form.addEventListener('input', render);
  form.addEventListener('submit', event => event.preventDefault());
  root.addEventListener('resize', () => {if (currentTrace) plot(currentTrace);});
  render();
})(typeof globalThis !== 'undefined' ? globalThis : this);
