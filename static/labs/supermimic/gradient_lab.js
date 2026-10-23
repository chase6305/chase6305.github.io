// A constructed scalar control problem, independent of the SuperMimic implementation.
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.SuperMimicGradientLab = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const defaults = Object.freeze({theta: .1, gain: 2, offset: 0, learningRate: .01, detach: false});
  const presets = Object.freeze({
    full: defaults,
    detached: Object.freeze({...defaults, detach: true}),
    opposite: Object.freeze({...defaults, gain: -2, offset: .4}),
  });

  function validate(input) {
    if (!input || typeof input !== 'object' || Array.isArray(input)) throw Error('输入必须是参数对象。');
    const p = {...defaults, ...input};
    for (const [key, low, high] of [
      ['theta', -.5, .75], ['gain', -4, 4], ['offset', -1, 1], ['learningRate', .001, .1],
    ]) {
      if (typeof p[key] !== 'number' || !Number.isFinite(p[key]) || p[key] < low || p[key] > high) {
        throw Error(`${key} 必须在 ${low} 至 ${high} 之间。`);
      }
    }
    if (typeof p.detach !== 'boolean') throw Error('detach 必须为布尔值。');
    return p;
  }

  function rollout(theta, gain, offset) {
    const states = [0];
    const rewards = [];
    for (let i = 0; i < 2; i++) {
      const next = states[i] + gain * theta + offset;
      states.push(next);
      rewards.push(-((next - 1) ** 2));
    }
    return {states, rewards, totalReturn: rewards[0] + rewards[1]};
  }

  function gradients(trace, gain, detach) {
    const stateDerivatives = [0, gain, detach ? gain : 2 * gain];
    const contributions = trace.rewards.map((_, i) => -2 * (trace.states[i + 1] - 1) * stateDerivatives[i + 1]);
    return {stateDerivatives, contributions, total: contributions[0] + contributions[1]};
  }

  function evaluate(input = {}) {
    const p = validate(input);
    const reference = rollout(p.theta, 2, 0);
    const model = rollout(p.theta, p.gain, p.offset);
    const full = gradients(model, p.gain, false);
    const selected = gradients(model, p.gain, p.detach);
    const referenceGradient = gradients(reference, 2, false).total;
    const epsilon = 1e-6;
    const finiteDifference = (
      rollout(p.theta + epsilon, p.gain, p.offset).totalReturn
      - rollout(p.theta - epsilon, p.gain, p.offset).totalReturn
    ) / (2 * epsilon);
    const thetaAfter = p.theta + p.learningRate * selected.total;
    const referenceAfter = rollout(thetaAfter, 2, 0);
    const modelAfter = rollout(thetaAfter, p.gain, p.offset);
    const stateRMSE = Math.hypot(...model.states.slice(1).map((s, i) => s - reference.states[i + 1])) / Math.sqrt(2);
    return {
      input: p, reference, model, full, selected, referenceGradient, finiteDifference, stateRMSE,
      after: {
        theta: thetaAfter, reference: referenceAfter, model: modelAfter,
        referenceReturnChange: referenceAfter.totalReturn - reference.totalReturn,
        modelReturnChange: modelAfter.totalReturn - model.totalReturn,
      },
    };
  }

  return {defaults, presets, evaluate};
});

if (typeof module === 'object' && module.exports && require.main === module) {
  const fs = require('node:fs');
  const payload = process.argv[2] ? JSON.parse(fs.readFileSync(process.argv[2], 'utf8')) : {};
  console.log(JSON.stringify(module.exports.evaluate(payload.input || payload), null, 2));
}

if (typeof document !== 'undefined') {
  (function () {
    'use strict';
    const {presets, evaluate} = globalThis.SuperMimicGradientLab;
    const form = document.getElementById('gradient-form');
    const status = document.getElementById('status');
    const details = document.getElementById('result-details');
    const download = document.getElementById('download-case');
    const format = n => Math.abs(n) < 1e-10 ? '0' : Number(n.toPrecision(7)).toString();
    const put = (id, n) => { document.getElementById(id).textContent = format(n); };

    function render() {
      try {
        if (!form.checkValidity()) throw Error('请完整填写各项参数，并使用输入框标出的范围。');
        const result = evaluate({
          theta: form.elements.theta.valueAsNumber,
          gain: form.elements.gain.valueAsNumber,
          offset: form.elements.offset.valueAsNumber,
          learningRate: form.elements.learningRate.valueAsNumber,
          detach: form.elements.detach.checked,
        });
        details.hidden = false;
        for (const name of ['reference', 'model']) {
          result[name].states.forEach((s, i) => put(`${name}-state-${i}`, s));
          put(`${name}-return`, result[name].totalReturn);
          put(`${name}-after`, result.after[name].totalReturn);
        }
        for (const [id, value] of Object.entries({
          'state-error': result.stateRMSE, 'gradient-reference': result.referenceGradient,
          'gradient-full': result.full.total, 'gradient-selected': result.selected.total,
          'gradient-fd': result.finiteDifference, 'theta-after': result.after.theta,
          'reference-change': result.after.referenceReturnChange, 'model-change': result.after.modelReturnChange,
        })) put(id, value);
        for (let i = 0; i < 2; i++) {
          put(`derivative-${i + 1}`, result.selected.stateDerivatives[i + 1]);
          put(`contribution-${i + 1}`, result.selected.contributions[i]);
        }
        document.getElementById('selected-path').textContent = result.input.detach ? '切断中间状态路径' : '完整 BPTT';
        document.getElementById('detach-note').hidden = !result.input.detach;
        const change = result.after.referenceReturnChange;
        status.textContent = `参照回报 ${format(result.reference.totalReturn)} → ${format(result.after.reference.totalReturn)}`
          + `（${change > 1e-10 ? '提高' : change < -1e-10 ? '降低' : '不变'}）；模型回报 ${format(result.model.totalReturn)} → ${format(result.after.model.totalReturn)}。`
          + (result.input.detach ? '当前反向路径切断了第一步状态对第二步的影响。' : '完整替代目标的解析梯度与有限差分一致。');
        const payload = {schema: 'supermimic-scalar-gradient-v1', illustrative: true,
          assumptions: 's0=0; a0=a1=theta; reference: s_next=s+2a; model: s_next=s+gain*a+offset; r=-(s_next-1)^2; H=2; gamma=1; no termination or critic; model parameters fixed',
          input: result.input, result};
        download.href = 'data:application/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(payload, null, 2) + '\n');
        download.hidden = false;
      } catch (error) {
        status.textContent = error.message;
        details.hidden = true;
        download.hidden = true;
        download.removeAttribute('href');
      }
    }

    form.addEventListener('input', render);
    form.addEventListener('submit', event => event.preventDefault());
    document.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => {
      const preset = presets[button.dataset.preset];
      for (const [key, value] of Object.entries(preset)) {
        if (key === 'detach') form.elements[key].checked = value;
        else form.elements[key].value = value;
      }
      render();
    }));
    document.getElementById('interactive').hidden = false;
    render();
  })();
}
