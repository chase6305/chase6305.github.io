(function () {
  'use strict';
  function calculate(ratio = 3, angle = 0, epsilon = 0) {
    if (!Number.isFinite(ratio) || ratio < 1 || ratio > 10 ||
        !Number.isFinite(angle) || angle < 0 || angle > 90 ||
        !Number.isFinite(epsilon) || epsilon < 0 || epsilon > 10) {
      throw new RangeError('尺度比、方向或正则项超出允许范围。');
    }
    const theta = angle * Math.PI / 180, c = Math.cos(theta), s = Math.sin(theta);
    const major = ratio**2 + epsilon, minor = 1 + epsilon;
    // For delta_A=(2,0), delta_B=(0,-1), rotate into eigenvectors.
    const metricA = 2 * Math.hypot(c / Math.sqrt(major), s / Math.sqrt(minor));
    const metricB = Math.hypot(s / Math.sqrt(major), c / Math.sqrt(minor));
    const tolerance = 64 * Number.EPSILON * Math.max(1, metricA**2, metricB**2);
    const metricWinner = Math.abs(metricA**2-metricB**2) <= tolerance ? 'tie' : metricA < metricB ? 'A' : 'B';
    return {euclideanA:2, euclideanB:1, metricA, metricB,
      euclideanWinner:'B', metricWinner, condition:major/minor};
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {calculate};
  if (typeof document === 'undefined') return;
  const NS = 'http://www.w3.org/2000/svg';
  const names = {A:'A', B:'B', tie:'并列'};
  const format = value => value.toFixed(4);
  function element(name, attrs) {
    const node = document.createElementNS(NS,name);
    for (const [key,value] of Object.entries(attrs)) node.setAttribute(key,value);
    return node;
  }
  function chart(svg, raw, metric, candidate) {
    const fragment = document.createDocumentFragment();
    fragment.append(element('line',{x1:8,y1:56,x2:388,y2:56,class:'last-whitening__axis-line'}));
    fragment.append(element('rect',{x:8,y:7,width:raw/2*380,height:15,class:'last-whitening__raw'}));
    fragment.append(element('rect',{x:8,y:32,width:metric/2*380,height:15,class:'last-whitening__metric'}));
    svg.replaceChildren(fragment);
    svg.setAttribute('aria-label',`候选 ${candidate}：原坐标距离 ${format(raw)}，白化距离 ${format(metric)}`);
  }
  function setup(root) {
    if (root.dataset.ready) return;
    const form = root.querySelector('form');
    const fields = ['ratio','angle','epsilon'].map(name => form.elements.namedItem(name));
    const outputs = [...root.querySelectorAll('[data-value]')];
    const error = root.querySelector('[data-error]'), computed = root.querySelector('[data-computed]');
    function render() {
      const invalid = fields.filter(field => !field.validity.valid || !Number.isFinite(field.valueAsNumber));
      for (const field of fields) {
        if (invalid.includes(field)) field.setAttribute('aria-invalid','true');
        else field.removeAttribute('aria-invalid');
      }
      if (invalid.length) {
        error.textContent = '请填写范围内的有限数值：尺度比 1 到 10，方向 0° 到 90°，正则项 0 到 10。';
        error.hidden = false; computed.hidden = true;
        for (const output of outputs) output.textContent = '';
        for (const svg of root.querySelectorAll('svg')) svg.replaceChildren();
        root.querySelector('[data-note]').textContent = '';
        return;
      }
      const result = calculate(...fields.map(field => field.valueAsNumber));
      for (const output of outputs) {
        const value = result[output.dataset.value];
        output.textContent = typeof value === 'number' ? format(value) : names[value];
      }
      chart(root.querySelector('[data-chart="A"]'),2,result.metricA,'A');
      chart(root.querySelector('[data-chart="B"]'),1,result.metricB,'B');
      root.querySelector('[data-note]').textContent = result.metricWinner === 'tie' ?
        '两候选的白化距离并列；实际分词器需要固定一致的并列规则。' :
        `白化距离选择 ${result.metricWinner}。` + (result.metricWinner === 'A' ? '它的原坐标误差更大，说明度量选择与原坐标重建误差是两项不同判断。' : '当前它也具有更小的原坐标误差。');
      computed.hidden = false; error.hidden = true; error.textContent = '';
    }
    try { render(); } catch (_) { return; }
    form.addEventListener('submit',event => event.preventDefault());
    form.addEventListener('input',render);
    const presets = {default:[3,0,0],isotropic:[1,0,0],rotated:[3,90,0],tie:[2,0,0],regularized:[3,0,3]};
    for (const button of root.querySelectorAll('[data-preset]')) button.addEventListener('click',() => {
      presets[button.dataset.preset].forEach((value,index) => {fields[index].value = String(value)});
      render();
    });
    root.dataset.ready = 'true';
    root.querySelector('[data-interactive]').hidden = false;
    root.querySelector('[data-static-result]').hidden = true;
  }
  for (const root of document.querySelectorAll('[data-last-whitening]')) setup(root);
})();
