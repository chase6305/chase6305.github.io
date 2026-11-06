(function () {
  'use strict';
  const DATA_STD = .02;
  function calculate(mean, std, excitation = 0) {
    if (!Number.isFinite(mean) || mean < -.5 || mean > .5 ||
        !Number.isFinite(std) || std < .005 || std > .3 ||
        !Number.isFinite(excitation) || excitation < 0 || excitation > 1) {
      throw new RangeError('先验或独立激励超出允许范围。');
    }
    const p = 1 / DATA_STD**2, s = 1 / std**2, e2 = excitation**2;
    const a = p * (1 + e2) + s, b = p * (1 - e2), d = p * (1 + e2);
    const determinant = 4 * p**2 * e2 + s * p * (1 + e2);
    const kneeError = s * d / determinant * (mean - .2);
    const ankleError = -s * b / determinant * (mean - .2);
    return {knee:.2 + kneeError, ankle:-.1 + ankleError, kneeError,
      ankleError, kneeStd:Math.sqrt(d / determinant), ankleStd:Math.sqrt(a / determinant),
      dataRank:excitation === 0 ? 1 : 2,
      dataCondition:excitation === 0 ? Infinity : 1 / e2};
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {calculate};
  if (typeof document === 'undefined') return;
  const NS = 'http://www.w3.org/2000/svg';
  const format = (value, signed) => {
    const rounded = Number(value.toFixed(4));
    return (signed && rounded > 0 ? '+' : '') + rounded.toFixed(4).replace('-', '−');
  };
  function element(name, attributes) {
    const node = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    return node;
  }
  function chart(svg, mean, std, truth, label) {
    const x = value => 20 + 180 * (value + 1);
    const fragment = document.createDocumentFragment();
    fragment.append(element('line', {x1:20,y1:30,x2:380,y2:30,class:'omnicalib-prior__axis-line'}));
    for (const value of [-1,0,1]) fragment.append(element('line',{x1:x(value),y1:25,x2:x(value),y2:35,class:'omnicalib-prior__axis-line'}));
    fragment.append(element('line',{x1:x(truth),y1:7,x2:x(truth),y2:53,class:'omnicalib-prior__truth'}));
    fragment.append(element('line',{x1:x(mean-std),y1:30,x2:x(mean+std),y2:30,class:'omnicalib-prior__estimate'}));
    for (const value of [mean-std,mean+std]) fragment.append(element('line',{x1:x(value),y1:21,x2:x(value),y2:39,class:'omnicalib-prior__estimate'}));
    fragment.append(element('circle',{cx:x(mean),cy:30,r:5,class:'omnicalib-prior__point'}));
    svg.replaceChildren(fragment);
    svg.setAttribute('aria-label', `${label}：估计 ${format(mean)} 度，局部标准差 ${format(std)} 度，构造真值 ${format(truth)} 度`);
  }
  function setup(root) {
    if (root.dataset.ready) return;
    const form = root.querySelector('form');
    const fields = ['mean','std','excitation'].map(name => form.elements.namedItem(name));
    const error = root.querySelector('[data-error]');
    const computed = root.querySelector('[data-computed]');
    const outputs = [...root.querySelectorAll('[data-value]')];
    function render() {
      const invalid = fields.filter(field => !field.validity.valid || !Number.isFinite(field.valueAsNumber));
      for (const field of fields) {
        if (invalid.includes(field)) field.setAttribute('aria-invalid','true');
        else field.removeAttribute('aria-invalid');
      }
      if (invalid.length) {
        error.textContent = '请填写范围内的有限数值：均值 −0.5° 到 0.5°，标准差 0.005° 到 0.3°，独立激励 0 到 1。';
        error.hidden = false;computed.hidden = true;
        for (const output of outputs) output.textContent = '';
        for (const svg of root.querySelectorAll('svg')) svg.replaceChildren();
        root.querySelector('[data-note]').textContent = '';
        return;
      }
      const result = calculate(...fields.map(field => field.valueAsNumber));
      for (const output of outputs) output.textContent = format(result[output.dataset.value],output.dataset.value.endsWith('Error'));
      chart(root.querySelector('[data-chart="knee"]'),result.knee,result.kneeStd,.2,'膝偏置');
      chart(root.querySelector('[data-chart="ankle"]'),result.ankle,result.ankleStd,-.1,'踝偏置');
      const condition = result.dataRank === 1 ? '无穷大' : Number.isFinite(result.dataCondition) ? result.dataCondition.toPrecision(4) : '超出浮点范围';
      root.querySelector('[data-note]').textContent = `数据秩 ${result.dataRank}/2，数据条件数 ${condition}；加入先验后总秩 2/2。膝误差为其局部标准差的 ${(Math.abs(result.kneeError)/result.kneeStd).toFixed(2)} 倍。` +
        (result.dataRank === 1 ? '当前数据只约束偏置和；先验强度只改变标准差，不改变最优分解。' : '独立激励提供第二个参数组合，但弱激励或偏置先验仍可能造成明显误差。');
      computed.hidden = false;error.hidden = true;error.textContent = '';
    }
    try { render(); } catch (_) { return; }
    form.addEventListener('submit',event=>event.preventDefault());
    form.addEventListener('input',render);
    for (const button of root.querySelectorAll('[data-preset]')) button.addEventListener('click',()=>{
      const preset = button.dataset.preset;
      fields[0].value = preset==='aligned' ? '.2' : '.4';
      fields[1].value = preset==='confident' ? '.005' : '.02';
      fields[2].value = preset==='excited' ? '1' : '0';
      render();
    });
    root.dataset.ready = 'true';
    root.querySelector('[data-interactive]').hidden = false;
    root.querySelector('[data-static-result]').hidden = true;
  }
  for (const root of document.querySelectorAll('[data-omnicalib-prior]')) setup(root);
})();
