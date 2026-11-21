/* Fixed-probability IID Bernoulli groups. No model, simulator or RL update. */
(function () {
  "use strict";
  const defaults = {probability: 2, group: 8, groups: 128};
  const limits = {probability: [0, 100], group: [2, 64], groups: [1, 1024]};
  const keys = ["all_fail", "all_success", "mixed", "candidate_groups", "candidate_rollouts", "retained_success"];
  const percentages = new Set(["all_fail", "all_success", "mixed", "retained_success"]);
  function calculate(values) {
    for (const [name, [min, max]] of Object.entries(limits)) {
      const value = values[name];
      if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max ||
          (name !== 'probability' && !Number.isInteger(value))) {
        const error = new RangeError('Invalid ' + name); error.field = name; throw error;
      }
    }
    const p = values.probability / 100, g = values.group, q = Math.min(p, 1-p);
    const mixed = q === 0 ? 0 : -Math.expm1(g * Math.log1p(-q)) - Math.pow(q, g);
    const raw = mixed ? values.groups / mixed : null;
    const rollouts = raw === null ? null : raw * g;
    const retained = mixed ? p * -Math.expm1((g-1) * Math.log(p)) / mixed : null;
    const available = value => value === null || Number.isFinite(value) ? value : null;
    return {all_fail: Math.pow(1-p,g), all_success: Math.pow(p,g), mixed,
      candidate_groups: available(raw), candidate_rollouts: available(rollouts),
      retained_success: available(retained), accepted_rollouts: g * values.groups};
  }
  function formatResult(result, key) {
    const value = result[key];
    if (value === null) return result.mixed > 0 ? '超出数值范围' : '无混合组';
    const number = percentages.has(key) ? value * 100 : value;
    const suffix = percentages.has(key) ? '%' : '';
    if ((number > 0 && number < 0.0001) || number >= 1e9) return number.toExponential(3) + suffix;
    return number.toFixed(percentages.has(key) ? 4 : 2).replace(/\.?0+$/, '') + suffix;
  }
  function anchorBeforeEnhancement(element) {
    if (element.getBoundingClientRect().top >= 0) return null;
    const article = element.closest('.content');
    if (!article) return null;
    const target = [...article.querySelectorAll('h2,h3,p,pre,figure,table')].find(node => {
      if (element.contains(node) || !(element.compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING)) return false;
      const box = node.getBoundingClientRect();
      return box.width > 0 && box.height > 0 && box.bottom > 64 && box.top < innerHeight;
    });
    return target ? {target, top: target.getBoundingClientRect().top} : null;
  }
  function mount(element) {
    if (element.dataset.ready === 'true') return;
    if (typeof Number.isFinite !== 'function' || typeof Number.isInteger !== 'function' ||
        typeof Math.expm1 !== 'function' || typeof Math.log1p !== 'function' || typeof Object.fromEntries !== 'function') return;
    const controls = element.querySelector('[data-controls]'), form = element.querySelector('form');
    const error = element.querySelector('[data-error]'), status = element.querySelector('[data-status]');
    const staticNote = element.querySelector('[data-static]');
    const outputs = Object.fromEntries(keys.map(key => [key,element.querySelector('[data-value="'+key+'"]')]));
    if (!controls || !form || !error || !status || !staticNote || Object.values(outputs).some(v=>!v)) return;
    const anchor = anchorBeforeEnhancement(element);
    function clearError() {
      error.hidden = true; error.textContent = '';
      for (const name of Object.keys(limits)) {
        const input = form.elements.namedItem(name);
        input.removeAttribute('aria-invalid'); input.setAttribute('aria-describedby',element.id+'-help');
      }
    }
    function clearResults() { for (const output of Object.values(outputs)) output.textContent = '—'; }
    function update() {
      clearError();
      const values = Object.fromEntries(Object.keys(limits).map(name => {
        const text = form.elements.namedItem(name).value.trim();
        return [name,text === '' ? NaN : Number(text)];
      }));
      try {
        const result = calculate(values);
        for (const key of keys) outputs[key].textContent = formatResult(result,key);
        element.dataset.result = 'valid';
        status.textContent = result.mixed === 0 ? '所有组奖励一致，无法形成混合组。此算例没有策略更新或额外探索机制。' :
          '计算完成：混合组概率 ' + formatResult(result,'mixed') + '，候选 rollout 总数期望 ' + formatResult(result,'candidate_rollouts') +
          '；目标保留 ' + result.accepted_rollouts + ' 段轨迹。以上采用独立、固定成功概率假设，保留组成功比例不是评测成功率。';
        return true;
      } catch (failure) {
        clearResults(); element.dataset.result = 'invalid';
        const input = form.elements.namedItem(failure.field);
        if (input) {
          input.setAttribute('aria-invalid','true'); input.setAttribute('aria-describedby',element.id+'-help '+error.id);
          error.textContent = input.labels[0].querySelector('[data-label]').textContent + '：请输入 ' + input.min + ' 至 ' + input.max +
            (failure.field === 'probability' ? ' 的有限数值。' : ' 的整数。');
        } else error.textContent = '无法计算，请检查输入。';
        error.hidden = false; status.textContent = '尚无有效结果；请修正输入后重新计算。';
        return false;
      }
    }
    update();
    form.addEventListener('submit',event=>{event.preventDefault();update();});
    form.addEventListener('input',()=>{clearError();clearResults();element.dataset.result='pending';status.textContent='输入已更改，请计算当前配置。';});
    for (const button of element.querySelectorAll('[data-preset]')) button.addEventListener('click',()=>{
      const values = {...defaults,probability:button.dataset.preset === 'rare' ? 0.4 : button.dataset.preset === 'balanced' ? 50 : 2};
      for (const [name,value] of Object.entries(values)) form.elements.namedItem(name).value = value;
      update();
    });
    controls.hidden = false; staticNote.hidden = true; element.dataset.ready = 'true';
    if (anchor?.target.isConnected) {
      const delta = anchor.target.getBoundingClientRect().top-anchor.top;
      if (Math.abs(delta)>.5) window.scrollBy({top:delta,behavior:'instant'});
    }
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {calculate,formatResult,defaults,limits};
  if (typeof document !== 'undefined') {
    const initialize = ()=>document.querySelectorAll('[data-pear-sampling]').forEach(mount);
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded',initialize,{once:true});
    else initialize();
  }
})();
