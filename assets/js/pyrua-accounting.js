/* Constructed input-token accounting; no tokenizer, policy or robot model. */
(function () {
  "use strict";
  const limits = {operations: [1, 100], group: [1, 100], reply: [0, 100000], retained: [0, 100]};
  const defaults = {operations: 12, group: 3, reply: 1200, retained: 20};
  const keys = ["tool_calls", "code_calls", "tool_input", "grouped_input", "selected_input"];
  function total(prefix, replies) {
    return prefix * replies.length + replies.reduce((sum, value, index) => sum + (replies.length - 1 - index) * value, 0);
  }
  function calculate(values) {
    for (const [name, [min, max]] of Object.entries(limits)) {
      if (!Number.isInteger(values[name]) || values[name] < min || values[name] > max) {
        const error = new RangeError("Invalid " + name); error.field = name; throw error;
      }
    }
    const raw = [];
    for (let left = values.operations; left > 0; left -= values.group) raw.push(Math.min(left, values.group) * values.reply);
    const selected = raw.map(value => Math.floor((value * values.retained + 99) / 100));
    return {primitive_events: values.operations, tool_calls: values.operations, code_calls: raw.length,
      tool_input: total(2000, Array(values.operations).fill(values.reply)),
      grouped_input: total(2400, raw), selected_input: total(2400, selected)};
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
    if (element.dataset.ready === "true") return;
    if (typeof Number.isInteger !== 'function' || typeof Object.fromEntries !== 'function') return;
    const controls = element.querySelector('[data-controls]');
    const form = element.querySelector('form');
    const error = element.querySelector('[data-error]');
    const status = element.querySelector('[data-status]');
    const staticNote = element.querySelector('[data-static]');
    const outputs = Object.fromEntries(keys.map(key => [key, element.querySelector('[data-value="' + key + '"]')]));
    if (!controls || !form || !error || !status || !staticNote || Object.values(outputs).some(value => !value)) return;
    const anchor = anchorBeforeEnhancement(element);
    function clearError() {
      error.hidden = true; error.textContent = "";
      for (const name of Object.keys(limits)) {
        const input = form.elements.namedItem(name);
        input.removeAttribute('aria-invalid'); input.setAttribute('aria-describedby', element.id + '-help');
      }
    }
    function clearResults() {
      for (const output of Object.values(outputs)) output.textContent = '—';
    }
    function update() {
      clearError();
      const values = {};
      for (const name of Object.keys(limits)) {
        const input = form.elements.namedItem(name);
        values[name] = input.value.trim() === '' ? NaN : Number(input.value);
      }
      try {
        const result = calculate(values);
        for (const key of keys) outputs[key].textContent = String(result[key]);
        element.dataset.result = 'valid';
        status.textContent = '计算完成：相同的 ' + values.operations + ' 个原语事件，逐事件反馈为 ' + result.tool_calls + ' 次模型调用，分组后为 ' + result.code_calls + ' 次。结果只描述构造输入记账。';
        return true;
      } catch (failure) {
        clearResults(); element.dataset.result = 'invalid';
        const input = form.elements.namedItem(failure.field);
        if (input) {
          input.setAttribute('aria-invalid', 'true');
          input.setAttribute('aria-describedby', element.id + '-help ' + error.id);
          error.textContent = input.labels[0].querySelector('[data-label]').textContent + '：请输入 ' + input.min + ' 至 ' + input.max + ' 的整数。';
        } else error.textContent = '当前配置无法计算，请检查输入。';
        error.hidden = false;
        status.textContent = '尚无有效结果；请修正输入后重新计算。';
        return false;
      }
    }
    // Browser-restored invalid inputs stay visible and can be corrected.
    update();
    form.addEventListener('submit', event => {event.preventDefault(); update();});
    form.addEventListener('input', () => {clearError(); clearResults(); element.dataset.result = 'pending'; status.textContent = '输入已更改，请计算当前配置。';});
    for (const button of element.querySelectorAll('[data-preset]')) button.addEventListener('click', () => {
      const values = button.dataset.preset === 'single' ? {...defaults, group: 1, retained: 100} : defaults;
      for (const [name, value] of Object.entries(values)) form.elements.namedItem(name).value = value;
      update();
    });
    controls.hidden = false; staticNote.hidden = true; element.dataset.ready = 'true';
    if (anchor?.target.isConnected) {
      const delta = anchor.target.getBoundingClientRect().top - anchor.top;
      if (Math.abs(delta) > .5) window.scrollBy({top: delta, behavior: 'instant'});
    }
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {calculate, total, limits, defaults};
  if (typeof document !== 'undefined') {
    const initialize = () => document.querySelectorAll('[data-pyrua-accounting]').forEach(mount);
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize, {once:true});
    else initialize();
  }
})();
