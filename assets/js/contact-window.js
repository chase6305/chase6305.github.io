/* Integer-centimeter reachability with a fixed zero prefix and bounded steps. */
(function () {
  'use strict';

  const limits = {
    distance: [0, 100],
    step: [0, 100],
    intervals: [1, 12],
    pinned: [0, 12],
  };
  const fields = Object.keys(limits);
  const resultFields = ['late_path', 'late_gap', 'window_path', 'window_gap'];

  function validate(values) {
    for (const name of fields) {
      const [minimum, maximum] = limits[name];
      const value = values[name];
      if (!Number.isInteger(value) || value < minimum || value > maximum) {
        return {name, message: `请输入 ${minimum} 到 ${maximum} 的整数。`};
      }
    }
    if (values.pinned > values.intervals) {
      return {name: 'pinned', message: '固定间隔数最多等于到接触的总间隔数。'};
    }
    return null;
  }

  function calculate(values) {
    const invalid = validate(values);
    if (invalid) throw new RangeError(`${invalid.name}: ${invalid.message}`);
    const {distance, step, intervals, pinned} = values;
    const freeIntervals = intervals - pinned;
    const lateEnd = freeIntervals ? Math.min(distance, step) : 0;
    const windowEnd = Math.min(distance, freeIntervals * step);
    return {
      free_intervals: freeIntervals,
      late_gap: distance - lateEnd,
      window_gap: distance - windowEnd,
      late_path: [...Array(intervals).fill(0), lateEnd],
      window_path: Array.from({length: intervals + 1}, (_, index) =>
        Math.max(0, windowEnd - (intervals - index) * step)),
    };
  }

  function mount(card) {
    if (card.dataset.ready === 'true') return;
    const form = card.querySelector('form');
    const fallback = card.querySelector('[data-static]');
    const error = card.querySelector('[data-error]');
    const status = card.querySelector('[data-status]');
    if (!form || !fallback || !error || !status) return;

    const inputs = Object.fromEntries(fields.map(name => [name, form.elements.namedItem(name)]));
    const outputs = Object.fromEntries(resultFields.map(name => [name, card.querySelector(`[data-value="${name}"]`)]));
    if (Object.values(inputs).some(input => !input || !input.labels?.length) ||
        Object.values(outputs).some(output => !output)) return;

    const values = () => Object.fromEntries(fields.map(name => [name, inputs[name].valueAsNumber]));
    const defaults = Object.fromEntries(fields.map(name => [name,
      inputs[name].defaultValue === '' ? NaN : Number(inputs[name].defaultValue)]));
    if (validate(defaults)) return;

    function clearError() {
      for (const input of Object.values(inputs)) {
        input.removeAttribute('aria-invalid');
        input.removeAttribute('aria-describedby');
      }
      error.textContent = '';
    }

    function clearResults() {
      for (const output of Object.values(outputs)) output.textContent = '—';
    }

    function markPending() {
      clearError();
      clearResults();
      status.textContent = '参数已更改。按「计算」更新路径和间隙。';
    }

    form.addEventListener('input', markPending);

    form.addEventListener('submit', event => {
      event.preventDefault();
      clearError();
      const current = values();
      const invalid = validate(current);
      if (invalid) {
        clearResults();
        const input = inputs[invalid.name];
        input.setAttribute('aria-invalid', 'true');
        input.setAttribute('aria-describedby', error.id);
        error.textContent = `${input.labels[0].textContent.trim()}。${invalid.message}`;
        status.textContent = '';
        return;
      }
      const result = calculate(current);
      for (const name of resultFields) {
        const value = result[name];
        outputs[name].textContent = Array.isArray(value) ? value.join(' → ') : String(value);
      }
      status.textContent = `可调整 ${result.free_intervals} 个间隔。等到接触帧的末帧间隙为 ${result.late_gap} cm，窗口内调整为 ${result.window_gap} cm。`;
    });

    const restored = values();
    if (fields.some(name => restored[name] !== defaults[name])) markPending();
    form.hidden = false;
    fallback.hidden = true;
    card.dataset.ready = 'true';
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = {calculate};
  if (typeof document !== 'undefined') document.querySelectorAll('[data-contact-window]').forEach(mount);
})();
