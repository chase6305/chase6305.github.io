/* Analytic single-frequency lag experiment; no robot data or network calls. */
(function () {
  "use strict";
  const constants = Object.freeze({mass: 2, damping: .4, amplitude: .03, rate: 100, trainSamples: 400, validationSamples: 300});
  const presets = Object.freeze({
    delayed: Object.freeze({delayMs: 20, trainHz: .7, validationHz: 1.1, correctTrainingDelay: false}),
    aligned: Object.freeze({delayMs: 20, trainHz: .7, validationHz: 1.1, correctTrainingDelay: true})
  });
  function sum(values) {
    let total = 0, correction = 0;
    for (const value of values) {
      const next = total + value;
      correction += Math.abs(total) >= Math.abs(value) ? total - next + value : value - next + total;
      total = next;
    }
    return total + correction;
  }
  function dot(left, right) { return sum(left.map((value, index) => value * right[index])); }
  function solveTwoColumns(first, second, target) {
    // Column scaling, then modified Gram-Schmidt with one reorthogonalization.
    const scale0 = Math.hypot(...first), scale1 = Math.hypot(...second);
    const q0 = first.map(value => value / scale0), x1 = second.map(value => value / scale1);
    let r01 = dot(q0, x1), remainder = x1.map((value, index) => value - r01 * q0[index]);
    const correction = dot(q0, remainder);
    r01 += correction;
    remainder = remainder.map((value, index) => value - correction * q0[index]);
    const r11 = Math.hypot(...remainder);
    if (![scale0, scale1, r11].every(Number.isFinite) || scale0 <= 0 || scale1 <= 0 || r11 <= 1e-12) {
      throw new RangeError("The two state columns need independent finite directions");
    }
    const q1 = remainder.map(value => value / r11), beta1 = dot(q1, target) / r11;
    return [(dot(q0, target) - r01 * beta1) / scale0, beta1 / scale1];
  }
  function state(time, frequency) {
    const omega = 2 * Math.PI * frequency;
    return {velocity: constants.amplitude * omega * Math.cos(omega * time),
      acceleration: -constants.amplitude * omega * omega * Math.sin(omega * time)};
  }
  function force(time, frequency, delay = 0) {
    const sample = state(time - delay, frequency);
    return constants.mass * sample.acceleration + constants.damping * sample.velocity;
  }
  function calculate(values) {
    for (const [key, low, high] of [["delayMs", -50, 50], ["trainHz", .1, 2], ["validationHz", .1, 2]]) {
      if (!Number.isFinite(values[key]) || values[key] < low || values[key] > high) throw new RangeError("Invalid " + key);
    }
    if (typeof values.correctTrainingDelay !== "boolean") throw new TypeError("Correction choice must be boolean");
    const delay = values.delayMs / 1000, first = [], second = [], target = [];
    for (let index = 0; index < constants.trainSamples; index++) {
      const time = index / constants.rate, sample = state(time, values.trainHz);
      first.push(sample.acceleration); second.push(sample.velocity);
      target.push(force(time + (values.correctTrainingDelay ? delay : 0), values.trainHz, delay));
    }
    const [mass, damping] = solveTwoColumns(first, second, target);
    const training = target.map((value, index) => value - mass * first[index] - damping * second[index]);
    const validation = [];
    for (let index = 0; index < constants.validationSamples; index++) {
      const time = index / constants.rate, sample = state(time, values.validationHz);
      validation.push({time, reference: force(time, values.validationHz),
        logged: force(time, values.validationHz, delay), prediction: mass * sample.acceleration + damping * sample.velocity});
    }
    const rms = values => Math.hypot(...values) / Math.sqrt(values.length);
    return {mass, damping, trainRmse: rms(training),
      validationReferenceRmse: rms(validation.map(row => row.prediction - row.reference)),
      validationLoggedRmse: rms(validation.map(row => row.prediction - row.logged)), validation};
  }
  const svgNS = "http://www.w3.org/2000/svg";
  function draw(svg, result) {
    const width = Math.max(180, svg.getBoundingClientRect().width);
    const font = parseFloat(getComputedStyle(svg).fontSize), height = Math.max(220, font * 14);
    const extent = Math.max(.001, ...result.validation.flatMap(row => [Math.abs(row.reference), Math.abs(row.logged), Math.abs(row.prediction)])) * 1.12;
    const labels = [-extent, 0, extent].map(value => Number(value.toPrecision(3)).toString());
    const measure = document.createElementNS(svgNS, "text");
    measure.setAttribute("visibility", "hidden"); measure.setAttribute("aria-hidden", "true"); svg.append(measure);
    const labelWidth = Math.max(...labels.map(label => { measure.textContent = label; return measure.getBBox().width; }));
    measure.remove();
    const left = Math.ceil(Math.max(48, labelWidth + 14)), right = 14, top = 18, bottom = Math.max(36, font * 2.6);
    const plotWidth = Math.max(45, width - left - right), plotHeight = height - top - bottom;
    const last = result.validation.at(-1).time;
    const x = time => left + time / last * plotWidth;
    const y = force => top + (extent - force) / (2 * extent) * plotHeight;
    const nodes = [];
    function add(tag, attributes, text) {
      const node = document.createElementNS(svgNS, tag);
      for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
      if (text !== undefined) node.textContent = text;
      nodes.push(node); return node;
    }
    add("title", {}, "独立验证轨迹的力：对齐参考、带延迟记录与辨识模型预测");
    for (const [index, value] of [-extent, 0, extent].entries()) {
      add("line", {x1: left, x2: left + plotWidth, y1: y(value), y2: y(value), class: "identification-timing__grid"});
      add("text", {x: left - 8, y: y(value), "text-anchor": "end", "dominant-baseline": "middle"}, labels[index]);
    }
    const ticks = plotWidth < font * 7.5 ? [0, last] : [0, last / 2, last];
    for (const time of ticks) {
      add("line", {x1: x(time), x2: x(time), y1: top, y2: top + plotHeight, class: "identification-timing__grid"});
      add("text", {x: x(time), y: top + plotHeight + font * 1.5,
        "text-anchor": time === 0 ? "start" : time === last ? "end" : "middle"}, Number(time.toFixed(2)).toString());
    }
    for (const name of ["reference", "logged", "prediction"]) {
      add("path", {d: result.validation.map((row, index) => (index ? "L" : "M") + x(row.time).toFixed(3) + " " + y(row[name]).toFixed(3)).join(" "),
        class: "identification-timing__curve identification-timing__curve--" + name});
    }
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    svg.style.height = height + "px";
    svg.replaceChildren(...nodes);
  }
  function mount(element) {
    if (element.dataset.ready) return;
    const form = element.querySelector("form"), error = element.querySelector("[data-error]");
    const chart = element.querySelector("[data-chart]"), svg = chart.querySelector("svg"), note = element.querySelector("[data-note]");
    let lastResult = null, lastWidth = 0, lastFont = 0, lastSpacing = null;
    function number(value, residual) { return residual && Math.abs(value) < 1e-10 ? "< 10⁻¹⁰" : value.toFixed(residual ? 6 : 5); }
    function update() {
      const values = {correctTrainingDelay: form.elements.correctTrainingDelay.checked};
      let firstInvalid = null;
      for (const input of form.querySelectorAll('input[type="number"]')) {
        values[input.name] = input.value === "" ? NaN : Number(input.value);
        const bad = !input.checkValidity() || !Number.isFinite(values[input.name]);
        input.setAttribute("aria-invalid", String(bad));
        if (bad && !firstInvalid) firstInvalid = input;
      }
      let result;
      try { if (!firstInvalid) result = calculate(values); } catch (_) { /* Keep a usable static explanation on setup failure. */ }
      const valid = Boolean(result);
      error.hidden = valid;
      error.textContent = valid ? "" : firstInvalid
        ? firstInvalid.labels[0].textContent.trim() + "：" + (firstInvalid.validationMessage || "请填写范围内的有限数值。")
        : "无法计算；请检查延迟与频率范围。";
      for (const key of ["mass", "damping", "trainRmse", "validationLoggedRmse", "validationReferenceRmse"]) {
        element.querySelector('[data-value="' + key + '"]').textContent = valid ? number(result[key], key.endsWith("Rmse")) : "—";
      }
      chart.hidden = !valid;
      lastResult = result || null;
      if (valid) {
        draw(svg, result);
        const warning = result.mass <= 0 || result.damping < 0 ? "出现非正质量或负阻尼估计，不能把它直接作为物理参数。" : "";
        note.textContent = warning + (values.correctTrainingDelay
          ? "训练力已按已知延迟对齐。点线仍是未校正的验证记录；预测应与对齐参考比较。"
          : "训练残差很小，仍可有参数偏差。验证中的两种力通道都不参与拟合，应分别核对时序约定。");
      } else note.textContent = "修正输入后重新计算；不显示上一次曲线或结果。";
    }
    for (const button of element.querySelectorAll("[data-preset]")) button.addEventListener("click", () => {
      const preset = presets[button.dataset.preset];
      for (const [key, value] of Object.entries(preset)) {
        const input = form.elements.namedItem(key);
        if (input.type === "checkbox") input.checked = value; else input.value = value;
      }
      update();
    });
    form.addEventListener("submit", event => { event.preventDefault(); update(); });
    form.addEventListener("input", update);
    form.addEventListener("change", update);
    element.querySelector("[data-interactive]").hidden = false;
    update();
    if (!lastResult) { element.querySelector("[data-interactive]").hidden = true; return; }
    element.querySelector("[data-static-result]").hidden = true;
    element.dataset.ready = "true";
    const redraw = () => {
      if (!lastResult) return;
      const style = getComputedStyle(svg), width = svg.getBoundingClientRect().width, font = parseFloat(style.fontSize);
      const spacing = style.letterSpacing + ':' + style.wordSpacing;
      if (Math.abs(width - lastWidth) > .5 || Math.abs(font - lastFont) > .1 || spacing !== lastSpacing) {
        lastWidth = width; lastFont = font; lastSpacing = spacing; draw(svg, lastResult);
      }
    };
    if (typeof ResizeObserver === "function") new ResizeObserver(redraw).observe(svg);
    else window.addEventListener("resize", redraw);
    if (typeof MutationObserver === "function") new MutationObserver(redraw).observe(document.documentElement, {attributes: true, attributeFilter: ["style", "class"]});
    window.addEventListener("beforeprint", () => { if (lastResult) draw(svg, lastResult); });
    window.addEventListener("afterprint", redraw);
  }
  if (typeof module !== "undefined" && module.exports) module.exports = {calculate, presets, constants};
  if (typeof document !== "undefined") {
    const initialize = () => document.querySelectorAll("[data-identification-timing]").forEach(element => {
      try { mount(element); } catch (_) { element.querySelector("[data-interactive]").hidden = true; element.querySelector("[data-static-result]").hidden = false; }
    });
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, {once: true}); else initialize();
  }
})();
