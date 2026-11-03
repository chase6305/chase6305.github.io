/* Conditional LR arithmetic and separate component invariants for one article. */
(function () {
  "use strict";
  const limits = {
    batchOld: [1, 1e8, true], batchNew: [1, 1e8, true],
    criticalBatch: [0, 1e12, false], lrOld: [1e-8, 1, false],
    decayOld: [0, 1000, false], betaOld: [0, 0.9999999, false]
  };
  const presets = {
    fourfold: {batchOld: 256, batchNew: 1024, criticalBatch: 4096, lrOld: 0.001, decayOld: 0.1, betaOld: 0.95},
    saturation: {batchOld: 4096, batchNew: 16384, criticalBatch: 256, lrOld: 0.001, decayOld: 0.1, betaOld: 0.95},
    smaller: {batchOld: 1024, batchNew: 256, criticalBatch: 4096, lrOld: 0.001, decayOld: 0.1, betaOld: 0.95}
  };
  const keys = ["batch_ratio", "lr_ratio", "lr_new", "updates_ratio", "beta_keep_data", "decay_keep_step", "decay_keep_data"];
  function calculate(values) {
    for (const [name, [min, max, integer]] of Object.entries(limits)) {
      if (!Number.isFinite(values[name]) || values[name] < min || values[name] > max || (integer && !Number.isInteger(values[name]))) {
        const error = new RangeError("Invalid " + name); error.fields = [name]; throw error;
      }
    }
    if (values.lrOld * values.decayOld >= 1) {
      const error = new RangeError("Require lrOld * decayOld < 1");
      error.fields = ["lrOld", "decayOld"]; throw error;
    }
    const k = values.batchNew / values.batchOld;
    const ratio = Math.sqrt((1 + values.criticalBatch / values.batchOld) / (1 + values.criticalBatch / values.batchNew));
    const lr = values.lrOld * ratio;
    return {
      batch_ratio: k, lr_ratio: ratio, lr_new: lr, updates_ratio: 1 / k,
      beta_keep_data: values.betaOld ** k,
      decay_keep_step: values.decayOld / ratio,
      decay_keep_data: Math.max(0, -Math.expm1(k * Math.log1p(-values.lrOld * values.decayOld)) / lr)
    };
  }
  function format(value) {
    const rounded = Number(value.toPrecision(8));
    // Do not present a valid momentum just below one as exactly one.
    if (value > 0 && value < 1 && rounded === 1) return value.toPrecision(16);
    return rounded.toString();
  }
  function readingAnchor(element) {
    // Capture immediately before enhancement, never an earlier reader position.
    const article = element.closest('.content');
    if (!article || element.getBoundingClientRect().top >= 0) return null;
    let fragment = null;
    try { fragment = document.getElementById(decodeURIComponent(location.hash.slice(1))); } catch (_) {}
    const candidates = [document.activeElement, fragment,
      ...element.querySelectorAll('.muon-batch__results > div'),
      ...article.querySelectorAll('h2,h3,p,pre,figure,table')];
    const target = candidates.find(node => {
      if (!node || node.closest?.('.content') !== article || node === element) return false;
      const result = node.matches?.('.muon-batch__results > div');
      if (element.contains(node) && !result) return false;
      if (!result && !(element.compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING)) return false;
      const bounds = node.getBoundingClientRect();
      return bounds.width > 0 && bounds.height > 0 && bounds.bottom > 64 && bounds.top < innerHeight;
    });
    return target ? {target, top: target.getBoundingClientRect().top} : null;
  }
  function mount(element) {
    if (element.dataset.ready === "true") return;
    if (typeof Math.expm1 !== "function" || typeof Math.log1p !== "function") return;
    const controls = element.querySelector("[data-controls]");
    const form = element.querySelector("form");
    const error = element.querySelector("[data-error]");
    const status = element.querySelector("[data-status]");
    const staticNote = element.querySelector("[data-static]");
    const outputs = Object.fromEntries(keys.map(key => [key, element.querySelector('[data-value="' + key + '"]')]));
    if (!controls || !form || !error || !status || !staticNote || Object.values(outputs).some(e => !e)) return;
    const anchor = readingAnchor(element);
    function clearInvalid() {
      for (const input of form.elements) if (input.name) {
        input.removeAttribute("aria-invalid");
        input.setAttribute("aria-describedby", element.id + "-help");
      }
      error.hidden = true; error.textContent = "";
    }
    function clearResults() {
      for (const output of Object.values(outputs)) output.textContent = "—";
      element.dataset.result = "pending";
    }
    function update() {
      clearInvalid();
      const values = {};
      for (const name of Object.keys(limits)) {
        const input = form.elements.namedItem(name);
        values[name] = input.value.trim() === "" ? NaN : Number(input.value);
      }
      try {
        const result = calculate(values);
        for (const key of keys) outputs[key].textContent = format(result[key]);
        element.dataset.result = "valid";
        status.textContent = "计算完成：局部学习率倍率为 " + format(result.lr_ratio) + "，同样本数下更新次数约为原来的 " + format(result.updates_ratio) + " 倍。";
        return true;
      } catch (failure) {
        clearResults(); element.dataset.result = "invalid";
        for (const name of failure.fields || []) {
          const input = form.elements.namedItem(name);
          input.setAttribute("aria-invalid", "true");
          input.setAttribute("aria-describedby", element.id + "-help " + error.id);
        }
        const field = failure.fields?.length === 1 ? form.elements.namedItem(failure.fields[0]) : null;
        error.textContent = field ? field.labels[0].childNodes[0].textContent.trim() + "：请输入 " + field.min + " 至 " + field.max + " 范围内的" + (limits[field.name][2] ? "整数。" : "有限数值。")
          : "初始学习率与衰减系数的乘积必须小于 1。";
        error.hidden = false;
        status.textContent = "尚未得到有效结果；请修正输入后重新计算。";
        return false;
      }
    }
    // Keep the server-rendered reference if setup or feature detection fails.
    if (!update()) return;
    form.addEventListener("submit", event => { event.preventDefault(); update(); });
    form.addEventListener("input", () => {
      clearInvalid(); clearResults();
      status.textContent = "输入已更改，请计算当前配置。";
    });
    for (const button of element.querySelectorAll("[data-preset]")) {
      button.addEventListener("click", () => {
        for (const [name, value] of Object.entries(presets[button.dataset.preset])) form.elements.namedItem(name).value = value;
        update();
      });
    }
    staticNote.hidden = true;
    controls.hidden = false;
    element.dataset.ready = "true";
    if (anchor?.target.isConnected) {
      const difference = anchor.target.getBoundingClientRect().top - anchor.top;
      if (Math.abs(difference) > 0.5) window.scrollBy({top: difference, behavior: 'instant'});
    }
  }
  if (typeof module !== "undefined" && module.exports) module.exports = {calculate, limits, presets, format};
  if (typeof document !== "undefined") {
    const initialize = () => document.querySelectorAll("[data-muon-batch]").forEach(mount);
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, {once: true});
    else initialize();
  }
})();
