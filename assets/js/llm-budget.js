/* Pure payload arithmetic with progressive enhancement for the article. */
(function () {
  "use strict";
  const GiB = 2 ** 30;
  const presets = {
    example: {parameters: 7, weightBits: 16, layers: 32, kvHeads: 8, headDim: 128, kvBits: 16, batch: 1, tokens: 4096},
    qwen: {parameters: 7.615616512, weightBits: 16, layers: 28, kvHeads: 4, headDim: 128, kvBits: 16, batch: 1, tokens: 4096}
  };
  const bounds = {
    parameters: [0.000001, 10000], layers: [1, 1024], kvHeads: [1, 1024],
    headDim: [1, 8192], batch: [1, 4096], tokens: [1, 10000000]
  };
  function calculate(values) {
    for (const [key, [min, max]] of Object.entries(bounds)) {
      const value = values[key];
      if (!Number.isFinite(value) || value < min || value > max || (key !== "parameters" && !Number.isInteger(value))) {
        throw new RangeError("Invalid " + key);
      }
    }
    if (![4, 8, 16, 32].includes(values.weightBits) || ![8, 16, 32].includes(values.kvBits)) {
      throw new RangeError("Unsupported bit width");
    }
    const weights = values.parameters * 1e9 * values.weightBits / 8;
    const kv = 2 * values.layers * values.batch * values.tokens * values.kvHeads * values.headDim * values.kvBits / 8;
    return {weights, kv, total: weights + kv};
  }
  function mount(element) {
    if (element.dataset.ready) return;
    element.dataset.ready = "true";
    const form = element.querySelector("form");
    const error = element.querySelector(".llm-budget__error");
    const detail = element.querySelector("[data-detail]");
    const structure = element.querySelector("[data-structure-summary]");
    const format = new Intl.NumberFormat("zh-CN", {minimumFractionDigits: 2, maximumFractionDigits: 2});
    function update() {
      let valid = true;
      let firstInvalid = null;
      const values = {};
      for (const input of form.elements) {
        values[input.name] = input.value === "" ? NaN : Number(input.value);
        const bad = !input.checkValidity() || !Number.isFinite(values[input.name]);
        input.setAttribute("aria-invalid", String(bad));
        if (bad && !firstInvalid) firstInvalid = input;
        valid = valid && !bad;
      }
      let result;
      try { if (valid) result = calculate(values); } catch (_) { valid = false; }
      error.hidden = valid;
      const invalidLabel = firstInvalid?.labels?.[0]?.textContent.trim();
      error.textContent = valid ? "" : firstInvalid
        ? invalidLabel + "：" + (firstInvalid.validationMessage || "请选择有效数值。")
        : "请填写范围内的数值；层数、头数、维度、并发数与 token 数必须为整数。";
      for (const key of ["weights", "kv", "total"]) {
        element.querySelector('[data-value="' + key + '"]').textContent = valid ? format.format(result[key] / GiB) + " GiB" : "—";
      }
      element.querySelector('[data-bar="weights"]').style.width = valid ? result.weights / result.total * 100 + "%" : "0%";
      element.querySelector('[data-bar="kv"]').style.width = valid ? result.kv / result.total * 100 + "%" : "0%";
      detail.textContent = valid ? "权重 " + format.format(result.weights / 1e9) + " GB；KV " + format.format(result.kv / 2 ** 20) + " MiB。1 GiB = 2³⁰ 字节；合计按未舍入值计算。" : "修正输入后重新计算；不沿用上一次结果。";
      structure.textContent = valid ? values.layers + " 层 · " + values.kvHeads + " 个 KV 头 · 每头 " + values.headDim + " 维 · KV " + values.kvBits + "-bit" : "请检查展开后的结构字段";
    }
    form.addEventListener("submit", event => event.preventDefault());
    form.addEventListener("input", update);
    form.addEventListener("change", update);
    for (const button of element.querySelectorAll("[data-preset]")) {
      button.addEventListener("click", () => {
        for (const [key, value] of Object.entries(presets[button.dataset.preset])) form.elements.namedItem(key).value = value;
        update();
      });
    }
    element.querySelector(".llm-budget__interactive").hidden = false;
    update();
  }
  if (typeof module !== "undefined" && module.exports) module.exports = {calculate, presets};
  if (typeof document !== "undefined") {
    const initialize = () => document.querySelectorAll("[data-llm-budget]").forEach(mount);
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, {once: true});
    else initialize();
  }
})();
