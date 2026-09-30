/* Independent action and sampling clocks; no inference-latency model. */
(function () {
  "use strict";
  const presets = {
    default: {horizon: 50, execute: 25, frequency: 50, sampling: 10},
    slower: {horizon: 50, execute: 16, frequency: 20, sampling: 10},
    short: {horizon: 50, execute: 5, frequency: 50, sampling: 10}
  };
  const labels = {horizon: "预测长度 H", execute: "执行长度 E", frequency: "控制频率", sampling: "内部采样 N"};
  function calculate(values) {
    for (const name of Object.keys(labels)) {
      const value = values[name];
      const upper = name === "frequency" ? 200 : 100;
      if (!Number.isFinite(value) || value < 1 || value > upper ||
          (name !== "frequency" && !Number.isInteger(value))) {
        throw new RangeError(labels[name] + "须为 1～" + upper + (name === "frequency" ? " 的数值。" : " 的整数。"));
      }
    }
    if (values.execute > values.horizon) throw new RangeError("执行长度 E 不能超过预测长度 H；请调整两者之一。");
    return {
      period: 1000 / values.frequency,
      coverage: values.horizon / values.frequency,
      prefix: values.execute / values.frequency,
      replans: values.frequency / values.execute,
      lastTarget: (values.horizon - 1) / values.frequency,
      sampling: values.sampling
    };
  }
  function mount(root) {
    if (root.dataset.ready) return;
    const form = root.querySelector("form");
    const error = root.querySelector("[data-clocks-error]");
    const outputs = root.querySelectorAll("[data-clocks-value]");
    const format = new Intl.NumberFormat("zh-CN", {maximumFractionDigits: 3});
    const formatValue = value => format.format(value);
    function strip(name, count, execute = 0) {
      const target = root.querySelector('[data-clocks-strip="' + name + '"]');
      const fragment = document.createDocumentFragment();
      for (let i = 0; i < count; i++) {
        const cell = document.createElement("span");
        if (i < execute) cell.dataset.execute = "true";
        fragment.append(cell);
      }
      target.replaceChildren(fragment);
    }
    function update() {
      const values = {};
      for (const name of Object.keys(labels)) {
        const input = form.elements.namedItem(name);
        values[name] = input.value.trim() === "" ? NaN : Number(input.value);
        input.setAttribute("aria-invalid", String(!input.checkValidity()));
      }
      try {
        const result = calculate(values);
        error.hidden = true;
        error.textContent = "";
        for (const output of outputs) {
          const key = output.dataset.clocksValue;
          output.textContent = formatValue(result[key]) + (key === "period" ? " ms" : key === "replans" ? " 次/秒" : " 秒");
        }
        strip("sampling", values.sampling);
        strip("actions", values.horizon, values.execute);
        root.querySelector("[data-clocks-generation]").textContent = "共 " + values.sampling + " 次更新，每次同时修改全部 " + values.horizon + " 行；这些格子的宽度不表示实际推理耗时。";
        root.querySelector("[data-clocks-execution]").textContent = "实色：执行第 0～" + (values.execute - 1) + " 行，共 " + formatValue(result.prefix) + " 秒。" +
          (values.execute < values.horizon ? "斜纹：剩余 " + (values.horizon - values.execute) + " 行暂不执行，重新观测后规划。" : "本轮执行整个动作块后重新观测。") +
          "整张预测表的最后一行（第 " + (values.horizon - 1) + " 行）位于起点后 " + formatValue(result.lastTarget) + " 秒。";
        root.querySelector("[data-clocks-diagrams]").hidden = false;
        root.dataset.clocksResult = JSON.stringify(result);
      } catch (exception) {
        if (values.execute > values.horizon) form.elements.namedItem("execute").setAttribute("aria-invalid", "true");
        error.textContent = exception.message;
        error.hidden = false;
        for (const output of outputs) output.textContent = "—";
        root.querySelector("[data-clocks-diagrams]").hidden = true;
        delete root.dataset.clocksResult;
      }
    }
    form.addEventListener("submit", event => event.preventDefault());
    form.addEventListener("input", update);
    form.addEventListener("change", update);
    for (const button of root.querySelectorAll("[data-clocks-preset]")) {
      button.addEventListener("click", () => {
        for (const [name, value] of Object.entries(presets[button.dataset.clocksPreset])) form.elements.namedItem(name).value = value;
        update();
      });
    }
    update();
    root.querySelector("[data-clocks-interactive]").hidden = false;
    root.querySelector("[data-clocks-fallback]").hidden = true;
    root.dataset.ready = "true";
  }
  if (typeof module !== "undefined" && module.exports) module.exports = {calculate, presets};
  if (typeof document !== "undefined") {
    const init = () => document.querySelectorAll("[data-vla-clocks]").forEach(mount);
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
  }
}());
