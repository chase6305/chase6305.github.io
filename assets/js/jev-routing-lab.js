/* Fixed offline records: no model calls, no task-success estimator. */
(function () {
  "use strict";
  const groupNames = {validation: "合成验证组", heldout: "合成留出组", shifted: "合成分布变化组"};
  const selectedThreshold = .8;

  function validateRecords(records) {
    if (!Array.isArray(records) || !records.length || records.length > 10000) throw Error("Invalid records");
    const ids = new Set();
    for (const row of records) {
      if (!row || typeof row.id !== "string" || !row.id || ids.has(row.id) ||
          !Object.hasOwn(groupNames, row.group) || typeof row.acceptable !== "boolean" ||
          !Number.isFinite(row.routing_score) || row.routing_score < 0 || row.routing_score > 1) {
        throw Error("Invalid record");
      }
      ids.add(row.id);
    }
    if (Object.keys(groupNames).some(group => !records.some(row => row.group === group))) throw Error("Missing group");
  }

  function calculate(records, group, threshold, fallbackCost) {
    if (!Object.hasOwn(groupNames, group) || !Number.isFinite(threshold) || threshold < 0 || threshold > 1 ||
        !Number.isFinite(fallbackCost) || fallbackCost < 1 || fallbackCost > 40) throw Error("Invalid controls");
    const rows = records.filter(row => row.group === group);
    const accepted = rows.filter(row => row.routing_score >= threshold);
    const errors = accepted.filter(row => !row.acceptable).length;
    const fallback = rows.length - accepted.length;
    return {total: rows.length, accepted: accepted.length, errors, fallback,
      error: accepted.length ? errors / accepted.length : null,
      cost: 1 + fallback / rows.length * fallbackCost};
  }

  async function mount(root) {
    if (root.dataset.routingMounted) return;
    root.dataset.routingMounted = "true";
    const status = root.querySelector("[data-routing-status]");
    status.hidden = false;
    try {
      const response = await fetch(root.dataset.records);
      if (!response.ok) throw Error("Records unavailable");
      const payload = await response.json();
      const records = payload.records;
      validateRecords(records);
      const form = root.querySelector("form");
      const controls = form.elements;
      const metric = name => root.querySelector('[data-routing-metric="' + name + '"]');
      const update = () => {
        const group = controls.group.value;
        const threshold = Number(controls.threshold.value);
        const fallbackCost = Number(controls.cost.value);
        const result = calculate(records, group, threshold, fallbackCost);
        root.querySelector('[data-routing-control="threshold"]').textContent = threshold.toFixed(2);
        root.querySelector('[data-routing-control="cost"]').textContent = fallbackCost + " 单位";
        metric("acceptance").textContent = result.accepted + " / " + result.total;
        metric("error").textContent = result.error === null ? "未定义（无人通过）" :
          (100 * result.error).toFixed(1) + "%（" + result.errors + " / " + result.accepted + "）";
        metric("fallback").textContent = String(result.fallback);
        metric("cost").textContent = result.cost.toFixed(2) + " 单位";
        const difference = result.cost - fallbackCost;
        root.querySelector("[data-routing-comparison]").textContent =
          "直接交给上层：每条 " + fallbackCost + " 单位；当前路由" +
          (Math.abs(difference) < 1e-9 ? "与其费用相同。" :
            (difference > 0 ? "多花 " : "少花 ") + Math.abs(difference).toFixed(2) + " 单位 / 条。");
        root.querySelector("[data-routing-note]").textContent = groupNames[group] + "。" +
          (result.accepted === 0 ? "拒绝全部判断仍支付选择器费用；不能将错误率记为零。" :
            group === "shifted" ? "该组高分段包含较多错误：把阈值从 0.80 调到 0.90，检查错误率是否反而升高。" :
              "阈值 0.80 由合成验证组选出；留出组用来观察固定阈值的结果。") +
          (threshold !== selectedThreshold ? "当前是手动探索值，不是重新验证的部署阈值。" : "");
        // Numeric values also make browser/Python parity checks unambiguous.
        root.dataset.routingResult = JSON.stringify(result);
      };
      form.addEventListener("submit", event => event.preventDefault());
      form.addEventListener("input", update);
      form.addEventListener("change", update);
      root.querySelector("[data-routing-reset]").addEventListener("click", () => {
        controls.threshold.value = String(selectedThreshold);
        update();
      });
      update();
      root.querySelector("[data-routing-interactive]").hidden = false;
      status.hidden = true;
    } catch (error) {
      status.textContent = "交互记录暂时无法读取，请使用下方静态结果或下载脚本。";
    }
  }
  document.querySelectorAll("[data-jev-routing-lab]").forEach(mount);
})();
