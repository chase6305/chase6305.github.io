/* Analytic two-mode transport, matched to the article's standard-library example. */
(function () {
  "use strict";
  const STEPS = 1000;
  const END = 0.02;
  const DELTA = (END - 1) / STEPS;
  function velocity(x, s) {
    if (!Number.isFinite(x) || !Number.isFinite(s) || s <= 0 || s > 1) {
      throw new RangeError("流时间须在 (0, 1] 内，当前位置须为有限数值。");
    }
    return (x - Math.tanh((1 - s) * x / (s * s))) / s;
  }
  function trajectory(noise) {
    if (!Number.isFinite(noise) || Math.abs(noise) > 2) {
      throw new RangeError("本算例的初始噪声须在 −2 到 2 之间。");
    }
    let x = noise;
    const points = [{s: 1, x}];
    for (let i = 0; i < STEPS; i++) {
      x += DELTA * velocity(x, 1 + i * DELTA);
      points.push({s: 1 + (i + 1) * DELTA, x});
    }
    return points;
  }
  function snapshot(points, progress) {
    if (!Number.isInteger(progress) || progress < 0 || progress > 100) {
      throw new RangeError("查看进度须为 0 到 100 的整数。");
    }
    const index = progress * STEPS / 100;
    const current = points[index];
    return {index, s: current.s, x: current.x, velocity: velocity(current.x, current.s),
      end: points[STEPS].x, next: index < STEPS ? points[index + 1].x : null};
  }
  function mount(root) {
    if (root.dataset.ready) return;
    const noise = root.querySelector('[name="flow-noise"]');
    const progress = root.querySelector('[name="flow-progress"]');
    const format = value => (Math.abs(value) < 0.0000005 ? 0 : value).toFixed(6);
    const xPixel = i => 46 + i / STEPS * 332;
    const yPixel = x => 106 - x / 2.2 * 84;
    const path = points => points.map((p, i) => (i ? "L" : "M") + xPixel(i).toFixed(3) + "," + yPixel(p.x).toFixed(3)).join(" ");
    let cachedNoise, points;
    function update() {
      const initial = Number(noise.value);
      if (initial !== cachedNoise) {
        points = trajectory(initial);
        cachedNoise = initial;
        root.querySelector('[data-flow-path="full"]').setAttribute("d", path(points));
        const square = root.querySelector("[data-flow-start]");
        square.setAttribute("x", xPixel(0) - 4);
        square.setAttribute("y", yPixel(initial) - 4);
      }
      const result = snapshot(points, Number(progress.value));
      root.querySelector("[data-flow-noise-label]").textContent = initial.toFixed(2);
      root.querySelector("[data-flow-progress-label]").textContent = progress.value + "%";
      noise.setAttribute("aria-valuetext", "初始噪声 " + initial.toFixed(2));
      progress.setAttribute("aria-valuetext", "查看进度 " + progress.value + "%，流时间 " + result.s.toFixed(3));
      for (const key of ["s", "x", "velocity", "end"]) {
        root.querySelector('[data-flow-value="' + key + '"]').textContent = format(result[key]);
      }
      root.querySelector('[data-flow-path="visited"]').setAttribute("d", path(points.slice(0, result.index + 1)));
      const dot = root.querySelector("[data-flow-current]");
      dot.setAttribute("cx", xPixel(result.index));
      dot.setAttribute("cy", yPixel(result.x));
      root.querySelector("[data-flow-step]").textContent = result.next === null ?
        "已完成 1,000 次 Euler 更新，停在 s = 0.02；没有继续走到离散分布的奇异终点。" :
        "已查看第 " + result.index + " 次更新。下一小步：" + format(result.x) + " + (−0.00098) × " +
        format(result.velocity) + " ≈ " + format(result.next) + "。";
      root.querySelector("[data-flow-note]").textContent = initial === 0 ?
        "恰好从 0 出发会停在对称中心。连续高斯恰好抽到这个单点的概率为零；手动选择它是为了检查对称性。" :
        "这里只改变初始噪声，目标分布与速度场均保持不变。曲线在生成空间中移动，不表示机器人正在走这条路线。";
      root.querySelector("[data-flow-chart-description]").textContent =
        "方块为初始噪声 " + initial.toFixed(2) + "，圆点为当前值 " + format(result.x) +
        "。完整曲线在流时间 0.02 停止，末端值 " + format(result.end) + "；目标模式位于负一和正一。";
      root.dataset.flowResult = JSON.stringify(result);
    }
    noise.addEventListener("input", update);
    progress.addEventListener("input", update);
    for (const button of root.querySelectorAll("[data-flow-preset]")) {
      button.addEventListener("click", () => {
        noise.value = button.dataset.flowPreset;
        progress.value = "100";
        update();
      });
    }
    update();
    root.querySelector("[data-flow-interactive]").hidden = false;
    root.querySelector("[data-flow-fallback]").hidden = true;
    root.dataset.ready = "true";
  }
  if (typeof module !== "undefined" && module.exports) module.exports = {velocity, trajectory, snapshot};
  if (typeof document !== "undefined") {
    const init = () => document.querySelectorAll("[data-vla-flow]").forEach(mount);
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
  }
}());
