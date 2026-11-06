/* Centered LetterBox arithmetic only: no models, image uploads or network calls. */
(function () {
  "use strict";
  function roundEven(value) {
    const lower = Math.floor(value), fraction = value - lower;
    return fraction === .5 ? lower + (lower % 2 === 0 ? 0 : 1) : Math.round(value);
  }
  function calculate({width, height, target, minimal}) {
    for (const [name, value] of Object.entries({width, height})) {
      const label = name === "width" ? "原图宽度" : "原图高度";
      if (!Number.isInteger(value) || value < 1 || value > 8192) throw Error(`${label}必须是 1–8192 的整数。`);
    }
    if (!Number.isInteger(target) || target < 32 || target > 2048 || target % 32) {
      throw Error("目标边长必须是 32–2048 的整数，且是 32 的倍数。");
    }
    if (typeof minimal !== "boolean") throw Error("填充模式必须是布尔值。");
    const gain = Math.min(target / width, target / height);
    const resizedWidth = Math.max(1, roundEven(width * gain));
    const resizedHeight = Math.max(1, roundEven(height * gain));
    let horizontal = target - resizedWidth, vertical = target - resizedHeight;
    if (minimal) { horizontal %= 32; vertical %= 32; }
    const left = Math.floor(horizontal / 2), top = Math.floor(vertical / 2);
    const right = horizontal - left, bottom = vertical - top;
    const outputWidth = resizedWidth + horizontal, outputHeight = resizedHeight + vertical;
    const original = [width * .25, height * .25, width * .75, height * .75];
    const network = original.map((value, i) => value * gain + (i % 2 ? top : left));
    const restored = network.map((value, i) => (value - (i % 2 ? top : left)) / gain);
    const paddingPercent = 100 * (1 - resizedWidth * resizedHeight / (outputWidth * outputHeight));
    return {gain, resizedWidth, resizedHeight, outputWidth, outputHeight, left, top, right, bottom,
      original, network, restored, paddingPercent};
  }
  if (typeof module !== "undefined" && module.exports) module.exports = {calculate, roundEven};
  if (typeof document === "undefined") return;

  const namespace = "http://www.w3.org/2000/svg";
  function node(tag, attributes) {
    const element = document.createElementNS(namespace, tag);
    for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
    return element;
  }
  function draw(svg, values, id) {
    const {outputWidth: width, outputHeight: height, resizedWidth, resizedHeight, left, top, network} = values;
    const pattern = node("pattern", {id, width: 12, height: 12, patternUnits: "userSpaceOnUse"});
    pattern.append(node("path", {d: "M-3,3 L3,-3 M0,12 L12,0 M9,15 L15,9",
      class: "ultralytics-geometry__hatch", "stroke-width": 1, fill: "none"}));
    const defs = node("defs", {}); defs.append(pattern);
    const outer = node("rect", {x: 0, y: 0, width, height, class: "ultralytics-geometry__padding"});
    const hatch = node("rect", {x: 0, y: 0, width, height, fill: `url(#${id})`});
    const image = node("rect", {x: left, y: top, width: resizedWidth, height: resizedHeight,
      class: "ultralytics-geometry__image"});
    const box = node("rect", {x: network[0], y: network[1], width: network[2]-network[0],
      height: network[3]-network[1], class: "ultralytics-geometry__box", "stroke-dasharray": "7 4"});
    for (const shape of [outer, image, box]) {
      shape.setAttribute("stroke-width", "2");
      shape.setAttribute("vector-effect", "non-scaling-stroke");
    }
    svg.setAttribute("viewBox", `-4 -4 ${width+8} ${height+8}`);
    svg.replaceChildren(defs, outer, hatch, image, box);
  }
  function setup(root) {
    const interactive = root.querySelector("[data-interactive]");
    const form = root.querySelector("form"), computed = root.querySelector("[data-computed]");
    const svg = root.querySelector("svg"), error = root.querySelector("[data-error]");
    const fields = Object.fromEntries(["width", "height", "target", "minimal"].map(name => [name, form.elements.namedItem(name)]));
    const outputs = Object.fromEntries([...root.querySelectorAll("[data-value]")].map(element => [element.dataset.value, element]));
    const number = value => Number(value.toFixed(6)).toString();
    function update() {
      for (const name of ["width", "height", "target"]) {
        if (fields[name].validity.valid) fields[name].removeAttribute("aria-invalid");
        else fields[name].setAttribute("aria-invalid", "true");
      }
      try {
        const result = calculate({width: fields.width.valueAsNumber, height: fields.height.valueAsNumber,
          target: fields.target.valueAsNumber, minimal: fields.minimal.checked});
        draw(svg, result, root.id + "-hatch");
        const values = {gain: number(result.gain), resized: `${result.resizedWidth} × ${result.resizedHeight}`,
          size: `${result.outputWidth} × ${result.outputHeight}`,
          padding: [result.left, result.top, result.right, result.bottom].join(", "),
          original: result.original.map(number).join(", "), network: result.network.map(number).join(", "),
          restored: result.restored.map(number).join(", "), paddingPercent: number(result.paddingPercent) + "%"};
        for (const [key, value] of Object.entries(values)) outputs[key].value = value;
        computed.hidden = false; error.hidden = true; error.textContent = "";
        return true;
      } catch (failure) {
        for (const output of Object.values(outputs)) output.value = "";
        svg.replaceChildren(); computed.hidden = true;
        error.textContent = failure.message || "无法计算，请检查输入。"; error.hidden = false;
        return false;
      }
    }
    // Preserve the static default if SVG or the initial calculation cannot run.
    if (!svg || typeof svg.replaceChildren !== "function" || !update()) return;
    form.addEventListener("submit", event => event.preventDefault());
    form.addEventListener("input", update); form.addEventListener("change", update);
    for (const button of root.querySelectorAll("[data-preset]")) {
      button.addEventListener("click", () => {
        fields.width.value = "1280"; fields.height.value = "720"; fields.target.value = "640";
        fields.minimal.checked = button.dataset.preset === "minimal";
        update();
      });
    }
    root.querySelector("[data-static-result]").hidden = true;
    interactive.hidden = false;
  }
  for (const root of document.querySelectorAll("[data-ultralytics-geometry]")) {
    if (root.dataset.geometryReady) continue;
    root.dataset.geometryReady = "true";
    try { setup(root); } catch (_) { /* The published static example remains readable. */ }
  }
}());
