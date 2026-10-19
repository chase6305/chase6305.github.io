/* Constructed rectified cameras; no image, network or model inference. */
(function (root) {
  'use strict';
  const reference = Object.freeze({fx: 800, fy: 800, cx: 320, cy: 240,
    width: 640, height: 480, baseline_m: .12, depth_z_m: 2,
    left_uv: Object.freeze([400, 280]), right_uv: Object.freeze([352, 280])});
  const defaults = Object.freeze({baseline_m: .12, depth_z_m: 2, reverse_pose: false,
    crop_resize: false, stale_intrinsics: false, match_offset_px: 0});
  const presets = Object.freeze({
    correct: {...defaults},
    reversed: {...defaults, reverse_pose: true},
    updated: {...defaults, crop_resize: true},
    stale: {...defaults, crop_resize: true, stale_intrinsics: true},
    scaled: {...defaults, baseline_m: .24, depth_z_m: 4},
    baseline: {...defaults, baseline_m: .24},
    mismatch: {...defaults, match_offset_px: 100}
  });
  function calculate(patch = {}) {
    if (patch === null || typeof patch !== 'object' || Array.isArray(patch)) throw new Error('输入必须是参数对象。');
    const input = {...defaults, ...patch};
    for (const name of Object.keys(patch)) {
      if (!Object.prototype.hasOwnProperty.call(defaults, name)) throw new Error('未知输入字段：' + name);
    }
    for (const [name, low, high] of [['baseline_m', .01, .5],
      ['depth_z_m', .1, 10], ['match_offset_px', -200, 200]]) {
      const value = input[name];
      if (typeof value !== 'number' || !Number.isFinite(value) || value < low || value > high) {
        throw new Error({baseline_m:'重建基线须为 0.01～0.50 m。',
          depth_z_m:'预测深度须为 0.10～10 m。',
          match_offset_px:'匹配水平偏差须为 −200～200 原图像素。'}[name]);
      }
    }
    for (const name of ['reverse_pose', 'crop_resize', 'stale_intrinsics']) {
      if (typeof input[name] !== 'boolean') throw new Error('开关输入必须是布尔值。');
    }
    if (input.stale_intrinsics && !input.crop_resize) throw new Error('先启用裁剪缩放，再选择沿用旧内参。');
    const scale = input.crop_resize ? .5 : 1;
    const map = ([u, v]) => input.crop_resize ? [(u - 80) * .5, (v - 40) * .5] : [u, v];
    const left = map(reference.left_uv), observed = map(reference.right_uv);
    const matched = map([reference.right_uv[0] + input.match_offset_px, reference.right_uv[1]]);
    const fx = input.crop_resize && !input.stale_intrinsics ? 400 : 800;
    const sign = input.reverse_pose ? 1 : -1;
    // With identical pinhole intrinsics and parallel axes, backprojection
    // and reprojection reduce to u_R = u_L + fx * t_x / Z, v_R = v_L.
    const projected = [left[0] + sign * fx * input.baseline_m / input.depth_z_m, left[1]];
    const width = input.crop_resize ? 280 : 640, height = input.crop_resize ? 220 : 480;
    const inFrame = ([u, v]) => u >= 0 && u < width && v >= 0 && v < height;
    return {input, grid: {width, height, scale_from_original: scale},
      left_uv: left, observed_right_uv: observed, matched_right_uv: matched,
      projected_right_uv: projected,
      reprojection_error_px: Math.hypot(projected[0] - observed[0], projected[1] - observed[1]),
      match_reprojection_error_px: Math.hypot(projected[0] - matched[0], projected[1] - matched[1]),
      vertical_epipolar_error_px: Math.abs(matched[1] - left[1]),
      baseline_scale: input.baseline_m / reference.baseline_m,
      depth_scale: input.depth_z_m / reference.depth_z_m,
      depth_relative_error_percent: (input.depth_z_m / reference.depth_z_m - 1) * 100,
      projected_in_frame: inFrame(projected), matched_in_frame: inFrame(matched)};
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {reference, defaults, presets, calculate};
  if (!root.document) return;
  const doc = root.document, form = doc.getElementById('geometry-form');
  if (!form) return;
  const field = name => form.elements.namedItem(name);
  const format = n => Number(n.toPrecision(7)).toString();
  const point = uv => '(' + uv.map(format).join(', ') + ')';
  const svgNS = 'http://www.w3.org/2000/svg';
  let current = null;
  function shape(tag, attributes, text) {
    const element = doc.createElementNS(svgNS, tag);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function plot(result) {
    const svg = doc.getElementById('pixel-chart');
    const width = Math.max(240, Math.floor(svg.parentElement.clientWidth));
    const values = [result.observed_right_uv[0], result.matched_right_uv[0], result.projected_right_uv[0]];
    const low = Math.min(0, ...values) - 20, high = Math.max(result.grid.width, ...values) + 20;
    const x = u => 20 + (width - 40) * (u - low) / (high - low);
    svg.setAttribute('viewBox', `0 0 ${width} 212`);
    svg.replaceChildren(shape('title', {id:'plot-title'}, '真实观测、候选匹配与重建投影的横向像素位置'),
      shape('desc', {id:'plot-description'}, `当前图像宽 ${result.grid.width} 像素。真实观测 ${point(result.observed_right_uv)}；候选匹配 ${point(result.matched_right_uv)}；重建投影 ${point(result.projected_right_uv)}。三行只展开显示同一条水平极线上的位置，纵向间隔不表示图像坐标差。`));
    const labels = ['真实观测', '候选匹配', '重建投影'];
    values.forEach((value, index) => {
      const y = 37 + index * 54;
      svg.append(shape('line', {x1:x(0), x2:x(result.grid.width), y1:y, y2:y, class:'pixel-axis'}));
      svg.append(shape('line', {x1:x(result.observed_right_uv[0]), x2:x(result.observed_right_uv[0]),
        y1:y - 8, y2:y + 8, class:'reference-guide'}));
      svg.append(shape('text', {x:20, y:y - 16}, labels[index]));
      if (index === 0) svg.append(shape('circle', {cx:x(value), cy:y, r:6, class:'observed-point'}));
      else if (index === 1) svg.append(shape('rect', {x:x(value) - 6, y:y - 6, width:12, height:12, class:'matched-point'}));
      else svg.append(shape('path', {d:`M ${x(value)} ${y-8} l 8 8 l -8 8 l -8 -8 Z`, class:'projected-point'}));
    });
    for (const value of [0, result.grid.width]) svg.append(shape('text', {x:x(value), y:194, 'text-anchor':'middle'}, format(value)));
    svg.append(shape('text', {x:width/2, y:211, 'text-anchor':'middle'}, '横向像素 u（当前图像网格）'));
  }
  function render() {
    const status = doc.getElementById('status'), details = doc.getElementById('details');
    field('stale_intrinsics').disabled = !field('crop_resize').checked;
    if (!field('crop_resize').checked) field('stale_intrinsics').checked = false;
    try {
      const input = Object.fromEntries(Object.entries(defaults).map(([name, value]) =>
        [name, typeof value === 'boolean' ? field(name).checked : field(name).valueAsNumber]));
      const result = calculate(input);
      current = result; details.hidden = false;
      let message;
      if (!result.projected_in_frame) message = '重建投影在图像外；实际评估应剔除此点，下面仍展示公式计算值。';
      else if (result.reprojection_error_px > 1e-9) message = '当前重建投影与构造观测不一致。';
      else if (Math.abs(result.depth_scale - 1) > 1e-9 || Math.abs(result.baseline_scale - 1) > 1e-9) message = '重投影相容，但独立的米制尺度不一致。';
      else message = '当前重建投影与构造观测相容；这只检查了一个点。';
      if (status.textContent !== message) status.textContent = message;
      status.dataset.state = result.reprojection_error_px <= 1e-9 && Math.abs(result.depth_scale - 1) <= 1e-9 && Math.abs(result.baseline_scale - 1) <= 1e-9 ? 'consistent' : 'review';
      const values = {observed:point(result.observed_right_uv), matched:point(result.matched_right_uv), projected:point(result.projected_right_uv),
        reprojection:format(result.reprojection_error_px) + ' px', matchError:format(result.match_reprojection_error_px) + ' px',
        epipolar:format(result.vertical_epipolar_error_px) + ' px', baselineScale:format(result.baseline_scale) + ' 倍',
        depthScale:format(result.depth_scale) + ' 倍', depthError:format(result.depth_relative_error_percent) + '%',
        grid:result.grid.width + ' × ' + result.grid.height + ' px'};
      for (const [key, value] of Object.entries(values)) doc.getElementById('value-' + key).textContent = value;
      doc.getElementById('match-warning').hidden = result.input.match_offset_px === 0;
      doc.getElementById('bounds-warning').hidden = result.projected_in_frame && result.matched_in_frame;
      const download = doc.getElementById('download-case');
      download.href = 'data:application/json;charset=utf-8,' + encodeURIComponent(JSON.stringify({illustrative:true, reference, ...result}, null, 2));
      plot(result);
    } catch (error) {
      current = null; details.hidden = true;
      status.textContent = error.message; status.dataset.state = 'invalid';
      doc.getElementById('download-case').removeAttribute('href');
    }
  }
  doc.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => {
    for (const [name, value] of Object.entries(presets[button.dataset.preset])) {
      if (typeof value === 'boolean') field(name).checked = value;
      else field(name).value = value;
    }
    render();
  }));
  form.addEventListener('input', render);
  form.addEventListener('submit', event => event.preventDefault());
  root.addEventListener('resize', () => {if (current) plot(current);});
  doc.getElementById('interactive').hidden = false;
  render();
})(typeof globalThis !== 'undefined' ? globalThis : this);
