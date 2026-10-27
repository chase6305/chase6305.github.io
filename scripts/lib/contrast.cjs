// Approximate painted colors on uniform CSS backgrounds, including alpha.
// Images, gradients, SVG diagrams and mathematical layout need visual review.
function contrastAudit(selector = 'body') {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 1;
  const context = canvas.getContext('2d', {willReadFrequently: true});
  const cache = new Map();
  const rgba = color => {
    if (!cache.has(color)) {
      context.clearRect(0, 0, 1, 1);
      context.fillStyle = color;
      context.fillRect(0, 0, 1, 1);
      cache.set(color, [...context.getImageData(0, 0, 1, 1).data].map(v => v / 255));
    }
    return [...cache.get(color)];
  };
  const over = (a, b) => [0, 1, 2].map(i => a[i] * a[3] + b[i] * (1 - a[3]));
  const luminance = c => c.map(x => x <= .04045 ? x / 12.92 : ((x + .055) / 1.055) ** 2.4)
    .reduce((sum, x, i) => sum + x * [.2126, .7152, .0722][i], 0);
  const backgrounds = new Map();
  const background = element => {
    if (!element) return [1, 1, 1];
    if (!backgrounds.has(element)) {
      backgrounds.set(element, over(rgba(getComputedStyle(element).backgroundColor), background(element.parentElement)));
    }
    return backgrounds.get(element);
  };
  const issues = [];
  let checked = 0;
  const root = document.querySelector(selector);
  if (!root) return {checked: 0, issues: []};
  root.querySelectorAll('*').forEach(element => {
    if (element.closest('script,style,svg,.katex,[aria-hidden=true],[inert],button:disabled,input:disabled')) return;
    const hasText = [...element.childNodes].some(n => n.nodeType === 3 && n.textContent.trim());
    const placeholder = !hasText && element.matches('input[placeholder],textarea[placeholder]') && !element.value;
    if (!hasText && !placeholder) return;
    if (element.closest('.sr-only,.hx\\:sr-only')) return;
    const style = getComputedStyle(element, placeholder ? '::placeholder' : null), rect = element.getBoundingClientRect();
    if (!rect.width || !rect.height || style.visibility !== 'visible') return;
    let opacity = 1;
    for (let parent = element; parent; parent = parent.parentElement) opacity *= Number(getComputedStyle(parent).opacity);
    if (placeholder) opacity *= Number(style.opacity);
    if (opacity < .05) return;
    const bg = background(element), fg = rgba(style.color);
    fg[3] *= opacity;
    const [light, dark] = [luminance(over(fg, bg)), luminance(bg)].sort((a, b) => b - a);
    const ratio = (light + .05) / (dark + .05);
    const size = parseFloat(style.fontSize), weight = Number(style.fontWeight);
    const minimum = size >= 24 || (size >= 18.666 && weight >= 700) ? 3 : 4.5;
    checked++;
    // The canvas round-trip uses 8-bit colors. Keep a small measurement tolerance.
    if (ratio < minimum - .05) issues.push({
      tag: element.tagName, class: typeof element.className === 'string' ? element.className : '',
      text: (placeholder ? element.getAttribute('placeholder') : element.textContent).trim().slice(0, 100),
      ...(placeholder ? {pseudo: '::placeholder'} : {}), ratio: Number(ratio.toFixed(3)), minimum,
      color: style.color, background: bg.map(x => Math.round(x * 255)),
    });
  });
  return {checked, issues};
}


module.exports = {contrastAudit};
