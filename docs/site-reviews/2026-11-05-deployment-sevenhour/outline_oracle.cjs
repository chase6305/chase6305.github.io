// Adapted from scripts/check_site_focus.cjs; repository-owned outline color oracle.
function inspectOutline(selector) {
  const element = document.querySelector(selector), style = getComputedStyle(element);
  const canvas = document.createElement('canvas');canvas.width = canvas.height = 1;
  const context = canvas.getContext('2d', {willReadFrequently: true});
  const rgba = color => {
    context.clearRect(0, 0, 1, 1);context.fillStyle = color;context.fillRect(0, 0, 1, 1);
    return [...context.getImageData(0, 0, 1, 1).data].map(value => value / 255);
  };
  const over = (a, b) => [0, 1, 2].map(index => a[index] * a[3] + b[index] * (1-a[3]));
  const gradients = [];
  const background = node => {
    if (!node) return [1, 1, 1];
    const css = getComputedStyle(node);
    if (css.backgroundImage !== 'none') gradients.push(css.backgroundImage);
    return over(rgba(css.backgroundColor), background(node.parentElement));
  };
  const bg = background(element.parentElement), foreground = over(rgba(style.outlineColor), bg);
  const luminance = color => color.map(value => value <= .04045 ? value / 12.92 : ((value + .055) / 1.055) ** 2.4)
    .reduce((total, value, index) => total + value * [.2126, .7152, .0722][index], 0);
  const values = [luminance(foreground), luminance(bg)].sort((a, b) => b-a);
  const bounds = element.getBoundingClientRect();
  return {focused: document.activeElement === element, visible: element.matches(':focus-visible'),
    outline: style.outlineColor, outlineStyle: style.outlineStyle, outlineWidth: parseFloat(style.outlineWidth),
    ratio: (values[0]+.05)/(values[1]+.05), background: bg, gradients,
    insideViewport: bounds.left >= -1 && bounds.right <= document.documentElement.clientWidth+1};
}
module.exports={inspectOutline};
