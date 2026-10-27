// Keep sticky navigation and chapter offsets aligned when larger text wraps.
document.addEventListener('DOMContentLoaded', () => {
  const navigation = document.querySelector('.hextra-nav-container nav');
  if (!navigation) return;
  const root = document.documentElement;
  const updateHeight = () => {
    const height = Math.ceil(navigation.getBoundingClientRect().height);
    if (height > 0 && root.style.getPropertyValue('--hextra-navbar-height') !== height + 'px') {
      root.style.setProperty('--hextra-navbar-height', height + 'px');
    }
  };
  updateHeight();
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(updateHeight).observe(navigation);
  else window.addEventListener('resize', updateHeight);
  document.fonts?.ready.then(updateHeight);
});
