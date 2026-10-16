// Reuse the blog's saved appearance; CSS follows the system without a preference.
(function () {
  function apply() {
    let preference;
    try { preference = localStorage.getItem('color-theme'); } catch { /* Use CSS fallback. */ }
    if (['light', 'warm', 'dark'].includes(preference)) {
      document.documentElement.dataset.theme = preference;
    } else {
      delete document.documentElement.dataset.theme;
    }
  }
  apply();
  window.addEventListener('storage', event => {
    if (event.key === 'color-theme' || event.key === null) apply();
  });
})();
