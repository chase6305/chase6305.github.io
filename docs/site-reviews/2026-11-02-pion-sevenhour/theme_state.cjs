// Use the site's real theme control when scripts run; model the same complete
// root state for script-disabled CSS checks. The class alone is insufficient:
// native initial colors also follow the root color-scheme property.
function applyReviewTheme(theme, scripts = true) {
  if (!['light', 'dark', 'warm'].includes(theme)) throw Error('Unknown review theme');
  const root = document.documentElement;
  if (scripts) {
    const option = document.querySelector('[data-item="'+theme+'"]');
    if (!option) throw Error('Native theme option missing');
    option.click();
  } else {
    root.classList.remove('light', 'dark', 'warm');
    root.classList.add(theme === 'dark' ? 'dark' : 'light');
    if (theme === 'warm') root.classList.add('warm');
    root.style.colorScheme = theme === 'dark' ? 'dark' : 'light';
  }
  const expected = theme === 'dark' ? 'dark' : 'light';
  const scheme = getComputedStyle(root).colorScheme;
  const configuredScheme = root.style.colorScheme;
  const forcedColors = matchMedia('(forced-colors: active)').matches;
  // Forced colors may expose both supported schemes regardless of the selected theme.
  const computedMatches = scheme === expected || (forcedColors && scheme.split(/\s+/).includes(expected));
  if (!computedMatches || configuredScheme !== expected || root.classList.contains('dark') !== (theme === 'dark') ||
      root.classList.contains('warm') !== (theme === 'warm')) {
    throw Error('Review theme state is inconsistent: '+root.className+' / '+scheme);
  }
  return {theme, nativeControl: scripts, className: root.className, colorScheme: scheme, configuredColorScheme: configuredScheme, forcedColors};
}

module.exports = {applyReviewTheme};
