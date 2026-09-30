// Follow the site's resolved theme, including manual and system selections.
// This overrides Hextra's system-only favicon handler.
(function () {
  const icons = document.querySelectorAll("link[data-favicon-light]");
  if (!icons.length) return;

  function updateFavicons() {
    const dark = document.documentElement.classList.contains("dark");
    icons.forEach((icon) => {
      const href = dark ? icon.dataset.faviconDark : icon.dataset.faviconLight;
      if (icon.getAttribute("href") !== href) icon.setAttribute("href", href);
    });
  }

  updateFavicons();
  new MutationObserver(updateFavicons).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class"],
  });
})();
