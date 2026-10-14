(function () {
  const defaultTheme = '{{ site.Params.theme.default | default `system`}}';
  const themes = ["light", "warm", "dark"];
  const toggles = document.querySelectorAll(".hextra-theme-toggle");
  const options = document.querySelectorAll(".hextra-theme-toggle-options button[role=menuitemradio]");
  let selectedTheme;

  function applyTheme(theme) {
    theme = themes.includes(theme) ? theme : "system";
    selectedTheme = theme;
    toggles.forEach((button) => button.parentElement.dataset.theme = theme);
    options.forEach((option) => option.setAttribute("aria-checked", option.dataset.item === theme ? "true" : "false"));
  }

  function switchTheme(theme, persist = true) {
    setTheme(theme);
    applyTheme(theme);
    if (persist) {
      try {
        localStorage.setItem("color-theme", selectedTheme);
      } catch {
        // The selection still applies to the current page.
      }
    }
  }

  switchTheme(readThemePreference(), false);

  function closeMenu(toggle, restoreFocus = false) {
    toggle.dataset.state = "closed";
    toggle.setAttribute("aria-expanded", "false");
    toggle.nextElementSibling.classList.add("hx:hidden");
    if (restoreFocus) toggle.focus();
  }

  function openMenu(toggle, focusLast = false) {
    toggles.forEach(other => { if (other !== toggle) closeMenu(other); });
    if (toggle.dataset.state !== "open") {
      toggle.dataset.state = "open";
      toggleMenu(toggle);
    }
    toggle.setAttribute("aria-expanded", "true");
    const items = toggle.nextElementSibling.querySelectorAll("button");
    (focusLast ? items[items.length - 1] : items[0])?.focus();
  }

  options.forEach((option) => option.addEventListener("click", (event) => {
    event.preventDefault();
    switchTheme(option.dataset.item);
    closeMenu(option.closest(".hextra-theme-toggle-options").previousElementSibling, true);
  }));

  toggles.forEach((toggle, index) => {
    const menu = toggle.nextElementSibling;
    menu.id = "theme-menu-" + index;
    toggle.setAttribute("aria-controls", menu.id);
    toggle.addEventListener("click", (event) => {
      event.preventDefault();
      if (toggle.dataset.state === "open") closeMenu(toggle);
      else openMenu(toggle);
    });
    toggle.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        openMenu(toggle, event.key === "ArrowUp");
      }
    });
    menu.addEventListener("focusout", (event) => {
      if (event.relatedTarget && !menu.contains(event.relatedTarget) && event.relatedTarget !== toggle) {
        closeMenu(toggle);
      }
    });
  });

  window.addEventListener("resize", () => toggles.forEach(resizeMenu));
  document.addEventListener("click", (event) => {
    if (event.target.closest(".hextra-theme-toggle") === null) {
      toggles.forEach(toggle => closeMenu(toggle));
    }
  });

  document.querySelectorAll('.hextra-theme-toggle-options[role="menu"]').forEach((menu) => {
    menu.addEventListener("keydown", (event) => {
      const items = Array.from(menu.querySelectorAll("button"));
      const index = items.indexOf(document.activeElement);
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        const offset = event.key === "ArrowDown" ? 1 : -1;
        items[(index + offset + items.length) % items.length].focus();
      } else if (event.key === "Home" || event.key === "End") {
        event.preventDefault();
        (event.key === "Home" ? items[0] : items[items.length - 1]).focus();
      } else if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        closeMenu(menu.previousElementSibling, true);
      }
    });
  });

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (selectedTheme === "system") setTheme("system");
  });

  window.addEventListener("storage", (event) => {
    if (event.key === "color-theme" || event.key === null) {
      switchTheme(event.newValue || defaultTheme, false);
    }
  });
})();
