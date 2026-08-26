(function () {
  const defaultTheme = '{{ site.Params.theme.default | default `system`}}';
  const themes = ["light", "warm", "dark"];
  const toggles = document.querySelectorAll(".hextra-theme-toggle");
  const options = document.querySelectorAll(".hextra-theme-toggle-options button[role=menuitemradio]");

  function applyTheme(theme) {
    theme = themes.includes(theme) ? theme : "system";
    toggles.forEach((button) => button.parentElement.dataset.theme = theme);
    options.forEach((option) => option.setAttribute("aria-checked", option.dataset.item === theme ? "true" : "false"));
    localStorage.setItem("color-theme", theme);
  }

  function switchTheme(theme) {
    setTheme(theme);
    applyTheme(theme);
  }

  switchTheme("color-theme" in localStorage ? localStorage.getItem("color-theme") : defaultTheme);

  options.forEach((option) => option.addEventListener("click", (event) => {
    event.preventDefault();
    switchTheme(option.dataset.item);
  }));

  toggles.forEach((toggle) => toggle.addEventListener("click", (event) => {
    event.preventDefault();
    toggle.dataset.state = toggle.dataset.state === "open" ? "closed" : "open";
    toggleMenu(toggle);
    const open = toggle.dataset.state === "open";
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    if (open) toggle.nextElementSibling.querySelector("button")?.focus();
  }));

  window.addEventListener("resize", () => toggles.forEach(resizeMenu));
  document.addEventListener("click", (event) => {
    if (event.target.closest(".hextra-theme-toggle") === null) {
      toggles.forEach((toggle) => {
        toggle.dataset.state = "closed";
        toggle.setAttribute("aria-expanded", "false");
        toggle.nextElementSibling.classList.add("hx:hidden");
      });
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
      } else if (event.key === "Escape") {
        event.preventDefault();
        const toggle = menu.previousElementSibling;
        toggle.dataset.state = "closed";
        toggle.setAttribute("aria-expanded", "false");
        menu.classList.add("hx:hidden");
        toggle.focus();
      }
    });
  });

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (localStorage.getItem("color-theme") === "system") setTheme("system");
  });
})();
