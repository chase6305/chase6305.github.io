// Apply the saved theme before paint to avoid a color flash.
function setTheme(theme) {
  document.documentElement.classList.remove("light", "warm", "dark");

  if (theme === "warm") {
    document.documentElement.classList.add("light", "warm");
    document.documentElement.style.colorScheme = "light";
    return;
  }

  if (theme !== "light" && theme !== "dark") {
    theme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  document.documentElement.classList.add(theme);
  document.documentElement.style.colorScheme = theme;
}

setTheme("color-theme" in localStorage ? localStorage.getItem("color-theme") : '{{ site.Params.theme.default | default `system`}}')
