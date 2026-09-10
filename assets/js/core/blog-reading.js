// Native details works without JS; close the mobile TOC after choosing a chapter.
document.addEventListener("DOMContentLoaded", () => {
  const tables = document.querySelectorAll(".content table");
  const formulas = [...document.querySelectorAll('.content .katex-display, .content mjx-container[display="true"]')];
  const hints = new Map();
  const updateOverflow = () => {
    tables.forEach(table => {
      if (table.scrollWidth > table.clientWidth) table.setAttribute("tabindex", "0");
      else table.removeAttribute("tabindex");
    });
    formulas.forEach((formula, index) => {
      const overflowing = formula.clientWidth > 0 && formula.scrollWidth > formula.clientWidth + 1;
      let hint = hints.get(formula);
      if (overflowing && !hint) {
        hint = document.createElement("span");
        hint.className = "blog-math-scroll-hint";
        hint.id = "blog-math-scroll-hint-" + index;
        hint.textContent = "左右滚动查看完整公式；键盘可用 Tab 聚焦后按方向键。";
        formula.before(hint);
        hints.set(formula, hint);
      }
      if (hint) hint.hidden = !overflowing;
      if (overflowing) {
        formula.setAttribute("tabindex", "0");
        formula.setAttribute("role", "region");
        formula.setAttribute("aria-label", "可横向滚动的公式");
        formula.setAttribute("aria-describedby", hint.id);
      } else {
        ["tabindex", "role", "aria-label", "aria-describedby"].forEach(name => formula.removeAttribute(name));
      }
    });
  };
  let resizeFrame;
  const scheduleUpdate = () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(updateOverflow);
  };
  updateOverflow();
  window.addEventListener("resize", scheduleUpdate);
  // Font loading and opening a collapsed section may change formula width.
  document.fonts?.ready.then(scheduleUpdate);
  document.addEventListener("toggle", scheduleUpdate, true);
  if (typeof ResizeObserver !== "undefined") {
    const observer = new ResizeObserver(scheduleUpdate);
    formulas.forEach(formula => observer.observe(formula));
  }
  const toc = document.querySelector(".blog-mobile-toc");
  if (!toc) return;
  toc.addEventListener("click", event => {
    const link = event.target.closest("a[href^='#']");
    if (!link) return;
    const anchor = document.getElementById(decodeURIComponent(link.hash.slice(1)));
    if (!anchor) return;
    const target = anchor.closest("h1, h2, h3, h4, h5, h6") || anchor;
    toc.open = false;
    requestAnimationFrame(() => {
      target.setAttribute("tabindex", "-1");
      target.focus({preventScroll: true});
      target.scrollIntoView({block: "start"});
    });
  });
});
