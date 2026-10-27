// Native details works without JS; close the mobile TOC after choosing a chapter.
document.addEventListener("DOMContentLoaded", () => {
  const tables = document.querySelectorAll(".content table");
  const formulas = [...document.querySelectorAll('.content .katex-display, .content mjx-container[display="true"]')];
  const hints = new Map();
  const updateOverflow = () => {
    // Measure the whole page before inserting hints or changing attributes.
    // Interleaving these phases forces repeated layouts on formula-heavy pages.
    const tableMeasurements = [...tables].map(table => ({
      table, overflowing: table.scrollWidth > table.clientWidth
    }));
    const formulaMeasurements = formulas.map(formula => {
      const width = formula.clientWidth;
      return {formula, overflowing: width > 0 && formula.scrollWidth > width + 1};
    });
    tableMeasurements.forEach(({table, overflowing}) => {
      if (overflowing && table.tabIndex !== 0) table.setAttribute("tabindex", "0");
      else if (!overflowing && table.hasAttribute("tabindex")) table.removeAttribute("tabindex");
    });
    formulaMeasurements.forEach(({formula, overflowing}, index) => {
      let hint = hints.get(formula);
      if (overflowing && !hint) {
        hint = document.createElement("span");
        hint.className = "blog-math-scroll-hint";
        hint.id = "blog-math-scroll-hint-" + index;
        hint.textContent = "左右滚动查看完整公式；键盘可用 Tab 聚焦后按方向键。";
        formula.before(hint);
        hints.set(formula, hint);
      }
      if (hint && hint.hidden !== !overflowing) hint.hidden = !overflowing;
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
    [...formulas, ...tables].forEach(element => observer.observe(element));
  }

  // Include supplemental explanations and answers in PDF/paper output, then
  // restore exactly the sections the reader had left closed on screen.
  const details = [...document.querySelectorAll('.content details:not(.blog-mobile-toc)')];
  let printing = false;
  let closedBeforePrint = [];
  const preparePrint = () => {
    if (printing) return;
    printing = true;
    closedBeforePrint = details.filter(section => !section.open);
    closedBeforePrint.forEach(section => { section.open = true; });
  };
  const restorePrint = () => {
    if (!printing) return;
    closedBeforePrint.forEach(section => { section.open = false; });
    closedBeforePrint = [];
    printing = false;
  };
  window.addEventListener('beforeprint', preparePrint);
  window.addEventListener('afterprint', restorePrint);
  window.matchMedia('print').addEventListener('change', event => {
    if (event.matches) preparePrint();
    else restorePrint();
  });
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
