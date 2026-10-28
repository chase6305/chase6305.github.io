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
      else if (!overflowing && table.hasAttribute("tabindex") && document.activeElement !== table) table.removeAttribute("tabindex");
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
        formula.setAttribute("aria-label", `第 ${index + 1} 个可横向滚动的公式`);
        formula.setAttribute("aria-describedby", hint.id);
      } else {
        if (document.activeElement !== formula) formula.removeAttribute("tabindex");
        ["role", "aria-label", "aria-describedby"].forEach(name => formula.removeAttribute(name));
      }
    });
  };
  let resizeFrame;
  const scheduleUpdate = () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(updateOverflow);
  };
  // Reflow must not blur a region the reader is using. Drop its temporary
  // keyboard entry only after focus has moved elsewhere.
  [...tables, ...formulas].forEach(element => element.addEventListener("blur", scheduleUpdate));
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
  const printQuery = window.matchMedia('print');
  let printing = false;
  let closedBeforePrint = [];
  let printScroll = null;
  let printScrollFrame = 0;
  const cancelPrintScroll = () => {
    printScroll = null;
    cancelAnimationFrame(printScrollFrame);
    printScrollFrame = 0;
  };
  const preparePrint = () => {
    if (printing) return;
    printing = true;
    cancelAnimationFrame(printScrollFrame);
    // A native print layout can clamp mobile scroll offsets. Retain the screen
    // position before switching into the paper layout.
    const record = printScroll || {
      x: scrollX, y: scrollY, width: innerWidth, url: location.href,
      fontSize: getComputedStyle(document.documentElement).fontSize
    };
    record.focusArmed = false;
    printScroll = record;
    queueMicrotask(() => { if (printScroll === record) record.focusArmed = true; });
    closedBeforePrint = details.filter(section => !section.open);
    closedBeforePrint.forEach(section => { section.open = true; });
  };
  const restorePrint = () => {
    if (!printing || printQuery.matches) return;
    closedBeforePrint.forEach(section => { section.open = false; });
    closedBeforePrint = [];
    printing = false;
    const record = printScroll;
    if (!record) return;
    record.focusArmed = false;
    // Capture synchronous overlay restoration after all print listeners finish,
    // then cancel if a later task deliberately focuses another reading target.
    queueMicrotask(() => {
      if (printScroll !== record) return;
      record.focus = document.activeElement;
      record.focusArmed = true;
    });
    // Let screen CSS, disclosures and the other overlays finish restoring.
    printScrollFrame = requestAnimationFrame(() => {
      if (printScroll !== record) return;
      printScrollFrame = requestAnimationFrame(() => {
        if (printScroll !== record) return;
        printScroll = null;
        printScrollFrame = 0;
        if (innerWidth === record.width && location.href === record.url
          && getComputedStyle(document.documentElement).fontSize === record.fontSize
          && (document.activeElement === record.focus || document.activeElement === document.body)) {
          window.scrollTo({left: record.x, top: record.y, behavior: 'instant'});
        }
      });
    });
  };
  window.addEventListener('beforeprint', preparePrint);
  window.addEventListener('afterprint', restorePrint);
  printQuery.addEventListener('change', event => {
    if (event.matches) preparePrint();
    else restorePrint();
  });
  // A reader action after printing takes precedence over the saved position.
  for (const type of ['keydown', 'pointerdown', 'wheel', 'touchstart']) {
    document.addEventListener(type, cancelPrintScroll, {capture: true, passive: true});
  }
  document.addEventListener('focusin', event => {
    if (printScroll?.focusArmed && event.target !== document.body
      && (!printing || printQuery.matches)) cancelPrintScroll();
  });
  const toc = document.querySelector(".blog-mobile-toc");
  if (!toc) return;
  toc.addEventListener("click", event => {
    if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest("a[href^='#']");
    if (!link) return;
    const anchor = document.getElementById(decodeURIComponent(link.hash.slice(1)));
    if (!anchor) return;
    const target = anchor.closest("h1, h2, h3, h4, h5, h6") || anchor;
    event.preventDefault();
    toc.open = false;
    // Location.hash retains native fragment history and hashchange semantics.
    // Closing details, focusing and scrolling are synchronous: a later input
    // cannot be overridden by a queued chapter callback.
    window.location.hash = link.hash;
    target.setAttribute("tabindex", "-1");
    target.focus({preventScroll: true});
    target.scrollIntoView({block: "start"});
  });
});
