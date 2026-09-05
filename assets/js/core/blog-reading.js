// Native details works without JS; close the mobile TOC after choosing a chapter.
document.addEventListener("DOMContentLoaded", () => {
  const tables = document.querySelectorAll(".content table");
  const updateTables = () => {
    tables.forEach(table => {
      if (table.scrollWidth > table.clientWidth) table.setAttribute("tabindex", "0");
      else table.removeAttribute("tabindex");
    });
  };
  updateTables();
  let resizeFrame;
  window.addEventListener("resize", () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(updateTables);
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
