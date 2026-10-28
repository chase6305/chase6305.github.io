// Copyright (c) 2023 Xin; MIT. Hextra 38d18a5 search-dialog override.
// License: /licenses/hextra.txt
// Search command palette UI. Owns the <dialog>, keyboard shortcuts, result
// rendering, and animations. Reads data from `window.hextraSearch` (defined
// in flexsearch.js).

// {{ $noResultsFound := (T "noResultsFound") | default "No results found." }}

(function () {
  const resultsFoundTemplate = '{{ (T "resultsFound") | default "%d results found" }}';
  const noResultsText = '{{ $noResultsFound }}';
  const loadingText = '{{ T "searchLoading" }}';
  const unavailableText = '{{ T "searchUnavailable" }}';

  const EDITABLE_TAGS = ['INPUT', 'SELECT', 'TEXTAREA'];
  const isMac = /iPad|iPhone|Macintosh/.test(navigator.userAgent);

  let dialog;
  let input;
  let resultsEl;
  let emptyEl;
  let statusEl;
  let viewportEl;
  let innerEl;
  let closeAnimationListener = null;
  let closeTimer = null;
  let collapseTimer = null;
  let searchRequest = 0;
  let opener = null;
  let printFocus = null;

  document.addEventListener('DOMContentLoaded', init);

  function init() {
    dialog = document.getElementById('hextra-search-dialog');
    if (!dialog || !window.HTMLDialogElement?.prototype.showModal
      || typeof dialog.showModal !== 'function' || !window.hextraSearch) return;

    input = dialog.querySelector('.hextra-search-input');
    input.maxLength = 120;
    resultsEl = dialog.querySelector('.hextra-search-results');
    emptyEl = dialog.querySelector('[data-search-empty]');
    statusEl = dialog.querySelector('.hextra-search-status');
    viewportEl = dialog.querySelector('[data-search-viewport]');
    innerEl = dialog.querySelector('[data-search-inner]');

    document.querySelectorAll('[data-search-open]').forEach(trigger => {
      trigger.setAttribute('aria-haspopup', 'dialog');
      trigger.setAttribute('aria-controls', dialog.id);
      trigger.setAttribute('aria-keyshortcuts', 'Control+k Meta+k');
      const hint = trigger.querySelector('.hextra-search-trigger-kbd');
      if (hint) {
        hint.hidden = false;
        const label = trigger.querySelector('.hextra-search-trigger-label');
        trigger.setAttribute('aria-label', `${label.textContent} ${hint.textContent}`);
      }
    });
    document.querySelectorAll('[data-search-full-help]').forEach(help => { help.hidden = false; });

    if (isMac) {
      document.querySelectorAll('.hextra-search-trigger-kbd').forEach(el => {
        while (el.firstChild) el.removeChild(el.firstChild);
        const cmd = document.createElement('span');
        cmd.style.fontSize = '0.75rem';
        cmd.textContent = '⌘';
        el.appendChild(cmd);
        el.appendChild(document.createTextNode('K'));
        const trigger = el.closest('[data-search-open]');
        const label = trigger.querySelector('.hextra-search-trigger-label');
        trigger.setAttribute('aria-label', `${label.textContent} ${el.textContent}`);
      });
    }

    document.querySelectorAll('[data-search-open]').forEach(btn => {
      btn.addEventListener('click', event => {
        if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        openDialog();
      });
    });

    const closeBtn = dialog.querySelector('[data-search-close]');
    if (closeBtn) closeBtn.addEventListener('click', () => closeDialog());

    document.addEventListener('keydown', handleGlobalKeyDown);

    input.addEventListener('input', runSearch);
    dialog.addEventListener('keydown', handleDialogKeyDown);

    // Click on backdrop closes the dialog.
    dialog.addEventListener('click', (e) => {
      if (e.target === dialog) closeDialog();
    });

    // Keep the previous query and results when reopening — only flip aria state.
    dialog.addEventListener('close', () => {
      // A queued close can belong to an earlier, already reopened session.
      if (!dialog.open) input.setAttribute('aria-expanded', 'false');
    });

    // Intercept the native Escape "cancel" so we can play the close animation.
    dialog.addEventListener('cancel', (e) => {
      e.preventDefault();
      closeDialog();
    });

    // Printing must capture the page, with no modal or backdrop above it.
    // Skip dismissal animations: the print snapshot may happen immediately.
    const printQuery = window.matchMedia('print');
    const preparePrint = () => {
      if (!dialog.open) return;
      const target = opener;
      closeDialog({ immediate: true });
      printFocus = { target, afterClose: document.activeElement };
    };
    const restorePrintFocus = () => {
      if (printQuery.matches || !printFocus) return;
      const record = printFocus;
      printFocus = null;
      const target = record.target;
      if (!dialog.open && document.hasFocus() && target?.isConnected
        && target.getClientRects().length && getComputedStyle(target).visibility !== 'hidden'
        && (document.activeElement === record.afterClose || document.activeElement === document.body)) {
        target.focus({ preventScroll: true });
      }
    };
    window.addEventListener('beforeprint', preparePrint);
    window.addEventListener('afterprint', restorePrintFocus);
    printQuery.addEventListener('change', event => {
      if (event.matches) preparePrint();
      else restorePrintFocus();
    });
    // A hidden print-time navigation control cannot receive focus. Restore it
    // after screen styles return, unless the reader has already moved on.
    const cancelPrintFocus = () => { printFocus = null; };
    document.addEventListener('pointerdown', cancelPrintFocus, true);
    document.addEventListener('keydown', cancelPrintFocus, true);

    // Delegate result clicks so we can close the dialog on navigation.
    resultsEl.addEventListener('click', (e) => {
      const anchor = e.target.closest('a[role="option"]');
      if (anchor) closeDialog();
    });

    // Hover updates selection like a standard command palette.
    resultsEl.addEventListener('mousemove', (e) => {
      const anchor = e.target.closest('a[role="option"]');
      if (anchor) setActiveOption(anchor, { scroll: false });
    });

    // Result anchors are in the Tab sequence; keep aria-selected in sync with
    // the focused option so the highlighted row matches keyboard focus.
    resultsEl.addEventListener('focusin', (e) => {
      const anchor = e.target.closest('a[role="option"]');
      if (anchor) setActiveOption(anchor, { scroll: false });
    });

    // Dialog-level keyboard handling keeps input and result navigation in sync.
    // Enter on focused result anchors falls through to native anchor activation.
  }

  function isEditableElement(element) {
    const tag = element && element.tagName;
    return !!element && (EDITABLE_TAGS.includes(tag) || element.isContentEditable);
  }

  function setActiveDescendant(id) {
    if (id) input.setAttribute('aria-activedescendant', id);
    else input.removeAttribute('aria-activedescendant');
  }

  function handleGlobalKeyDown(e) {
    const inEditable = isEditableElement(document.activeElement);

    if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
      if (inEditable && !dialog.contains(document.activeElement)) return;
      e.preventDefault();
      // A dialog mid-dismiss is logically "closed" for toggle purposes — without
      // this, a rapid second Cmd+K is eaten by closeDialog's early-return.
      const isClosing = dialog.dataset.state === 'closing';
      if (dialog.open && !isClosing) closeDialog();
      else openDialog();
      return;
    }

    if (dialog.open) return;

    if (e.key === '/') {
      if (inEditable) return;
      e.preventDefault();
      openDialog();
    }
  }

  function openDialog() {
    // Mid-dismiss? Cancel the close animation and stay open — the in animation
    // re-plays so the dialog visibly snaps back instead of completing the fade.
    if (dialog.open && dialog.dataset.state === 'closing') {
      cancelClose();
      input.focus();
      input.select();
      return;
    }
    if (dialog.open) return;
    printFocus = null;
    opener = document.activeElement;
    dialog.showModal();
    input.setAttribute('aria-expanded', 'true');
    // Focus in this activation; a later frame can steal focus from a result.
    // Select the existing query so typing immediately replaces it.
    input.focus();
    input.select();
    if (window.hextraSearch) window.hextraSearch.preload().catch(() => {});
  }

  function cancelClose() {
    if (closeAnimationListener) {
      dialog.removeEventListener('animationend', closeAnimationListener);
      closeAnimationListener = null;
    }
    if (closeTimer !== null) {
      clearTimeout(closeTimer);
      closeTimer = null;
    }
    delete dialog.dataset.state;
  }

  function closeDialog({ immediate = false } = {}) {
    if (!dialog.open || (!immediate && dialog.dataset.state === 'closing')) return;
    cancelCollapse();

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (immediate || reducedMotion) {
      cancelClose();
      dialog.close();
      input.setAttribute('aria-expanded', 'false');
      return;
    }

    dialog.dataset.state = 'closing';
    const finalize = () => {
      if (closeTimer !== null) {
        clearTimeout(closeTimer);
        closeTimer = null;
      }
      if (closeAnimationListener) {
        dialog.removeEventListener('animationend', closeAnimationListener);
        closeAnimationListener = null;
      }
      if (dialog.dataset.state === 'closing') {
        delete dialog.dataset.state;
        dialog.close();
      }
    };
    // The backdrop animation is the longest — wait for it so neither layer
    // gets cut off when the dialog is removed from the top layer.
    closeAnimationListener = (e) => {
      if (e.animationName !== 'hextra-search-backdrop-out') return;
      finalize();
    };
    dialog.addEventListener('animationend', closeAnimationListener);
    // animationend on ::backdrop doesn't bubble reliably across browsers
    // (Firefox in particular); force-close shortly after the 300ms animation
    // so the dialog can't get stranded.
    closeTimer = setTimeout(finalize, 400);
  }

  function setViewportExpanded(expanded) {
    if (!viewportEl) return;
    viewportEl.dataset.expanded = expanded ? 'true' : 'false';
  }

  function cancelCollapse() {
    if (collapseTimer !== null) {
      clearTimeout(collapseTimer);
      collapseTimer = null;
    }
  }

  function clearResults() {
    while (resultsEl.firstChild) resultsEl.removeChild(resultsEl.firstChild);
  }

  function handleDialogKeyDown(e) {
    const anchor = e.target.closest('a[role="option"]');
    const fromInput = e.target === input;
    if (!fromInput && !anchor) return;

    // During IME composition the user is still selecting a candidate; Enter
    // commits the candidate (not the result), arrow keys cycle candidates.
    if (fromInput && (e.isComposing || e.keyCode === 229)) return;
    if (fromInput && !input.value.trim() && e.key !== 'Escape') return;

    const focusResults = !!anchor;
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        moveSelection(1, { current: anchor, focus: focusResults });
        break;
      case 'ArrowUp':
        e.preventDefault();
        moveSelection(-1, { current: anchor, focus: focusResults });
        break;
      case 'Home': {
        if (fromInput) return;
        const all = getOptions();
        if (!all.length) return;
        e.preventDefault();
        selectOption(all[0], { focus: focusResults });
        break;
      }
      case 'End': {
        if (fromInput) return;
        const all = getOptions();
        if (!all.length) return;
        e.preventDefault();
        selectOption(all[all.length - 1], { focus: focusResults });
        break;
      }
      case 'Enter': {
        if (!fromInput) return;
        const opt = getActiveOption();
        if (!opt) return;
        e.preventDefault();
        opt.click();
        break;
      }
      case 'Escape':
        // input[type=search] clears its value on Escape and swallows the event,
        // so close the dialog explicitly here.
        e.preventDefault();
        closeDialog();
        break;
    }
  }

  function getOptions() {
    return Array.from(resultsEl.querySelectorAll('a[role="option"]'));
  }

  function getActiveOption() {
    return resultsEl.querySelector('a[aria-selected="true"]');
  }

  function setActiveOption(el, opts) {
    const scroll = !opts || opts.scroll !== false;
    getOptions().forEach((o) => {
      o.setAttribute('aria-selected', o === el ? 'true' : 'false');
    });
    if (el) {
      setActiveDescendant(el.id);
      if (scroll) el.scrollIntoView({ block: 'nearest' });
    } else {
      setActiveDescendant();
    }
  }

  function focusOption(el) {
    setActiveOption(el, { scroll: false });
    el.focus({ preventScroll: true });
    el.scrollIntoView({ block: 'nearest' });
  }

  function selectOption(el, opts) {
    if (opts && opts.focus) focusOption(el);
    else setActiveOption(el);
  }

  function moveSelection(delta, opts) {
    const options = getOptions();
    if (!options.length) return;
    const current = (opts && opts.current) || getActiveOption();
    let i = options.indexOf(current);
    if (i === -1) {
      i = delta > 0 ? 0 : options.length - 1;
    } else {
      i = (i + delta + options.length) % options.length;
    }
    selectOption(options[i], opts);
  }

  async function runSearch(e) {
    if (e && e.isComposing) return;

    const query = input.value.trim();
    const request = ++searchRequest;
    if (!query) {
      setActiveOption(null, {scroll: false});
      resultsEl.inert = true;
      if (statusEl) statusEl.textContent = '';
      setViewportExpanded(false);
      // Defer DOM clear until the viewport finishes collapsing — otherwise the
      // results vanish first and the viewport snaps to the inner padding before
      // animating the last few pixels to 0.
      cancelCollapse();
      collapseTimer = setTimeout(() => {
        collapseTimer = null;
        clearResults();
        if (emptyEl) emptyEl.hidden = true;
      }, 250);
      return;
    }
    if (!window.hextraSearch) return;
    cancelCollapse();
    clearResults();
    resultsEl.inert = true;
    setActiveDescendant();
    if (emptyEl) {
      emptyEl.textContent = loadingText;
      emptyEl.hidden = false;
    }
    if (statusEl) statusEl.textContent = loadingText;
    setViewportExpanded(true);
    let results;
    try {
      results = await window.hextraSearch.search(query);
    } catch (err) {
      // Stale failure for an old query — ignore. Otherwise log and fall
      // through to an empty render so the user sees feedback instead of stale
      // results.
      if (request !== searchRequest || input.value.trim() !== query) return;
      console.warn('[hextra-search]', err);
      if (emptyEl) {
        emptyEl.textContent = unavailableText;
        emptyEl.hidden = false;
      }
      if (statusEl) statusEl.textContent = unavailableText;
      return;
    }
    // Drop stale results if the input changed during the await.
    if (request !== searchRequest || input.value.trim() !== query) return;
    renderResults(results, query);
  }

  function appendHighlightedText(container, text, query) {
    if (!text) return;
    if (!query) {
      container.textContent = text;
      return;
    }
    const escapedQuery = [...new Set(window.chaseSearchTerms.queryTerms(query))]
      .sort((a, b) => b.length - a.length)
      .map(term => term.replace(/[-\\^$*+?.()|[\]{}]/g, '\\$&'))
      .join('|');
    if (!escapedQuery) {
      container.textContent = text;
      return;
    }
    const regex = new RegExp(escapedQuery, 'giu');
    let lastIndex = 0;
    let match;
    while ((match = regex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        container.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
      }
      const span = document.createElement('span');
      span.className = 'hextra-search-match';
      span.textContent = match[0];
      container.appendChild(span);
      lastIndex = match.index + match[0].length;
    }
    if (lastIndex < text.length) {
      container.appendChild(document.createTextNode(text.slice(lastIndex)));
    }
  }

  function createBreadcrumbSeparator() {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.classList.add('hextra-search-crumb-separator');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'm9 18 6-6-6-6');
    svg.appendChild(path);
    return svg;
  }

  function appendBreadcrumb(container, parts) {
    for (let i = 0; i < parts.length; i++) {
      if (i > 0) container.appendChild(createBreadcrumbSeparator());

      const part = document.createElement('span');
      part.textContent = parts[i];
      container.appendChild(part);
    }
  }

  function renderResults(results, query) {
    cancelCollapse();
    clearResults();
    if (innerEl) innerEl.scrollTop = 0;

    if (!results.length) {
      if (emptyEl) {
        emptyEl.textContent = noResultsText;
        emptyEl.hidden = false;
      }
      if (statusEl) statusEl.textContent = noResultsText;
      setActiveDescendant();
      setViewportExpanded(true);
      return;
    }

    resultsEl.inert = false;
    if (emptyEl) emptyEl.hidden = true;

    const fragment = document.createDocumentFragment();
    let isFirstOption = true;
    let optionCount = 0;
    for (let i = 0; i < results.length; i++) {
      const result = results[i];

      const li = document.createElement('li');
      li.setAttribute('role', 'presentation');

      const link = document.createElement('a');
      link.id = result.id;
      link.href = result.route;
      link.setAttribute('role', 'option');
      link.setAttribute('aria-selected', isFirstOption ? 'true' : 'false');
      isFirstOption = false;
      optionCount++;

      if (result.breadcrumbs && result.breadcrumbs.length) {
        const crumb = document.createElement('div');
        crumb.className = 'hextra-search-crumb';
        appendBreadcrumb(crumb, result.breadcrumbs);
        link.appendChild(crumb);
      }

      const title = document.createElement('div');
      title.id = `${result.id}-title`;
      title.className = 'hextra-search-title';
      appendHighlightedText(title, result.title, query);
      link.appendChild(title);

      li.appendChild(link);
      fragment.appendChild(li);

      for (let j = 0; j < result.matches.length; j++) {
        const match = result.matches[j];
        const hasExcerpt = match.content && match.content !== match.title;
        if (match.title === result.title && !hasExcerpt) continue;

        const excerptLi = document.createElement('li');
        excerptLi.setAttribute('role', 'presentation');
        excerptLi.classList.add('hextra-search-child');

        const excerptLink = document.createElement('a');
        excerptLink.id = match.id;
        excerptLink.href = match.route;
        excerptLink.setAttribute('role', 'option');
        excerptLink.setAttribute('aria-describedby', title.id);
        excerptLink.setAttribute('aria-selected', isFirstOption ? 'true' : 'false');
        isFirstOption = false;
        optionCount++;

        if (match.title && match.title !== result.title) {
          const title = document.createElement('div');
          title.className = 'hextra-search-title';
          appendHighlightedText(title, match.title, query);
          excerptLink.appendChild(title);
        }

        if (hasExcerpt) {
          const excerpt = document.createElement('div');
          excerpt.className = 'hextra-search-excerpt';
          appendHighlightedText(excerpt, match.content, query);
          excerptLink.appendChild(excerpt);
        }

        excerptLi.appendChild(excerptLink);
        fragment.appendChild(excerptLi);
      }
    }
    resultsEl.appendChild(fragment);

    const firstOption = resultsEl.querySelector('a[role="option"]');
    setActiveDescendant(firstOption && firstOption.id);

    if (statusEl) {
      statusEl.textContent = resultsFoundTemplate.replace('%d', optionCount.toString());
    }

    setViewportExpanded(true);
  }
})();
