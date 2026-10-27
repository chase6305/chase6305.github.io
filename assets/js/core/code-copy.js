// Copy button for code blocks

document.addEventListener('DOMContentLoaded', function () {
  const getCopyIcon = () => {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.innerHTML = `
      <path stroke-linecap="round" stroke-linejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
    `;
    svg.setAttribute('fill', 'none');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    return svg;
  }

  const getSuccessIcon = () => {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.innerHTML = `
      <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
    `;
    svg.setAttribute('fill', 'none');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    return svg;
  }

  // Recheck after text zoom, font loading and opening collapsed examples.
  const codeBlocks = [...document.querySelectorAll('.hextra-code-block pre, .highlight pre')];
  const updateScrollableCodeBlocks = () => {
    const measurements = codeBlocks.map(pre => ({
      pre, overflowing: pre.clientWidth > 0 &&
        (pre.scrollWidth > pre.clientWidth || pre.scrollHeight > pre.clientHeight)
    }));
    measurements.forEach(({pre, overflowing}) => {
      if (overflowing && pre.tabIndex !== 0) pre.setAttribute('tabindex', '0');
      else if (!overflowing && pre.hasAttribute('tabindex') && document.activeElement !== pre) pre.removeAttribute('tabindex');
    });
  };
  let resizeRaf;
  const scheduleCodeCheck = () => {
    cancelAnimationFrame(resizeRaf);
    resizeRaf = requestAnimationFrame(updateScrollableCodeBlocks);
  };
  updateScrollableCodeBlocks();
  // Manual-copy fallback may focus a short block. Retain that focus until the
  // reader leaves; removing tabindex during the next layout would blur it.
  codeBlocks.forEach(pre => pre.addEventListener('blur', scheduleCodeCheck));
  window.addEventListener('resize', scheduleCodeCheck);
  document.addEventListener('toggle', scheduleCodeCheck, true);
  document.fonts?.ready.then(scheduleCodeCheck);
  if (typeof ResizeObserver !== 'undefined') {
    const observer = new ResizeObserver(scheduleCodeCheck);
    codeBlocks.forEach(pre => observer.observe(pre));
    // User text spacing can change inner overflow while a capped pre keeps its size.
    const content = document.querySelector('.content');
    if (content) observer.observe(content);
  }

  document.querySelectorAll('.hextra-code-copy-btn').forEach(function (button) {
    // Add copy and success icons
    button.querySelector('.hextra-copy-icon')?.appendChild(getCopyIcon());
    button.querySelector('.hextra-success-icon')?.appendChild(getSuccessIcon());

    const chinese = (button.closest('[lang]')?.lang || document.documentElement.lang).startsWith('zh');
    const originalLabel = chinese ? '复制代码' : button.getAttribute('aria-label') || 'Copy code';
    const copiedLabel = chinese ? '已复制' : button.dataset.copiedLabel || 'Copied!';
    button.setAttribute('aria-label', originalLabel);
    const status = document.createElement('div');
    status.className = 'blog-code-copy-status';
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');
    status.setAttribute('aria-atomic', 'true');
    button.closest('.hextra-code-block')?.before(status);
    let resetTimer;
    let copyPending = false;

    button.addEventListener('click', async function (e) {
      e.preventDefault();
      if (copyPending) return;
      clearTimeout(resetTimer);
      button.classList.remove('copied');
      button.setAttribute('aria-label', originalLabel);
      status.textContent = '';
      // Get the code target
      const target = button.parentElement.previousElementSibling;
      let codeElement;
      if (target?.tagName === 'CODE') {
        codeElement = target;
      } else if (target) {
        // Select the last code element in case line numbers are present
        const codeElements = target.querySelectorAll('code');
        codeElement = codeElements[codeElements.length - 1];
      }
      if (!codeElement) {
        status.textContent = chinese ? '未找到代码，请手动选择。' : 'Code not found. Please select it manually.';
        return;
      }
      // Keep the focused button in the keyboard order while permission or
      // clipboard work is pending. Native disabled can blur it on a later frame.
      const focusBeforeCopy = document.activeElement;
      const beforeSelection = window.getSelection();
      const selectionBeforeCopy = beforeSelection ? [beforeSelection.anchorNode,
        beforeSelection.anchorOffset, beforeSelection.focusNode, beforeSelection.focusOffset] : null;
      copyPending = true;
      button.setAttribute('aria-disabled', 'true');
      button.setAttribute('aria-busy', 'true');
      try {
        if (typeof navigator.clipboard?.writeText !== 'function') {
          throw new Error('Clipboard API unavailable');
        }
        // Preserve blank lines, indentation and multiline string contents exactly.
        await navigator.clipboard.writeText(codeElement.textContent);
        button.classList.add('copied');
        button.setAttribute('aria-label', copiedLabel);
        status.textContent = copiedLabel;
        resetTimer = setTimeout(() => {
          button.classList.remove('copied');
          button.setAttribute('aria-label', originalLabel);
          status.textContent = '';
        }, 1500);
      } catch {
        // Permission denial and non-secure origins still allow manual copying.
        const selection = window.getSelection();
        const pre = codeElement.closest('pre');
        const selectionUnchanged = selection && selectionBeforeCopy &&
          [selection.anchorNode, selection.anchorOffset, selection.focusNode, selection.focusOffset]
            .every((value, index) => value === selectionBeforeCopy[index]);
        const canSelect = selectionUnchanged && document.hasFocus() &&
          document.activeElement === focusBeforeCopy && codeElement.isConnected;
        if (canSelect) {
          if (pre) {
            pre.setAttribute('tabindex', '0');
            pre.focus({preventScroll: true});
          }
          const range = document.createRange();
          range.selectNodeContents(codeElement);
          selection.removeAllRanges();
          selection.addRange(range);
        }
        status.textContent = chinese
          ? canSelect ? '自动复制不可用，请手动复制已选中的代码。' : '自动复制不可用，请手动选择并复制代码。'
          : 'Automatic copying is unavailable. Please ' + (canSelect ? 'copy the selected' : 'select and copy the') + ' code manually.';
      } finally {
        copyPending = false;
        button.removeAttribute('aria-disabled');
        button.removeAttribute('aria-busy');
      }
    });
  });
});
