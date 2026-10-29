// One data request at a time; retry initial loading without moving reader focus.
(function (root) {
  'use strict';
  function mount({element, status, retry, url, validate, activate, loadingMessage, failureMessage}) {
    let pending = false, ready = false, activationStarted = false, attempts = 0;
    const load = async () => {
      if (pending || ready || activationStarted) return;
      pending = true;
      element.dataset.loadState = 'loading';
      element.setAttribute('aria-busy', 'true');
      status.hidden = false;
      status.textContent = loadingMessage;
      retry.setAttribute('aria-disabled', 'true');
      let phase = 'data';
      const controller = typeof root.AbortController === 'function' ? new root.AbortController() : null;
      const timer = controller ? root.setTimeout(() => controller.abort(), 15000) : null;
      try {
        const options = attempts++ ? {cache: 'reload'} : {};
        if (controller) options.signal = controller.signal;
        const response = await root.fetch(url, options);
        if (!response.ok) throw Error('Lab data unavailable');
        const data = await response.json();
        validate(data);
        if (timer !== null) root.clearTimeout(timer);
        // Activation installs listeners once. A setup failure should retain the
        // static fallback rather than installing a second partial interface.
        phase = 'activate';
        activationStarted = true;
        activate(data);
        ready = true;
        element.dataset.loadState = 'ready';
        status.textContent = '交互实验已载入。';
        status.hidden = true;
        if (element.ownerDocument.activeElement === retry) {
          retry.textContent = '实验已载入';
          // Retain a focused entry until blur, including an inactive browser tab.
          // Tabbing then reaches the newly available settings naturally.
          retry.hidden = false;
        } else {
          retry.hidden = true;
        }
      } catch (_) {
        element.dataset.loadState = 'error';
        status.hidden = false;
        status.textContent = failureMessage;
        const focusedFailure = phase === 'activate' && element.ownerDocument.activeElement === retry;
        retry.hidden = phase !== 'data' && !focusedFailure;
        if (focusedFailure) retry.textContent = '交互暂不可用';
        retry.setAttribute('aria-disabled', phase === 'data' ? 'false' : 'true');
      } finally {
        if (timer !== null) root.clearTimeout(timer);
        pending = false;
        element.setAttribute('aria-busy', 'false');
      }
    };
    retry.addEventListener('click', event => { event.preventDefault(); load(); });
    retry.addEventListener('blur', () => { if (ready || activationStarted) retry.hidden = true; });
    return load();
  }
  root.ChaseLabDataLoader = {mount};
})(typeof globalThis !== 'undefined' ? globalThis : this);
