// An immediately usable preview with a full-resolution upgrade in the background.
(function () {
  function initialize() {
    if (!window.HTMLDialogElement?.prototype.showModal) return;
    const eligible = [...document.querySelectorAll('[data-zoomable]')].filter(image => {
      const link = image.closest('a');
      return !link || link.classList.contains('post-image-link');
    });
    if (!eligible.length) return;

    const dialog = document.createElement('dialog');
    dialog.id = 'post-image-viewer';
    dialog.className = 'post-image-viewer';
    dialog.dataset.imageViewer = '';
    dialog.setAttribute('aria-labelledby', 'post-image-viewer-title');
    dialog.setAttribute('aria-modal', 'true');
    dialog.setAttribute('aria-describedby', 'post-image-viewer-status');
    const toolbar = document.createElement('div');
    toolbar.className = 'post-image-viewer__toolbar';
    const title = document.createElement('h2');
    title.id = 'post-image-viewer-title';
    const actions = document.createElement('div');
    actions.className = 'post-image-viewer__actions';
    const originalLink = document.createElement('a');
    originalLink.target = '_blank';
    originalLink.rel = 'noopener';
    originalLink.dataset.imageViewerOriginal = '';
    const closeButton = document.createElement('button');
    closeButton.type = 'button';
    closeButton.autofocus = true;
    closeButton.dataset.imageViewerClose = '';
    actions.append(originalLink, closeButton);
    toolbar.append(title, actions);
    const stage = document.createElement('div');
    stage.className = 'post-image-viewer__stage';
    stage.dataset.imageViewerStage = '';
    const status = document.createElement('p');
    status.id = 'post-image-viewer-status';
    status.className = 'post-image-viewer__status';
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');
    dialog.append(toolbar, stage, status);
    document.body.append(dialog);

    let current = null;
    let savedOverflow = '';
    let printFocus = null;
    const messagesFor = chinese => chinese ? {
      title: '图片查看', original: '打开原图', close: '关闭',
      loading: '正在加载清晰原图…', ready: '清晰原图已显示。',
      error: '原图暂时无法加载。可以关闭后重试，或在新窗口打开原图。'
    } : {
      title: 'Image viewer', original: 'Open original', close: 'Close',
      loading: 'Loading the full-resolution image…', ready: 'Full-resolution image displayed.',
      error: 'The original could not load. Close and retry, or open it in a new tab.'
    };
    const setState = (record, state) => {
      if (current !== record) return;
      dialog.dataset.state = state;
      stage.setAttribute('aria-busy', state === 'loading' ? 'true' : 'false');
      status.textContent = record.messages[state];
    };
    const release = record => {
      if (record.loader) {
        record.loader.src = '';
        record.loader = null;
      }
      stage.replaceChildren();
      status.textContent = '';
      stage.setAttribute('aria-busy', 'false');
      document.documentElement.style.overflow = savedOverflow;
      if (record.opener?.isConnected) record.opener.focus({preventScroll: true});
    };
    const close = () => {
      const record = current;
      current = null;
      if (dialog.open) dialog.close();
      if (record) release(record);
    };
    closeButton.addEventListener('click', close);
    dialog.addEventListener('cancel', event => {
      event.preventDefault();
      close();
    });
    dialog.addEventListener('keydown', event => {
      if (event.key !== 'Tab') return;
      if (event.shiftKey && document.activeElement === originalLink) {
        event.preventDefault();closeButton.focus();
      } else if (!event.shiftKey && document.activeElement === closeButton) {
        event.preventDefault();originalLink.focus();
      }
    });
    dialog.addEventListener('close', () => {
      // A queued close event can belong to a session that was already replaced.
      if (!dialog.open && current) close();
    });
    let backdropPressed = false;
    const outside = event => {
      const bounds = dialog.getBoundingClientRect();
      return event.clientX < bounds.left || event.clientX > bounds.right
        || event.clientY < bounds.top || event.clientY > bounds.bottom;
    };
    dialog.addEventListener('pointerdown', event => { backdropPressed = event.target === dialog && outside(event); });
    dialog.addEventListener('click', event => {
      if (backdropPressed && event.target === dialog && outside(event)) close();
      backdropPressed = false;
    });
    stage.addEventListener('click', event => { if (event.target.tagName === 'IMG') close(); });

    const open = (source, opener) => {
      printFocus = null;
      if (current) close();
      const language = source.closest('[lang]')?.getAttribute('lang') || document.documentElement.lang;
      const messages = messagesFor(language.toLowerCase().startsWith('zh'));
      const original = new URL(source.dataset.zoomSrc || opener?.href || source.src, document.baseURI).href;
      const preview = source.currentSrc || source.src;
      const image = new Image();
      image.className = 'post-image-viewer__image';
      image.alt = source.alt;
      image.decoding = 'async';
      image.fetchPriority = 'high';
      const width = Number(source.getAttribute('width')) || source.naturalWidth || 1024;
      const height = Number(source.getAttribute('height')) || source.naturalHeight || 768;
      dialog.style.setProperty('--image-viewer-width', width + 'px');
      dialog.style.setProperty('--image-viewer-height', height + 'px');
      title.textContent = messages.title;
      originalLink.textContent = messages.original;
      originalLink.href = original;
      closeButton.textContent = messages.close;
      dialog.lang = language;
      const record = {image, opener, messages, loader: null};
      current = record;
      image.addEventListener('load', () => {
        if (current === record) dialog.dataset.previewReady = 'true';
      });
      image.addEventListener('error', () => { if (!record.loader) setState(record, 'error'); });
      dialog.dataset.previewReady = 'false';
      image.src = preview;
      stage.replaceChildren(image);
      savedOverflow = document.documentElement.style.overflow;
      document.documentElement.style.overflow = 'hidden';
      setState(record, 'loading');
      dialog.showModal();
      closeButton.focus({preventScroll: true});

      if (new URL(preview, document.baseURI).href === original) {
        image.decode().then(() => setState(record, 'ready'), () => setState(record, 'error'));
        return;
      }
      const loader = new Image();
      loader.decoding = 'async';
      loader.fetchPriority = 'low';
      record.loader = loader;
      loader.src = original;
      loader.decode().then(() => {
        if (current !== record) return;
        record.loader = null;
        image.src = original;
        return image.decode().then(() => setState(record, 'ready'), () => setState(record, 'error'));
      }).catch(() => {
        if (current !== record) return;
        record.loader = null;
        setState(record, 'error');
      });
    };
    eligible.forEach(image => {
      const link = image.closest('.post-image-link');
      const trigger = link || image;
      if (link) {
        link.setAttribute('aria-haspopup', 'dialog');
        link.setAttribute('aria-controls', dialog.id);
      }
      trigger.addEventListener('click', event => {
        if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        open(image, trigger);
      });
    });
    // Print the article itself; a pending original cannot block the print flow.
    const printQuery = window.matchMedia('print');
    const preparePrint = () => {
      if (!current) return;
      const target = current.opener;
      close();
      // Another modal can prevent close() from restoring the article link.
      printFocus = { target, afterClose: document.activeElement };
    };
    const restorePrintFocus = () => {
      if (printQuery.matches || !printFocus) return;
      const record = printFocus;
      printFocus = null;
      const target = record.target;
      if (!dialog.open && document.hasFocus() && target?.isConnected
        && target.getClientRects().length && getComputedStyle(target).visibility !== 'hidden'
        && !target.closest('[inert]')
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
    const cancelPrintFocus = () => { printFocus = null; };
    document.addEventListener('pointerdown', cancelPrintFocus, true);
    document.addEventListener('keydown', cancelPrintFocus, true);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize, {once: true});
  else initialize();
})();
