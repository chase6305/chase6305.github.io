// Copyright (c) 2023 Xin; MIT. Hextra 38d18a5 language menu override.
// License: /licenses/hextra.txt
(function () {
  const languageSwitchers = document.querySelectorAll('.hextra-language-switcher');
  const closeSwitcher = (switcher, focusSwitcher = false) => {
    switcher.dataset.state = 'closed';
    switcher.setAttribute('aria-expanded', 'false');
    const optionsElement = switcher.nextElementSibling;
    optionsElement.classList.add('hx:hidden');
    if (focusSwitcher) {
      switcher.focus();
    }
  };

  const openSwitcher = (switcher, focusTarget = "none") => {
    languageSwitchers.forEach(other => { if (other !== switcher) closeSwitcher(other); });
    switcher.dataset.state = 'open';
    switcher.setAttribute('aria-expanded', 'true');
    const optionsElement = switcher.nextElementSibling;
    if (optionsElement.classList.contains('hx:hidden')) {
      toggleMenu(switcher);
    } else {
      resizeMenu(switcher);
    }

    if (focusTarget !== "none") {
      const items = Array.from(optionsElement.querySelectorAll('[role="menuitem"]'));
      if (items.length > 0) {
        const target = focusTarget === "last" ? items[items.length - 1] : items[0];
        target.focus();
      }
    }
  };

  languageSwitchers.forEach((switcher) => {
    switcher.addEventListener('click', (e) => {
      e.preventDefault();

      if (switcher.dataset.state === 'open') {
        closeSwitcher(switcher);
      } else {
        openSwitcher(switcher, e.detail === 0 ? 'first' : 'none');
      }
    });

    switcher.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        openSwitcher(switcher, 'first');
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        openSwitcher(switcher, 'last');
      } else if (e.key === 'Escape' && switcher.dataset.state === 'open') {
        e.preventDefault();
        closeSwitcher(switcher, true);
      }
    });
  });

  document.querySelectorAll('.hextra-language-options[role=menu]').forEach((menu) => {
    menu.addEventListener('keydown', (e) => {
      const items = Array.from(menu.querySelectorAll('[role="menuitem"]'));
      if (items.length === 0) return;

      const currentIndex = items.indexOf(document.activeElement);
      let newIndex;

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          newIndex = (currentIndex + 1) % items.length;
          items[newIndex].focus();
          break;
        case 'ArrowUp':
          e.preventDefault();
          newIndex = (currentIndex - 1 + items.length) % items.length;
          items[newIndex].focus();
          break;
        case 'Home':
          e.preventDefault();
          items[0].focus();
          break;
        case 'End':
          e.preventDefault();
          items[items.length - 1].focus();
          break;
        case 'Escape': {
          e.preventDefault();
          const switcher = menu.previousElementSibling;
          if (switcher) {
            closeSwitcher(switcher, true);
          }
          break;
        }
      }
    });
  });

  window.addEventListener("resize", () => languageSwitchers.forEach(resizeMenu));

  // Dismiss this menu before an enclosing mobile navigation overlay. An open
  // dialog owns Escape until it closes; later focus outside the menu is retained.
  document.addEventListener('keydown', event => {
    if (event.defaultPrevented || event.key !== 'Escape' || document.querySelector('dialog[open]')) return;
    const open = [...languageSwitchers].filter(switcher => switcher.dataset.state === 'open');
    if (!open.length) return;
    event.preventDefault();
    open.forEach(switcher => closeSwitcher(switcher,
      switcher === document.activeElement || switcher.nextElementSibling.contains(document.activeElement)));
  });

  // Dismiss language switcher when clicking outside.
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.hextra-language-switcher') && !e.target.closest('.hextra-language-options')) {
      languageSwitchers.forEach((switcher) => {
        closeSwitcher(switcher);
      });
    }
  });
})();
