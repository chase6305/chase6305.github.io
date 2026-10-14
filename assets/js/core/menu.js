// Hamburger menu for mobile navigation

document.addEventListener('DOMContentLoaded', function () {
  const menu = document.querySelector('.hextra-hamburger-menu');
  const sidebarContainer = document.querySelector('.hextra-sidebar-container');
  const mobileQuery = window.matchMedia('(max-width: 767px)');
  if (!menu || !sidebarContainer) return;
  const background = document.querySelectorAll('#content, .hextra-footer');

  function isMenuOpen() {
    return menu.querySelector('svg').classList.contains('open');
  }

  // On mobile, the sidebar is off-screen so hide it from assistive tech
  function syncAriaHidden() {
    const open = mobileQuery.matches && isMenuOpen();
    sidebarContainer.inert = mobileQuery.matches && !open;
    background.forEach(element => { element.inert = open; });
    if (mobileQuery.matches) {
      sidebarContainer.setAttribute('aria-hidden', isMenuOpen() ? 'false' : 'true');
    } else {
      sidebarContainer.removeAttribute('aria-hidden');
    }
  }

  // Set initial state
  syncAriaHidden();
  mobileQuery.addEventListener('change', () => {
    if (!mobileQuery.matches && isMenuOpen()) {
      const restoreNavFocus = sidebarContainer.contains(document.activeElement);
      toggleMenu({restoreFocusOnClose: false});
      if (restoreNavFocus) document.querySelector('.hextra-nav-container a')?.focus();
    }
    syncAriaHidden();
  });

  function toggleMenu(options = {}) {
    const { focusOnOpen = true, restoreFocusOnClose = true } = options;

    // Toggle the hamburger menu
    menu.querySelector('svg').classList.toggle('open');

    // When the menu is open, we want to show the navigation sidebar
    sidebarContainer.classList.toggle('hx:max-md:[transform:translate3d(0,-100%,0)]');
    sidebarContainer.classList.toggle('hx:max-md:[transform:translate3d(0,0,0)]');

    // When the menu is open, we want to prevent the body from scrolling
    document.body.classList.toggle('hx:overflow-hidden');
    document.body.classList.toggle('hx:md:overflow-auto');

    // Sync aria-expanded and aria-hidden
    const isOpen = isMenuOpen();
    menu.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    syncAriaHidden();

    // Move focus into sidebar when opening, restore when closing
    if (isOpen) {
      if (focusOnOpen) {
        const firstFocusable = sidebarContainer.querySelector('a, button, input, [tabindex="0"]');
        if (firstFocusable) firstFocusable.focus();
      }
    } else if (restoreFocusOnClose) {
      menu.focus();
    }
  }

  menu.addEventListener('click', (e) => {
    e.preventDefault();
    // Pointer-initiated clicks on mobile should not force focus into the search input,
    // which opens the software keyboard immediately.
    toggleMenu({ focusOnOpen: e.detail === 0 });
  });

  // Close menu on Escape key (mobile only)
  document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented) return;
    if (e.key !== 'Escape') return;
    if (document.getElementById('hextra-search-dialog')?.open) return;
    if (mobileQuery.matches && isMenuOpen()) {
      toggleMenu();
    }
  });

  // Keep Tab in the visible navigation while the mobile overlay is open.
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Tab' || e.defaultPrevented || !mobileQuery.matches || !isMenuOpen()) return;
    if (document.getElementById('hextra-search-dialog')?.open) return;
    const focusable = [...document.querySelectorAll('.hextra-nav-container a, .hextra-nav-container button, .hextra-sidebar-container a, .hextra-sidebar-container button')]
      .filter(element => element.tabIndex >= 0 && !element.disabled && element.getClientRects().length && getComputedStyle(element).visibility !== 'hidden');
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last?.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first?.focus();
    }
  });

  // Select all anchor tags in the sidebar container
  const sidebarLinks = sidebarContainer.querySelectorAll('a');

  // Add click event listener to each anchor tag
  sidebarLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      // Check if the href attribute contains a hash symbol (links to a heading)
      if (link.getAttribute('href') && link.getAttribute('href').startsWith('#')) {
        // Only dismiss overlay on mobile view
        if (window.innerWidth < 768) {
          toggleMenu();
        }
      }
    });
  });
});
