const toggle = document.querySelector('.nav-toggle');
const navigation = document.querySelector('.primary-nav');
const more = document.querySelector('.nav-more');
if (toggle && navigation && more) {
  toggle.hidden = false;
  toggle.addEventListener('click', () => {
    const expanded = toggle.getAttribute('aria-expanded') !== 'true';
    toggle.setAttribute('aria-expanded', String(expanded));
    if (!expanded) more.open = false;
  });
  document.addEventListener('click', (event) => {
    if (!more.contains(event.target)) more.open = false;
  });
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    if (more.open) {
      more.open = false;
      more.querySelector('summary').focus();
    } else if (toggle.getAttribute('aria-expanded') === 'true') {
      toggle.setAttribute('aria-expanded', 'false');
      toggle.focus();
    }
  });
  const mobile = window.matchMedia('(max-width: 640px)');
  mobile.addEventListener('change', () => {
    toggle.setAttribute('aria-expanded', 'false');
    more.open = false;
  });
}
