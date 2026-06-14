/**
 * SmartFarm AI — Global JavaScript
 * Navbar hamburger toggle + toast notifications.
 */

'use strict';

// ── Navbar hamburger ───────────────────────────────────────────────────────
(function () {
  const btn  = document.getElementById('hamburgerBtn');
  const menu = document.getElementById('mobileMenu');
  if (!btn || !menu) return;

  btn.addEventListener('click', () => {
    const open = menu.style.display === 'flex';
    menu.style.display = open ? 'none' : 'flex';
    menu.style.flexDirection = 'column';
    btn.textContent = open ? '☰' : '✕';
  });

  document.addEventListener('click', (e) => {
    if (!btn.contains(e.target) && !menu.contains(e.target)) {
      menu.style.display = 'none';
      btn.textContent = '☰';
    }
  });
})();

// ── Toast notifications ────────────────────────────────────────────────────
function showToast(message, type = 'success') {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;
  toast.textContent = (type === 'success' ? '✅ ' : '❌ ') + message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity .3s';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
