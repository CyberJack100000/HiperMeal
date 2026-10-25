/* Loaded in <head> so the saved theme is applied before the page paints (no flash). */
(function () {
  try {
    var t = localStorage.getItem('hm_theme');
    if (!t && window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches) t = 'dark';
    if (t) document.documentElement.dataset.theme = t;
  } catch (e) { /* storage blocked: stay on the default theme */ }
})();
