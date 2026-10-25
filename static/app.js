/* HiperMeal front end. Plain JavaScript, no build step. Everything shown from the server or the AI is set with textContent, never as HTML. */
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const NS = 'http://www.w3.org/2000/svg';
const KEYS = ['kcal', 'protein_g', 'carbs_g', 'fat_g'];

function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === false || v == null) continue;
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'value') el.value = v;
    else if (k === 'checked' || k === 'disabled' || k === 'hidden') el[k] = !!v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid !== false && kid != null) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return el;
}
function svg(tag, attrs, ...kids) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  for (const kid of kids.flat()) if (kid != null) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return el;
}

function detailText(data, status) {
  const d = data && data.detail;
  return typeof d === 'string' ? d : Array.isArray(d) ? d.map((x) => x.msg).join(' ') : 'Request failed (' + status + ')';
}
async function api(path, options = {}) {
  let res;
  try { res = await fetch('/api' + path, options); } catch { throw new Error('Cannot reach the HiperMeal server. Is it still running?'); }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(detailText(data, res.status));
  return data;
}
/** POST a form with real upload progress (fetch cannot report it). Rejects with {cancelled:true} if aborted through the signal. */
function xhrPost(path, form, headers, { onUpload = () => {}, onSent = () => {}, signal } = {}) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open('POST', '/api' + path); x.timeout = 150000;
    for (const [k, v] of Object.entries(headers || {})) x.setRequestHeader(k, v);
    x.upload.onprogress = (e) => { if (e.lengthComputable) onUpload(e.loaded / e.total); };
    x.upload.onload = () => { onUpload(1); onSent(); };
    x.onload = () => {
      let data = {}; try { data = JSON.parse(x.responseText); } catch { /* not JSON */ }
      if (x.status >= 200 && x.status < 300) resolve(data); else reject(new Error(detailText(data, x.status)));
    };
    x.onerror = () => reject(new Error('Cannot reach the HiperMeal server. Is it still running?'));
    x.ontimeout = () => reject(new Error('The analysis took too long. Please try again.'));
    x.onabort = () => { const e = new Error('Cancelled.'); e.cancelled = true; reject(e); };
    if (signal) { if (signal.aborted) return x.abort(); signal.addEventListener('abort', () => x.abort(), { once: true }); }
    x.send(form);
  });
}
const post = (path, body, extra = {}) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json', ...extra }, body: JSON.stringify(body) });

/* The person's own Gemma API key. It lives only in this browser (this session, or remembered on this device if they choose), is sent only with the requests
   that need it, and the server passes it straight to Google without storing or logging it. */
const AIKey = {
  read() {
    for (const st of [sessionStorage, localStorage]) {
      try { const v = st.getItem('hm_ai'); if (v) { const o = JSON.parse(v); return { key: o.key || '', model: o.model || '', backup: o.backup || '', remember: st === localStorage }; } } catch { /* storage blocked or corrupt: treat as empty */ }
    }
    return { key: '', model: '', backup: '', remember: false };
  },
  save(key, model, remember, backup = '') {
    this.clear();
    try { (remember ? localStorage : sessionStorage).setItem('hm_ai', JSON.stringify({ key, model, backup })); return true; } catch { return false; }
  },
  clear() { for (const st of [sessionStorage, localStorage]) { try { st.removeItem('hm_ai'); } catch { /* ignore */ } } },
  headers(over) {
    const { key, model, backup } = over || this.read(); const hd = {};
    if (key) hd['X-Gemini-Key'] = key;
    if (model) hd['X-Gemma-Model'] = model;
    if (backup) hd['X-Gemma-Backup-Model'] = backup;
    return hd;
  },
  mask(k) { return k.length > 10 ? k.slice(0, 4) + '…' + k.slice(-4) : '••••'; },
  valid(k) { return /^[\x21-\x7e]{16,256}$/.test(k); },
};
const aiReady = () => !!AIKey.read().key || !!(S.health && S.health.ai_configured);
const insecureTransport = () => location.protocol === 'http:' && !['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
const IMAGE_EXTS = ['jpg', 'jpeg', 'jpe', 'png', 'webp', 'gif', 'bmp', 'tif', 'tiff', 'heic', 'heif', 'avif'];
const isImageFile = (f) => (f.type && f.type.startsWith('image/')) || IMAGE_EXTS.includes((f.name.split('.').pop() || '').toLowerCase());   // HEIC often has no type
const formatsText = () => ((S.health && S.health.image_formats) || ['JPG', 'PNG', 'WEBP']).join(', ');
const CUISINES = ['indian', 'chinese', 'italian', 'mexican', 'middle eastern', 'japanese', 'western'];
const getHint = () => { try { return localStorage.getItem('hm_hint') || ''; } catch { return ''; } };

let toastTimer;
function notify(msg) {
  const t = $('#toast'); t.textContent = msg; t.classList.remove('hidden');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add('hidden'), 3800);
}

// ---------------------------------------------------------------------------------------------------- helpers
const pad = (n) => String(n).padStart(2, '0');
const localDate = (d = new Date()) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;   // the person's own day, not UTC
const parseDay = (s) => { const [y, m, d] = s.split('-').map(Number); return new Date(y, m - 1, d); };
const fmtDay = (s) => {
  const d = parseDay(s), t = localDate(), y = localDate(new Date(Date.now() - 86400000));
  return s === t ? 'Today' : s === y ? 'Yesterday' : d.toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'short' });
};
const r0 = (n) => Math.round(n || 0);
const macroLine = (t) => `${r0(t.kcal)} kcal · P ${r0(t.protein_g)} g · C ${r0(t.carbs_g)} g · F ${r0(t.fat_g)} g`;
const num = (v) => (v === '' || v == null ? NaN : Number(v));

const Theme = {
  get() { try { return localStorage.getItem('hm_theme'); } catch { return null; } },
  current() { return document.documentElement.dataset.theme || 'light'; },
  set(t) {
    try { t ? localStorage.setItem('hm_theme', t) : localStorage.removeItem('hm_theme'); } catch { /* storage blocked */ }
    if (t) document.documentElement.dataset.theme = t;
    else document.documentElement.dataset.theme = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    this.paint();
  },
  paint() { const b = $('#themeBtn'); const dark = this.current() === 'dark'; b.textContent = dark ? '☀ Light' : '☾ Dark'; b.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode'); },
};

// ---------------------------------------------------------------------------------------------------- state
const S = {
  meals: [], summary: null, targets: null, foodNames: [], health: null,
  draft: null,            // the photo being reviewed: {file, url, analysisId, aiEnabled, message, mealType, note, foods: [...]}
};

async function refreshData() {
  const [m, s] = await Promise.all([api('/meals?days=30&today=' + localDate()), api('/summary?days=7&today=' + localDate())]);
  S.meals = m.meals; S.summary = s; S.targets = s.targets;
}

// ---------------------------------------------------------------------------------------------------- draft helpers
/** The per-100-g values that will be used for a food: the table match, or the person's own numbers. null = not usable yet. */
function per100(f) {
  if (f.nutrition) return f.nutrition;
  const c = Object.fromEntries(KEYS.map((k) => [k, num(f.custom[k])]));
  if (KEYS.some((k) => !Number.isFinite(c[k]) || c[k] < 0)) return null;
  if (c.kcal > 900 || c.protein_g > 100 || c.carbs_g > 100 || c.fat_g > 100 || c.protein_g + c.carbs_g + c.fat_g > 100.5) return null;
  return c;
}
function draftTotals() {
  const t = { kcal: 0, protein_g: 0, carbs_g: 0, fat_g: 0 }; let missing = 0;
  for (const f of S.draft.foods) {
    const n = per100(f), g = num(f.grams);
    if (!n || !(g > 0)) { missing++; continue; }
    for (const k of KEYS) t[k] += n[k] * g / 100;
  }
  return { t, missing };
}
function newFood(f) {
  return { name: f.name, grams: f.grams ?? 100, confidence: f.confidence || '', nutrition: f.nutrition || null, matched_name: f.matched_name || null, match: f.match || null,
           preparation: f.preparation || '', reasoning: f.reasoning || '', aiNote: '', custom: { kcal: '', protein_g: '', carbs_g: '', fat_g: '' } };
}
async function lookup(food) {
  try {
    const d = await post('/nutrition/lookup', { food_name: food.name });
    Object.assign(food, { nutrition: d.nutrition, matched_name: d.matched_name, match: d.match, aiNote: '' });
  } catch { Object.assign(food, { nutrition: null, matched_name: null, match: null, aiNote: '' }); }
}

async function chooseFile(file) {
  if (!file) return;
  if (!isImageFile(file)) return notify(`That does not look like a photo. Use ${formatsText()}.`);
  if (file.size > 10 * 1024 * 1024) return notify('Image must be smaller than 10 MB.');
  clearDraft(false);
  S.draft = { file, url: URL.createObjectURL(file), analysisId: null, aiEnabled: false, message: '', mealType: '', note: '', foods: [], busy: false, error: '' };
  location.hash = '#/'; render();
  await analyse();
}
function clearDraft(redraw = true) {
  if (S.draft && S.draft.ctl) S.draft.ctl.abort();
  if (S.draft && S.draft.url && S.draft.url.startsWith('blob:')) URL.revokeObjectURL(S.draft.url);
  S.draft = null; if (redraw) render();
}
/** Bring a panel into view: on a phone the analysis and the review appear below the fold, and nothing would seem to happen. */
function showPanel(id) {
  const el = document.getElementById(id); if (!el) return;
  const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
  el.scrollIntoView({ behavior: calm ? 'auto' : 'smooth', block: 'start' });
}
let progressTimer = null;
function paintProgress() {
  const d = S.draft, bar = $('#progBar'), track = $('#progress'), note = $('#progNote');
  if (!d || !d.busy || !bar) return;
  const up = d.stage === 'upload';
  track.classList.toggle('indeterminate', !up);
  if (up) {
    bar.style.width = d.pct + '%'; track.setAttribute('aria-valuenow', d.pct); track.setAttribute('aria-valuetext', `Uploading photo, ${d.pct} percent`);
    note.textContent = `Uploading photo… ${d.pct}%`;
  } else {
    bar.style.width = ''; track.removeAttribute('aria-valuenow'); track.setAttribute('aria-valuetext', 'Analysing your photo');
    const sec = Math.floor((Date.now() - d.t0) / 1000);
    note.textContent = `${aiReady() ? 'Gemma 4 is looking at your meal' : 'Preparing demo suggestions'}… ${sec}s` + (sec >= 25 ? ' · still working; big photos and busy servers take longer' : ' · usually under a minute');
  }
  const stages = [$('#stg0'), $('#stg1')];
  stages[0].className = up ? 'active' : 'done'; stages[1].className = up ? '' : 'active';
  const sr = $('#progSr'); const say = up ? 'Uploading photo' : 'Analysing photo'; if (sr && sr.textContent !== say) sr.textContent = say;      // announced on stage change only, not every tick
}
async function analyse() {
  const d = S.draft;
  Object.assign(d, { busy: true, error: '', stage: 'upload', pct: 0, t0: Date.now(), ctl: new AbortController() });
  render(); showPanel('analysis'); clearInterval(progressTimer); progressTimer = setInterval(paintProgress, 250);
  try {
    const form = new FormData(); form.append('file', d.file);
    if (getHint()) form.append('hint', getHint());
    const r = await xhrPost('/analyze', form, AIKey.headers(), {
      signal: d.ctl.signal,
      onUpload: (f) => { d.pct = Math.round(f * 100); paintProgress(); },
      onSent: () => { d.stage = 'analyse'; d.t0 = Date.now(); paintProgress(); },
    });
    d.analysisId = r.analysis_id; d.aiEnabled = r.ai_enabled; d.message = r.message; d.foods = r.foods.map(newFood);
    d.fallbackUsed = !!r.fallback_used; d.modelUsed = r.model_used || '';
    if (d.url.startsWith('blob:')) URL.revokeObjectURL(d.url);
    d.url = `/api/photos/${r.analysis_id}`;                   // the server's cleaned JPEG: shows even for HEIC, which browsers cannot display
  } catch (e) { d.error = e.cancelled ? 'Analysis cancelled.' : e.message; }
  clearInterval(progressTimer); d.busy = false; d.ctl = null;
  if (S.draft === d) { render(); showPanel('analysis'); }
}
async function saveDraft() {
  const d = S.draft, { missing } = draftTotals();
  if (!d.foods.length) return notify('Add at least one food item.');
  if (missing) return notify('Every food needs a name, a portion, and nutrition values. Fix the items marked above.');
  try {
    await post('/meals', {
      analysis_id: d.analysisId, meal_type: d.mealType || null, note: d.note, local_date: localDate(),
      foods: d.foods.map((f) => ({ name: f.name.trim(), grams: Number(f.grams), ...(f.nutrition ? {} : { custom: Object.fromEntries(KEYS.map((k) => [k, Number(f.custom[k])])) }) })),
    });
    clearDraft(false); await refreshData(); notify('Meal saved.'); render();
  } catch (e) { notify('Could not save the meal: ' + e.message); }
}

// ---------------------------------------------------------------------------------------------------- shared pieces
function plate() {
  return svg('svg', { class: 'plate', viewBox: '0 0 200 200', role: 'img', 'aria-label': 'A plate of food' },
    svg('circle', { cx: 100, cy: 100, r: 96, fill: 'var(--soft)' }), svg('circle', { cx: 100, cy: 100, r: 76, fill: 'var(--card)', stroke: 'var(--line)', 'stroke-width': 3 }),
    svg('path', { d: 'M100 100 L100 40 A60 60 0 0 1 152 70 Z', fill: 'var(--orange)' }), svg('path', { d: 'M100 100 L152 70 A60 60 0 0 1 100 160 Z', fill: 'var(--green)' }),
    svg('path', { d: 'M100 100 L100 160 A60 60 0 0 1 48 70 Z', fill: 'var(--blue)' }), svg('path', { d: 'M100 100 L48 70 A60 60 0 0 1 100 40 Z', fill: 'var(--pink)' }),
    svg('circle', { cx: 100, cy: 100, r: 18, fill: 'var(--card)' }));
}
function thumb(m) {
  return m.image_url ? h('img', { class: 'thumb', src: m.image_url, alt: 'Photo of this meal', loading: 'lazy' }) : h('div', { class: 'thumb', 'aria-hidden': 'true' }, '🍽');
}
const mealTitle = (m) => m.foods.map((f) => f.name).join(', ');
const mealWhen = (m) => `${new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}${m.meal_type ? ' · ' + m.meal_type : ''}`;

function summaryPanel() {
  const t = S.summary.today, g = S.targets;
  const cp = t.protein_g * 4, cc = t.carbs_g * 4, cf = t.fat_g * 9, tot = cp + cc + cf;
  const a = tot ? (cp / tot) * 100 : 0, b = tot ? a + (cc / tot) * 100 : 0;
  const ring = h('div', { class: 'ring', style: tot ? `background:conic-gradient(var(--blue) 0 ${a}%,var(--orange) ${a}% ${b}%,var(--pink) ${b}% 100%)` : '', role: 'img', 'aria-label': `${r0(t.kcal)} of ${r0(g.kcal)} kilocalories logged today` },
    h('div', { class: 'ring-inner' }, h('strong', {}, r0(t.kcal)), h('span', {}, `of ${r0(g.kcal)} kcal`)));
  const bar = (label, v, target, cls) => h('div', { class: 'bar-row' }, h('div', { class: 'bar-label' }, h('span', { class: 'dot ' + cls }), label, h('b', {}, `${r0(v)} / ${r0(target)} g`)),
    h('div', { class: 'track' }, h('div', { class: 'fill ' + cls, style: `width:${Math.min(100, (v / target) * 100 || 0)}%` })));
  return h('div', { class: 'panel' }, h('div', { class: 'summary-body' }, ring,
    h('div', { class: 'bars' }, bar('Protein', t.protein_g, g.protein_g, 'c-p'), bar('Carbs', t.carbs_g, g.carbs_g, 'c-c'), bar('Fat', t.fat_g, g.fat_g, 'c-f'))),
    h('p', { class: 'notice' }, 'Targets are yours to set in Settings. They are general figures, not personal dietary advice.'));
}

// ---------------------------------------------------------------------------------------------------- views
function homeView() {
  const t = S.summary.today;
  const hero = h('section', { class: 'hero' },
    h('div', {}, h('div', { class: 'eyebrow' }, 'Your visual nutrition diary'), h('h1', { tabindex: -1 }, 'Understand your meals with the ', h('span', {}, 'power of AI')),
      h('p', {}, 'Upload a photo of your meal, check the foods and portions the AI suggests, and get a rough nutrition estimate. Review, edit, and save to track your day.')),
    h('div', { class: 'hero-visual' }, plate(), h('div', { class: 'cal-card' }, h('div', { class: 'small' }, 'Logged today'), h('h2', {}, r0(t.kcal), ' ', h('small', {}, 'kcal')),
      ...[['Protein', t.protein_g, 'c-p'], ['Carbs', t.carbs_g, 'c-c'], ['Fat', t.fat_g, 'c-f']].map(([n, v, c]) => h('div', { class: 'macro-line' }, h('span', { class: 'dot ' + c }), n, h('b', {}, r0(v) + ' g'))))));

  const input = h('input', { type: 'file', id: 'fileInput', accept: 'image/*,.heic,.heif,.avif,.tif,.tiff', class: 'hidden', 'aria-label': 'Choose a meal photo', onchange: (e) => { chooseFile(e.target.files[0]); e.target.value = ''; } });
  const drop = h('section', { class: 'upload', id: 'dropzone', 'aria-label': 'Upload a meal photo' }, h('div', { class: 'cloud', 'aria-hidden': 'true' }, '⇧'), h('h2', {}, 'Upload a photo of your meal'),
    h('p', {}, 'Drag and drop an image here, or choose a file to browse.'), h('small', {}, `${formatsText()} · Max 10 MB · Location data is removed from your photo`), input,
    h('div', { class: 'actions' }, h('button', { class: 'btn', id: 'chooseBtn', type: 'button', onclick: () => input.click() }, '＋ Choose image')),
    h('div', { class: 'field hint' }, h('label', { for: 'hint' }, 'Cuisine hint (optional, helps the AI)'),
      h('select', { id: 'hint', onchange: (e) => { try { e.target.value ? localStorage.setItem('hm_hint', e.target.value) : localStorage.removeItem('hm_hint'); } catch { /* ignore */ } } },
        h('option', { value: '' }, 'Let the AI decide'), CUISINES.map((c) => h('option', { value: c, selected: getHint() === c }, c[0].toUpperCase() + c.slice(1))))));
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('drag'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
  drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('drag'); chooseFile(e.dataTransfer.files[0]); });

  const steps = h('section', {}, h('div', { class: 'section-head' }, h('h2', {}, 'How it works')),
    h('ol', { class: 'steps' }, ...[['Upload meal photo', 'Choose a clear photo of your meal.'], ['AI suggests foods', 'Likely foods and portions appear for review.'], ['Review & edit', 'Fix names and portions. Add your own values for foods we do not know.'], ['Save and track', 'Your day, week and favourites build up automatically.']]
      .map(([a, b], i) => h('li', { class: 'step' }, h('div', { class: 'num' }, i + 1), h('strong', {}, a), h('p', {}, b)))));

  const recent = h('div', { class: 'panel' }, S.meals.length ? S.meals.slice(0, 4).map((m) => h('div', { class: 'meal-row' }, thumb(m), h('div', { class: 'info' }, h('strong', {}, mealTitle(m)), h('small', {}, `${fmtDay(m.local_date)} · ${mealWhen(m)} · ${r0(m.totals.kcal)} kcal`))))
    : h('div', { class: 'empty' }, h('div', { class: 'bowl', 'aria-hidden': 'true' }, '🍽'), h('strong', {}, 'No meals logged yet'), h('p', {}, 'Upload your first meal photo to get started.'), h('button', { class: 'btn quiet', type: 'button', onclick: () => input.click() }, '＋ Upload a meal')));
  const bottom = h('section', { class: 'bottom-grid' },
    h('div', {}, h('div', { class: 'section-head' }, h('h2', {}, 'Recent meals'), h('a', { class: 'text-btn', href: '#/history' }, 'View all →')), recent),
    h('div', {}, h('div', { class: 'section-head' }, h('h2', {}, "Today's summary")), summaryPanel()));
  return [hero, drop, S.draft && reviewPanel(), steps, bottom];
}

function foodRow(f, i) {
  const name = h('input', { value: f.name, list: 'foodNames', 'aria-label': 'Food item name', maxlength: 100 });
  name.addEventListener('change', async () => { f.name = name.value.trim(); await lookup(f); render(); });
  const qty = h('input', { type: 'number', min: 1, max: 5000, value: f.grams, 'aria-label': 'Portion in grams', inputmode: 'decimal' });
  qty.addEventListener('input', () => { f.grams = qty.value; updateTotals(); });
  let status;
  if (f.nutrition) {
    const how = { exact: 'Matched', alias: `Matched as “${f.matched_name}”`, variant: `Matched as “${f.matched_name}”`, close: `Closest match: “${f.matched_name}”. Check that this is right.` }[f.match];
    status = h('div', { class: 'match' + (f.match === 'close' ? ' warn' : '') }, f.match === 'exact' ? `Matched “${f.matched_name}” · ${r0(f.nutrition.kcal)} kcal per 100 g` : `${how} · ${r0(f.nutrition.kcal)} kcal per 100 g`);
  } else {
    const inputs = [['kcal', 'kcal'], ['protein_g', 'Protein g'], ['carbs_g', 'Carbs g'], ['fat_g', 'Fat g']].map(([k, label]) => {
      const el = h('input', { type: 'number', min: 0, step: 'any', value: f.custom[k], 'aria-label': `${f.name || 'Food'}: ${label} per 100 g`, inputmode: 'decimal' });
      el.addEventListener('input', () => { f.custom[k] = el.value; updateTotals(); });
      return h('div', { class: 'field' }, h('span', { class: 'lab' }, label), el);
    });
    const ask = aiReady() && h('button', { class: 'btn quiet', type: 'button', onclick: async (e) => {
      if (!f.name.trim()) return notify('Type the food name first.');
      e.target.disabled = true; e.target.textContent = 'Asking Gemma 4…';
      try {
        const r = await post('/nutrition/estimate', { food_name: f.name.trim() }, AIKey.headers());
        for (const k of KEYS) f.custom[k] = String(r.nutrition[k]);
        f.aiNote = r.note || 'AI estimate'; render(); notify('Estimate filled in. Please check it before saving.');
      } catch (err) { e.target.disabled = false; e.target.textContent = 'Ask Gemma 4 to estimate'; notify(err.message); }
    } }, 'Ask Gemma 4 to estimate');
    status = h('div', { class: 'grid' }, h('div', { class: 'match warn' }, `No nutrition record for “${f.name || 'this food'}”. Rename it to a known food, type its values per 100 g (from the package or a trusted source)` + (ask ? ', or ask Gemma 4 for a rough estimate.' : '.')),
      f.aiNote && h('div', { class: 'match warn' }, `Estimated by Gemma 4 (${f.aiNote}). These are rough values, so check them.`), h('div', { class: 'custom-grid' }, inputs), ask && h('div', {}, ask));
  }
  return h('div', { class: 'food' }, h('div', { class: 'food-row' }, h('div', { class: 'field' }, h('label', {}, 'Food item'), name), h('div', { class: 'field' }, h('label', {}, 'Portion (g)'), qty),
    h('button', { class: 'remove', type: 'button', 'aria-label': 'Remove ' + (f.name || 'item'), title: 'Remove', onclick: () => { S.draft.foods.splice(i, 1); render(); } }, '×')),
    (f.confidence && f.confidence !== 'manual-review') || f.reasoning ? h('div', { class: 'why' }, f.confidence && f.confidence !== 'manual-review' && h('span', { class: 'badge' }, 'AI confidence: ' + f.confidence),
      f.preparation && h('span', { class: 'badge' }, f.preparation), f.reasoning && h('span', {}, f.reasoning)) : false, status);
}
function updateTotals() {
  const el = $('#totalLine'); if (!el || !S.draft) return;
  const { t, missing } = draftTotals();
  el.textContent = missing ? `${macroLine(t)} · ${missing} item${missing > 1 ? 's' : ''} need${missing > 1 ? '' : 's'} attention` : 'Estimated ' + macroLine(t);
}
function previewImg(src) {
  const img = h('img', { class: 'preview', src, alt: 'Your meal photo' });
  img.addEventListener('error', () => img.replaceWith(h('div', { class: 'preview nopreview', role: 'img', 'aria-label': 'Preview not available' }, 'This browser cannot preview this file type. It will still be analysed.')));
  return img;
}
function reviewPanel() {
  const d = S.draft;
  const head = h('div', { class: 'section-head' }, h('h2', {}, 'Review detected foods'), h('button', { class: 'text-btn', type: 'button', onclick: () => { clearDraft(); notify('Photo removed.'); } }, 'Remove photo'));
  if (d.busy) return h('section', { id: 'analysis', style: 'margin-bottom:28px;scroll-margin-top:84px' }, h('div', { class: 'section-head' }, h('h2', {}, 'Analysing your photo')), h('div', { class: 'panel', 'aria-busy': 'true' }, h('div', { class: 'meal-result' }, previewImg(d.url),
    h('div', { class: 'grid' },
      h('div', { id: 'progress', class: 'progress', role: 'progressbar', 'aria-label': 'Photo analysis progress', 'aria-valuemin': 0, 'aria-valuemax': 100 }, h('div', { id: 'progBar', class: 'bar' })),
      h('p', { id: 'progNote', class: 'muted', style: 'margin:0' }),
      h('ol', { class: 'stages' }, h('li', { id: 'stg0' }, 'Upload photo'), h('li', { id: 'stg1' }, aiReady() ? 'Analyse with Gemma 4' : 'Prepare demo suggestions'), h('li', { id: 'stg2' }, 'Review foods and portions')),
      h('div', { class: 'sr', id: 'progSr', role: 'status' }),
      h('div', {}, h('button', { class: 'btn quiet', type: 'button', id: 'cancelBtn', onclick: () => d.ctl && d.ctl.abort() }, 'Cancel'))))));
  if (d.error) return h('section', { id: 'analysis', style: 'margin-bottom:28px;scroll-margin-top:84px' }, head, h('div', { class: 'panel' }, h('div', { class: 'banner', role: 'alert' }, d.error), h('div', { class: 'actions', style: 'justify-content:flex-start' }, h('button', { class: 'btn', type: 'button', onclick: analyse }, 'Try again'))));
  const type = h('select', { id: 'mealType', 'aria-label': 'Meal type' }, h('option', { value: '' }, 'Meal type (optional)'), ['breakfast', 'lunch', 'dinner', 'snack'].map((x) => h('option', { value: x, selected: d.mealType === x }, x[0].toUpperCase() + x.slice(1))));
  type.addEventListener('change', () => { d.mealType = type.value; });
  const note = h('input', { id: 'mealNote', value: d.note, maxlength: 300, placeholder: 'Note (optional)', 'aria-label': 'Note' }); note.addEventListener('input', () => { d.note = note.value; });
  const save = h('button', { class: 'btn', type: 'button', id: 'saveBtn', onclick: saveDraft }, 'Save meal');
  return h('section', { id: 'analysis', style: 'margin-bottom:28px;scroll-margin-top:84px' }, head, h('div', { class: 'panel' },
    d.fallbackUsed && h('div', { class: 'banner', role: 'status' }, `The main Gemma model was busy, so the backup model (${d.modelUsed}) analysed this photo. Check the foods and portions as usual.`),
    !d.aiEnabled && h('div', { class: 'banner' }, d.message, ' ', h('a', { href: '#/settings' }, 'Open Settings')), d.aiEnabled && d.message && h('p', { class: 'notice', style: 'margin:0 0 12px' }, d.message),
    h('div', { class: 'meal-result' }, previewImg(d.url),
      h('div', {}, h('div', { class: 'eyebrow', style: 'margin-bottom:10px' }, 'Editable food list'), h('div', { class: 'food-list' }, d.foods.map(foodRow)),
        h('div', { style: 'margin-top:10px' }, h('button', { class: 'btn quiet', type: 'button', id: 'addFood', onclick: () => { d.foods.push(newFood({ name: '', grams: 100 })); render(); $$last('.food input'); } }, '＋ Add food item')),
        h('div', { class: 'row2' }, type, note))),
    h('div', { class: 'result-footer' }, h('div', {}, h('strong', { id: 'totalLine' }), h('div', { class: 'notice', style: 'margin:4px 0 0' }, 'Estimates only. Portions and foods can be wrong, so check every item. Not medical advice.')), save)));
}
function $$last(sel) { const all = document.querySelectorAll(sel); const el = [...all].filter((e) => e.getAttribute('aria-label') === 'Food item name').pop(); if (el) el.focus(); }

function historyView() {
  const groups = new Map();
  for (const m of S.meals) { if (!groups.has(m.local_date)) groups.set(m.local_date, []); groups.get(m.local_date).push(m); }
  const body = S.meals.length ? [...groups].map(([day, ms]) => {
    const tot = ms.reduce((a, m) => { for (const k of KEYS) a[k] += m.totals[k]; return a; }, { kcal: 0, protein_g: 0, carbs_g: 0, fat_g: 0 });
    return h('section', { class: 'day' }, h('h2', {}, h('span', {}, fmtDay(day)), h('span', {}, macroLine(tot))), h('div', { class: 'panel' }, ms.map((m) => h('div', { class: 'card-meal' }, thumb(m),
      h('div', { class: 'info' }, h('strong', {}, mealTitle(m)), h('div', { class: 'muted', style: 'font-size:13px' }, `${mealWhen(m)} · ${macroLine(m.totals)}`),
        m.foods.map((f) => f.grams && h('span', { class: 'badge', style: 'margin:6px 6px 0 0' }, `${f.name} ${r0(f.grams)} g`)), m.note && h('div', { class: 'muted', style: 'font-size:13px;margin-top:6px' }, '“' + m.note + '”')),
      h('div', { class: 'del' }, h('button', { class: 'btn danger', type: 'button', 'aria-label': 'Delete this meal: ' + mealTitle(m), onclick: async () => {
        if (!confirm('Delete this meal and its photo?')) return;
        try { await api('/meals/' + m.id, { method: 'DELETE' }); await refreshData(); notify('Meal deleted.'); render(); } catch (e) { notify(e.message); }
      } }, 'Delete'))))));
  }) : [h('div', { class: 'panel empty' }, h('div', { class: 'bowl', 'aria-hidden': 'true' }, '🍽'), h('strong', {}, 'No meals saved in the last 30 days'), h('a', { class: 'btn', href: '#/' }, 'Log a meal'))];
  return [h('div', { class: 'page-head' }, h('div', {}, h('div', { class: 'eyebrow' }, 'Last 30 days'), h('h1', { tabindex: -1 }, 'Meal History'))), ...body];
}

let insightDays = 7;
function insightsView() {
  const root = h('div', {}); loadInsights(root); return [h('div', { class: 'page-head' }, h('div', {}, h('div', { class: 'eyebrow' }, 'Your patterns'), h('h1', { tabindex: -1 }, 'Nutrition Insights')),
    h('div', { class: 'field' }, h('label', { for: 'range' }, 'Period'), h('select', { id: 'range', onchange: (e) => { insightDays = Number(e.target.value); render(); } }, [7, 14, 30].map((n) => h('option', { value: n, selected: insightDays === n }, `Last ${n} days`))))), root];
}
async function loadInsights(root) {
  let s; try { s = await api(`/summary?days=${insightDays}&today=${localDate()}`); } catch (e) { return root.replaceChildren(h('div', { class: 'banner' }, e.message)); }
  const g = s.targets, a = s.average_per_logged_day;
  if (!s.logged_days) return root.replaceChildren(h('div', { class: 'panel empty' }, h('strong', {}, 'Nothing to chart yet'), h('p', {}, 'Save a few meals and your trends will appear here.'), h('a', { class: 'btn', href: '#/' }, 'Log a meal')));
  const stat = (v, l) => h('div', { class: 'stat' }, h('b', {}, v), h('span', {}, l));
  const W = 700, H = 250, L = 40, B = 28, T = 14, iw = W - L - 8, ih = H - B - T, max = Math.max(g.kcal * 1.15, ...s.days.map((d) => d.kcal)), y = (v) => T + ih - (v / max) * ih, bw = iw / s.days.length;
  const chart = svg('svg', { class: 'chart', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `Daily calories for the last ${insightDays} days against a target of ${r0(g.kcal)}. The table below has the same numbers.` },
    ...[0, 0.5, 1].map((p) => svg('g', {}, svg('line', { class: 'axis', x1: L, x2: W - 8, y1: y(max * p / 1.0), y2: y(max * p / 1.0) }), svg('text', { x: L - 6, y: y(max * p) + 4, 'text-anchor': 'end' }, r0(max * p)))),
    svg('line', { class: 'target', x1: L, x2: W - 8, y1: y(g.kcal), y2: y(g.kcal) }), svg('text', { x: W - 8, y: y(g.kcal) - 5, 'text-anchor': 'end' }, 'target'),
    ...s.days.map((d, i) => { const x = L + i * bw + bw * 0.15, hh = Math.max(d.kcal ? 2 : 0, ih - (y(d.kcal) - T)); return svg('g', {}, svg('rect', { class: 'bar' + (d.kcal > g.kcal ? ' over' : ''), x, y: T + ih - hh, width: bw * 0.7, height: hh, rx: 3 }, svg('title', {}, `${fmtDay(d.date)}: ${r0(d.kcal)} kcal`)),
      (insightDays <= 14 || i % 5 === 0) && svg('text', { x: x + bw * 0.35, y: H - 8, 'text-anchor': 'middle' }, parseDay(d.date).toLocaleDateString(undefined, insightDays <= 7 ? { weekday: 'short' } : { day: 'numeric', month: 'short' }))); }));
  const table = h('table', { class: 'sr' }, h('caption', {}, 'Calories per day'), h('tbody', {}, s.days.map((d) => h('tr', {}, h('th', { scope: 'row' }, d.date), h('td', {}, `${r0(d.kcal)} kcal`)))));
  root.replaceChildren(
    h('div', { class: 'stats' }, stat(r0(a.kcal), 'kcal per logged day'), stat(`${s.logged_days} / ${insightDays}`, 'days with meals'), stat(r0(a.protein_g) + ' g', 'protein per logged day'), stat(r0(a.carbs_g) + ' g · ' + r0(a.fat_g) + ' g', 'carbs · fat per logged day')),
    h('div', { class: 'panel', style: 'margin-bottom:22px' }, h('h2', { style: 'font-size:16px;margin-bottom:10px' }, 'Calories per day'), chart, table, h('p', { class: 'notice' }, 'Bars above the dashed line are over your daily calorie target (pink). Totals depend on rough nutrition estimates.')),
    h('div', { class: 'panel' }, h('h2', { style: 'font-size:16px;margin-bottom:10px' }, 'Most logged foods'), h('ul', { class: 'toplist' }, s.top_foods.map((f) => h('li', {}, h('span', {}, f.name), h('span', { class: 'muted' }, `${f.times}× · ${f.grams} g`))))));
}

function aiPanel() {
  const cur = AIKey.read(), hh = S.health || {};
  const keyIn = h('input', { id: 'aiKey', type: 'password', autocomplete: 'off', spellcheck: 'false', placeholder: cur.key ? `Saved: ${AIKey.mask(cur.key)} (paste a new key to replace it)` : 'Paste your Gemma API key', 'aria-label': 'Gemma API key' });
  const modelIn = h('input', { id: 'aiModel', value: cur.model, autocomplete: 'off', spellcheck: 'false', placeholder: hh.model || 'gemma-4-31b-it', 'aria-label': 'Model name' });
  const backupIn = h('input', { id: 'aiBackup', value: cur.backup, autocomplete: 'off', spellcheck: 'false', placeholder: hh.backup_model || 'none', 'aria-label': 'Backup model' });
  const remember = h('input', { id: 'aiRemember', type: 'checkbox', checked: cur.remember });
  const status = h('p', { class: 'match', id: 'aiStatus', role: 'status' });
  const state = cur.key ? `Using your key ${AIKey.mask(cur.key)} (${cur.remember ? 'remembered on this device' : 'this browser session only'}), model ${cur.model || hh.model || 'gemma-4-31b-it'}.`
    : hh.ai_configured ? `Using the key set on the server, model ${cur.model || hh.model}.` : 'No key yet: photos get demo suggestions instead of a real analysis.';
  const typed = () => {
    const key = keyIn.value.trim() || cur.key, model = modelIn.value.trim(), backup = backupIn.value.trim();
    if (key && !AIKey.valid(key)) { status.textContent = 'That does not look like an API key. Paste just the key, with no spaces.'; status.className = 'match warn'; return null; }
    if (model && !/^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$/.test(model)) { status.textContent = 'That does not look like a model name, for example gemma-4-31b-it.'; status.className = 'match warn'; return null; }
    if (backup && !/^(none|off)$/i.test(backup) && !/^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$/.test(backup)) { status.textContent = 'That does not look like a backup model name. Use a model name, or write none to turn the backup off.'; status.className = 'match warn'; return null; }
    return { key, model, backup };
  };
  const test = async (btn) => {
    const v = typed(); if (!v) return;
    if (!v.key && !hh.ai_configured) { status.textContent = 'Paste a key first.'; status.className = 'match warn'; return; }
    btn.disabled = true; status.className = 'match'; status.textContent = 'Checking with Google…';
    try { const r = await post('/ai/test', {}, AIKey.headers(v)); status.textContent = `Works. ${r.model} answered.`; status.className = 'match ok'; }
    catch (e) { status.textContent = e.message; status.className = 'match warn'; }
    btn.disabled = false;
  };
  return h('div', { class: 'panel stack', id: 'aiPanel' }, h('h2', { style: 'font-size:18px' }, 'AI photo recognition (Gemma 4)'),
    h('p', { style: 'margin:0;font-size:14px' }, state),
    insecureTransport() && h('div', { class: 'banner', role: 'alert' }, 'This page is not HTTPS, so a key you enter would travel unencrypted. Use the app on localhost or over HTTPS before entering a key.'),
    h('div', { class: 'form-grid' }, h('div', { class: 'field' }, h('label', { for: 'aiKey' }, 'Your Gemma API key'), keyIn), h('div', { class: 'field' }, h('label', { for: 'aiModel' }, 'Model'), modelIn),
      h('div', { class: 'field' }, h('label', { for: 'aiBackup' }, 'Backup model (used if the main one stays busy; write none to turn off)'), backupIn)),
    h('label', { class: 'check' }, remember, ' Remember the key on this device (otherwise it is forgotten when you close the tab)'),
    h('div', { class: 'actions', style: 'justify-content:flex-start' },
      h('button', { class: 'btn', type: 'button', id: 'aiSave', onclick: () => {
        const v = typed(); if (!v) return;
        if (!v.key) { status.textContent = 'Paste a key first.'; status.className = 'match warn'; return; }
        if (!AIKey.save(v.key, v.model, remember.checked, v.backup)) { status.textContent = 'This browser would not let the key be saved.'; status.className = 'match warn'; return; }
        notify('Key saved in this browser.'); render();
      } }, 'Save key'),
      h('button', { class: 'btn quiet', type: 'button', id: 'aiTest', onclick: (e) => test(e.target) }, 'Test key'),
      cur.key && h('button', { class: 'btn danger', type: 'button', id: 'aiRemove', onclick: () => { AIKey.clear(); notify('Key removed from this browser.'); render(); } }, 'Remove key')),
    status,
    h('p', { class: 'notice', style: 'margin:0' }, 'Get a key at ', h('a', { href: 'https://aistudio.google.com/apikey', target: '_blank', rel: 'noopener noreferrer' }, 'aistudio.google.com/apikey'),
      '. Your key is kept only in this browser. It is sent with each photo or estimate request to this HiperMeal server, which passes it straight to Google and does not store or log it. Photos you analyse go to Google under your own key and terms.'));
}

function settingsView() {
  const t = S.targets, f = {};
  const field = (k, label, max) => { f[k] = h('input', { id: 't_' + k, type: 'number', min: k === 'kcal' ? 500 : 0, max, step: 'any', value: t[k], inputmode: 'decimal' }); return h('div', { class: 'field' }, h('label', { for: 't_' + k }, label), f[k]); };
  const form = h('form', { class: 'panel stack', onsubmit: async (e) => {
    e.preventDefault();
    try { const r = await api('/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(Object.fromEntries(KEYS.map((k) => [k, Number(f[k].value)]))) }); S.targets = r.targets; await refreshData(); notify('Targets saved.'); }
    catch (err) { notify(err.message); }
  } }, h('div', {}, h('h2', { style: 'font-size:18px' }, 'Daily targets'), h('p', { class: 'muted', style: 'margin:4px 0 0;font-size:14px' }, 'Used for the bars and charts. Pick numbers that suit you; HiperMeal does not give dietary advice.')),
    h('div', { class: 'form-grid' }, field('kcal', 'Calories (kcal)', 10000), field('protein_g', 'Protein (g)', 500), field('carbs_g', 'Carbs (g)', 1500), field('fat_g', 'Fat (g)', 500)), h('div', {}, h('button', { class: 'btn', type: 'submit' }, 'Save targets')));
  const cur = Theme.get() || 'system';
  const radio = (v, label) => h('label', {}, h('input', { type: 'radio', name: 'theme', value: v, checked: cur === v, onchange: () => Theme.set(v === 'system' ? null : v) }), label);
  const look = h('div', { class: 'panel stack' }, h('h2', { style: 'font-size:18px' }, 'Appearance'), h('div', { class: 'seg', role: 'radiogroup', 'aria-label': 'Theme' }, radio('system', 'Match my device'), radio('light', 'Light'), radio('dark', 'Dark')));
  const hh = S.health || {};
  const privacy = h('div', { class: 'panel stack' }, h('h2', { style: 'font-size:18px' }, 'Your data'),
    h('p', { class: 'muted', style: 'margin:0;font-size:14px' }, 'Meals and photos are stored on the computer running HiperMeal, with no sign-in, so keep this app on your own machine or network. Photos are stripped of location data. A photo you analysed but never saved is deleted automatically after 24 hours. Deleting a meal deletes its photo.'),
    h('div', {}, h('button', { class: 'btn danger', type: 'button', id: 'wipeBtn', onclick: async () => {
      if (!confirm('Delete ALL meals and photos permanently?')) return;
      try { await api('/data', { method: 'DELETE' }); await refreshData(); notify('Everything was deleted.'); render(); } catch (e) { notify(e.message); }
    } }, 'Delete all my data')));
  const about = h('div', { class: 'panel stack' }, h('h2', { style: 'font-size:18px' }, 'About this install'),
    h('p', { class: 'muted', style: 'margin:0;font-size:13px' }, `${hh.nutrition_records || '…'} foods in the built-in table. Values are rounded approximations, not a verified food database.`));
  return [h('div', { class: 'page-head' }, h('div', {}, h('div', { class: 'eyebrow' }, 'Make it yours'), h('h1', { tabindex: -1 }, 'Settings'))), h('div', { class: 'stack' }, aiPanel(), form, look, privacy, about)];
}

// ---------------------------------------------------------------------------------------------------- router
const ROUTES = { '': homeView, '/': homeView, '/history': historyView, '/insights': insightsView, '/settings': settingsView };
let lastRoute = null;
function render() {
  const path = location.hash.replace(/^#/, '');
  const view = ROUTES[path];
  const name = view ? (path === '' || path === '/' ? 'home' : path.slice(1)) : null;
  for (const a of document.querySelectorAll('[data-route]')) { if (a.dataset.route === name) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); }
  const root = $('#view');
  if (!view) { root.replaceChildren(h('div', { class: 'panel stack' }, h('h1', { tabindex: -1 }, 'Page not found'), h('p', { class: 'muted' }, 'There is nothing at this address.'), h('div', {}, h('a', { class: 'btn', href: '#/' }, 'Back to Home')))); return; }
  const keep = document.activeElement && document.activeElement.id;
  root.replaceChildren(...[].concat(view()).filter(Boolean));
  const dl = $('#foodNames'); if (!dl) root.append(h('datalist', { id: 'foodNames' }, S.foodNames.map((n) => h('option', { value: n }))));
  updateTotals(); paintProgress();
  if (lastRoute !== path) { const t = $('h1', root); if (t) { t.tabIndex = -1; t.focus({ preventScroll: true }); } window.scrollTo(0, 0); }
  else if (keep) { const el = document.getElementById(keep); if (el) el.focus(); }
  lastRoute = path;
}

async function boot() {
  $('#themeBtn').addEventListener('click', () => Theme.set(Theme.current() === 'dark' ? 'light' : 'dark'));
  Theme.paint();
  window.addEventListener('hashchange', render);
  window.addEventListener('beforeunload', (e) => { if (S.draft && S.draft.foods.length) { e.preventDefault(); e.returnValue = ''; } });     // an unsaved review would be lost
  try {
    const [, foods, health] = await Promise.all([refreshData(), api('/foods'), api('/health')]);
    S.foodNames = foods.foods; S.health = health;
  } catch (e) {
    $('#view').replaceChildren(h('div', { class: 'panel stack' }, h('h1', {}, 'Could not load HiperMeal'), h('div', { class: 'banner', role: 'alert' }, e.message), h('div', {}, h('button', { class: 'btn', type: 'button', onclick: () => location.reload() }, 'Try again'))));
    return;
  }
  render();
}
boot();
