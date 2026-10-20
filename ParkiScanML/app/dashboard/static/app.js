/* ParkiScan Neuroverse (game + digital twin) — clinician dashboard (vanilla JS + Chart.js 4, works offline). */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const C = { s1: css('--s1'), s2: css('--s2'), s3: css('--s3'), ink: css('--ink'), ink2: css('--ink-2'), ink3: css('--ink-3'), line: css('--line'), crit: css('--critical'),
              q1: css('--q1'), q2: css('--q2'), q3: css('--q3'), good: css('--good'), serious: css('--serious') };
  const pct = (x, d = 0) => (x == null || isNaN(x)) ? '–' : (100 * x).toFixed(d) + '%';
  const fx = (x, d = 1) => (x == null || isNaN(x)) ? '–' : Number(x).toFixed(d);
  const clockOf = (h) => { const m = ((Math.round((13 + h) * 60) % 1440) + 1440) % 1440; return String(Math.floor(m / 60)).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0'); };
  const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  // STATIC = the exported offline copy (GitHub Pages / shared link): API responses are pre-computed JSON files and the
  // what-if is looked up from a pre-simulated grid of scenarios instead of being simulated live.
  const STATIC = !!window.PARKISCAN_STATIC, wiCache = {};
  const enc = (s) => encodeURIComponent(s);
  async function staticApi(p, opt) {
    const get = async (f) => { const r = await fetch('data/' + f); if (!r.ok) throw new Error(f + ' → ' + r.status); return r.json(); };
    let m;
    if (p === '/twin/patients') return get('patients.json');
    if (p === '/twin/model_card') return get('model_card.json');
    if ((m = p.match(/^\/twin\/([^/?]+)\/replay(?:\?run=(.+))?$/))) return get('replay/' + decodeURIComponent(m[1]) + (m[2] ? '__' + decodeURIComponent(m[2]) : '') + '.json');
    if ((m = p.match(/^\/twin\/([^/?]+)\/whatif$/))) {
      const id = decodeURIComponent(m[1]), b = JSON.parse(opt.body);
      if (!wiCache[id]) wiCache[id] = await get('whatif/' + id + '.json');
      const key = [b.dose_shift_min || 0, b.extra_dose_clock_h ?? '', b.sleep_delta_h || 0].join('|');
      const hit = wiCache[id].scenarios[key];
      if (!hit) throw new Error('This offline copy holds pre-simulated scenarios only — run the service for any combination.');
      return hit;
    }
    if ((m = p.match(/^\/twin\/([^/?]+)$/))) return get('twin/' + decodeURIComponent(m[1]) + '.json');
    throw new Error('not available offline: ' + p);
  }
  const api = async (p, opt) => { if (STATIC) return staticApi(p, opt); const r = await fetch(p, opt); if (!r.ok) throw new Error(p + ' → ' + r.status); return r.json(); };

  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.font.size = 11;
  Chart.defaults.color = C.ink3;
  Chart.defaults.animation = false;
  Chart.defaults.plugins.legend.display = false;
  Chart.defaults.plugins.tooltip.backgroundColor = C.ink;
  Chart.defaults.plugins.tooltip.padding = 8;
  Chart.defaults.plugins.tooltip.boxPadding = 4;

  /* guides plugin: now-line, horizontal lines, horizontal bands, vertical bands */
  Chart.register({
    id: 'guides',
    beforeDatasetsDraw(chart, _a, o) {
      const { ctx, chartArea: a, scales: { x, y } } = chart; if (!o || !a) return;
      ctx.save();
      (o.hBands || []).forEach((b) => { ctx.fillStyle = b.color; const y0 = y.getPixelForValue(b.y1), y1 = y.getPixelForValue(b.y0); ctx.fillRect(a.left, Math.max(a.top, y0), a.right - a.left, Math.min(a.bottom, y1) - Math.max(a.top, y0)); });
      (o.xBands || []).forEach((b) => { ctx.fillStyle = b.color; const x0 = Math.max(a.left, x.getPixelForValue(b.x0)), x1 = Math.min(a.right, x.getPixelForValue(b.x1)); if (x1 > x0) ctx.fillRect(x0, a.top, x1 - x0, a.bottom - a.top); });
      (o.hLines || []).forEach((l) => { const py = y.getPixelForValue(l.y); ctx.strokeStyle = l.color; ctx.lineWidth = 1.5; ctx.setLineDash(l.dash || [4, 3]); ctx.beginPath(); ctx.moveTo(a.left, py); ctx.lineTo(a.right, py); ctx.stroke(); });
      if (o.nowX != null) {
        const px = x.getPixelForValue(o.nowX); ctx.setLineDash([]); ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(px, a.top); ctx.lineTo(px, a.bottom); ctx.stroke();
        ctx.fillStyle = C.ink2; ctx.font = '600 11px ' + Chart.defaults.font.family; ctx.textAlign = 'left'; ctx.fillText(o.nowLabel || 'now', px + 4, o.nowBottom ? a.bottom - 6 : a.top + 11);
      }
      ctx.restore();
    },
  });

  const charts = {};
  function chart(id, cfg) { if (charts[id]) charts[id].destroy(); charts[id] = new Chart($('#' + id), cfg); return charts[id]; }
  const gridY = { color: C.line, drawTicks: false }, noGrid = { display: false };
  const hourAxis = (min, max) => ({ type: 'linear', min, max, grid: noGrid, border: { color: C.line },
    ticks: { stepSize: 6, callback: (v) => clockOf(v), maxRotation: 0 } });

  let patients = [], current = null, detail = null, filter = 'all', modelCard = null, replayData = null, replayTimer = null, whatifData = null;

  /* ---------------- sidebar */
  async function loadPatients() {
    const r = await api('/twin/patients');
    patients = r.patients.sort((a, b) => b.risk_peak_24h - a.risk_peak_24h);
    renderList();
    const first = patients.find((p) => p.kind === 'daphnet' && p.status !== 'stable') || patients[0];
    if (first) select(first.id);
  }
  function renderList() {
    const q = $('#search').value.trim().toLowerCase();
    const ul = $('#plist'); ul.innerHTML = '';
    patients.filter((p) => (filter === 'all' || p.status === filter) && (!q || (p.name + p.id).toLowerCase().includes(q))).forEach((p) => {
      const li = document.createElement('li'); li.dataset.id = p.id; if (current === p.id) li.classList.add('sel');
      const src = p.kind === 'daphnet' ? 'Daphnet ' + p.daphnet_subject : p.kind === 'game' ? 'Game session' : 'Synthetic';
      li.innerHTML = `<div class="nm">${esc(p.name)}</div><div class="rk">${pct(p.risk_peak_24h)}</div>
        <div class="sub">${p.age} ${p.sex} · H&amp;Y ${p.hoehn_yahr} · ${esc(src)}</div><div><span class="status ${p.status}">${p.status.toUpperCase()}</span></div>`;
      li.onclick = () => select(p.id); ul.appendChild(li);
    });
  }
  $('#search').oninput = renderList;
  $$('.chip').forEach((b) => b.onclick = () => { $$('.chip').forEach((x) => x.classList.remove('active')); b.classList.add('active'); filter = b.dataset.f; renderList(); });

  /* ---------------- tabs */
  $$('.tab').forEach((t) => t.onclick = () => showTab(t.dataset.tab));
  function showTab(name) {
    $$('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === name));
    $$('.pane').forEach((p) => p.hidden = p.dataset.pane !== name);
    if (name === 'replay') loadReplay();
    if (name === 'model') renderModelCard();
    if (name === 'wearables') renderWearables();
  }

  /* ---------------- patient */
  async function select(id) {
    current = id; renderList(); stopReplay(); replayData = null; whatifData = null;
    detail = await api('/twin/' + encodeURIComponent(id));
    $('#empty').hidden = true; $('#patient').hidden = false;
    renderHead(); renderKpis(); renderRisk(); renderEffect(); renderAlerts(); setupWhatIf(); renderEhr(); renderGame();
    $('#tab-replay').hidden = detail.kind !== 'daphnet';
    $('#tab-game').hidden = !(detail.game && detail.game.length);
    const active = $('.tab.active').dataset.tab;
    if ((active === 'replay' && detail.kind !== 'daphnet') || (active === 'game' && !(detail.game && detail.game.length))) showTab('overview'); else showTab(active);
  }

  function renderHead() {
    const p = detail.patient;
    $('#p-avatar').textContent = p.name.split(/\s+/).filter((w) => /[A-Za-z]/.test(w[0] || '')).map((w) => w[0].toUpperCase()).join('').slice(0, 2) || '•';
    $('#p-name').textContent = p.name;
    $('#p-meta').textContent = `${p.age} y · ${p.sex === 'F' ? 'female' : 'male'} · ${p.city}, ${p.state} · ${p.synthetic_id} · Parkinson's disease, ${p.disease_duration_y} y · H&Y ${p.hoehn_yahr}`;
    const tags = [`MDS-UPDRS III ${p.updrs3_off}`, `NFOG-Q ${p.nfogq}/24`, `Falls (12 m) ${p.falls_12m}`, `MoCA ${p.moca}`, `LEDD ${p.ledd_mg} mg`];
    const conds = [['hypertension', 'Hypertension'], ['type2_diabetes', 'Type 2 diabetes'], ['depression', 'Depression'], ['orthostatic_hypotension', 'Orthostatic hypotension'], ['rem_sleep_behaviour_disorder', 'RBD']].filter(([k]) => p[k]).map(([, v]) => v);
    const gen = [p.gba_carrier ? 'GBA variant' : null, p.lrrk2_g2019s ? 'LRRK2 G2019S' : null].filter(Boolean);
    $('#p-tags').innerHTML = tags.map((t) => `<span class="tag">${esc(t)}</span>`).join('') + conds.map((t) => `<span class="tag c">${esc(t)}</span>`).join('') + gen.map((t) => `<span class="tag g">${esc(t)} (simulated)</span>`).join('');
    const pv = detail.provenance;
    $('#p-prov').innerHTML = `<div><b>Record:</b> ${esc(pv.ehr)}</div><div><b>Wearables:</b> ${esc(pv.wearables)}</div>${pv.game ? `<div><b>Game:</b> ${esc(pv.game)}</div>` : ''}`;
  }

  function kpi(k, v, d, unit = '') { return `<div class="kpi"><div class="k">${k}</div><div class="v">${v}<small>${unit}</small></div><div class="d">${d}</div></div>`; }
  function renderKpis() {
    const c = detail.card, f = detail.forecast, n = detail.nights[detail.nights.length - 1], p = detail.patient;
    const hrvRel = n ? n.rmssd_ms / p.hrv_baseline_ms - 1 : NaN;
    $('#kpis').innerHTML = [
      kpi('Freezing risk now', pct(c.risk_now), `<span class="status ${c.status}">${c.status.toUpperCase()}</span> next 2 h`),
      kpi('Peak next 24 h', pct(c.risk_peak_24h), 'at ' + c.peak_at),
      kpi('Expected freezes', fx(f.expected_fog_episodes_24h, 1), 'episodes, next 24 h (model)'),
      kpi('Predicted OFF time', fx(f.off_hours_awake_24h, 1), 'awake hours, next 24 h', ' h'),
      kpi('Sleep last night', n ? fx(n.total_sleep_h, 1) : '–', n ? `debt ${fx(n.sleep_debt_h, 1)} h · ${n.awakenings} awakenings` : '', ' h'),
      kpi('Overnight HRV', n ? fx(n.rmssd_ms, 0) : '–', isNaN(hrvRel) ? '' : `${hrvRel >= 0 ? '+' : ''}${(100 * hrvRel).toFixed(0)}% vs baseline`, ' ms'),
    ].join('');
  }

  function series(xs, ys) { return xs.map((x, i) => ({ x, y: ys[i] })); }
  // asleep 22:00–06:30 (now = 13:00): relative hours −15…−6.5 and +9…+17.5
  const nights = () => [{ x0: -15, x1: -6.5 }, { x0: 9, x1: 17.5 }].map((b) => ({ ...b, color: 'rgba(18,34,58,0.045)' }));
  function renderRisk() {
    const s = detail.series, thr = detail.forecast.risk_threshold;
    const ds = [{ label: 'Twin forecast', data: series(s.t_h, s.risk), borderColor: C.s1, borderWidth: 2, pointRadius: 0, tension: 0.25,
      segment: { borderDash: (ctx) => ctx.p0.parsed.x >= 0 ? [5, 4] : undefined } }];
    if (whatifData) ds.push({ label: 'What-if', data: series(whatifData.t_h, whatifData.risk_scenario), borderColor: C.s2, borderWidth: 2, pointRadius: 0, tension: 0.25 });
    const ymax = Math.max(thr * 2.2, ...s.risk, ...(whatifData ? whatifData.risk_scenario : [0])) * 1.08;
    chart('c-risk', { type: 'line', data: { datasets: ds }, options: {
      maintainAspectRatio: false, interaction: { mode: 'nearest', axis: 'x', intersect: false },
      scales: { x: hourAxis(-24, 24), y: { min: 0, max: ymax, grid: gridY, border: { display: false }, ticks: { callback: (v) => pct(v) } } },
      plugins: { guides: { nowX: 0, nowLabel: 'now ' + detail.now_clock, xBands: nights(), hLines: [{ y: thr, color: C.crit }] },
        tooltip: { callbacks: { title: (it) => clockOf(it[0].parsed.x) + (it[0].parsed.x >= 0 ? ' (forecast)' : ''), label: (it) => `${it.dataset.label}: ${pct(it.parsed.y, 1)}` } } } } });
  }

  function renderEffect() {
    const s = detail.series;
    const ds = [{ label: 'Dopaminergic effect', data: series(s.t_h, s.effect), borderColor: C.s3, borderWidth: 2, pointRadius: 0, tension: 0.2,
      segment: { borderDash: (ctx) => ctx.p0.parsed.x >= 0 ? [5, 4] : undefined } }];
    if (whatifData) ds.push({ label: 'What-if', data: series(whatifData.t_h, whatifData.effect_scenario), borderColor: C.s2, borderWidth: 2, pointRadius: 0, tension: 0.2 });
    const doses = (whatifData ? detail.doses.filter((d) => d.t_h < 0).concat(whatifData.doses_scenario) : detail.doses);
    ds.push({ type: 'scatter', label: 'Levodopa dose', data: doses.map((d) => ({ x: d.t_h, y: 1.06, mg: d.mg, clock: d.clock })), pointStyle: 'triangle', rotation: 180, radius: 6, hoverRadius: 8,
      backgroundColor: C.ink2, borderColor: '#fff', borderWidth: 1 });
    chart('c-effect', { type: 'line', data: { datasets: ds }, options: {
      maintainAspectRatio: false, interaction: { mode: 'nearest', axis: 'x', intersect: false },
      scales: { x: hourAxis(-24, 24), y: { min: 0, max: 1.12, grid: gridY, border: { display: false }, ticks: { stepSize: 0.25, callback: (v) => v > 1 ? '' : v === 0 ? 'OFF 0' : v === 1 ? 'ON 1' : v } } },
      plugins: { guides: { nowX: 0, nowLabel: 'now', nowBottom: true, xBands: nights(), hBands: [{ y0: 0, y1: 0.35, color: 'rgba(236,131,90,0.10)' }] },
        tooltip: { callbacks: { title: (it) => clockOf(it[0].parsed.x), label: (it) => it.dataset.type === 'scatter' ? `Levodopa ${it.raw.mg} mg at ${it.raw.clock}` : `${it.dataset.label}: ${fx(it.parsed.y, 2)}` } } } } });
  }

  function renderAlerts() {
    const a = detail.alerts; const ul = $('#alerts');
    $('#alert-count').textContent = a.length ? a.length + ' window' + (a.length > 1 ? 's' : '') : '';
    if (!a.length) { ul.innerHTML = '<li class="none">No alert windows in the next 24 h. Routine monitoring.</li>'; return; }
    const maxR = Math.max(...a.flatMap((x) => x.drivers.map((d) => d.risk_reduction)), 0.01);
    ul.innerHTML = a.map((x) => `<li class="${x.level}"><div class="when">${x.from}–${x.to} <span class="status ${x.level === 'high' ? 'high' : 'elevated'}">${x.level.toUpperCase()} · peak ${pct(x.peak)}</span></div>
      <div class="msg">${esc(x.message)}</div><div class="drv">${x.drivers.map((d) => `<div><span class="bar" style="width:${Math.max(6, 70 * d.risk_reduction / maxR)}px"></span>${esc(d.factor)} <span class="muted">(−${pct(d.risk_reduction, 1)} if resolved)</span></div>`).join('')}</div></li>`).join('');
  }

  function alertBox(msg) { const t = $('#w-table'); t.hidden = false; t.querySelector('tbody').innerHTML = `<tr><td colspan="4" class="muted">${esc(msg)}</td></tr>`; }
  /* ---------------- what-if */
  function setupWhatIf() {
    const sel = $('#w-extra'); sel.innerHTML = '<option value="">— none —</option>';
    const extras = STATIC ? (window.PARKISCAN_STATIC.extra || []) : [];
    for (let h = 13.5; h <= 21.5; h += 0.5) if (!STATIC || extras.includes(h)) sel.insertAdjacentHTML('beforeend', `<option value="${h}">${clockOf(h - 13)}</option>`);
    if (STATIC) {
      const g = window.PARKISCAN_STATIC;
      Object.assign($('#w-shift'), { min: Math.min(...g.shift), max: Math.max(...g.shift), step: g.shift[1] - g.shift[0] });
      Object.assign($('#w-sleep'), { min: Math.min(...g.sleep), max: Math.max(...g.sleep), step: g.sleep[1] - g.sleep[0] });
      $('#w-mg').disabled = true; $('#w-mg').title = 'live service only';
    }
    $('#w-shift').value = 0; $('#w-sleep').value = 0; $('#w-mg').value = ''; sync(); $('#w-table').hidden = true;
  }
  function sync() { const v = +$('#w-shift').value; $('#o-shift').textContent = (v > 0 ? '+' : '') + v + ' min'; const s = +$('#w-sleep').value; $('#o-sleep').textContent = (s >= 0 ? '+' : '') + s + ' h'; }
  $('#w-shift').oninput = sync; $('#w-sleep').oninput = sync;
  $('#w-reset').onclick = () => { setupWhatIf(); whatifData = null; renderRisk(); renderEffect(); };
  $('#w-run').onclick = async () => {
    const body = { dose_shift_min: +$('#w-shift').value, sleep_delta_h: +$('#w-sleep').value };
    if ($('#w-extra').value) body.extra_dose_clock_h = +$('#w-extra').value;
    if ($('#w-mg').value) body.dose_mg = +$('#w-mg').value;
    $('#w-run').textContent = 'Simulating…';
    try { whatifData = await api('/twin/' + encodeURIComponent(current) + '/whatif', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); }
    catch (e) { $('#w-run').textContent = 'Simulate on the twin'; alertBox(e.message); return; }
    finally { $('#w-run').textContent = 'Simulate on the twin'; }
    renderRisk(); renderEffect();
    const b = whatifData.baseline, s = whatifData.scenario;
    const rows = [['Expected freezes (24 h)', 'expected_fog_episodes_24h', 1, ''], ['OFF time awake', 'off_hours_awake_24h', 1, ' h'], ['Time above alert threshold', 'alert_hours', 1, ' h'], ['Peak 2-h risk', 'peak_risk', null, '']];
    $('#w-table tbody').innerHTML = rows.map(([lab, k, d, u]) => {
      const fmt = (v) => d == null ? pct(v, 1) : fx(v, d) + u; const diff = s[k] - b[k];
      const cls = Math.abs(diff) < 1e-3 ? '' : diff < 0 ? 'better' : 'worse';
      return `<tr><td>${lab}</td><td>${fmt(b[k])}</td><td>${fmt(s[k])}</td><td class="${cls}">${diff > 0 ? '+' : ''}${d == null ? pct(diff, 1) : fx(diff, d) + u}</td></tr>`;
    }).join('');
    $('#w-table').hidden = false;
  };

  /* ---------------- wearables */
  function renderWearables() {
    if (!detail) return;
    const h = detail.history_hourly, nights = detail.nights;
    const days = {}; h.forEach((r) => { const d = Math.floor((r.t_h + 13 + 144) / 24); (days[d] = days[d] || { steps: 0, inst: 0 }); days[d].steps += r.steps; days[d].inst += r.instability; });
    const keys = Object.keys(days).map(Number).sort((a, b) => a - b).slice(-6);
    const lab = keys.map((d, i) => i === keys.length - 1 ? 'Today' : `D-${keys.length - 1 - i}`);
    const bar = (id, data, color, fmt) => chart(id, { type: 'bar', data: { labels: lab, datasets: [{ data, backgroundColor: color, borderRadius: 4, borderSkipped: 'start', maxBarThickness: 34 }] },
      options: { maintainAspectRatio: false, scales: { x: { grid: noGrid }, y: { grid: gridY, border: { display: false }, beginAtZero: true } }, plugins: { tooltip: { callbacks: { label: (it) => fmt(it.parsed.y) } } } } });
    bar('c-steps', keys.map((d) => days[d].steps), C.s1, (v) => Math.round(v).toLocaleString() + ' steps');
    bar('c-inst', keys.map((d) => +days[d].inst.toFixed(2)), C.serious, (v) => 'instability ' + v.toFixed(2));
    const nl = nights.map((n, i) => i === nights.length - 1 ? 'Last night' : `N-${nights.length - 1 - i}`);
    chart('c-sleep', { type: 'bar', data: { labels: nl, datasets: [
      { label: 'Deep', data: nights.map((n) => n.deep_min / 60), backgroundColor: C.q1, borderColor: '#fff', borderWidth: { top: 2 }, maxBarThickness: 34 },
      { label: 'REM', data: nights.map((n) => n.rem_min / 60), backgroundColor: C.q2, borderColor: '#fff', borderWidth: { top: 2 }, maxBarThickness: 34 },
      { label: 'Light', data: nights.map((n) => n.light_min / 60), backgroundColor: C.q3, borderRadius: { topLeft: 4, topRight: 4 }, maxBarThickness: 34 }] },
      options: { maintainAspectRatio: false, scales: { x: { stacked: true, grid: noGrid }, y: { stacked: true, grid: gridY, border: { display: false }, ticks: { callback: (v) => v + ' h' } } },
        plugins: { tooltip: { mode: 'index', callbacks: { label: (it) => `${it.dataset.label}: ${fx(it.parsed.y, 1)} h` } } } } });
    const base = detail.patient.hrv_baseline_ms; $('#hrv-base').textContent = 'baseline ' + base + ' ms';
    chart('c-hrv', { type: 'line', data: { labels: nl, datasets: [{ data: nights.map((n) => n.rmssd_ms), borderColor: C.s1, borderWidth: 2, pointRadius: 4, pointBackgroundColor: C.s1, pointBorderColor: '#fff', pointBorderWidth: 2 }] },
      options: { maintainAspectRatio: false, scales: { x: { grid: noGrid }, y: { grid: gridY, border: { display: false }, ticks: { callback: (v) => v + ' ms' } } },
        plugins: { guides: { hLines: [{ y: base, color: C.ink3, dash: [3, 3] }] }, tooltip: { callbacks: { label: (it) => `RMSSD ${it.parsed.y} ms` } } } } });
  }

  /* ---------------- replay (real Daphnet) */
  async function loadReplay(run) {
    if (!detail || detail.kind !== 'daphnet') return;
    if (replayData && !run) { drawReplay(replayData.t_s.length); return; }
    stopReplay();
    try { replayData = await api('/twin/' + encodeURIComponent(current) + '/replay' + (run ? '?run=' + encodeURIComponent(run) : '')); }
    catch (e) { $('#r-note').textContent = 'Replay unavailable: ' + e.message; return; }
    const sel = $('#r-run'); sel.innerHTML = replayData.runs.map((r) => `<option ${r === replayData.run ? 'selected' : ''}>${r}</option>`).join('');
    sel.onchange = () => loadReplay(sel.value);
    $('#r-note').textContent = replayData.note + ' Horizon: freezing onset within ' + replayData.horizon_s + ' s. Shaded = annotated freezing (video-rated by physiotherapists).';
    drawReplay(replayData.t_s.length);
  }
  function fogBands(r, upto) {
    const b = []; let s = null;
    for (let i = 0; i < upto; i++) { if (r.fog[i] && s === null) s = r.t_s[i]; if (!r.fog[i] && s !== null) { b.push({ x0: s, x1: r.t_s[i], color: 'rgba(208,59,59,0.14)' }); s = null; } }
    if (s !== null) b.push({ x0: s, x1: r.t_s[upto - 1], color: 'rgba(208,59,59,0.14)' });
    return b;
  }
  function drawReplay(upto) {
    const r = replayData; if (!r) return;
    const pts = r.t_s.slice(0, upto).map((t, i) => ({ x: t, y: r.risk[i] }));
    const al = r.t_s.slice(0, upto).map((t, i) => r.alarm[i] ? { x: t, y: 1.03 } : null).filter(Boolean);
    chart('c-replay', { type: 'line', data: { datasets: [
      { label: 'Freeze risk (next ' + r.horizon_s + ' s)', data: pts, borderColor: C.s1, borderWidth: 1.6, pointRadius: 0 },
      { type: 'scatter', label: 'Alarm', data: al, pointStyle: 'rect', radius: 2.5, backgroundColor: C.crit, borderWidth: 0 }] },
      options: { maintainAspectRatio: false, interaction: { mode: 'nearest', axis: 'x', intersect: false },
        scales: { x: { type: 'linear', min: r.t_s[0], max: r.t_s[r.t_s.length - 1], grid: noGrid, title: { display: true, text: 'time into recording (min:s)', color: C.ink3 || '#73859a', font: { size: 11 } }, ticks: { callback: (v) => Math.floor(v / 60) + ':' + String(Math.floor(v % 60)).padStart(2, '0') } },
          y: { min: 0, max: 1.08, grid: gridY, border: { display: false }, ticks: { callback: (v) => v > 1 ? '' : pct(v) } } },
        plugins: { guides: { xBands: fogBands(r, upto), hLines: [{ y: r.threshold, color: C.crit }] },
          tooltip: { callbacks: { title: (it) => 't = ' + it[0].parsed.x.toFixed(1) + ' s', label: (it) => it.dataset.type === 'scatter' ? 'Early-warning alarm' : 'risk ' + pct(it.parsed.y, 1) } } } } });
    // event stats for the shown part
    let eps = 0, warned = 0, leads = [];
    for (let i = 1; i < upto; i++) if (r.fog[i] && !r.fog[i - 1]) {
      eps++; const on = r.t_s[i]; let first = null;
      for (let j = i - 1; j >= 0 && r.t_s[j] >= on - 5; j--) if (r.alarm[j]) first = r.t_s[j];
      if (first !== null) { warned++; leads.push(on - first); }
    }
    leads.sort((a, b) => a - b);
    const m = r.model;
    $('#r-stats').innerHTML = [kpi('Freezes in view', eps, 'annotated onsets'), kpi('Warned before onset', eps ? warned + ' / ' + eps : '–', 'alarm within 5 s before'),
      kpi('Median lead', leads.length ? fx(leads[Math.floor(leads.length / 2)], 1) : '–', 'seconds of warning', ' s'),
      kpi('Model (all 10 people)', pct(m.early_warning_rate), `warned · ${fx(m.false_cues_per_hour ?? m.false_alarms_per_hour, 0)} false cues / h of walking`)].join('');
  }
  function stopReplay() { if (replayTimer) { clearInterval(replayTimer); replayTimer = null; } $('#r-play').textContent = '▶ Play'; }
  $('#r-play').onclick = () => {
    if (!replayData) return;
    if (replayTimer) { stopReplay(); return; }
    let i = 2; const n = replayData.t_s.length; $('#r-play').textContent = '❚❚ Pause';
    const fps = 15, step = () => Math.max(1, Math.round(+$('#r-speed').value * 2 / fps));   // windows are 0.5 s apart
    replayTimer = setInterval(() => { i = Math.min(n, i + step()); drawReplay(i); if (i >= n) stopReplay(); }, 1000 / fps);
  };

  /* ---------------- game biomarkers */
  function renderGame() {
    const g = detail.game || []; const box = $('#game');
    if (!g.length) { box.innerHTML = '<p class="muted">No game sessions yet. Play ParkiScan Neuroverse with the analysis service running — results arrive here automatically at the end of Level 3.</p>'; return; }
    const last = g[g.length - 1];
    const items = Object.entries(last.metrics || {}).filter(([, v]) => v && v.value != null && !isNaN(v.value));
    box.innerHTML = `<p class="muted" style="grid-column:1/-1">Session ${esc(last.received || '')} · ${items.length} measures${(last.flags || []).length ? ' · flags: ' + esc(last.flags.join(', ')) : ''}</p>` +
      items.map(([k, v]) => `<div class="gm"><div class="t">${esc(v.label || k)}</div><div class="v">${fx(v.value, v.digits ?? 2)} <span class="u">${esc(v.unit || '')}</span></div><div class="u">${esc(v.task || '')}</div></div>`).join('');
  }

  /* ---------------- EHR */
  function renderEhr() {
    const E = detail.fhir.entry.map((e) => e.resource);
    const cond = E.filter((r) => r.resourceType === 'Condition'), med = E.filter((r) => r.resourceType === 'MedicationRequest'), obs = E.filter((r) => r.resourceType === 'Observation');
    $('#t-cond').innerHTML = '<tr><th>Condition</th><th>Code</th><th>Since</th></tr>' + cond.map((c) => `<tr><td>${esc(c.code.text)}</td><td class="muted">${esc(c.code.coding[0].system.includes('snomed') ? 'SNOMED ' + c.code.coding[0].code : '—')}</td><td>${esc(c.onsetDateTime)}</td></tr>`).join('');
    $('#t-med').innerHTML = '<tr><th>Medication</th><th>Instruction</th></tr>' + med.map((m) => `<tr><td>${esc(m.medicationCodeableConcept.text)}</td><td>${esc(m.dosageInstruction[0].text)}</td></tr>`).join('');
    $('#ledd').textContent = 'LEDD ' + detail.patient.ledd_mg + ' mg/day';
    $('#t-obs').innerHTML = '<tr><th>Observation</th><th>Value</th><th>Date</th></tr>' + obs.map((o) => `<tr><td>${esc(o.code.text)}</td><td class="num">${o.valueQuantity ? fx(o.valueQuantity.value, 1) + ' ' + esc(o.valueQuantity.unit) : esc(o.valueString)}</td><td>${esc(o.effectiveDateTime)}</td></tr>`).join('');
    $('#fhir').textContent = JSON.stringify(detail.fhir, null, 2); $('#fhir').hidden = true;
  }
  $('#fhir-btn').onclick = () => { const p = $('#fhir'); p.hidden = !p.hidden; $('#fhir-btn').textContent = p.hidden ? 'View FHIR bundle' : 'Hide FHIR bundle'; };

  /* ---------------- model card */
  async function renderModelCard() {
    if (!modelCard) modelCard = await api('/twin/model_card');
    const f = modelCard.fog_forecast_report, t = modelCard.twin_risk_report;
    if (f) {
      const d = f.deployed, ev = d.cue_rule || d.event_level, oc = d.operating_curve || [];
      const hs = Object.keys(f.results);
      const names = [...new Set(hs.flatMap((h) => Object.keys(f.results[h])))];
      $('#m-fog').innerHTML = `<div class="big-stat"><div><b>${pct(ev.early_warning_rate)}</b><span>freezes warned before onset (${ev.warned_before_onset}/${ev.episodes})</span></div>
        <div><b>${fx(ev.median_lead_s, 1)} s</b><span>median warning lead time</span></div><div><b>${fx(ev.false_cues_per_hour ?? ev.false_alarms_per_hour, 0)}</b><span>false cues per hour of walking (10 s cue)</span></div></div>
        <table class="metric-tbl"><tr><th>ROC AUC (95% CI)</th><th>freeze-index baseline</th>${names.map((n) => `<th>${esc(n)}</th>`).join('')}</tr>
        ${hs.map((h) => `<tr><td>onset within ${h.slice(1)} s</td><td>${fx(f.baselines[h].freeze_index_auc, 2)}</td>${names.map((n) => { const v = f.results[h][n]; return `<td>${v ? fx(v.auc, 2) + ` <span class="muted">(${fx(v.auc_ci95_subject_bootstrap[0], 2)}–${fx(v.auc_ci95_subject_bootstrap[1], 2)})</span>` : '–'}</td>`; }).join('')}</tr>`).join('')}</table>
        ${oc.length ? `<h4 class="sub">Operating curve — the clinician chooses sensitivity vs false cues</h4><table class="metric-tbl"><tr><th>window specificity</th><th>freezes warned</th><th>false cues / h</th><th>median lead</th></tr>${oc.map((o) => `<tr><td>${fx(o.window_specificity, 2)}</td><td>${pct(o.early_warning_rate)} <span class="muted">(${o.warned_before_onset}/${o.episodes})</span></td><td>${fx(o.false_cues_per_hour, 0)}</td><td>${fx(o.median_lead_s, 1)} s</td></tr>`).join('')}</table>` : ''}
        <p class="note">${esc(f.dataset)} · ${esc(f.validation)} · deployed: ${esc(d.configuration)}, horizon ${d.horizon_s} s. The lab protocol provoked ≈ ${fx(ev.episodes / Math.max(1e-9, ev.non_fog_hours), 0)} freezes per hour of walking, so home false-cue rates will differ.</p>`;
    } else $('#m-fog').innerHTML = '<p class="muted">Run python -m training.fog_forecast_train</p>';
    if (t) {
      $('#m-twin').innerHTML = `<div class="big-stat"><div><b>${fx(t.results['digital_twin (ehr+wearable+pk)'].auc, 2)}</b><span>ROC AUC, fused twin (patient-grouped CV)</span></div>
        <div><b>${pct(t.top_decile.capture)}</b><span>of freeze windows caught by top-10% alerts</span></div><div><b>${fx(t.top_decile.lift, 1)}×</b><span>lift over base rate</span></div></div>
        <table class="metric-tbl"><tr><th>inputs</th><th>ROC AUC</th><th>PR AUC</th><th>Brier</th></tr>${Object.entries(t.results).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${fx(v.auc, 3)}</td><td>${fx(v.pr_auc, 3)}</td><td>${fx(v.brier, 4)}</td></tr>`).join('')}</table>
        <p class="note">${esc(t.data)}. Labels come from the documented simulator — this shows that fusing records, wearables and pharmacology recovers more risk than any single stream; it is not clinical accuracy.</p>`;
    } else $('#m-twin').innerHTML = '<p class="muted">Run python -m training.twin_risk_train</p>';
  }

  loadPatients().catch((e) => { $('#empty').textContent = 'Could not reach the twin service: ' + e.message; });
})();
