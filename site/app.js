(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const eur = (v, d = 3) => v.toLocaleString("es-ES", { minimumFractionDigits: d, maximumFractionDigits: d });
  const hh = (h) => `${String(h % 24).padStart(2, "0")}:00`;
  const range = (a, b) => `${hh(a)}–${hh(b)}`;

  let data = null;
  let current = 0;

  fetch("data/forecast.json", { cache: "no-store" })
    .then((r) => {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    })
    .then((json) => {
      data = json;
      const tomorrow = data.days.findIndex((d) => d.label === "Mañana");
      current = tomorrow >= 0 ? tomorrow : 0;
      renderTabs();
      renderDay();
      renderBacktest();
      renderSpark();
      renderMeta();
    })
    .catch(() => {
      $("day-title").textContent = "Aún no hay datos";
      $("day-source").textContent = "El primer cálculo diario todavía no se ha ejecutado.";
    });

  function renderTabs() {
    const nav = $("tabs");
    nav.innerHTML = "";
    data.days.forEach((d, i) => {
      const b = document.createElement("button");
      b.className = "tab";
      b.type = "button";
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", String(i === current));
      const pill = d.source === "official"
        ? '<span class="pill pill-official">Oficial</span>'
        : '<span class="pill pill-forecast">Previsión</span>';
      b.innerHTML = `${d.label} ${pill}`;
      b.addEventListener("click", () => {
        current = i;
        renderTabs();
        renderDay();
      });
      nav.appendChild(b);
    });
  }

  function renderDay() {
    const d = data.days[current];
    const s = d.summary;
    const isForecast = d.source === "forecast";
    $("day-title").textContent = d.pretty_date + (d.is_offday ? " · fin de semana o festivo" : "");
    $("day-source").textContent = isForecast
      ? `Previsión del modelo a ${d.horizon_days} ${d.horizon_days === 1 ? "día" : "días"} vista. Red Eléctrica publica el precio oficial hacia las 20:15 del día anterior.`
      : "Precio oficial publicado por Red Eléctrica.";
    $("k-window").textContent = range(s.best_window_start, s.best_window_end);
    $("k-window-price").textContent = `${eur(s.best_window_mean)} €/kWh de media`;
    $("k-min").textContent = hh(s.min_hour);
    $("k-min-price").textContent = `${eur(s.min)} €/kWh`;
    $("k-mean").textContent = eur(s.mean);
    $("k-max").textContent = hh(s.max_hour);
    $("k-max-price").textContent = `${eur(s.max)} €/kWh`;
    $("legend-band").hidden = !isForecast;
    renderChart(d);
  }

  function renderChart(d) {
    const chart = $("chart");
    const tip = $("tooltip");
    chart.innerHTML = "";
    const isForecast = d.source === "forecast";
    const lows = d.hours.map((h) => (isForecast ? h.low : h.price));
    const highs = d.hours.map((h) => (isForecast ? h.high : h.price));
    const lo = Math.min(0, ...lows);
    const hi = Math.max(...highs) * 1.05 || 1;
    const pct = (v) => ((v - lo) / (hi - lo)) * 100;
    const cheap = new Set(d.summary.cheapest_hours);
    const ws = d.summary.best_window_start;
    const nowHour = d.label === "Hoy" ? new Date().getHours() : -1;

    d.hours.forEach((h) => {
      const col = document.createElement("div");
      col.className = "col";
      if (cheap.has(h.hour)) col.classList.add("cheap");
      if (h.hour >= ws && h.hour < ws + 3) col.classList.add("window");
      if (h.hour === d.summary.max_hour) col.classList.add("peak");
      if (h.hour === nowHour) col.classList.add("now");
      col.tabIndex = 0;
      const bar = document.createElement("div");
      bar.className = "bar";
      bar.style.height = `${Math.max(pct(h.price) - pct(lo), 0.8)}%`;
      col.appendChild(bar);
      if (isForecast) {
        const band = document.createElement("div");
        band.className = "band";
        band.style.bottom = `${pct(h.low)}%`;
        band.style.height = `${Math.max(pct(h.high) - pct(h.low), 0.5)}%`;
        col.appendChild(band);
      }
      const label = document.createElement("span");
      label.className = "hr";
      label.textContent = String(h.hour).padStart(2, "0");
      col.appendChild(label);

      const show = () => {
        tip.innerHTML = `${range(h.hour, h.hour + 1)}<br><b>${eur(h.price, 4)} €/kWh</b>` +
          (isForecast ? `<br><span style="opacity:.75">80 %: ${eur(h.low)} – ${eur(h.high)}</span>` : "");
        tip.hidden = false;
        const card = chart.parentElement.getBoundingClientRect();
        const r = col.getBoundingClientRect();
        const x = Math.min(Math.max(r.left - card.left + r.width / 2 - tip.offsetWidth / 2, 8), card.width - tip.offsetWidth - 8);
        tip.style.left = `${x}px`;
        tip.style.top = `${r.top - card.top + r.height * (1 - pct(isForecast ? h.high : h.price) / 100) - tip.offsetHeight - 6}px`;
      };
      col.addEventListener("mouseenter", show);
      col.addEventListener("focus", show);
      col.addEventListener("mouseleave", () => (tip.hidden = true));
      col.addEventListener("blur", () => (tip.hidden = true));
      chart.appendChild(col);
    });
    let pad = chart.parentElement.querySelector(".chart-pad");
    if (!pad) {
      pad = document.createElement("div");
      pad.className = "chart-pad";
      chart.after(pad);
    }
  }

  function renderBacktest() {
    const bt = data.backtest;
    const grid = $("bt-grid");
    grid.innerHTML = "";
    const first = bt["1"];
    $("bt-intro").textContent =
      `Validación sobre los últimos ${first.test_days} días, que el modelo no vio al entrenar. ` +
      "Error medio absoluto (MAE) frente a dos referencias: repetir el precio de hace k días y repetir el de la semana pasada. Más corto es mejor.";
    ["1", "2"].forEach((k) => {
      const m = bt[k];
      if (!m) return;
      const rows = [
        ["Modelo ML", m.mae_model, "model"],
        [`Repetir hace ${k} ${k === "1" ? "día" : "días"}`, m.mae_persistence, ""],
        ["Repetir hace 7 días", m.mae_weekly, ""],
      ];
      const max = Math.max(...rows.map((r) => r[1]));
      const card = document.createElement("div");
      card.className = "bt";
      card.innerHTML = `<h3>${k === "1" ? "Previsión a 1 día" : "Previsión a 2 días"}</h3>` +
        rows.map(([name, v, cls]) => `
          <div class="bt-row ${cls}">
            <span>${name}</span>
            <div class="bt-track"><div class="bt-fill" style="width:${(v / max) * 100}%"></div></div>
            <span class="bt-num">${eur(v)}</span>
          </div>`).join("") +
        `<div class="bt-foot"><strong>${m.skill >= 0 ? "−" : "+"}${Math.abs(Math.round(m.skill * 100))} %</strong> de error frente a la mejor referencia · ` +
        `el precio real cae dentro del intervalo del 80 % el ${Math.round(m.coverage_80 * 100)} % de las horas.</div>`;
      grid.appendChild(card);
    });
    const live = data.live || {};
    const parts = Object.entries(live).map(([k, v]) => `a ${k} ${k === "1" ? "día" : "días"}: MAE ${eur(v.mae)} €/kWh en ${v.days} días`);
    $("live").textContent = parts.length
      ? `Historial real de previsiones publicadas (se acumula cada día) — ${parts.join(" · ")}.`
      : "El historial real de previsiones publicadas empezará a acumularse a partir de mañana.";
  }

  function renderSpark() {
    const pts = data.recent_daily;
    if (!pts || !pts.length) return;
    const W = Math.max(340, Math.min(900, $("spark").clientWidth - 28)), H = 220, P = { l: 44, r: 18, t: 10, b: 26 };
    const lo = Math.min(0, ...pts.map((p) => p.min));
    const hi = Math.max(...pts.map((p) => p.max)) * 1.05;
    const x = (i) => P.l + (i / Math.max(pts.length - 1, 1)) * (W - P.l - P.r);
    const y = (v) => P.t + (1 - (v - lo) / (hi - lo)) * (H - P.t - P.b);
    const area = pts.map((p, i) => `${x(i)},${y(p.max)}`).join(" ") + " " +
      pts.map((p, i) => `${x(i)},${y(p.min)}`).reverse().join(" ");
    const line = pts.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.mean)}`).join("");
    const ticks = [lo, (lo + hi) / 2, hi].map((v) =>
      `<line class="grid" x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}"/>` +
      `<text class="axis" x="${P.l - 6}" y="${y(v) + 3}" text-anchor="end">${eur(v, 2)}</text>`).join("");
    const anchors = ["start", "middle", "end"];
    const labels = [0, Math.floor(pts.length / 2), pts.length - 1].map((i, j) => {
      const [, m, dd] = pts[i].date.split("-");
      return `<text class="axis" x="${x(i)}" y="${H - 6}" text-anchor="${anchors[j]}">${dd}/${m}</text>`;
    }).join("");
    $("spark").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Precio medio diario de los últimos 30 días">
      ${ticks}<polygon class="range" points="${area}"/><path class="line" d="${line}"/>${labels}</svg>`;
  }

  function renderMeta() {
    const g = new Date(data.generated_at);
    $("meta").textContent =
      `Actualizado ${g.toLocaleString("es-ES", { dateStyle: "long", timeStyle: "short" })} · ` +
      `entrenado con ${data.training_days.toLocaleString("es-ES")} días de datos desde ${data.first_day} · ` +
      `último día oficial: ${data.last_official_day}.`;
  }
})();
