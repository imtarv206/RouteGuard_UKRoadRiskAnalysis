/* ─────────────────────────────────────────────────────────────
   map.js  –  RouteGuard frontend logic
   ───────────────────────────────────────────────────────────── */

const API_BASE = "http://localhost:8000";   // ← change if deployed elsewhere

/* ─── STATE ─────────────────────────────────────────────────── */
const state = {
  origin: null,
  dest:   null,
  time_from: null,   // hour 0-23 or null
  time_to:   null,   // hour 0-23 or null
  day:    "any",
  weather:"any",
  routeCoords: [],
  lastResult: null,  // Store analysis result for exports
};

/* ─── MAP INIT ───────────────────────────────────────────────── */
const map = L.map("map", {
  center: [51.5, -0.12],   // UK default view
  zoom: 11,
  zoomControl: true,
});

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 18,
}).addTo(map);

/* Layer groups */
const layers = {
  markers:   L.layerGroup().addTo(map),
  route:     L.layerGroup().addTo(map),
  riskPts:   L.layerGroup().addTo(map),
  heatmap:   null,
};

/* ─── CUSTOM ICONS ───────────────────────────────────────────── */
function makeIcon(color, label) {
  return L.divIcon({
    className: "",
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    html: `<div style="
      width:32px;height:32px;border-radius:50% 50% 50% 0;
      background:${color};border:3px solid #fff;
      transform:rotate(-45deg);
      box-shadow:0 2px 8px rgba(0,0,0,.4);
      display:flex;align-items:center;justify-content:center;
    "><span style="transform:rotate(45deg);font-size:11px;font-weight:700;color:#0b0f1a;">${label}</span></div>`,
  });
}

const ICON_ORIGIN = makeIcon("#22c55e", "A");
const ICON_DEST   = makeIcon("#38bdf8", "B");

/* ─── MAP CLICK ──────────────────────────────────────────────── */
map.on("click", (e) => {
  const { lat, lng } = e.latlng;

  if (!state.origin) {
    state.origin = { lat, lng };
    setMarker("origin", lat, lng);
    updatePointUI("origin", lat, lng);
    setTip("คลิกแผนที่เพื่อเลือกปลายทาง");
  } else if (!state.dest) {
    state.dest = { lat, lng };
    setMarker("dest", lat, lng);
    updatePointUI("dest", lat, lng);
    setTip("กด วิเคราะห์เส้นทาง เพื่อประเมินความเสี่ยง");
    drawRoute();
    enableAnalyze();
  }
});

/* ─── MARKERS ────────────────────────────────────────────────── */
const markers = {};
function setMarker(type, lat, lng) {
  if (markers[type]) layers.markers.removeLayer(markers[type]);
  markers[type] = L.marker([lat, lng], {
    icon: type === "origin" ? ICON_ORIGIN : ICON_DEST,
    zIndexOffset: 1000,
  }).addTo(layers.markers);
}

/* ─── ROUTE (OSRM) ───────────────────────────────────────────── */
async function drawRoute() {
  if (!state.origin || !state.dest) return;
  layers.route.clearLayers();

  const { origin: o, dest: d } = state;
  const url = `https://router.project-osrm.org/route/v1/driving/${o.lng},${o.lat};${d.lng},${d.lat}?overview=full&geometries=geojson`;

  try {
    const res  = await fetch(url);
    const data = await res.json();
    if (data.code !== "Ok") throw new Error("OSRM error");

    const coords = data.routes[0].geometry.coordinates; // [lng,lat]
    state.routeCoords = coords.map(([lng, lat]) => ({ lat, lng }));

    const latlngs = state.routeCoords.map(p => [p.lat, p.lng]);
    L.polyline(latlngs, {
      color: "#38bdf8",
      weight: 4,
      opacity: .8,
      dashArray: null,
    }).addTo(layers.route);

    // Fit map to route
    map.fitBounds(L.latLngBounds(latlngs), { padding: [60, 60] });
  } catch {
    // Fallback: straight line
    state.routeCoords = [state.origin, state.dest];
    L.polyline([[o.lat, o.lng], [d.lat, d.lng]], {
      color: "#38bdf8", weight: 4, opacity: .8, dashArray: "8 6",
    }).addTo(layers.route);
  }
}

/* ─── ANALYZE ────────────────────────────────────────────────── */

async function analyzeRoute() {
  if (!state.origin || !state.dest || !state.routeCoords.length) return;

  showLoading(true);
  layers.riskPts.clearLayers();

  const body = {
    route:       state.routeCoords,
    time_from:   state.time_from,   // null = any
    time_to:     state.time_to,
    day_of_week: state.day,
    weather:     state.weather,
  };

  try {
    const resRisk = await fetch(`${API_BASE}/analyze`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });

    if (!resRisk.ok) throw new Error(await resRisk.text());
    const data = await resRisk.json();

    renderResult(data);
    plotRiskPoints(data.risk_points);

  } catch (err) {
    alert("ไม่สามารถเชื่อมต่อ backend ได้\n" + err.message);
  } finally {
    showLoading(false);
  }
}

/* ─── FILTER SUMMARY ─────────────────────────────────────────── */
function renderFilterSummary() {
  const from = state.time_from;
  const to   = state.time_to;

  const dayLabel = {
    any: "ทุกวัน",
    Monday: "วันจันทร์", Tuesday: "วันอังคาร", Wednesday: "วันพุธ",
    Thursday: "วันพฤหัส", Friday: "วันศุกร์", Saturday: "วันเสาร์", Sunday: "วันอาทิตย์",
  };
  const weatherLabel = { any: "ทุกสภาพ", Fine: "แจ่มใส", Rain: "ฝนตก", Fog: "หมอก" };

  let timeText = "ทุกช่วงเวลา";
  if (from !== null && to !== null) {
    timeText = `${String(from).padStart(2,"0")}:00 – ${String(to).padStart(2,"0")}:59 น.`;
  } else if (from !== null) {
    timeText = `ตั้งแต่ ${String(from).padStart(2,"0")}:00 น.`;
  } else if (to !== null) {
    timeText = `ถึง ${String(to).padStart(2,"0")}:59 น.`;
  }

  const tags = [
    { icon: "🕐", label: "เวลา", val: timeText, hi: from !== null || to !== null },
    { icon: "📅", label: "วัน",  val: dayLabel[state.day] || state.day, hi: state.day !== "any" },
    { icon: "🌤️", label: "อากาศ", val: weatherLabel[state.weather] || state.weather, hi: state.weather !== "any" },
  ];

  document.getElementById("filterSummary").innerHTML = tags.map(t => `
    <div class="fs-tag ${t.hi ? "highlight" : ""}">
      <span class="fs-icon">${t.icon}</span>
      <span>${t.label}:</span>
      <span class="fs-val">${t.val}</span>
    </div>
  `).join("");
}

/* ─── RENDER RESULT ──────────────────────────────────────────── */
function renderResult(data) {
  document.getElementById("resultPanel").style.display = "flex";
  state.lastResult = data;  // Save result for exports
  renderRouteSummary();
  renderFilterSummary();

  const badge = document.getElementById("riskBadge");
  badge.className = `risk-badge ${data.risk_level}`;

  document.getElementById("riskLabel").textContent = {
    LOW: "ความเสี่ยงต่ำ", MEDIUM: "ความเสี่ยงปานกลาง", HIGH: "ความเสี่ยงสูง",
  }[data.risk_level] || data.risk_level;

  document.getElementById("riskScore").textContent = data.overall_score;

  // Meter
  const fill = document.getElementById("riskFill");
  const colors = { LOW: "#22c55e", MEDIUM: "#f59e0b", HIGH: "#ef4444" };
  fill.style.width = `${data.overall_score}%`;
  fill.style.background = colors[data.risk_level];

  // Stats with color coding
  const s = data.stats;
  const statGrid = document.getElementById("statGrid");
  statGrid.innerHTML = `
    <div class="stat-item">
      <label>ระยะทาง</label>
      <span>${s.route_km} km</span>
    </div>
    <div class="stat-item">
      <label>จุดตรวจสอบ</label>
      <span>${s.route_points_checked}</span>
    </div>
    <div class="stat-item">
      <label>อุบัติเหตุใกล้เคียง</label>
      <span>${s.total_nearby || 0}</span>
    </div>
    <div class="stat-item">
      <label>อุบัติเหตุร้าย</label>
      <span class="fatal">${s.fatal_nearby}</span>
    </div>
    <div class="stat-item">
      <label>บาดเจ็บสาหัส</label>
      <span class="serious">${s.serious_nearby}</span>
    </div>
    ${s.ml_severity_prob != null ? `
    <div class="stat-item">
      <label>🤖 ML ความรุนแรง</label>
      <span class="${s.ml_severity_prob > 0.2 ? 'fatal' : s.ml_severity_prob > 0.12 ? 'serious' : ''}">${(s.ml_severity_prob * 100).toFixed(1)}%</span>
    </div>` : ''}
  `;

  // Recommendations with smart icons
  const recItems = data.recommendations.map(r => {
    let icon = "⚠️";
    
    // Detect recommendation type by keywords
    if (r.includes("ชั่วโมง") || r.includes("เวลา") || r.includes("17:00") || r.includes("hour")) {
      icon = "🕐";
    } else if (r.includes("อากาศ") || r.includes("ฝน") || r.includes("หมอก") || r.includes("rain") || r.includes("fog")) {
      icon = "💧";
    } else if (r.includes("รถ") || r.includes("เดินทาง") || r.includes("ความเร็ว") || r.includes("vehicle") || r.includes("speed")) {
      icon = "🚗";
    } else if (r.includes("ปลอดภัย") || r.includes("ระมัดระวัง") || r.includes("safe")) {
      icon = "👤";
    } else if (r.includes("ทะเบียน") || r.includes("ข้อมูล")) {
      icon = "📋";
    } else if (r.includes("ช่วย") || r.includes("ติดต่อ") || r.includes("help")) {
      icon = "📞";
    }
    
    return `<div class="rec-item" data-icon="${icon}">${r}</div>`;
  });
  
  const recsBox = document.getElementById("recsBox");
  recsBox.innerHTML = recItems.join("");

  // Scroll result into view
  document.getElementById("resultPanel").scrollIntoView({ behavior: "smooth" });
}

/* ─── RENDER ROUTE SUMMARY ───────────────────────────────────── */
function renderRouteSummary() {
  if (!state.origin || !state.dest) return;
  
  const html = `
    <div class="route-point">
      <div class="route-point-icon origin">A</div>
      <div>
        <div class="route-point-label">ต้นทาง</div>
        <div class="route-point-coords">${state.origin.lat.toFixed(4)}, ${state.origin.lng.toFixed(4)}</div>
      </div>
    </div>
    <div class="route-divider">↓</div>
    <div class="route-point">
      <div class="route-point-icon dest">B</div>
      <div>
        <div class="route-point-label">ปลายทาง</div>
        <div class="route-point-coords">${state.dest.lat.toFixed(4)}, ${state.dest.lng.toFixed(4)}</div>
      </div>
    </div>
  `;
  
  document.getElementById("routeSummary").innerHTML = html;
}

/* ─── EXPORT TO JSON ─────────────────────────────────────────── */
function exportSummaryJson() {
  if (!state.lastResult) {
    alert("ไม่มีข้อมูลผลการวิเคราะห์");
    return;
  }
  
  const exportData = {
    timestamp: new Date().toISOString(),
    route: {
      origin: state.origin,
      destination: state.dest,
      distance_km: state.lastResult.stats.route_km,
      points_checked: state.lastResult.stats.route_points_checked,
    },
    filters: {
      time_from: state.time_from,
      time_to: state.time_to,
      day_of_week: state.day,
      weather: state.weather,
    },
    risk_assessment: {
      overall_score: state.lastResult.overall_score,
      risk_level: state.lastResult.risk_level,
      fatal_accidents_nearby: state.lastResult.stats.fatal_nearby,
      serious_accidents_nearby: state.lastResult.stats.serious_nearby,
    },
    risk_points: state.lastResult.risk_points,
    recommendations: state.lastResult.recommendations,
  };
  
  const json = JSON.stringify(exportData, null, 2);
  const blob = new Blob([json], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `route_risk_assessment_${Date.now()}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/* ─── EXPORT TO PDF ──────────────────────────────────────────── */
function exportSummaryPdf() {
  if (!state.lastResult) {
    alert("ไม่มีข้อมูลผลการวิเคราะห์");
    return;
  }
  
  // Simple HTML-based PDF export (using print-to-PDF)
  const styles = `
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; background: white; color: #000; }
      h1, h2 { color: #333; margin-top: 20px; margin-bottom: 10px; }
      .section { margin-bottom: 20px; page-break-inside: avoid; }
      .label { color: #666; font-size: 12px; text-transform: uppercase; font-weight: bold; }
      .value { font-size: 14px; color: #000; margin: 5px 0; }
      .badge { display: inline-block; padding: 8px 12px; border-radius: 4px; font-weight: bold; }
      .badge.LOW { background-color: #22c55e; color: white; }
      .badge.MEDIUM { background-color: #f59e0b; color: white; }
      .badge.HIGH { background-color: #ef4444; color: white; }
      .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 10px 0; }
      .stat-box { border: 1px solid #ddd; padding: 10px; }
      .recs { margin: 10px 0; }
      .rec-item { margin: 5px 0; padding: 8px; background: #f5f5f5; border-left: 3px solid #0066cc; }
      table { width: 100%; border-collapse: collapse; }
      th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
      th { background: #f5f5f5; }
      .timestamp { color: #999; font-size: 12px; margin-top: 20px; }
    </style>
  `;
  
  const dayLabels = {
    any: "ทุกวัน",
    Monday: "จันทร์", Tuesday: "อังคาร", Wednesday: "พุธ",
    Thursday: "พฤหัส", Friday: "ศุกร์", Saturday: "เสาร์", Sunday: "อาทิตย์",
  };
  const weatherLabels = { any: "ทุกสภาพ", Fine: "แจ่มใส", Rain: "ฝนตก", Fog: "หมอก" };
  
  let timeText = "ทุกช่วงเวลา";
  if (state.time_from !== null && state.time_to !== null) {
    timeText = `${String(state.time_from).padStart(2,"0")}:00 – ${String(state.time_to).padStart(2,"0")}:59`;
  }
  
  const content = `
    <h1>📋 รายงานการประเมินความเสี่ยงเส้นทาง</h1>
    
    <div class="section">
      <h2>🚗 ข้อมูลเส้นทาง</h2>
      <div class="label">ต้นทาง</div>
      <div class="value">${state.origin.lat.toFixed(4)}, ${state.origin.lng.toFixed(4)}</div>
      <div class="label">ปลายทาง</div>
      <div class="value">${state.dest.lat.toFixed(4)}, ${state.dest.lng.toFixed(4)}</div>
      <div class="label">ระยะทาง</div>
      <div class="value">${state.lastResult.stats.route_km} km</div>
    </div>
    
    <div class="section">
      <h2>🔍 เงื่อนไขการระเมิน</h2>
      <table>
        <tr><th>เกณฑ์</th><th>ค่า</th></tr>
        <tr><td>เวลาเดินทาง</td><td>${timeText} น.</td></tr>
        <tr><td>วัน</td><td>${dayLabels[state.day]}</td></tr>
        <tr><td>สภาพอากาศ</td><td>${weatherLabels[state.weather]}</td></tr>
      </table>
    </div>
    
    <div class="section">
      <h2>⚠️ ผลการประเมินความเสี่ยง</h2>
      <div style="margin: 10px 0;">
        <span class="badge ${state.lastResult.risk_level}">${state.lastResult.risk_level === 'LOW' ? 'ความเสี่ยงต่ำ' : state.lastResult.risk_level === 'MEDIUM' ? 'ความเสี่ยงปานกลาง' : 'ความเสี่ยงสูง'}</span>
        <div style="font-size: 24px; font-weight: bold; margin-top: 10px;">คะแนน: ${state.lastResult.overall_score} / 100</div>
      </div>
    </div>
    
    <div class="section">
      <h2>📊 สถิติบนเส้นทาง</h2>
      <div class="stats">
        <div class="stat-box">
          <div class="label">จุดตรวจสอบ</div>
          <div class="value">${state.lastResult.stats.route_points_checked}</div>
        </div>
        <div class="stat-box">
          <div class="label">อุบัติเหตุร้าย</div>
          <div class="value" style="color: #ef4444;">${state.lastResult.stats.fatal_nearby}</div>
        </div>
        <div class="stat-box">
          <div class="label">บาดเจ็บสาหัส</div>
          <div class="value" style="color: #f59e0b;">${state.lastResult.stats.serious_nearby}</div>
        </div>
        <div class="stat-box">
          <div class="label">อุบัติเหตุทั้งหมด</div>
          <div class="value">${(state.lastResult.stats.fatal_nearby + state.lastResult.stats.serious_nearby)}</div>
        </div>
      </div>
    </div>
    
    <div class="section">
      <h2>💡 ข้อเสนอแนะ</h2>
      <div class="recs">
        ${state.lastResult.recommendations.map(r => `<div class="rec-item">• ${r}</div>`).join("")}
      </div>
    </div>
    
    <div class="timestamp">สร้างเมื่อ: ${new Date().toLocaleString("th-TH")}</div>
  `;
  
  const printWindow = window.open("", "_blank");
  printWindow.document.write(styles + content);
  printWindow.document.close();
  setTimeout(() => {
    printWindow.print();
  }, 250);
}

/* ─── PRINT SUMMARY ──────────────────────────────────────────── */
function printSummary() {
  if (!state.lastResult) {
    alert("ไม่มีข้อมูลผลการวิเคราะห์");
    return;
  }
  
  const dayLabels = {
    any: "ทุกวัน",
    Monday: "จันทร์", Tuesday: "อังคาร", Wednesday: "พุธ",
    Thursday: "พฤหัส", Friday: "ศุกร์", Saturday: "เสาร์", Sunday: "อาทิตย์",
  };
  const weatherLabels = { any: "ทุกสภาพ", Fine: "แจ่มใส", Rain: "ฝนตก", Fog: "หมอก" };
  
  let timeText = "ทุกช่วงเวลา";
  if (state.time_from !== null && state.time_to !== null) {
    timeText = `${String(state.time_from).padStart(2,"0")}:00 – ${String(state.time_to).padStart(2,"0")}:59`;
  }
  
  const printContent = `
    <div style="font-family: Arial, sans-serif; padding: 20px; background: white; color: black;">
      <h1 style="text-align: center;">📋 รายงานการประเมินความเสี่ยงเส้นทาง</h1>
      
      <h2>🚗 ข้อมูลเส้นทาง</h2>
      <p><strong>ต้นทาง:</strong> ${state.origin.lat.toFixed(4)}, ${state.origin.lng.toFixed(4)}</p>
      <p><strong>ปลายทาง:</strong> ${state.dest.lat.toFixed(4)}, ${state.dest.lng.toFixed(4)}</p>
      <p><strong>ระยะทาง:</strong> ${state.lastResult.stats.route_km} km</p>
      
      <h2>🔍 เงื่อนไขการระเมิน</h2>
      <p><strong>เวลาเดินทาง:</strong> ${timeText} น.</p>
      <p><strong>วัน:</strong> ${dayLabels[state.day]}</p>
      <p><strong>สภาพอากาศ:</strong> ${weatherLabels[state.weather]}</p>
      
      <h2>⚠️ ผลการประเมินความเสี่ยง</h2>
      <p><strong>ระดับความเสี่ยง:</strong> ${state.lastResult.risk_level === 'LOW' ? 'ต่ำ' : state.lastResult.risk_level === 'MEDIUM' ? 'ปานกลาง' : 'สูง'}</p>
      <p><strong>คะแนน:</strong> ${state.lastResult.overall_score} / 100</p>
      
      <h2>📊 สถิติบนเส้นทาง</h2>
      <p><strong>จุดตรวจสอบ:</strong> ${state.lastResult.stats.route_points_checked}</p>
      <p><strong>อุบัติเหตุร้าย:</strong> ${state.lastResult.stats.fatal_nearby}</p>
      <p><strong>บาดเจ็บสาหัส:</strong> ${state.lastResult.stats.serious_nearby}</p>
      
      <h2>💡 ข้อเสนอแนะ</h2>
      <ul>
        ${state.lastResult.recommendations.map(r => `<li>${r}</li>`).join("")}
      </ul>
      
      <p style="margin-top: 30px; color: #999; font-size: 12px;">
        สร้างเมื่อ: ${new Date().toLocaleString("th-TH")}
      </p>
    </div>
  `;
  
  const printWindow = window.open("", "_blank");
  printWindow.document.write(printContent);
  printWindow.document.close();
  setTimeout(() => {
    printWindow.print();
  }, 250);
}

/* ─── RISK POINTS ON MAP ─────────────────────────────────────── */
function plotRiskPoints(points) {
  const colors = { LOW: "#22c55e", MEDIUM: "#f59e0b", HIGH: "#ef4444" };

  points.forEach(p => {
    const c = colors[p.risk_level] || "#94a3b8";
    L.circleMarker([p.lat, p.lng], {
      radius:      p.risk_level === "HIGH" ? 10 : p.risk_level === "MEDIUM" ? 7 : 5,
      fillColor:   c,
      color:       "#fff",
      weight:      1.5,
      fillOpacity: .75,
    })
      .bindPopup(`
        <b style="color:${c}">${p.risk_level}</b><br>
        คะแนน: ${p.score}<br>
        อุบัติเหตุใกล้เคียง: ${p.nearby_accidents} จุด
      `)
      .addTo(layers.riskPts);
  });
}

/* ─── HOTSPOT HEATMAP (on load) ─────────────────────────────── */
async function loadHotspots() {
  try {
    const res  = await fetch(`${API_BASE}/hotspots?limit=500`);
    const data = await res.json();
    const heat = data.hotspots.map(h => [h.Latitude, h.Longitude, h.count / 10]);
    if (layers.heatmap) map.removeLayer(layers.heatmap);
    layers.heatmap = L.heatLayer(heat, {
      radius: 18, blur: 20, maxZoom: 14,
      gradient: { 0.3: "#22c55e", 0.6: "#f59e0b", 1.0: "#ef4444" },
    }).addTo(map);
  } catch { /* backend not running yet */ }
}

async function loadStats() {
  try {
    const res  = await fetch(`${API_BASE}/stats`);
    const data = await res.json();
    document.getElementById("totalAccidents").textContent =
      `${data.total_accidents.toLocaleString()} อุบัติเหตุ`;
  } catch { /* ignore */ }
}

/* ─── CHIP GROUPS ────────────────────────────────────────────── */
function setupChips(groupId, stateKey) {
  document.getElementById(groupId).querySelectorAll(".chip").forEach(btn => {
    btn.addEventListener("click", () => {
      document.getElementById(groupId).querySelectorAll(".chip").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state[stateKey] = btn.dataset.val;
    });
  });
}
setupChips("dayChips",     "day");
setupChips("weatherChips", "weather");

/* ─── TIME INPUT (24h) ───────────────────────────────────────── */
function setupTimeInput() {
  const fromEl = document.getElementById("timeFrom");
  const toEl   = document.getElementById("timeTo");
  const preview = document.getElementById("timePreview");

  function clamp(el) {
    let v = parseInt(el.value);
    if (isNaN(v)) return;
    if (v < 0)  { el.value = 0;  v = 0; }
    if (v > 23) { el.value = 23; v = 23; }
  }

  function update() {
    clamp(fromEl); clamp(toEl);
    const from = fromEl.value.trim();
    const to   = toEl.value.trim();

    if (from === "" && to === "") {
      state.time_from = null;
      state.time_to   = null;
      preview.textContent = "ทุกช่วงเวลา";
    } else {
      const f = from !== "" ? parseInt(from) : 0;
      const t = to   !== "" ? parseInt(to)   : 23;
      state.time_from = f;
      state.time_to   = t;
      const label = f <= t
        ? `${String(f).padStart(2,"0")}:00 – ${String(t).padStart(2,"0")}:59 น.`
        : `${String(f).padStart(2,"0")}:00 – 23:59 และ 00:00 – ${String(t).padStart(2,"0")}:59 น.`;
      preview.textContent = label;
    }
  }

  fromEl.addEventListener("input", update);
  toEl.addEventListener("input",   update);
}
setupTimeInput();

function clearTimeInput() {
  document.getElementById("timeFrom").value = "";
  document.getElementById("timeTo").value   = "";
  state.time_from = null;
  state.time_to   = null;
  document.getElementById("timePreview").textContent = "ทุกช่วงเวลา";
}

/* ─── UI HELPERS ─────────────────────────────────────────────── */
function updatePointUI(type, lat, lng) {
  const el = document.getElementById(type === "origin" ? "originText" : "destText");
  el.textContent = `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
}

function setTip(msg) {
  document.getElementById("mapTip").textContent = msg;
}

function enableAnalyze() {
  document.getElementById("analyzeBtn").disabled = false;
}

function showLoading(show) {
  document.getElementById("loadingOverlay").style.display = show ? "flex" : "none";
}

function clearPoint(type) {
  if (type === "origin") {
    state.origin = null;
    if (markers.origin) { layers.markers.removeLayer(markers.origin); markers.origin = null; }
    document.getElementById("originText").textContent = "ยังไม่ได้เลือก";
    document.getElementById("originSearch").value = "";
    document.getElementById("originDropdown").classList.remove("open");
  } else {
    state.dest = null;
    if (markers.dest) { layers.markers.removeLayer(markers.dest); markers.dest = null; }
    document.getElementById("destText").textContent = "ยังไม่ได้เลือก";
    document.getElementById("destSearch").value = "";
    document.getElementById("destDropdown").classList.remove("open");
  }
  layers.route.clearLayers();
  layers.riskPts.clearLayers();
  document.getElementById("analyzeBtn").disabled = true;
  document.getElementById("resultPanel").style.display = "none";
  setTip("คลิกแผนที่เพื่อเลือก" + (type === "origin" ? "ต้นทาง" : "ปลายทาง"));
}

function clearAll() {
  clearPoint("origin");
  clearPoint("dest");
  state.routeCoords = [];
  setTip("คลิกแผนที่เพื่อเลือกต้นทาง");
}

/* ─── PLACE SEARCH (Nominatim Geocoding) ────────────────────── */
let searchTimers = { origin: null, dest: null };

function setupPlaceSearch(type) {
  const inputId  = type === "origin" ? "originSearch" : "destSearch";
  const dropId   = type === "origin" ? "originDropdown" : "destDropdown";
  const input    = document.getElementById(inputId);
  const dropdown = document.getElementById(dropId);

  input.addEventListener("input", () => {
    const q = input.value.trim();
    clearTimeout(searchTimers[type]);

    if (q.length < 2) {
      dropdown.classList.remove("open");
      dropdown.innerHTML = "";
      return;
    }

    // Debounce 400ms to respect Nominatim rate limit
    searchTimers[type] = setTimeout(() => searchNominatim(q, dropdown, type), 400);
  });

  // Close dropdown on outside click
  document.addEventListener("click", (e) => {
    if (!input.contains(e.target) && !dropdown.contains(e.target)) {
      dropdown.classList.remove("open");
    }
  });
}

async function searchNominatim(query, dropdown, type) {
  dropdown.innerHTML = '<div class="search-loading">กำลังค้นหา…</div>';
  dropdown.classList.add("open");

  try {
    // Bias towards UK
    const url = `https://nominatim.openstreetmap.org/search?` +
      `q=${encodeURIComponent(query)}` +
      `&format=json&limit=5&addressdetails=1` +
      `&viewbox=-8.5,49.5,2.0,61.0&bounded=0`;

    const res = await fetch(url, {
      headers: { "Accept-Language": "en" }
    });
    const results = await res.json();

    if (!results.length) {
      dropdown.innerHTML = '<div class="search-loading">ไม่พบผลลัพธ์</div>';
      return;
    }

    dropdown.innerHTML = results.map((r, i) => {
      const name = r.display_name.split(",")[0];
      const addr = r.display_name.split(",").slice(1, 3).join(",").trim();
      return `<div class="search-dropdown-item" data-idx="${i}">
        <div class="place-name">${name}</div>
        <div class="place-addr">${addr}</div>
      </div>`;
    }).join("");

    dropdown.querySelectorAll(".search-dropdown-item").forEach((el, i) => {
      el.addEventListener("click", () => {
        const r = results[i];
        const lat = parseFloat(r.lat);
        const lng = parseFloat(r.lon);
        selectPlace(type, lat, lng, r.display_name.split(",")[0]);
        dropdown.classList.remove("open");
      });
    });

  } catch {
    dropdown.innerHTML = '<div class="search-loading">เกิดข้อผิดพลาด</div>';
  }
}

function selectPlace(type, lat, lng, name) {
  const inputId = type === "origin" ? "originSearch" : "destSearch";
  document.getElementById(inputId).value = name;

  if (type === "origin") {
    state.origin = { lat, lng };
    setMarker("origin", lat, lng);
    updatePointUI("origin", lat, lng);
    if (state.dest) {
      drawRoute();
      enableAnalyze();
      setTip("กด วิเคราะห์เส้นทาง เพื่อประเมินความเสี่ยง");
    } else {
      setTip("เลือกปลายทาง");
      map.setView([lat, lng], 14);
    }
  } else {
    state.dest = { lat, lng };
    setMarker("dest", lat, lng);
    updatePointUI("dest", lat, lng);
    if (state.origin) {
      drawRoute();
      enableAnalyze();
      setTip("กด วิเคราะห์เส้นทาง เพื่อประเมินความเสี่ยง");
    } else {
      setTip("เลือกต้นทาง");
      map.setView([lat, lng], 14);
    }
  }
}

setupPlaceSearch("origin");
setupPlaceSearch("dest");

/* ─── INIT ───────────────────────────────────────────────────── */
loadStats();
loadHotspots();