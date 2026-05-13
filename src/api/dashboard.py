"""Small SOC dashboard served by the ARGUS API."""

from __future__ import annotations


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ARGUS SOC Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-2: #eceff3;
      --ink: #15181d;
      --muted: #626a73;
      --line: #d9dee5;
      --red: #b42318;
      --amber: #b25e09;
      --green: #157347;
      --blue: #1f6feb;
      --focus: #111827;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 64px;
      padding: 12px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--surface);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .status {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
      color: var(--muted);
      white-space: nowrap;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--amber);
      flex: 0 0 auto;
    }
    .dot.ok { background: var(--green); }
    .dot.fail { background: var(--red); }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      min-height: calc(100vh - 65px);
    }
    .workspace {
      min-width: 0;
      padding: 16px 20px 24px;
    }
    .filters {
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
      align-items: end;
    }
    label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    input, select, button {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
      color: var(--ink);
      padding: 7px 9px;
      font: inherit;
    }
    button {
      cursor: pointer;
      font-weight: 700;
      background: var(--focus);
      color: #ffffff;
      border-color: var(--focus);
    }
    button.secondary {
      background: var(--surface);
      color: var(--ink);
      border-color: var(--line);
    }
    button:focus, input:focus, select:focus {
      outline: 2px solid var(--blue);
      outline-offset: 1px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    .metric strong {
      display: block;
      margin-top: 4px;
      font-size: 22px;
      line-height: 1.1;
    }
    .table-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
    }
    table {
      width: 100%;
      min-width: 980px;
      border-collapse: collapse;
    }
    th, td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
      vertical-align: middle;
    }
    th {
      position: sticky;
      top: 0;
      background: var(--surface-2);
      z-index: 1;
      color: #2d333b;
      font-size: 12px;
      text-transform: uppercase;
    }
    tbody tr {
      cursor: pointer;
    }
    tbody tr:hover, tbody tr.selected {
      background: #fff7ed;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 76px;
      height: 24px;
      border-radius: 999px;
      padding: 0 8px;
      font-size: 12px;
      font-weight: 800;
      border: 1px solid transparent;
    }
    .badge.CRITICAL { color: #ffffff; background: var(--red); }
    .badge.HIGH { color: #ffffff; background: #c2410c; }
    .badge.MEDIUM { color: #3b2500; background: #f4c542; }
    .badge.LOW { color: #0f5132; background: #cfe9dc; border-color: #9dd2b7; }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    aside {
      min-width: 0;
      border-left: 1px solid var(--line);
      background: var(--surface);
      padding: 16px;
      overflow: auto;
    }
    .panel-title {
      margin: 0 0 12px;
      font-size: 15px;
      font-weight: 800;
    }
    .detail-grid {
      display: grid;
      grid-template-columns: 120px minmax(0, 1fr);
      gap: 8px 10px;
      align-items: start;
      margin-bottom: 16px;
    }
    .detail-grid dt {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .detail-grid dd {
      margin: 0;
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .risk-line {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
    }
    .bar {
      height: 8px;
      background: var(--surface-2);
      border-radius: 999px;
      overflow: hidden;
    }
    .bar > span {
      display: block;
      height: 100%;
      background: var(--amber);
      width: 0%;
    }
    .error {
      display: none;
      margin-bottom: 12px;
      padding: 10px 12px;
      border: 1px solid #f1a7a1;
      border-radius: 8px;
      color: var(--red);
      background: #fff1f0;
    }
    .empty {
      padding: 28px;
      color: var(--muted);
      text-align: center;
    }
    @media (max-width: 1100px) {
      main { grid-template-columns: 1fr; }
      aside { border-left: 0; border-top: 1px solid var(--line); }
      .filters { grid-template-columns: repeat(3, minmax(120px, 1fr)); }
      .metrics { grid-template-columns: repeat(3, minmax(120px, 1fr)); }
    }
    @media (max-width: 720px) {
      header { align-items: flex-start; flex-direction: column; }
      .filters, .metrics { grid-template-columns: 1fr 1fr; }
      .workspace { padding: 12px; }
      aside { padding: 12px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>ARGUS SOC Dashboard</h1>
    <div class="status"><span id="healthDot" class="dot"></span><span id="healthText">Checking API</span></div>
  </header>
  <main>
    <section class="workspace">
      <form id="filters" class="filters">
        <label>Run ID<input id="replayRunId" name="replay_run_id" autocomplete="off"></label>
        <label>User<input id="userId" name="user_id" autocomplete="off"></label>
        <label>Class<select id="alertClass" name="alert_class"><option value="">Any</option><option>CRITICAL</option><option>HIGH</option><option>MEDIUM</option><option>LOW</option></select></label>
        <label>Min Severity<input id="minSeverity" name="min_severity" type="number" min="0" max="1" step="0.05"></label>
        <label>Limit<input id="limit" name="limit" type="number" min="1" max="500" value="50"></label>
        <button type="submit">Refresh</button>
      </form>
      <div id="error" class="error"></div>
      <section class="metrics" aria-label="Alert metrics">
        <div class="metric"><span>Total</span><strong id="metricTotal">0</strong></div>
        <div class="metric"><span>Critical</span><strong id="metricCritical">0</strong></div>
        <div class="metric"><span>High</span><strong id="metricHigh">0</strong></div>
        <div class="metric"><span>Max Severity</span><strong id="metricSeverity">0.000</strong></div>
        <div class="metric"><span>Users</span><strong id="metricUsers">0</strong></div>
      </section>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Class</th>
              <th>Severity</th>
              <th>Probability</th>
              <th>User</th>
              <th>Host</th>
              <th>Session</th>
              <th>Technique</th>
              <th>Run ID</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody id="alertsBody"></tbody>
        </table>
        <div id="empty" class="empty">No alerts match the active filters.</div>
      </div>
    </section>
    <aside>
      <h2 class="panel-title">Alert Detail</h2>
      <dl id="details" class="detail-grid"></dl>
      <h2 class="panel-title">UEBA Risk</h2>
      <div id="riskPanel"></div>
    </aside>
  </main>
  <script>
    const state = { alerts: [], selectedAlertId: null, timer: null };
    const allowedAlertClasses = new Set(["CRITICAL", "HIGH", "MEDIUM", "LOW"]);
    const $ = (id) => document.getElementById(id);

    function fmtNumber(value, digits = 3) {
      const n = Number(value);
      return Number.isFinite(n) ? n.toFixed(digits) : "";
    }

    function text(value) {
      return value === null || value === undefined || value === "" ? "-" : String(value);
    }

    function clearNode(node) {
      node.replaceChildren();
    }

    function makeNode(tag, options = {}) {
      const node = document.createElement(tag);
      if (options.className) node.className = options.className;
      if (options.text !== undefined) node.textContent = text(options.text);
      return node;
    }

    function safeAlertClass(value) {
      const raw = String(value || "").toUpperCase();
      return allowedAlertClasses.has(raw) ? raw : "LOW";
    }

    function appendCell(row, value, className) {
      const cell = document.createElement("td");
      if (className) cell.className = className;
      cell.textContent = text(value);
      row.appendChild(cell);
      return cell;
    }

    function setError(message) {
      const box = $("error");
      box.textContent = message || "";
      box.style.display = message ? "block" : "none";
    }

    function queryFromFilters() {
      const params = new URLSearchParams();
      for (const id of ["replayRunId", "userId", "alertClass", "minSeverity", "limit"]) {
        const el = $(id);
        if (el.value.trim()) params.set(el.name, el.value.trim());
      }
      return params.toString();
    }

    async function fetchJson(url) {
      const response = await fetch(url, {
        credentials: "same-origin",
        headers: { "accept": "application/json" },
      });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return response.json();
    }

    async function refreshHealth() {
      try {
        const health = await fetchJson("/health");
        $("healthDot").className = health.phase3_model_loaded ? "dot ok" : "dot fail";
        $("healthText").textContent = health.phase3_model_loaded
          ? `API ready | ES alerts ${health.elasticsearch_alerts_enabled ? "on" : "off"}`
          : "API missing Phase 3 model";
      } catch (err) {
        $("healthDot").className = "dot fail";
        $("healthText").textContent = "API unreachable";
      }
    }

    async function refreshAlerts() {
      setError("");
      try {
        const payload = await fetchJson(`/phase3/alerts?${queryFromFilters()}`);
        state.alerts = payload.alerts || [];
        renderMetrics();
        renderTable();
        if (state.alerts.length) selectAlert(state.selectedAlertId || state.alerts[0].alert_id);
        else renderDetails(null);
      } catch (err) {
        setError(`Alert query failed: ${err.message}`);
      }
    }

    function renderMetrics() {
      const alerts = state.alerts;
      const users = new Set(alerts.map((a) => a.user_id).filter(Boolean));
      const maxSeverity = alerts.reduce((m, a) => Math.max(m, Number(a.composite_severity) || 0), 0);
      $("metricTotal").textContent = alerts.length;
      $("metricCritical").textContent = alerts.filter((a) => a.alert_class === "CRITICAL").length;
      $("metricHigh").textContent = alerts.filter((a) => a.alert_class === "HIGH").length;
      $("metricSeverity").textContent = fmtNumber(maxSeverity);
      $("metricUsers").textContent = users.size;
    }

    function renderTable() {
      const tbody = $("alertsBody");
      clearNode(tbody);
      $("empty").style.display = state.alerts.length ? "none" : "block";
      const fragment = document.createDocumentFragment();
      for (const alert of state.alerts) {
        const tr = document.createElement("tr");
        tr.dataset.alertId = alert.alert_id || "";
        if (tr.dataset.alertId === state.selectedAlertId) tr.classList.add("selected");

        const classCell = document.createElement("td");
        const badge = makeNode("span", { className: `badge ${safeAlertClass(alert.alert_class)}`, text: alert.alert_class });
        classCell.appendChild(badge);
        tr.appendChild(classCell);
        appendCell(tr, fmtNumber(alert.composite_severity));
        appendCell(tr, fmtNumber(alert.attack_probability));
        appendCell(tr, alert.user_id, "mono");
        appendCell(tr, alert.host_id, "mono");
        appendCell(tr, alert.session_id, "mono");
        appendCell(tr, alert.technique_id || alert.fallback_technique_id);
        appendCell(tr, alert.replay_run_id, "mono");
        appendCell(tr, alert["@timestamp"]);
        tr.addEventListener("click", () => selectAlert(alert.alert_id));
        fragment.appendChild(tr);
      }
      tbody.appendChild(fragment);
    }

    async function selectAlert(alertId) {
      const alert = state.alerts.find((item) => item.alert_id === alertId) || state.alerts[0];
      state.selectedAlertId = alert ? alert.alert_id : null;
      renderTable();
      renderDetails(alert || null);
      if (alert && alert.user_id) await renderRisk(alert.user_id);
    }

    function appendDetail(details, label, value) {
      details.appendChild(makeNode("dt", { text: label }));
      details.appendChild(makeNode("dd", { className: "mono", text: value }));
    }

    function renderDetails(alert) {
      const details = $("details");
      clearNode(details);
      if (!alert) {
        appendDetail(details, "Status", "No alert selected");
        clearNode($("riskPanel"));
        return;
      }
      const rows = [
        ["Alert ID", alert.alert_id],
        ["Class", alert.alert_class],
        ["Severity", fmtNumber(alert.composite_severity)],
        ["Attack Prob", fmtNumber(alert.attack_probability)],
        ["Technique", alert.technique_id || alert.fallback_technique_id],
        ["Technique Source", alert.technique_source],
        ["User", alert.user_id],
        ["Host", alert.host_id],
        ["Session", alert.session_id],
        ["Run ID", alert.replay_run_id],
        ["Timestamp", alert["@timestamp"]],
      ];
      for (const [label, value] of rows) appendDetail(details, label, value);
    }

    async function renderRisk(userId) {
      const panel = $("riskPanel");
      clearNode(panel);
      try {
        const payload = await fetchJson(`/phase3/ueba/risks/${encodeURIComponent(userId)}`);
        const risk = Number(payload.risk) || 0;
        const riskLine = makeNode("div", { className: "risk-line" });
        riskLine.appendChild(makeNode("span", { className: "mono", text: userId }));
        riskLine.appendChild(makeNode("strong", { text: fmtNumber(risk) }));
        const bar = makeNode("div", { className: "bar" });
        const fill = document.createElement("span");
        fill.style.width = `${Math.max(0, Math.min(100, risk * 100))}%`;
        bar.appendChild(fill);
        panel.appendChild(riskLine);
        panel.appendChild(bar);
      } catch (err) {
        const error = makeNode("div", { className: "error", text: `UEBA query failed: ${err.message}` });
        error.style.display = "block";
        panel.appendChild(error);
      }
    }

    $("filters").addEventListener("submit", (event) => {
      event.preventDefault();
      refreshAlerts();
    });
    refreshHealth();
    refreshAlerts();
    state.timer = window.setInterval(() => {
      refreshHealth();
      refreshAlerts();
    }, 30000);
  </script>
</body>
</html>
"""


__all__ = ["DASHBOARD_HTML"]
