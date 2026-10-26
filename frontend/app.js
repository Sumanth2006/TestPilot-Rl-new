/* TestPilot-RL dashboard
 * Static results are read from RESULTS_BASE (your project's /results folder).
 * Uploads are POSTed to API_BASE + "/evaluate" (see API contract in README / chat). */
const CONFIG = {
  RESULTS_BASE: "/results",   // serves results/*.csv and results/figures/*.png
  API_BASE: "/api",
};

const DATASETS = {
  synthetic:    { summary:"summary.csv", paired:"paired_synthetic_comparison.csv", raw:"raw_results.csv" },
  paintcontrol: { summary:"real_world_paintcontrol_summary.csv", paired:"real_world_paintcontrol_paired.csv", raw:"real_world_paintcontrol_raw.csv" },
  iofrol:       { summary:"real_world_iofrol_summary.csv", paired:"real_world_iofrol_paired.csv", raw:"real_world_iofrol_raw.csv" },
};
const FIGURES = [
  ["baseline_comparison.png","Baseline comparison (synthetic)"],
  ["training_curve.png","Training curve"],
  ["budget_sensitivity.png","Budget sensitivity"],
  ["reward_ablation.png","Reward ablation"],
  ["agent_ablation.png","Agent ablation"],
  ["real_world_paintcontrol.png","Real-world: PaintControl"],
  ["real_world_iofrol.png","Real-world: IOF/ROL"],
];
const METRIC_LABEL = {
  napfd:"NAPFD", apfd:"APFD", bug_detection_rate:"Bug detection rate",
  bugs_per_cost_unit:"Bugs per cost unit", coverage_rate:"Coverage rate", unique_bugs_detected:"Unique bugs detected",
};
const PALETTE = ["#94a3b8","#f59e0b","#ef4444","#10b981","#3b82f6","#8b5cf6","#ec4899","#14b8a6"];
const colorOf = (() => { const m = {}; let i = 0; return n => (m[n] ??= PALETTE[i++ % PALETTE.length]); })();

const $ = id => document.getElementById(id);
const cache = {};
const charts = {};

/* ---------- helpers ---------- */
async function loadCSV(name) {
  if (cache[name]) return cache[name];
  const url = `${CONFIG.RESULTS_BASE}/${name}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url} (HTTP ${res.status})`);
  const text = await res.text();
  const parsed = Papa.parse(text, { header:true, dynamicTyping:true, skipEmptyLines:true });
  return (cache[name] = parsed.data);
}
function banner(msg) { const b = $("banner"); b.textContent = msg; b.classList.toggle("hidden", !msg); }
function fmt(v, d = 4) { return (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d); }
function mean(a) { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN; }
function groupBy(rows, key) { const m = new Map(); rows.forEach(r => { const k = r[key]; if (!m.has(k)) m.set(k, []); m.get(k).push(r); }); return m; }
function drawChart(id, cfg) {
  charts[id]?.destroy();
  const ctx = $(id).getContext("2d");
  charts[id] = new Chart(ctx, cfg);
}
const baseOpts = (extra = {}) => ({
  responsive:true, maintainAspectRatio:false, animation:{ duration:250 },
  plugins:{ legend:{ labels:{ boxWidth:12 } } }, ...extra,
});

/* ---------- tabs ---------- */
document.querySelectorAll(".tab").forEach(btn => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  btn.classList.add("active");
  $("tab-" + btn.dataset.tab).classList.add("active");
  const t = btn.dataset.tab;
  if (t === "ablations") renderAblations();
  if (t === "training") renderTraining();
  if (t === "figures") renderFigures();
}));

/* ---------- shared renderers (used for existing + uploaded results) ---------- */
function renderKpis(el, summary, metric) {
  const key = metric + "_mean";
  const rows = [...summary].sort((a, b) => b[key] - a[key]);
  const best = rows[0];
  const dqn = summary.find(r => /Dueling/i.test(r.strategy));
  const rnd = summary.find(r => /Random/i.test(r.strategy));
  const k = [];
  k.push(["Best strategy", best.strategy, `${METRIC_LABEL[metric]} ${fmt(best[key])}`]);
  if (dqn) k.push(["Dueling DQN", fmt(dqn[key]), `±${fmt(dqn[metric + "_ci95"])} (95% CI)`]);
  if (dqn && rnd && rnd[key]) {
    const lift = (dqn[key] - rnd[key]) / rnd[key] * 100;
    k.push(["DQN vs. Random", (lift >= 0 ? "+" : "") + lift.toFixed(1) + "%", METRIC_LABEL[metric]]);
  }
  k.push(["Evaluations", summary[0].num_evaluations ?? "–", `${summary[0].num_unique_builds ?? "–"} unique builds/cycles`]);
  el.innerHTML = k.map(([l, v, s]) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
}
function renderCompare(canvasId, summary, metric) {
  const m = metric + "_mean", c = metric + "_ci95";
  drawChart(canvasId, {
    type:"bar",
    data:{ labels:summary.map(r => r.strategy),
      datasets:[{ label:METRIC_LABEL[metric], data:summary.map(r => r[m]),
        backgroundColor:summary.map(r => colorOf(r.strategy)), borderRadius:6 }] },
    options:baseOpts({
      plugins:{ legend:{ display:false }, tooltip:{ callbacks:{ label: ctx => {
        const r = summary[ctx.dataIndex]; return `${fmt(r[m])} ± ${fmt(r[c])}`; } } } },
      scales:{ y:{ beginAtZero:true } },
    }),
  });
}
function renderSummaryTable(el, summary, metric) {
  const cols = [["bug_detection_rate","BDR"],["napfd","NAPFD"],["apfd","APFD"],["bugs_per_cost_unit","Bugs/Cost"],["coverage_rate","Coverage"],["unique_bugs_detected","Unique bugs"]]
    .filter(([k]) => summary[0][k + "_mean"] !== undefined);
  const bestVal = Math.max(...summary.map(r => r[metric + "_mean"]));
  el.innerHTML = `<thead><tr><th>Strategy</th>${cols.map(c => `<th>${c[1]}</th>`).join("")}</tr></thead><tbody>` +
    summary.map(r => `<tr class="${r[metric + "_mean"] === bestVal ? "best" : ""}"><td>${r.strategy}</td>` +
      cols.map(([k]) => `<td>${fmt(r[k + "_mean"])} <span class="muted">±${fmt(r[k + "_ci95"], 3)}</span></td>`).join("") + `</tr>`).join("") + "</tbody>";
}
const trendState = {};   // per-canvas hidden strategies
function renderTrend(canvasId, raw, metric, chipsEl) {
  const strategies = [...new Set(raw.map(r => r.strategy))];
  const hidden = (trendState[canvasId] ??= new Set(strategies.filter(s => /Coverage|Failure/.test(s))));
  const builds = [...new Set(raw.map(r => r.build_id))];
  const byStrat = groupBy(raw, "strategy");
  const datasets = strategies.map(s => {
    const byBuild = groupBy(byStrat.get(s), "build_id");
    return { label:s, data:builds.map(b => mean((byBuild.get(b) || []).map(r => r[metric]))),
      borderColor:colorOf(s), backgroundColor:colorOf(s), borderWidth:1.6, pointRadius:0, tension:.2, hidden:hidden.has(s) };
  });
  if (chipsEl) {
    chipsEl.innerHTML = strategies.map(s => `<span class="chip ${hidden.has(s) ? "off" : ""}" data-s="${s}"><i style="background:${colorOf(s)}"></i>${s}</span>`).join("");
    chipsEl.querySelectorAll(".chip").forEach(ch => ch.addEventListener("click", () => {
      const s = ch.dataset.s; hidden.has(s) ? hidden.delete(s) : hidden.add(s);
      renderTrend(canvasId, raw, metric, chipsEl);
    }));
  }
  drawChart(canvasId, {
    type:"line", data:{ labels:builds.map((_, i) => i + 1), datasets },
    options:baseOpts({ plugins:{ legend:{ display:!chipsEl } }, interaction:{ mode:"index", intersect:false },
      scales:{ x:{ title:{ display:true, text:"Build / cycle index" }, ticks:{ maxTicksLimit:12 } },
               y:{ title:{ display:true, text:METRIC_LABEL[metric] }, beginAtZero:true } } }),
  });
}

/* ---------- RESULTS tab ---------- */
async function renderResults() {
  const ds = DATASETS[$("datasetSelect").value], metric = $("metricSelect").value;
  try {
    const [summary, paired, raw] = await Promise.all([loadCSV(ds.summary), loadCSV(ds.paired), loadCSV(ds.raw)]);
    banner("");
    renderKpis($("kpis"), summary, metric);
    renderCompare("cmpChart", summary, metric);
    renderSummaryTable($("summaryTable"), summary, metric);
    renderTrend("trendChart", raw, metric, $("strategyChips"));
    renderPaired(paired);
  } catch (e) {
    banner(`${e.message}. Make sure your server exposes the project's results/ folder at "${CONFIG.RESULTS_BASE}".`);
  }
}
function renderPaired(rows) {
  const sign = v => `<span class="${v > 0 ? "pos" : v < 0 ? "neg" : ""}">${v > 0 ? "+" : ""}${fmt(v)}</span>`;
  $("pairedTable").innerHTML = `<thead><tr><th>Compared against</th><th>Builds</th><th>Δ BDR</th><th>Δ APFD</th><th>Δ NAPFD</th><th>Won</th><th>Tied</th><th>Lost</th></tr></thead><tbody>` +
    rows.map(r => `<tr><td>${r.compared_against}</td><td>${r.num_paired_builds}</td>
      <td>${sign(r.paired_bdr_diff_mean)} <span class="muted">±${fmt(r.paired_bdr_diff_ci95, 3)}</span></td>
      <td>${sign(r.paired_apfd_diff_mean)} <span class="muted">±${fmt(r.paired_apfd_diff_ci95, 3)}</span></td>
      <td>${sign(r.paired_napfd_diff_mean)} <span class="muted">±${fmt(r.paired_napfd_diff_ci95, 3)}</span></td>
      <td>${r.builds_won}</td><td>${r.builds_tied}</td><td>${r.builds_lost}</td></tr>`).join("") + "</tbody>";
}
$("datasetSelect").addEventListener("change", renderResults);
$("metricSelect").addEventListener("change", renderResults);

/* ---------- ABLATIONS tab ---------- */
async function renderAblations() {
  try {
    const bm = $("budgetMetric").value;
    const budget = await loadCSV("budget_sensitivity.csv");
    const ratios = [...new Set(budget.map(r => r.budget_ratio))].sort((a, b) => a - b);
    const byS = groupBy(budget, "strategy");
    drawChart("budgetChart", { type:"line",
      data:{ labels:ratios.map(r => (r * 100).toFixed(0) + "%"),
        datasets:[...byS].map(([s, rows]) => ({ label:s, borderColor:colorOf(s), backgroundColor:colorOf(s), tension:.2,
          data:ratios.map(r => mean(rows.filter(x => x.budget_ratio === r).map(x => x[bm]))) })) },
      options:baseOpts({ scales:{ x:{ title:{ display:true, text:"Time budget (% of total test time)" } }, y:{ title:{ display:true, text:METRIC_LABEL[bm] }, beginAtZero:true } } }) });
    await Promise.all([
      variantBar("rewardChart", "reward_ablation.csv", "ablation_variant"),
      variantBar("stateChart", "state_ablation.csv", "state_variant"),
      variantBar("agentChart", "agent_ablation.csv", "agent_variant"),
    ]);
    banner("");
  } catch (e) { banner(e.message); }
}
async function variantBar(id, file, key) {
  const rows = await loadCSV(file), g = groupBy(rows, key);
  const labels = [...g.keys()];
  drawChart(id, { type:"bar",
    data:{ labels, datasets:[
      { label:"NAPFD", data:labels.map(l => mean(g.get(l).map(r => r.napfd))), backgroundColor:PALETTE[4], borderRadius:5 },
      { label:"Bug detection rate", data:labels.map(l => mean(g.get(l).map(r => r.bug_detection_rate))), backgroundColor:PALETTE[3], borderRadius:5 } ] },
    options:baseOpts({ scales:{ y:{ beginAtZero:true } } }) });
}
$("budgetMetric").addEventListener("change", renderAblations);

/* ---------- TRAINING tab ---------- */
async function renderTraining() {
  const metric = $("trainMetric").value, w = +$("trainWindow").value;
  $("trainWindowVal").textContent = w;
  try {
    const rows = await loadCSV("training_history.csv");
    const eps = [...new Set(rows.map(r => r.episode))].sort((a, b) => a - b);
    const byEp = groupBy(rows, "episode");
    const raw = eps.map(e => mean(byEp.get(e).map(r => r[metric])));
    const smooth = raw.map((_, i) => mean(raw.slice(Math.max(0, i - w + 1), i + 1)));
    drawChart("trainChart", { type:"line",
      data:{ labels:eps, datasets:[
        { label:"Raw", data:raw, borderColor:"rgba(148,163,184,.5)", pointRadius:0, borderWidth:1 },
        { label:`Moving avg (${w})`, data:smooth, borderColor:PALETTE[4], pointRadius:0, borderWidth:2.2 } ] },
      options:baseOpts({ scales:{ x:{ title:{ display:true, text:"Episode" }, ticks:{ maxTicksLimit:12 } } } }) });
    banner("");
  } catch (e) { banner(e.message); }
}
$("trainMetric").addEventListener("change", renderTraining);
$("trainWindow").addEventListener("input", renderTraining);

/* ---------- FIGURES tab ---------- */
function renderFigures() {
  if ($("figureGrid").children.length) return;
  $("figureGrid").innerHTML = FIGURES.map(([f, cap]) =>
    `<figure><img loading="lazy" src="${CONFIG.RESULTS_BASE}/figures/${f}" alt="${cap}"><figcaption>${cap}</figcaption></figure>`).join("");
  $("figureGrid").addEventListener("click", e => {
    if (e.target.tagName !== "IMG") return;
    const lb = document.createElement("div"); lb.className = "lightbox";
    lb.innerHTML = `<img src="${e.target.src}">`; lb.onclick = () => lb.remove(); document.body.appendChild(lb);
  });
}

/* ---------- UPLOAD & EVALUATE ---------- */
let pickedFile = null, lastUpload = null;
const REQUIRED = ["Id","Name","Duration","Verdict","Cycle"];
const drop = $("drop"), fileInput = $("fileInput");
drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("keydown", e => { if (e.key === "Enter") fileInput.click(); });
["dragenter","dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave","drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => handleFile(e.dataTransfer.files[0]));
fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

function handleFile(file) {
  if (!file) return;
  pickedFile = file;
  $("fileName").textContent = `${file.name} (${(file.size / 1e6).toFixed(2)} MB)`;
  const pv = $("preview"); pv.classList.remove("hidden", "err");
  Papa.parse(file, { header:true, delimiter:";", preview:5000, skipEmptyLines:true, complete: res => {
    const cols = res.meta.fields || [];
    const missing = REQUIRED.filter(c => !cols.includes(c));
    if (missing.length) {
      pv.classList.add("err");
      pv.innerHTML = `Missing required column(s): <b>${missing.join(", ")}</b>. Expected a semicolon-separated file with: Id;Name;Duration;CalcPrio;LastRun;LastResults;Verdict;Cycle`;
      $("runBtn").disabled = true; return;
    }
    const rows = res.data, cycles = new Set(rows.map(r => r.Cycle)), tests = new Set(rows.map(r => r.Name));
    const fails = rows.filter(r => String(r.Verdict).trim() === "1").length;
    pv.innerHTML = `Preview of first ${rows.length.toLocaleString()} rows: <b>${cycles.size}</b> cycles, <b>${tests.size}</b> unique tests, <b>${fails}</b> failures. Full statistics are computed on the server.`;
    $("runBtn").disabled = false;
  }});
}

$("runBtn").addEventListener("click", async () => {
  if (!pickedFile) return;
  const fd = new FormData();
  fd.append("file", pickedFile);
  fd.append("budget_ratio", $("optBudget").value);
  fd.append("train_ratio", $("optSplit").value);
  fd.append("episodes", $("optEpisodes").value);
  fd.append("mode", $("optMode").value);
  $("runBtn").disabled = true; $("progress").classList.remove("hidden");
  $("progressText").textContent = $("optMode").value === "retrain" ? "Training and evaluating… this can take several minutes" : "Evaluating…";
  try {
    const res = await fetch(`${CONFIG.API_BASE}/evaluate`, { method:"POST", body:fd });
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { const j = await res.json(); msg = j.error || j.detail || msg; } catch {}
      throw new Error(msg);
    }
    lastUpload = await res.json();
    banner("");
    renderUpload();
  } catch (e) {
    banner(`Evaluation failed: ${e.message}. Is the backend running at "${CONFIG.API_BASE}/evaluate"?`);
  } finally {
    $("progress").classList.add("hidden"); $("runBtn").disabled = false;
  }
});

function renderUpload() {
  const d = lastUpload; if (!d) return;
  const metric = $("uMetric").value;
  $("uploadResults").classList.remove("hidden");
  const info = [];
  if (d.dataset) {
    info.push(["Executions", d.dataset.total_executions?.toLocaleString(), `${d.dataset.unique_test_cases} unique tests`]);
    info.push(["CI cycles", d.dataset.total_ci_cycles, `${fmt(d.dataset.avg_tests_per_cycle, 1)} tests/cycle`]);
    info.push(["Failure rate", fmt(d.dataset.failure_rate_pct, 2) + "%", `${d.dataset.total_failures} failures`]);
  }
  if (d.split) info.push(["Held-out cycles evaluated", d.split.evaluated_cycles, `${d.split.excluded_zero_fail ?? 0} zero-failure cycles excluded`]);
  const infoHtml = info.map(([l, v, s]) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v ?? "–"}</div><div class="s">${s}</div></div>`).join("");
  const tmp = document.createElement("div"); renderKpis(tmp, d.summary, metric);
  $("uKpis").innerHTML = infoHtml + tmp.innerHTML;
  renderCompare("uCmpChart", d.summary, metric);
  renderSummaryTable($("uTable"), d.summary, metric);
  if (d.per_cycle?.length) renderTrend("uTrendChart", d.per_cycle, metric, null);
  $("uploadResults").scrollIntoView({ behavior:"smooth" });
}
$("uMetric").addEventListener("change", renderUpload);
$("dlBtn").addEventListener("click", () => {
  if (!lastUpload) return;
  const csv = Papa.unparse(lastUpload.summary);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type:"text/csv" }));
  a.download = "evaluation_summary.csv"; a.click();
});

/* ---------- init ---------- */
renderResults();
