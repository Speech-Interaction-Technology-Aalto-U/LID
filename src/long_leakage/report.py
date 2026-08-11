"""Create a self-contained interactive longitudinal aggregation report."""

import json
from pathlib import Path

import numpy as np


METHOD_STYLES = {
    "direct_mean": {
        "label": "Direct mean",
        "color": "#111827",
        "shape": "circle",
    },
    "sum_llr": {
        "label": "Summed LLR (T=1)",
        "color": "#D97706",
        "shape": "diamond",
    },
    "average_llr": {
        "label": "Averaged LLR (T=k)",
        "color": "#2563EB",
        "shape": "star",
    },
    "temperature_scaled_brier": {
        "label": "Global T (Brier fit)",
        "color": "#DB2777",
        "shape": "square",
    },
    "global_temperature": {
        "label": "Global T (slider)",
        "color": "#DC2626",
        "shape": "ring",
    },
    "count_adjusted_dynamic": {
        "label": "Count-adjusted (sliders)",
        "color": "#0F766E",
        "shape": "triangle",
    },
    "similarity_adjusted_dynamic": {
        "label": "Embedding-adjusted (sliders)",
        "color": "#7E22CE",
        "shape": "cross",
    },
}


def _histograms(value_groups, edges):
    rows = []
    for values in value_groups:
        counts = np.histogram(values, bins=edges)[0].astype(float)
        rows.append((counts / counts.sum()).tolist())
    return rows


def _ordered_method_values(comparison, speakers, method, metric):
    values = (
        comparison.loc[comparison["method"] == method, ["trial_spk", metric]]
        .set_index("trial_spk")[metric]
        .reindex(speakers)
    )
    if values.isna().any():
        missing = values[values.isna()].index.tolist()
        raise ValueError(
            f"Missing {method} {metric} values for speakers: {missing[:10]}"
        )
    return values.astype(float).tolist()


def _report_payload(
    experiment_name,
    evidence,
    dataset,
    results,
    comparison,
    calibration,
    sweep,
):
    speakers = list(evidence.speaker_ids)
    raw_mated = evidence.mated_dataframe()
    raw_mated = raw_mated[raw_mated["stage"] == "before"]
    raw_lid_groups = []
    raw_probability_groups = []
    for speaker in speakers:
        speaker_rows = raw_mated[raw_mated["trial_spk"] == speaker]
        raw_lid_groups.append(speaker_rows["LID"].to_numpy(dtype=float))
        raw_probability_groups.append(speaker_rows["p"].to_numpy(dtype=float))

    central_lids = comparison.loc[
        comparison["method"] != "sum_llr", "LID"
    ].to_numpy(dtype=float)
    individual_lids = raw_mated["LID"].to_numpy(dtype=float)
    lid_minimum = float(min(individual_lids.min(), central_lids.min()))
    lid_maximum = float(np.log2(evidence.n_enrolments))
    lid_span = max(lid_maximum - lid_minimum, 1.0)
    lid_edges = np.linspace(lid_minimum - 0.025 * lid_span, lid_maximum, 51)
    probability_edges = np.linspace(0.0, 1.0, 51)

    markers = {}
    for method in (
        "direct_mean",
        "sum_llr",
        "average_llr",
        "temperature_scaled_brier",
    ):
        markers[method] = {
            "p": _ordered_method_values(comparison, speakers, method, "p"),
            "LID": _ordered_method_values(
                comparison,
                speakers,
                method,
                "LID",
            ),
        }

    method_calibration = calibration["methods"]
    return {
        "experiment": experiment_name,
        "n_enrolments": evidence.n_enrolments,
        "speakers": speakers,
        "trial_counts": dataset.counts.astype(int).tolist(),
        "embedding_redundancy": dataset.embedding_redundancy.tolist(),
        "domains": {
            "LID": [float(lid_edges[0]), float(lid_edges[-1])],
            "p": [0.0, 1.0],
        },
        "bin_edges": {
            "LID": lid_edges.tolist(),
            "p": probability_edges.tolist(),
        },
        "histograms": {
            "LID": _histograms(raw_lid_groups, lid_edges),
            "p": _histograms(raw_probability_groups, probability_edges),
        },
        "individual_values": {
            "LID": individual_lids.tolist(),
            "p": raw_mated["p"].astype(float).tolist(),
        },
        "static_markers": markers,
        "method_styles": METHOD_STYLES,
        "aggregation": {
            "sum_logits": dataset.sum_llr.tolist(),
            "target_indices": dataset.target_indices.tolist(),
            "temperature_minimum": 0.05,
            "temperature_maximum": float(sweep["temperature"].iloc[-1]),
            "global_fits": {
                "nll": method_calibration["temperature_scaled"][
                    "fitted_parameters"
                ]["temperature"],
                "brier": method_calibration["temperature_scaled_brier"][
                    "fitted_parameters"
                ]["temperature"],
            },
            "count_fit": method_calibration["count_adjusted"][
                "fitted_parameters"
            ],
            "similarity_fit": method_calibration["similarity_adjusted"][
                "fitted_parameters"
            ],
        },
        "calibration": calibration,
        "temperature_sweep": {
            "temperature": sweep["temperature"].astype(float).tolist(),
            "nll": sweep["development_multiclass_nll_nats"].astype(float).tolist(),
            "brier": sweep["development_multiclass_brier"].astype(float).tolist(),
        },
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Longitudinal LID aggregation - __EXPERIMENT__</title>
<style>
:root {
  color-scheme: light;
  --ink: #17202a;
  --muted: #5f6b76;
  --line: #d7dde2;
  --soft: #f4f6f7;
  --paper: #ffffff;
  --red: #dc2626;
  --teal: #0f766e;
  --purple: #7e22ce;
}
* { box-sizing: border-box; }
html { background: var(--paper); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: 0; max-width: 100%; overflow-x: clip; }
body { margin: 0; min-width: 320px; max-width: 100%; overflow-x: clip; }
header { border-bottom: 1px solid var(--line); padding: 20px max(20px, calc((100vw - 1500px) / 2)); }
h1 { font-size: 25px; line-height: 1.2; margin: 0; font-weight: 720; }
.kicker { color: var(--muted); font-size: 11px; font-weight: 750; margin-bottom: 5px; text-transform: uppercase; }
.status { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 11px; }
.status span { border: 1px solid var(--line); border-radius: 3px; color: var(--muted); font-size: 11px; padding: 4px 7px; }
main { max-width: 1500px; min-width: 0; width: 100%; margin: 0 auto; }
.workbench { background: var(--soft); border-bottom: 1px solid #aeb7bf; position: sticky; top: 0; z-index: 10; }
.view-row { align-items: center; border-bottom: 1px solid var(--line); display: flex; flex-wrap: wrap; gap: 16px 28px; padding: 9px 20px; }
.view-control { align-items: center; display: flex; gap: 9px; }
.control-label { color: var(--muted); font-size: 10px; font-weight: 760; text-transform: uppercase; white-space: nowrap; }
.segmented { display: inline-flex; border: 1px solid #aeb7bf; border-radius: 4px; overflow: hidden; }
.segmented button { background: white; border: 0; border-right: 1px solid #aeb7bf; color: var(--ink); cursor: pointer; font: inherit; min-height: 31px; padding: 0 11px; }
.segmented button:last-child { border-right: 0; }
.segmented button.active { background: var(--ink); color: white; }
.parameter-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); }
.parameter-group { border-right: 1px solid var(--line); min-width: 0; padding: 10px 16px 11px; }
.parameter-group:last-child { border-right: 0; }
.parameter-title { align-items: center; display: flex; font-size: 12px; font-weight: 740; gap: 7px; margin-bottom: 8px; }
.parameter-title .swatch { height: 9px; width: 9px; }
.parameter-line { align-items: center; display: grid; gap: 8px; grid-template-columns: 24px minmax(90px, 1fr) 78px; margin-top: 6px; }
.parameter-symbol { color: var(--muted); font-size: 11px; font-weight: 760; }
input[type="range"] { accent-color: currentColor; min-width: 0; width: 100%; }
input[type="number"] { background: white; border: 1px solid #aeb7bf; border-radius: 3px; color: var(--ink); font: inherit; height: 29px; padding: 0 6px; width: 78px; }
.fit-row { align-items: center; display: flex; flex-wrap: wrap; gap: 6px; justify-content: space-between; margin-top: 8px; min-height: 29px; }
.effective-range { color: var(--muted); font-size: 10px; font-variant-numeric: tabular-nums; }
button.command { background: white; border: 1px solid #aeb7bf; border-radius: 3px; color: var(--ink); cursor: pointer; font: inherit; font-size: 11px; min-height: 29px; padding: 0 9px; }
button.command:hover { border-color: var(--ink); }
.method-bar { border-bottom: 1px solid var(--line); display: flex; flex-wrap: wrap; gap: 8px 17px; min-width: 0; padding: 10px 20px; }
.method-toggle { align-items: center; cursor: pointer; display: inline-flex; font-size: 12px; gap: 6px; white-space: nowrap; }
.method-toggle input { accent-color: var(--ink); }
.swatch { border-radius: 2px; display: inline-block; flex: 0 0 auto; height: 9px; width: 9px; }
.metrics-section { border-bottom: 1px solid var(--line); padding: 12px 20px 14px; }
.section-heading { align-items: baseline; display: flex; flex-wrap: wrap; gap: 8px 16px; justify-content: space-between; margin-bottom: 8px; }
h2 { font-size: 15px; margin: 0; }
.subtle { color: var(--muted); font-size: 10px; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 11px; width: 100%; }
th { color: var(--muted); font-size: 9px; text-align: right; text-transform: uppercase; }
th:first-child, td:first-child { text-align: left; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 7px; text-align: right; white-space: nowrap; }
td { font-variant-numeric: tabular-nums; }
.method-name { align-items: center; display: inline-flex; gap: 7px; }
.plot-shell { max-width: 100%; overflow-x: auto; padding: 10px 8px 6px; position: relative; }
#heatmap { display: block; min-width: 900px; width: 100%; }
.tooltip { background: rgba(23, 32, 42, 0.96); border-radius: 3px; color: white; display: none; font-size: 11px; line-height: 1.45; max-width: 280px; padding: 7px 9px; pointer-events: none; position: fixed; z-index: 30; }
.histogram-section { border-top: 1px solid var(--line); padding: 18px 20px 22px; }
.histogram-legend { color: var(--muted); display: flex; font-size: 10px; gap: 15px; }
.histogram-grid { display: grid; gap: 0; grid-template-columns: repeat(3, minmax(0, 1fr)); }
.histogram-panel { border-right: 1px solid var(--line); min-width: 0; padding: 10px 13px 0; }
.histogram-panel:last-child { border-right: 0; }
.histogram-panel h3 { font-size: 12px; margin: 0 0 3px; }
.histogram-caption { color: var(--muted); font-size: 9px; min-height: 14px; }
.histogram { display: block; height: 220px; width: 100%; }
.diagnostics { border-top: 1px solid var(--line); display: grid; grid-template-columns: minmax(360px, 0.85fr) minmax(620px, 1.45fr); }
.diagnostic-section { min-width: 0; padding: 18px 20px; }
.diagnostic-section + .diagnostic-section { border-left: 1px solid var(--line); }
#objective { display: block; height: 250px; width: 100%; }
.explanation { color: #46515c; font-size: 10px; line-height: 1.5; margin: 8px 0 0; }
.explanation strong { color: var(--ink); }
footer { border-top: 1px solid var(--line); color: var(--muted); font-size: 10px; padding: 12px 20px 22px; }
@media (max-width: 900px) {
  .workbench { position: static; }
  .parameter-grid { grid-template-columns: 1fr; }
  .parameter-group { border-bottom: 1px solid var(--line); border-right: 0; }
  .parameter-group:last-child { border-bottom: 0; }
  .histogram-grid, .diagnostics { grid-template-columns: 1fr; }
  .histogram-panel { border-bottom: 1px solid var(--line); border-right: 0; }
  .diagnostic-section + .diagnostic-section { border-left: 0; border-top: 1px solid var(--line); }
}
@media (max-width: 520px) {
  h1 { font-size: 20px; }
  .view-row { align-items: flex-start; flex-direction: column; gap: 8px; }
  .parameter-line { grid-template-columns: 24px minmax(80px, 1fr) 82px; }
  input[type="number"] { width: 82px; }
}
</style>
</head>
<body>
<header>
  <div class="kicker">Longitudinal attacker confidence</div>
  <h1>__EXPERIMENT__ aggregation explorer</h1>
  <div class="status"><span>Parameters: development only</span><span>Evaluation: test speakers</span><span>Similarity: trial embeddings</span><span>Temperature convention: sum(LLR) / T</span></div>
</header>
<main>
  <section class="workbench" aria-label="Aggregation controls">
    <div class="view-row">
      <div class="view-control"><span class="control-label">Metric</span><div class="segmented" role="group" aria-label="Metric"><button type="button" class="active" data-metric="LID">LID</button><button type="button" data-metric="p">Probability</button></div></div>
      <div class="view-control"><label class="control-label" for="density-slider">Graph row height</label><input id="density-slider" type="range" min="14" max="34" value="17"><span id="density-value" class="effective-range">17 px</span></div>
    </div>
    <div class="parameter-grid">
      <div class="parameter-group" style="color:var(--red)">
        <div class="parameter-title"><span class="swatch" style="background:var(--red)"></span>Global temperature</div>
        <div class="parameter-line"><label class="parameter-symbol" for="global-temperature">T</label><input id="global-temperature" type="range" min="0" max="1000"><input id="global-temperature-number" type="number" min="0.05" step="0.01"></div>
        <div class="fit-row"><div><button id="global-use-nll" type="button" class="command">Use NLL fit</button> <button id="global-use-brier" type="button" class="command">Use Brier fit</button></div><span id="global-fit-values" class="effective-range"></span></div>
      </div>
      <div class="parameter-group" style="color:var(--teal)">
        <div class="parameter-title"><span class="swatch" style="background:var(--teal)"></span>Count-adjusted temperature</div>
        <div class="parameter-line"><label class="parameter-symbol" for="count-temperature">T</label><input id="count-temperature" type="range" min="0" max="1000"><input id="count-temperature-number" type="number" min="0.05" step="0.01"></div>
        <div class="parameter-line"><label class="parameter-symbol" for="count-rho">rho</label><input id="count-rho" type="range" min="0" max="1000"><input id="count-rho-number" type="number" min="0" max="1" step="0.01"></div>
        <div class="fit-row"><button id="count-use-fit" type="button" class="command">Use fitted values</button><span id="count-effective" class="effective-range"></span></div>
      </div>
      <div class="parameter-group" style="color:var(--purple)">
        <div class="parameter-title"><span class="swatch" style="background:var(--purple)"></span>Embedding-adjusted temperature</div>
        <div class="parameter-line"><label class="parameter-symbol" for="similarity-temperature">T</label><input id="similarity-temperature" type="range" min="0" max="1000"><input id="similarity-temperature-number" type="number" min="0.05" step="0.01"></div>
        <div class="parameter-line"><label class="parameter-symbol" for="similarity-rho">rho</label><input id="similarity-rho" type="range" min="0" max="1000"><input id="similarity-rho-number" type="number" min="0" max="1" step="0.01"></div>
        <div class="fit-row"><button id="similarity-use-fit" type="button" class="command">Use fitted values</button><span id="similarity-effective" class="effective-range"></span></div>
      </div>
    </div>
  </section>
  <section id="method-bar" class="method-bar" aria-label="Visible plot markers"></section>
  <section class="metrics-section">
    <div class="section-heading"><h2>Method metrics</h2><span class="subtle">Slider rows update immediately; each aggregated row represents one test speaker.</span></div>
    <div class="table-wrap"><table><thead><tr><th>Method</th><th>Current parameters</th><th>Mean target p</th><th>ALID</th><th>PDR</th><th>LID max</th></tr></thead><tbody id="metric-rows"></tbody></table></div>
  </section>
  <section class="plot-shell"><svg id="heatmap" role="img" aria-label="Longitudinal disclosure by trial speaker"></svg></section>
  <section class="histogram-section">
    <div class="section-heading"><h2>Before and after aggregation</h2><div class="histogram-legend"><span><span class="swatch" style="background:#9ca3af"></span> Individual trials</span><span>Colored bars: aggregated speakers</span></div></div>
    <div class="histogram-grid">
      <div class="histogram-panel"><h3 style="color:var(--red)">Global temperature</h3><div id="global-histogram-caption" class="histogram-caption"></div><svg id="global-histogram" class="histogram"></svg></div>
      <div class="histogram-panel"><h3 style="color:var(--teal)">Count-adjusted</h3><div id="count-histogram-caption" class="histogram-caption"></div><svg id="count-histogram" class="histogram"></svg></div>
      <div class="histogram-panel"><h3 style="color:var(--purple)">Embedding-adjusted</h3><div id="similarity-histogram-caption" class="histogram-caption"></div><svg id="similarity-histogram" class="histogram"></svg></div>
    </div>
  </section>
  <section class="diagnostics">
    <div class="diagnostic-section">
      <div class="section-heading"><h2>Development temperature objective</h2><div class="segmented" role="group" aria-label="Calibration objective"><button type="button" class="active" data-objective="nll">NLL</button><button type="button" data-objective="brier">Brier</button></div></div>
      <svg id="objective" role="img" aria-label="Development calibration objective by temperature"></svg>
      <p class="explanation"><strong>NLL</strong> is <code>-log(q_true)</code>. Every nonmated logit still matters because it appears in the softmax denominator of <code>q_true</code>. <strong>Brier</strong> explicitly sums squared probability error over all enrolment candidates.</p>
    </div>
    <div class="diagnostic-section">
      <h2>Development calibration diagnostics</h2>
      <p class="explanation"><strong>LOO expectation:</strong> fit on all but one development speaker and score the held-out speaker, repeated once per speaker. LOO loss will often be above fit loss; a large gap or a wide refit range indicates instability. Lower NLL and Brier are better.</p>
      <div class="table-wrap"><table><thead><tr><th>Method</th><th>Objective</th><th>Fitted parameters</th><th>LOO refit range</th><th>Fit NLL</th><th>LOO NLL</th><th>Fit Brier</th><th>LOO Brier</th></tr></thead><tbody id="diagnostic-rows"></tbody></table></div>
    </div>
  </section>
  <footer>Slider values are sensitivity analyses, not newly fitted evaluation results. Sum and average are fixed reference rules, not absolute upper or lower disclosure bounds. Values outside the individual-trial heatmap domain are visually clamped; tooltips report the actual value.</footer>
</main>
<div id="tooltip" class="tooltip"></div>
<script id="report-data" type="application/json">__DATA__</script>
<script>
(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("report-data").textContent);
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.getElementById("heatmap");
  const tooltip = document.getElementById("tooltip");
  const tMin = data.aggregation.temperature_minimum;
  const tMax = data.aggregation.temperature_maximum;
  const state = {
    metric: "LID",
    objective: "nll",
    rowHeight: 17,
    globalTemperature: data.aggregation.global_fits.nll,
    countTemperature: data.aggregation.count_fit.temperature,
    countRho: data.aggregation.count_fit.rho,
    similarityTemperature: data.aggregation.similarity_fit.temperature,
    similarityRho: data.aggregation.similarity_fit.rho,
  };
  const visible = new Set(Object.keys(data.method_styles));

  function node(name, attrs = {}, text = null) {
    const item = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([key, value]) => item.setAttribute(key, value));
    if (text !== null) item.textContent = text;
    return item;
  }
  function format(value, digits = 3) {
    if (value === null || value === undefined || !Number.isFinite(value)) return "n/a";
    return Number(value).toFixed(digits);
  }
  function logSumExp(values) {
    const maximum = Math.max(...values);
    return maximum + Math.log(values.reduce((sum, value) => sum + Math.exp(value - maximum), 0));
  }
  function temperatureFromRange(value) {
    return Math.exp(Math.log(tMin) + (Number(value) / 1000) * (Math.log(tMax) - Math.log(tMin)));
  }
  function rangeFromTemperature(value) {
    return 1000 * (Math.log(value) - Math.log(tMin)) / (Math.log(tMax) - Math.log(tMin));
  }
  function aggregate(temperature, rho, factors) {
    const probabilities = [], lids = [], effectiveTemperatures = [];
    data.aggregation.sum_logits.forEach((row, index) => {
      const effective = temperature * (1 + rho * factors[index]);
      const scaled = row.map(value => value / effective);
      const logDenominator = logSumExp(scaled);
      const targetLogP = scaled[data.aggregation.target_indices[index]] - logDenominator;
      effectiveTemperatures.push(effective);
      probabilities.push(Math.exp(targetLogP));
      lids.push(Math.log2(data.n_enrolments) + targetLogP / Math.log(2));
    });
    return {p: probabilities, LID: lids, temperatures: effectiveTemperatures};
  }
  function currentSeries() {
    const zeros = data.trial_counts.map(() => 0);
    return {
      ...data.static_markers,
      global_temperature: aggregate(state.globalTemperature, 0, zeros),
      count_adjusted_dynamic: aggregate(state.countTemperature, state.countRho, data.trial_counts.map(value => value - 1)),
      similarity_adjusted_dynamic: aggregate(state.similarityTemperature, state.similarityRho, data.embedding_redundancy),
    };
  }
  function viridis(value) {
    const stops = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
    const position = Math.max(0, Math.min(1, value)) * (stops.length - 1);
    const index = Math.min(stops.length - 2, Math.floor(position));
    const fraction = position - index;
    const color = stops[index].map((channel, i) => Math.round(channel + fraction * (stops[index + 1][i] - channel)));
    return `rgb(${color.join(",")})`;
  }
  function starPoints(x, y, outer, inner) {
    const points = [];
    for (let i = 0; i < 10; i += 1) {
      const radius = i % 2 === 0 ? outer : inner;
      const angle = -Math.PI / 2 + i * Math.PI / 5;
      points.push(`${x + radius * Math.cos(angle)},${y + radius * Math.sin(angle)}`);
    }
    return points.join(" ");
  }
  function marker(shape, x, y, color, scale) {
    const group = node("g");
    const radius = Math.max(3.2, Math.min(5.7, scale));
    if (shape === "circle") group.append(node("circle", {cx:x, cy:y, r:radius, fill:"white", stroke:color, "stroke-width":1.8}));
    if (shape === "diamond") group.append(node("polygon", {points:`${x},${y-radius-1} ${x+radius+1},${y} ${x},${y+radius+1} ${x-radius-1},${y}`, fill:"white", stroke:color, "stroke-width":1.8}));
    if (shape === "star") group.append(node("polygon", {points:starPoints(x,y,radius+1.5,Math.max(2,radius/2)), fill:"white", stroke:color, "stroke-width":1.6}));
    if (shape === "square") group.append(node("rect", {x:x-radius, y:y-radius, width:2*radius, height:2*radius, fill:"white", stroke:color, "stroke-width":1.8}));
    if (shape === "triangle") group.append(node("polygon", {points:`${x},${y-radius-1} ${x+radius+1},${y+radius} ${x-radius-1},${y+radius}`, fill:"white", stroke:color, "stroke-width":1.8}));
    if (shape === "cross") {
      group.append(node("line", {x1:x-radius, y1:y-radius, x2:x+radius, y2:y+radius, stroke:color, "stroke-width":2.2}));
      group.append(node("line", {x1:x+radius, y1:y-radius, x2:x-radius, y2:y+radius, stroke:color, "stroke-width":2.2}));
    }
    if (shape === "ring") group.append(node("circle", {cx:x, cy:y, r:radius+0.6, fill:"none", stroke:color, "stroke-width":2.5}));
    group.append(node("circle", {cx:x, cy:y, r:Math.max(8,radius+3), fill:"transparent"}));
    return group;
  }
  function showTooltip(event, lines) {
    tooltip.innerHTML = lines.filter(Boolean).join("<br>");
    tooltip.style.display = "block";
    tooltip.style.left = `${Math.min(window.innerWidth - 295, event.clientX + 12)}px`;
    tooltip.style.top = `${Math.max(8, event.clientY - 14)}px`;
  }
  function hideTooltip() { tooltip.style.display = "none"; }
  function drawHeatmap(series) {
    svg.replaceChildren();
    const width = 1440;
    const rowHeight = state.rowHeight;
    const margin = {left:170, right:112, top:28, bottom:68};
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = rowHeight * data.speakers.length;
    const height = margin.top + plotHeight + margin.bottom;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const domain = data.domains[state.metric];
    const edges = data.bin_edges[state.metric];
    const histograms = data.histograms[state.metric];
    const maxFrequency = Math.max(...histograms.flat());
    const x = value => margin.left + (value - domain[0]) / (domain[1] - domain[0]) * plotWidth;

    svg.append(node("rect", {x:margin.left, y:margin.top, width:plotWidth, height:plotHeight, fill:"#f8fafc", stroke:"#26323d", "stroke-width":1}));
    histograms.forEach((row, rowIndex) => {
      row.forEach((frequency, binIndex) => {
        const left = x(edges[binIndex]), right = x(edges[binIndex + 1]);
        const rect = node("rect", {x:left, y:margin.top + rowIndex * rowHeight, width:Math.max(0.4,right-left), height:rowHeight, fill:viridis(frequency/maxFrequency)});
        rect.addEventListener("mousemove", event => showTooltip(event, [`Speaker ${data.speakers[rowIndex]}`, `${data.trial_counts[rowIndex]} trials`, `Relative frequency ${format(frequency,3)}`]));
        rect.addEventListener("mouseleave", hideTooltip); svg.append(rect);
      });
      svg.append(node("line", {x1:margin.left, y1:margin.top+(rowIndex+1)*rowHeight, x2:margin.left+plotWidth, y2:margin.top+(rowIndex+1)*rowHeight, stroke:"rgba(255,255,255,0.20)", "stroke-width":1}));
      svg.append(node("text", {x:margin.left-10, y:margin.top+rowIndex*rowHeight+rowHeight/2+3.5, "text-anchor":"end", fill:"#29333d", "font-size":Math.max(8,Math.min(11,rowHeight*.48))}, `${data.speakers[rowIndex]} (${data.trial_counts[rowIndex]})`));
    });
    const baseline = state.metric === "LID" ? 0 : 1/data.n_enrolments;
    if (baseline >= domain[0] && baseline <= domain[1]) svg.append(node("line", {x1:x(baseline), y1:margin.top, x2:x(baseline), y2:margin.top+plotHeight, stroke:"#ef4444", "stroke-width":1.8, "stroke-dasharray":"5 5"}));
    for (let tick = 0; tick <= 8; tick += 1) {
      const value = domain[0] + tick/8*(domain[1]-domain[0]);
      const tx = x(value);
      svg.append(node("line", {x1:tx, y1:margin.top+plotHeight, x2:tx, y2:margin.top+plotHeight+6, stroke:"#26323d"}));
      svg.append(node("text", {x:tx, y:margin.top+plotHeight+22, "text-anchor":"middle", fill:"#46515c", "font-size":11}, state.metric === "LID" ? format(value,1) : format(value,2)));
    }
    const entries = Object.keys(data.method_styles);
    const offsetStep = Math.min(2.7, Math.max(1.2, rowHeight/8));
    const center = (entries.length - 1) / 2;
    entries.forEach((method, methodIndex) => {
      if (!visible.has(method)) return;
      const style = data.method_styles[method], values = series[method][state.metric];
      values.forEach((actual, rowIndex) => {
        const plotted = Math.max(domain[0], Math.min(domain[1], actual));
        const y = margin.top + rowIndex*rowHeight + rowHeight/2 + (methodIndex-center)*offsetStep;
        const item = marker(style.shape, x(plotted), y, style.color, rowHeight*.22);
        item.addEventListener("mousemove", event => showTooltip(event, [style.label, `Speaker ${data.speakers[rowIndex]}`, `${state.metric === "LID" ? "LID" : "Target p"} ${format(actual,4)}`, actual !== plotted ? "Marker clamped to plot boundary" : ""]));
        item.addEventListener("mouseleave", hideTooltip); svg.append(item);
      });
    });
    svg.append(node("text", {x:margin.left+plotWidth/2, y:height-16, "text-anchor":"middle", fill:"#17202a", "font-size":14, "font-weight":650}, state.metric === "LID" ? "Local Information Disclosure (bits)" : "Mated enrolment probability"));
    svg.append(node("text", {x:24, y:margin.top+plotHeight/2, transform:`rotate(-90 24 ${margin.top+plotHeight/2})`, "text-anchor":"middle", fill:"#17202a", "font-size":14, "font-weight":650}, "Trial speaker"));
    const barX = margin.left + plotWidth + 32;
    const defs = node("defs"), gradient = node("linearGradient", {id:"colorbar", x1:"0", y1:"1", x2:"0", y2:"0"});
    [[0,"rgb(68,1,84)"],[.25,"rgb(59,82,139)"],[.5,"rgb(33,145,140)"],[.75,"rgb(94,201,98)"],[1,"rgb(253,231,37)"]].forEach(([offset,color]) => gradient.append(node("stop", {offset, "stop-color":color})));
    defs.append(gradient); svg.append(defs);
    svg.append(node("rect", {x:barX, y:margin.top, width:17, height:plotHeight, fill:"url(#colorbar)", stroke:"#26323d"}));
    svg.append(node("text", {x:barX+27, y:margin.top+4, fill:"#46515c", "font-size":10}, format(maxFrequency,2)));
    svg.append(node("text", {x:barX+27, y:margin.top+plotHeight, fill:"#46515c", "font-size":10}, "0"));
    svg.append(node("text", {x:barX+70, y:margin.top+plotHeight/2, transform:`rotate(-90 ${barX+70} ${margin.top+plotHeight/2})`, "text-anchor":"middle", fill:"#17202a", "font-size":11}, "Within-speaker relative frequency"));
  }
  function methodMetrics(values) {
    return {
      meanP: values.p.reduce((a,b)=>a+b,0)/values.p.length,
      alid: values.LID.reduce((a,b)=>a+b,0)/values.LID.length,
      pdr: values.LID.filter(value=>value>0).length/values.LID.length,
      maximum: Math.max(...values.LID),
    };
  }
  function parameterText(method) {
    if (method === "direct_mean") return "mean of trial metrics";
    if (method === "sum_llr") return "T = 1";
    if (method === "average_llr") return "T = k";
    if (method === "temperature_scaled_brier") return `T = ${format(data.aggregation.global_fits.brier,3)}`;
    if (method === "global_temperature") return `T = ${format(state.globalTemperature,3)}`;
    if (method === "count_adjusted_dynamic") return `T = ${format(state.countTemperature,3)}, rho = ${format(state.countRho,3)}`;
    return `T = ${format(state.similarityTemperature,3)}, rho = ${format(state.similarityRho,3)}`;
  }
  function updateMetricTable(series) {
    const body = document.getElementById("metric-rows"); body.replaceChildren();
    Object.keys(data.method_styles).forEach(method => {
      const style = data.method_styles[method], metrics = methodMetrics(series[method]);
      const row = document.createElement("tr");
      const name = document.createElement("td");
      const wrapper = document.createElement("span"); wrapper.className="method-name";
      const swatch = document.createElement("span"); swatch.className="swatch"; swatch.style.background=style.color;
      wrapper.append(swatch,document.createTextNode(style.label)); name.append(wrapper); row.append(name);
      [parameterText(method),format(metrics.meanP,4),`${format(metrics.alid,3)} bits`,`${format(100*metrics.pdr,1)}%`,`${format(metrics.maximum,3)} bits`].forEach(value=>{const td=document.createElement("td");td.textContent=value;row.append(td);});
      body.append(row);
    });
  }
  function histogramCounts(values, domain, nBins) {
    const counts = Array(nBins).fill(0), span = domain[1]-domain[0];
    values.forEach(value => {
      const index = Math.max(0,Math.min(nBins-1,Math.floor((value-domain[0])/span*nBins)));
      counts[index] += 1/values.length;
    });
    return counts;
  }
  function drawHistogram(elementId, after, color) {
    const chart = document.getElementById(elementId); chart.replaceChildren();
    const width=500,height=220,margin={left:44,right:12,top:12,bottom:38},nBins=32;
    chart.setAttribute("viewBox",`0 0 ${width} ${height}`);
    const domain=data.domains[state.metric], before=histogramCounts(data.individual_values[state.metric],domain,nBins), aggregated=histogramCounts(after[state.metric],domain,nBins);
    const maximum=Math.max(...before,...aggregated,0.01),plotWidth=width-margin.left-margin.right,plotHeight=height-margin.top-margin.bottom;
    const x=index=>margin.left+index/nBins*plotWidth,y=value=>margin.top+(maximum-value)/maximum*plotHeight,binWidth=plotWidth/nBins;
    chart.append(node("line",{x1:margin.left,y1:margin.top+plotHeight,x2:width-margin.right,y2:margin.top+plotHeight,stroke:"#64707b"}));
    before.forEach((value,index)=>chart.append(node("rect",{x:x(index),y:y(value),width:Math.max(1,binWidth-.5),height:margin.top+plotHeight-y(value),fill:"#9ca3af",opacity:.42})));
    aggregated.forEach((value,index)=>chart.append(node("rect",{x:x(index)+binWidth*.19,y:y(value),width:Math.max(1,binWidth*.62),height:margin.top+plotHeight-y(value),fill:color,opacity:.70})));
    for(let tick=0;tick<=4;tick+=1){const value=domain[0]+tick/4*(domain[1]-domain[0]),tx=margin.left+tick/4*plotWidth;chart.append(node("text",{x:tx,y:height-17,"text-anchor":"middle",fill:"#5f6b76","font-size":9},state.metric==="LID"?format(value,1):format(value,2)));}
    chart.append(node("text",{x:margin.left-6,y:margin.top+4,"text-anchor":"end",fill:"#5f6b76","font-size":9},`${format(100*maximum,1)}%`));
    chart.append(node("text",{x:width/2,y:height-2,"text-anchor":"middle",fill:"#17202a","font-size":10},state.metric==="LID"?"LID (bits)":"Target probability"));
  }
  function drawHistograms(series) {
    drawHistogram("global-histogram",series.global_temperature,"#dc2626");
    drawHistogram("count-histogram",series.count_adjusted_dynamic,"#0f766e");
    drawHistogram("similarity-histogram",series.similarity_adjusted_dynamic,"#7e22ce");
    document.getElementById("global-histogram-caption").textContent=`T=${format(state.globalTemperature,3)}`;
    document.getElementById("count-histogram-caption").textContent=`T=${format(state.countTemperature,3)}, rho=${format(state.countRho,3)}`;
    document.getElementById("similarity-histogram-caption").textContent=`T=${format(state.similarityTemperature,3)}, rho=${format(state.similarityRho,3)}`;
  }
  function drawObjective() {
    const chart=document.getElementById("objective");chart.replaceChildren();
    const width=560,height=250,margin={left:55,right:18,top:12,bottom:40};
    chart.setAttribute("viewBox",`0 0 ${width} ${height}`);
    const temperatures=data.temperature_sweep.temperature,losses=data.temperature_sweep[state.objective];
    const logMin=Math.log(temperatures[0]),logMax=Math.log(temperatures[temperatures.length-1]),lossMin=Math.min(...losses),lossMax=Math.max(...losses),paddedMin=lossMin-.04*(lossMax-lossMin||1),paddedMax=lossMax+.04*(lossMax-lossMin||1);
    const x=value=>margin.left+(Math.log(value)-logMin)/(logMax-logMin)*(width-margin.left-margin.right),y=value=>margin.top+(paddedMax-value)/(paddedMax-paddedMin)*(height-margin.top-margin.bottom);
    chart.append(node("line",{x1:margin.left,y1:height-margin.bottom,x2:width-margin.right,y2:height-margin.bottom,stroke:"#64707b"}));chart.append(node("line",{x1:margin.left,y1:margin.top,x2:margin.left,y2:height-margin.bottom,stroke:"#64707b"}));
    chart.append(node("polyline",{points:temperatures.map((value,index)=>`${x(value)},${y(losses[index])}`).join(" "),fill:"none",stroke:state.objective==="nll"?"#0f766e":"#db2777","stroke-width":2.3}));
    const fitted=state.objective==="nll"?data.aggregation.global_fits.nll:data.aggregation.global_fits.brier;
    chart.append(node("line",{x1:x(fitted),y1:margin.top,x2:x(fitted),y2:height-margin.bottom,stroke:"#17202a","stroke-width":1.5,"stroke-dasharray":"4 4"}));
    chart.append(node("line",{x1:x(state.globalTemperature),y1:margin.top,x2:x(state.globalTemperature),y2:height-margin.bottom,stroke:"#dc2626","stroke-width":2}));
    [temperatures[0],fitted,temperatures[temperatures.length-1]].forEach(value=>chart.append(node("text",{x:x(value),y:height-15,"text-anchor":"middle",fill:"#5f6b76","font-size":9},format(value,value<1?2:1))));
    chart.append(node("text",{x:width/2,y:height-1,"text-anchor":"middle",fill:"#17202a","font-size":10},"Temperature T (log scale)"));
    chart.append(node("text",{x:12,y:height/2,transform:`rotate(-90 12 ${height/2})`,"text-anchor":"middle",fill:"#17202a","font-size":10},state.objective==="nll"?"Development NLL":"Development Brier"));
    chart.append(node("text",{x:margin.left-7,y:y(lossMin)+3,"text-anchor":"end",fill:"#5f6b76","font-size":9},format(lossMin,3)));
  }
  function updateEffectiveRanges(series) {
    const count=series.count_adjusted_dynamic.temperatures,similarity=series.similarity_adjusted_dynamic.temperatures;
    document.getElementById("count-effective").textContent=`effective T ${format(Math.min(...count),2)}-${format(Math.max(...count),2)}`;
    document.getElementById("similarity-effective").textContent=`effective T ${format(Math.min(...similarity),2)}-${format(Math.max(...similarity),2)}`;
  }
  function update() {
    const series=currentSeries();
    updateEffectiveRanges(series);updateMetricTable(series);drawHeatmap(series);drawHistograms(series);drawObjective();
  }
  function syncTemperature(prefix,value) {
    const clamped=Math.max(tMin,Math.min(tMax,Number(value)));
    document.getElementById(`${prefix}-temperature`).value=rangeFromTemperature(clamped);
    document.getElementById(`${prefix}-temperature-number`).value=clamped.toFixed(clamped<1?3:2);
    return clamped;
  }
  function syncRho(prefix,value) {
    const clamped=Math.max(0,Math.min(1,Number(value)));
    document.getElementById(`${prefix}-rho`).value=1000*clamped;
    document.getElementById(`${prefix}-rho-number`).value=clamped.toFixed(3);
    return clamped;
  }
  function bindTemperature(prefix,stateKey) {
    document.getElementById(`${prefix}-temperature`).addEventListener("input",event=>{state[stateKey]=syncTemperature(prefix,temperatureFromRange(event.target.value));update();});
    document.getElementById(`${prefix}-temperature-number`).addEventListener("change",event=>{state[stateKey]=syncTemperature(prefix,event.target.value);update();});
  }
  function bindRho(prefix,stateKey) {
    document.getElementById(`${prefix}-rho`).addEventListener("input",event=>{state[stateKey]=syncRho(prefix,Number(event.target.value)/1000);update();});
    document.getElementById(`${prefix}-rho-number`).addEventListener("change",event=>{state[stateKey]=syncRho(prefix,event.target.value);update();});
  }
  Object.entries(data.method_styles).forEach(([method,style])=>{const label=document.createElement("label");label.className="method-toggle";const checkbox=document.createElement("input");checkbox.type="checkbox";checkbox.checked=true;checkbox.addEventListener("change",()=>{checkbox.checked?visible.add(method):visible.delete(method);update();});const swatch=document.createElement("span");swatch.className="swatch";swatch.style.background=style.color;label.append(checkbox,swatch,document.createTextNode(style.label));document.getElementById("method-bar").append(label);});
  document.querySelectorAll("[data-metric]").forEach(button=>button.addEventListener("click",()=>{state.metric=button.dataset.metric;document.querySelectorAll("[data-metric]").forEach(item=>item.classList.toggle("active",item===button));update();}));
  document.querySelectorAll("[data-objective]").forEach(button=>button.addEventListener("click",()=>{state.objective=button.dataset.objective;document.querySelectorAll("[data-objective]").forEach(item=>item.classList.toggle("active",item===button));drawObjective();}));
  document.getElementById("density-slider").addEventListener("input",event=>{state.rowHeight=Number(event.target.value);document.getElementById("density-value").textContent=`${state.rowHeight} px`;drawHeatmap(currentSeries());});
  bindTemperature("global","globalTemperature");bindTemperature("count","countTemperature");bindTemperature("similarity","similarityTemperature");bindRho("count","countRho");bindRho("similarity","similarityRho");
  document.getElementById("global-use-nll").addEventListener("click",()=>{state.globalTemperature=syncTemperature("global",data.aggregation.global_fits.nll);update();});
  document.getElementById("global-use-brier").addEventListener("click",()=>{state.globalTemperature=syncTemperature("global",data.aggregation.global_fits.brier);update();});
  document.getElementById("count-use-fit").addEventListener("click",()=>{state.countTemperature=syncTemperature("count",data.aggregation.count_fit.temperature);state.countRho=syncRho("count",data.aggregation.count_fit.rho);update();});
  document.getElementById("similarity-use-fit").addEventListener("click",()=>{state.similarityTemperature=syncTemperature("similarity",data.aggregation.similarity_fit.temperature);state.similarityRho=syncRho("similarity",data.aggregation.similarity_fit.rho);update();});
  const diagnostics=document.getElementById("diagnostic-rows");
  Object.values(data.calibration.methods).forEach(summary=>{const tr=document.createElement("tr"),loo=summary.leave_one_speaker_out_metrics||{},parameters=Object.entries(summary.fitted_parameters).map(([name,value])=>`${name}=${format(value,4)}`).join(", ")||"fixed",stability=summary.leave_one_speaker_out_parameter_stability,range=stability?Object.entries(stability).map(([name,values])=>`${name} ${format(values.minimum,3)}-${format(values.maximum,3)}`).join(", "):"fixed";[summary.label,summary.fit_objective?summary.fit_objective.replace("multiclass_",""):"fixed",parameters,range,format(summary.full_development_metrics.multiclass_nll_nats,4),format(loo.multiclass_nll_nats,4),format(summary.full_development_metrics.multiclass_brier,4),format(loo.multiclass_brier,4)].forEach(value=>{const td=document.createElement("td");td.textContent=value;tr.append(td);});diagnostics.append(tr);});
  document.getElementById("global-fit-values").textContent=`NLL ${format(data.aggregation.global_fits.nll,2)} | Brier ${format(data.aggregation.global_fits.brier,2)}`;
  state.globalTemperature=syncTemperature("global",state.globalTemperature);state.countTemperature=syncTemperature("count",state.countTemperature);state.countRho=syncRho("count",state.countRho);state.similarityTemperature=syncTemperature("similarity",state.similarityTemperature);state.similarityRho=syncRho("similarity",state.similarityRho);update();
})();
</script>
</body>
</html>
"""


def write_interactive_report(
    output_path,
    experiment_name,
    evidence,
    dataset,
    results,
    comparison,
    calibration,
    sweep,
):
    """Write the interactive report as a dependency-free HTML document."""
    output_path = Path(output_path)
    payload = _report_payload(
        experiment_name,
        evidence,
        dataset,
        results,
        comparison,
        calibration,
        sweep,
    )
    serialized = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__EXPERIMENT__", experiment_name).replace(
        "__DATA__",
        serialized,
    )
    output_path.write_text(html)
    return output_path
