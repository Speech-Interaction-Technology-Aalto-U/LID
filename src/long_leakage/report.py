"""Create a self-contained interactive longitudinal aggregation report."""

import json
from pathlib import Path

import numpy as np

from .calibration import METHOD_METADATA, METHOD_ORDER


METHOD_STYLES = {
    "direct_mean": {
        "label": "Direct mean",
        "color": "#111827",
        "shape": "circle",
    },
    "sum_llr": {
        "label": "Summed LLR",
        "color": "#D97706",
        "shape": "diamond",
    },
    "average_llr": {
        "label": "Averaged LLR",
        "color": "#2563EB",
        "shape": "star",
    },
    "temperature_scaled": {
        "label": "Fitted temperature",
        "color": "#DC2626",
        "shape": "square",
    },
    "count_adjusted": {
        "label": "Count-adjusted",
        "color": "#0F766E",
        "shape": "triangle",
    },
    "similarity_adjusted": {
        "label": "Similarity-adjusted",
        "color": "#9333EA",
        "shape": "cross",
    },
    "custom_temperature": {
        "label": "Slider temperature",
        "color": "#E11D48",
        "shape": "ring",
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
    for method in ("direct_mean", *METHOD_ORDER):
        markers[method] = {
            "p": _ordered_method_values(comparison, speakers, method, "p"),
            "LID": _ordered_method_values(
                comparison,
                speakers,
                method,
                "LID",
            ),
        }

    fitted_temperature = calibration["methods"]["temperature_scaled"][
        "fitted_parameter"
    ]
    temperature_maximum = max(
        100.0,
        2.0 * float(dataset.counts.max()),
        3.0 * float(fitted_temperature),
        float(sweep["temperature"].iloc[-1]),
    )
    return {
        "experiment": experiment_name,
        "n_enrolments": evidence.n_enrolments,
        "speakers": speakers,
        "trial_counts": dataset.counts.astype(int).tolist(),
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
        "markers": markers,
        "method_styles": METHOD_STYLES,
        "custom_temperature": {
            "minimum": 0.05,
            "maximum": temperature_maximum,
            "fitted": fitted_temperature,
            "sum_logits": dataset.sum_llr.tolist(),
            "target_indices": dataset.target_indices.tolist(),
        },
        "calibration": calibration,
        "temperature_sweep": {
            "temperature": sweep["temperature"].astype(float).tolist(),
            "nll": sweep["development_multiclass_nll_nats"].astype(float).tolist(),
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
  --teal: #0f766e;
  --coral: #dc2626;
}
* { box-sizing: border-box; }
html { background: var(--paper); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: 0; max-width: 100%; overflow-x: hidden; }
body { margin: 0; min-width: 320px; max-width: 100%; overflow-x: hidden; }
header { border-bottom: 1px solid var(--line); padding: 22px max(20px, calc((100vw - 1500px) / 2)); }
h1 { font-size: 26px; line-height: 1.2; margin: 0; font-weight: 720; }
.kicker { color: var(--muted); font-size: 12px; font-weight: 700; margin-bottom: 5px; text-transform: uppercase; }
.status { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.status span { border: 1px solid var(--line); border-radius: 3px; color: var(--muted); font-size: 12px; padding: 4px 7px; }
main { max-width: 1500px; min-width: 0; width: 100%; margin: 0 auto; }
.controls { align-items: end; background: var(--soft); border-bottom: 1px solid var(--line); display: grid; gap: 18px; grid-template-columns: auto minmax(300px, 1fr) 130px auto; padding: 16px 20px; }
.control label, .control-title { color: var(--muted); display: block; font-size: 11px; font-weight: 750; margin-bottom: 7px; text-transform: uppercase; }
.segmented { display: inline-flex; border: 1px solid #aeb7bf; border-radius: 4px; overflow: hidden; }
.segmented button { background: white; border: 0; border-right: 1px solid #aeb7bf; color: var(--ink); cursor: pointer; font: inherit; min-height: 36px; padding: 0 13px; }
.segmented button:last-child { border-right: 0; }
.segmented button.active { background: var(--ink); color: white; }
.slider-row { align-items: center; display: grid; gap: 10px; grid-template-columns: minmax(0, 1fr) 118px; min-width: 0; }
input[type="range"] { accent-color: var(--coral); min-width: 0; width: 100%; }
input[type="number"] { background: white; border: 1px solid #aeb7bf; border-radius: 3px; color: var(--ink); font: inherit; height: 36px; padding: 0 8px; width: 118px; }
button.command { background: white; border: 1px solid #aeb7bf; border-radius: 3px; color: var(--ink); cursor: pointer; font: inherit; height: 36px; padding: 0 12px; }
button.command:hover { border-color: var(--ink); }
.scale-value { font-variant-numeric: tabular-nums; min-height: 36px; padding-top: 8px; white-space: nowrap; }
.method-bar { border-bottom: 1px solid var(--line); display: flex; flex-wrap: wrap; gap: 8px 18px; min-width: 0; padding: 12px 20px; }
.method-toggle { align-items: center; cursor: pointer; display: inline-flex; font-size: 13px; gap: 7px; white-space: nowrap; }
.method-toggle input { accent-color: var(--ink); }
.swatch { border-radius: 2px; display: inline-block; height: 10px; width: 10px; }
.metrics { border-bottom: 1px solid var(--line); display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); }
.metric { border-right: 1px solid var(--line); min-height: 76px; padding: 13px 16px; }
.metric:last-child { border-right: 0; }
.metric-name { color: var(--muted); font-size: 11px; font-weight: 750; text-transform: uppercase; }
.metric-value { font-size: 21px; font-variant-numeric: tabular-nums; font-weight: 680; margin-top: 5px; overflow-wrap: anywhere; }
.plot-shell { overflow-x: auto; padding: 16px 12px 8px; position: relative; }
#heatmap { display: block; min-width: 1050px; width: 100%; }
.tooltip { background: rgba(23, 32, 42, 0.96); border-radius: 3px; color: white; display: none; font-size: 12px; line-height: 1.45; max-width: 280px; padding: 7px 9px; pointer-events: none; position: fixed; z-index: 20; }
.diagnostics { border-top: 1px solid var(--line); display: grid; grid-template-columns: minmax(360px, 0.9fr) minmax(560px, 1.4fr); }
.diagnostic-section { min-width: 0; padding: 20px; }
.diagnostic-section + .diagnostic-section { border-left: 1px solid var(--line); }
h2 { font-size: 16px; margin: 0 0 14px; }
#objective { display: block; height: 270px; width: 100%; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 12px; width: 100%; }
th { color: var(--muted); font-size: 10px; text-align: right; text-transform: uppercase; }
th:first-child, td:first-child { text-align: left; }
th, td { border-bottom: 1px solid var(--line); padding: 8px 7px; text-align: right; white-space: nowrap; }
td { font-variant-numeric: tabular-nums; }
footer { border-top: 1px solid var(--line); color: var(--muted); font-size: 11px; padding: 14px 20px 24px; }
@media (max-width: 900px) {
  .controls { grid-template-columns: 1fr; }
  .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .metric:nth-child(3) { border-right: 0; }
  .metric:nth-child(-n+3) { border-bottom: 1px solid var(--line); }
  .diagnostics { grid-template-columns: 1fr; }
  .diagnostic-section + .diagnostic-section { border-left: 0; border-top: 1px solid var(--line); }
}
@media (max-width: 520px) {
  h1 { font-size: 21px; }
  .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .metric:nth-child(2n) { border-right: 0; }
  .metric:nth-child(3) { border-right: 1px solid var(--line); }
  .metric:nth-child(-n+4) { border-bottom: 1px solid var(--line); }
}
</style>
</head>
<body>
<header>
  <div class="kicker">Longitudinal attacker confidence</div>
  <h1>__EXPERIMENT__ aggregation sensitivity</h1>
  <div class="status"><span>Parameters: development only</span><span>Evaluation: test speakers</span><span>Temperature: sum(LLR) / T</span></div>
</header>
<main>
  <section class="controls" aria-label="Plot controls">
    <div class="control">
      <span class="control-title">Metric</span>
      <div class="segmented" role="group" aria-label="Metric">
        <button type="button" class="active" data-metric="LID">LID</button>
        <button type="button" data-metric="p">Probability</button>
      </div>
    </div>
    <div class="control">
      <label for="temperature-slider">Slider temperature T</label>
      <div class="slider-row"><input id="temperature-slider" type="range" min="0" max="1000" value="500"><input id="temperature-number" type="number" min="0.05" step="0.01"></div>
    </div>
    <div class="control"><span class="control-title">Evidence scale</span><div id="scale-value" class="scale-value"></div></div>
    <button type="button" id="use-fitted" class="command">Use fitted T</button>
  </section>
  <section id="method-bar" class="method-bar" aria-label="Visible methods"></section>
  <section class="metrics" aria-label="Slider temperature summary">
    <div class="metric"><div class="metric-name">Temperature</div><div id="metric-temperature" class="metric-value"></div></div>
    <div class="metric"><div class="metric-name">Mean target p</div><div id="metric-probability" class="metric-value"></div></div>
    <div class="metric"><div class="metric-name">ALID</div><div id="metric-alid" class="metric-value"></div></div>
    <div class="metric"><div class="metric-name">PDR</div><div id="metric-pdr" class="metric-value"></div></div>
    <div class="metric"><div class="metric-name">LID max</div><div id="metric-max" class="metric-value"></div></div>
    <div class="metric"><div class="metric-name">Fitted dev T</div><div id="metric-fitted" class="metric-value"></div></div>
  </section>
  <section class="plot-shell"><svg id="heatmap" role="img" aria-label="Longitudinal disclosure by trial speaker"></svg></section>
  <section class="diagnostics">
    <div class="diagnostic-section"><h2>Development temperature objective</h2><svg id="objective" role="img" aria-label="Development multiclass log loss by temperature"></svg></div>
    <div class="diagnostic-section"><h2>Development calibration diagnostics</h2><div class="table-wrap"><table><thead><tr><th>Method</th><th>Parameter</th><th>Value</th><th>Fit NLL</th><th>LOO NLL</th><th>LOO Brier</th></tr></thead><tbody id="diagnostic-rows"></tbody></table></div></div>
  </section>
  <footer>Custom slider values are sensitivity results, not fitted parameters. NLL is multiclass negative log likelihood in nats; LOO holds out one development trial speaker at a time.</footer>
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
  const slider = document.getElementById("temperature-slider");
  const number = document.getElementById("temperature-number");
  const tMin = data.custom_temperature.minimum;
  const tMax = data.custom_temperature.maximum;
  const fittedT = data.custom_temperature.fitted;
  let metric = "LID";
  let currentT = fittedT;
  const visible = new Set(Object.keys(data.method_styles));
  const markerOffsets = [-9, -6, -3, 0, 3, 6, 9];

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
  function customValues() {
    const probabilities = [];
    const lids = [];
    data.custom_temperature.sum_logits.forEach((row, index) => {
      const scaled = row.map(value => value / currentT);
      const logDenominator = logSumExp(scaled);
      const targetLogP = scaled[data.custom_temperature.target_indices[index]] - logDenominator;
      probabilities.push(Math.exp(targetLogP));
      lids.push(Math.log2(data.n_enrolments) + targetLogP / Math.log(2));
    });
    return {p: probabilities, LID: lids};
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
  function marker(shape, x, y, color) {
    const group = node("g");
    if (shape === "circle") group.append(node("circle", {cx:x, cy:y, r:5.2, fill:"white", stroke:color, "stroke-width":2}));
    if (shape === "diamond") group.append(node("polygon", {points:`${x},${y-6} ${x+6},${y} ${x},${y+6} ${x-6},${y}`, fill:"white", stroke:color, "stroke-width":2}));
    if (shape === "star") group.append(node("polygon", {points:starPoints(x,y,7,3), fill:"white", stroke:color, "stroke-width":1.8}));
    if (shape === "square") group.append(node("rect", {x:x-5, y:y-5, width:10, height:10, fill:"white", stroke:color, "stroke-width":2}));
    if (shape === "triangle") group.append(node("polygon", {points:`${x},${y-6} ${x+6},${y+5} ${x-6},${y+5}`, fill:"white", stroke:color, "stroke-width":2}));
    if (shape === "cross") {
      group.append(node("line", {x1:x-5, y1:y-5, x2:x+5, y2:y+5, stroke:color, "stroke-width":2.4}));
      group.append(node("line", {x1:x+5, y1:y-5, x2:x-5, y2:y+5, stroke:color, "stroke-width":2.4}));
    }
    if (shape === "ring") group.append(node("circle", {cx:x, cy:y, r:6, fill:"none", stroke:color, "stroke-width":2.8}));
    group.append(node("circle", {cx:x, cy:y, r:10, fill:"transparent"}));
    return group;
  }
  function showTooltip(event, lines) {
    tooltip.innerHTML = lines.join("<br>");
    tooltip.style.display = "block";
    tooltip.style.left = `${Math.min(window.innerWidth - 295, event.clientX + 12)}px`;
    tooltip.style.top = `${Math.max(8, event.clientY - 14)}px`;
  }
  function hideTooltip() { tooltip.style.display = "none"; }
  function drawHeatmap(custom) {
    svg.replaceChildren();
    const width = 1440;
    const rowHeight = 34;
    const margin = {left:170, right:110, top:34, bottom:76};
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = rowHeight * data.speakers.length;
    const height = margin.top + plotHeight + margin.bottom;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const domain = data.domains[metric];
    const edges = data.bin_edges[metric];
    const histograms = data.histograms[metric];
    const maxFrequency = Math.max(...histograms.flat());
    const x = value => margin.left + (value - domain[0]) / (domain[1] - domain[0]) * plotWidth;

    svg.append(node("rect", {x:margin.left, y:margin.top, width:plotWidth, height:plotHeight, fill:"#f8fafc", stroke:"#26323d", "stroke-width":1}));
    histograms.forEach((row, rowIndex) => {
      row.forEach((frequency, binIndex) => {
        const left = x(edges[binIndex]);
        const right = x(edges[binIndex + 1]);
        const rect = node("rect", {x:left, y:margin.top + rowIndex * rowHeight, width:Math.max(0.4,right-left), height:rowHeight, fill:viridis(frequency/maxFrequency)});
        rect.addEventListener("mousemove", event => showTooltip(event, [`Speaker ${data.speakers[rowIndex]}`, `${data.trial_counts[rowIndex]} trials`, `Relative frequency ${format(frequency,3)}`]));
        rect.addEventListener("mouseleave", hideTooltip);
        svg.append(rect);
      });
      svg.append(node("line", {x1:margin.left, y1:margin.top+(rowIndex+1)*rowHeight, x2:margin.left+plotWidth, y2:margin.top+(rowIndex+1)*rowHeight, stroke:"rgba(255,255,255,0.20)", "stroke-width":1}));
      svg.append(node("text", {x:margin.left-10, y:margin.top+rowIndex*rowHeight+rowHeight/2+4, "text-anchor":"end", fill:"#29333d", "font-size":12}, `${data.speakers[rowIndex]} (${data.trial_counts[rowIndex]})`));
    });
    const baseline = metric === "LID" ? 0 : 1/data.n_enrolments;
    if (baseline >= domain[0] && baseline <= domain[1]) svg.append(node("line", {x1:x(baseline), y1:margin.top, x2:x(baseline), y2:margin.top+plotHeight, stroke:"#ef4444", "stroke-width":2, "stroke-dasharray":"5 5"}));
    for (let tick = 0; tick <= 8; tick += 1) {
      const value = domain[0] + tick/8*(domain[1]-domain[0]);
      const tx = x(value);
      svg.append(node("line", {x1:tx, y1:margin.top+plotHeight, x2:tx, y2:margin.top+plotHeight+7, stroke:"#26323d"}));
      svg.append(node("text", {x:tx, y:margin.top+plotHeight+25, "text-anchor":"middle", fill:"#46515c", "font-size":12}, metric === "LID" ? format(value,1) : format(value,2)));
    }
    const entries = [...Object.keys(data.markers), "custom_temperature"];
    entries.forEach((method, methodIndex) => {
      if (!visible.has(method)) return;
      const style = data.method_styles[method];
      const values = method === "custom_temperature" ? custom[metric] : data.markers[method][metric];
      values.forEach((actual, rowIndex) => {
        const plotted = Math.max(domain[0], Math.min(domain[1], actual));
        const item = marker(style.shape, x(plotted), margin.top+rowIndex*rowHeight+rowHeight/2+markerOffsets[methodIndex], style.color);
        item.style.cursor = "default";
        item.addEventListener("mousemove", event => showTooltip(event, [style.label, `Speaker ${data.speakers[rowIndex]}`, `${metric === "LID" ? "LID" : "Target p"} ${format(actual,4)}`, actual !== plotted ? "Marker clamped to plot boundary" : ""]));
        item.addEventListener("mouseleave", hideTooltip);
        svg.append(item);
      });
    });
    svg.append(node("text", {x:margin.left+plotWidth/2, y:height-18, "text-anchor":"middle", fill:"#17202a", "font-size":15, "font-weight":650}, metric === "LID" ? "Local Information Disclosure (bits)" : "Mated enrolment probability"));
    const yLabel = node("text", {x:25, y:margin.top+plotHeight/2, transform:`rotate(-90 25 ${margin.top+plotHeight/2})`, "text-anchor":"middle", fill:"#17202a", "font-size":15, "font-weight":650}, "Trial speaker");
    svg.append(yLabel);
    const barX = margin.left + plotWidth + 35;
    const defs = node("defs");
    const gradient = node("linearGradient", {id:"colorbar", x1:"0", y1:"1", x2:"0", y2:"0"});
    [[0,"rgb(68,1,84)"],[.25,"rgb(59,82,139)"],[.5,"rgb(33,145,140)"],[.75,"rgb(94,201,98)"],[1,"rgb(253,231,37)"]].forEach(([offset,color]) => gradient.append(node("stop", {offset, "stop-color":color})));
    defs.append(gradient); svg.append(defs);
    svg.append(node("rect", {x:barX, y:margin.top, width:18, height:plotHeight, fill:"url(#colorbar)", stroke:"#26323d"}));
    svg.append(node("text", {x:barX+30, y:margin.top+4, fill:"#46515c", "font-size":11}, format(maxFrequency,2)));
    svg.append(node("text", {x:barX+30, y:margin.top+plotHeight, fill:"#46515c", "font-size":11}, "0"));
    svg.append(node("text", {x:barX+75, y:margin.top+plotHeight/2, transform:`rotate(-90 ${barX+75} ${margin.top+plotHeight/2})`, "text-anchor":"middle", fill:"#17202a", "font-size":12}, "Within-speaker relative frequency"));
  }
  function drawObjective() {
    const objective = document.getElementById("objective");
    objective.replaceChildren();
    const width = 560, height = 270, margin = {left:54,right:18,top:12,bottom:42};
    objective.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const temperatures = data.temperature_sweep.temperature;
    const losses = data.temperature_sweep.nll;
    const logMin = Math.log(temperatures[0]), logMax = Math.log(temperatures[temperatures.length-1]);
    const lossMin = Math.min(...losses), lossMax = Math.max(...losses);
    const paddedMin = lossMin - .04*(lossMax-lossMin || 1), paddedMax = lossMax + .04*(lossMax-lossMin || 1);
    const x = value => margin.left + (Math.log(value)-logMin)/(logMax-logMin)*(width-margin.left-margin.right);
    const y = value => margin.top + (paddedMax-value)/(paddedMax-paddedMin)*(height-margin.top-margin.bottom);
    objective.append(node("line", {x1:margin.left,y1:height-margin.bottom,x2:width-margin.right,y2:height-margin.bottom,stroke:"#64707b"}));
    objective.append(node("line", {x1:margin.left,y1:margin.top,x2:margin.left,y2:height-margin.bottom,stroke:"#64707b"}));
    const points = temperatures.map((value,index) => `${x(value)},${y(losses[index])}`).join(" ");
    objective.append(node("polyline", {points,fill:"none",stroke:"#0f766e","stroke-width":2.4}));
    objective.append(node("line", {x1:x(fittedT),y1:margin.top,x2:x(fittedT),y2:height-margin.bottom,stroke:"#17202a","stroke-width":1.5,"stroke-dasharray":"4 4"}));
    objective.append(node("line", {id:"objective-current",x1:x(currentT),y1:margin.top,x2:x(currentT),y2:height-margin.bottom,stroke:"#e11d48","stroke-width":2}));
    [temperatures[0], fittedT, temperatures[temperatures.length-1]].forEach(value => objective.append(node("text", {x:x(value),y:height-16,"text-anchor":"middle",fill:"#5f6b76","font-size":10}, format(value,value<1?2:1))));
    objective.append(node("text", {x:width/2,y:height-1,"text-anchor":"middle",fill:"#17202a","font-size":11}, "Temperature T (log scale)"));
    objective.append(node("text", {x:12,y:height/2,transform:`rotate(-90 12 ${height/2})`,"text-anchor":"middle",fill:"#17202a","font-size":11}, "Development NLL"));
    objective.append(node("text", {x:margin.left-7,y:y(lossMin)+4,"text-anchor":"end",fill:"#5f6b76","font-size":10}, format(lossMin,3)));
  }
  function updateObjectiveLine() {
    const line = document.getElementById("objective-current");
    if (!line) return;
    const temperatures = data.temperature_sweep.temperature;
    const width = 560, left = 54, right = 18;
    const position = left + (Math.log(currentT)-Math.log(temperatures[0]))/(Math.log(temperatures[temperatures.length-1])-Math.log(temperatures[0]))*(width-left-right);
    line.setAttribute("x1", position); line.setAttribute("x2", position);
  }
  function update() {
    const custom = customValues();
    document.getElementById("scale-value").textContent = `${format(1/currentT,4)} x`;
    document.getElementById("metric-temperature").textContent = format(currentT,3);
    document.getElementById("metric-probability").textContent = format(custom.p.reduce((a,b)=>a+b,0)/custom.p.length,4);
    document.getElementById("metric-alid").textContent = `${format(custom.LID.reduce((a,b)=>a+b,0)/custom.LID.length,3)} bits`;
    document.getElementById("metric-pdr").textContent = `${format(100*custom.LID.filter(value=>value>0).length/custom.LID.length,1)}%`;
    document.getElementById("metric-max").textContent = `${format(Math.max(...custom.LID),3)} bits`;
    document.getElementById("metric-fitted").textContent = format(fittedT,3);
    drawHeatmap(custom); updateObjectiveLine();
  }
  function sliderFromTemperature(value) { return 1000*(Math.log(value)-Math.log(tMin))/(Math.log(tMax)-Math.log(tMin)); }
  function temperatureFromSlider(value) { return Math.exp(Math.log(tMin)+(value/1000)*(Math.log(tMax)-Math.log(tMin))); }
  function setTemperature(value) {
    currentT = Math.max(tMin, Math.min(tMax, Number(value)));
    slider.value = sliderFromTemperature(currentT);
    number.value = currentT.toFixed(currentT < 1 ? 3 : 2);
    update();
  }
  Object.entries(data.method_styles).forEach(([method, style]) => {
    const label = document.createElement("label"); label.className = "method-toggle";
    const checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.checked = true;
    checkbox.addEventListener("change", () => { checkbox.checked ? visible.add(method) : visible.delete(method); update(); });
    const swatch = document.createElement("span"); swatch.className = "swatch"; swatch.style.background = style.color;
    label.append(checkbox, swatch, document.createTextNode(style.label)); document.getElementById("method-bar").append(label);
  });
  document.querySelectorAll("[data-metric]").forEach(button => button.addEventListener("click", () => {
    metric = button.dataset.metric;
    document.querySelectorAll("[data-metric]").forEach(item => item.classList.toggle("active", item === button));
    update();
  }));
  slider.addEventListener("input", () => setTemperature(temperatureFromSlider(Number(slider.value))));
  number.addEventListener("change", () => setTemperature(number.value));
  document.getElementById("use-fitted").addEventListener("click", () => setTemperature(fittedT));
  const rows = document.getElementById("diagnostic-rows");
  data.calibration.methods && Object.entries(data.calibration.methods).forEach(([method, summary]) => {
    const tr = document.createElement("tr");
    const loo = summary.leave_one_speaker_out_metrics || {};
    const cells = [summary.label, summary.parameter || "fixed", summary.fitted_parameter === null ? "-" : format(summary.fitted_parameter,4), format(summary.full_development_metrics.multiclass_nll_nats,4), format(loo.multiclass_nll_nats,4), format(loo.multiclass_brier,4)];
    cells.forEach(value => { const td = document.createElement("td"); td.textContent = value; tr.append(td); }); rows.append(tr);
  });
  drawObjective(); setTemperature(fittedT);
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
