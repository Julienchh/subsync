"""Local Web UI for SubSync."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import urllib.parse
import webbrowser

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUN_PY = _REPO_ROOT / "run.py"

_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>subsync</title>
<style>
  :root {
    --bg: #09090b;
    --surface: #111113;
    --surface-hover: #18181b;
    --surface-subtle: #141417;
    --border: #232329;
    --border-strong: #32323a;
    --text: #f4f4f5;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --primary: #e4e4e7;
    --primary-fg: #09090b;
    --ok: #10b981;
    --ok-bg: rgba(16, 185, 129, 0.1);
    --warn: #f59e0b;
    --warn-bg: rgba(245, 158, 11, 0.1);
    --err: #ef4444;
    --err-bg: rgba(239, 68, 68, 0.1);
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    font-size: 13px;
    line-height: 1.45;
    -webkit-font-smoothing: antialiased;
  }
  header {
    height: 52px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .brand {
    display: flex;
    align-items: baseline;
    gap: 12px;
  }
  .brand-title {
    font-family: var(--mono);
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text);
  }
  .brand-sub {
    font-size: 12px;
    color: var(--text-muted);
  }
  .toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .layout {
    max-width: 1560px;
    margin: 0 auto;
    padding: 20px 24px;
    display: grid;
    grid-template-columns: 360px 1fr;
    gap: 20px;
    align-items: start;
  }
  @media (max-width: 1080px) {
    .layout { grid-template-columns: 1fr; }
  }
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }
  .panel-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface-subtle);
  }
  .panel-title {
    margin: 0;
    font-size: 12px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: 0.01em;
  }
  .panel-body {
    padding: 16px;
  }
  .field {
    margin-bottom: 12px;
  }
  .field:last-child {
    margin-bottom: 0;
  }
  label {
    display: block;
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 5px;
    font-weight: 500;
  }
  input[type="text"], input[type="number"], select {
    width: 100%;
    height: 32px;
    padding: 0 10px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    font-size: 12.5px;
    font-family: inherit;
    outline: none;
    transition: border-color 0.12s;
  }
  input[type="text"]:focus, input[type="number"]:focus, select:focus {
    border-color: var(--border-strong);
  }
  .mono-input {
    font-family: var(--mono);
    font-size: 12px;
  }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .grid-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
  }
  .input-group {
    display: flex;
    gap: 6px;
  }
  .input-group input {
    flex: 1;
  }
  .checkbox-list {
    margin: 14px 0;
    padding: 10px 12px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    display: flex;
    flex-direction: column;
    gap: 7px;
  }
  .checkbox-list label {
    margin: 0;
    color: var(--text);
    font-weight: 400;
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    user-select: none;
  }
  input[type="checkbox"] {
    accent-color: var(--text);
    margin: 0;
  }
  code {
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--text-secondary);
  }
  button {
    height: 32px;
    padding: 0 12px;
    border-radius: 4px;
    border: 1px solid var(--border);
    background: var(--surface-hover);
    color: var(--text);
    font-size: 12.5px;
    font-weight: 500;
    font-family: inherit;
    cursor: pointer;
    transition: background 0.12s, border-color 0.12s;
    white-space: nowrap;
  }
  button:hover {
    background: #222227;
    border-color: var(--border-strong);
  }
  .btn-primary {
    width: 100%;
    height: 34px;
    background: var(--primary);
    color: var(--primary-fg);
    border-color: var(--primary);
    font-weight: 600;
  }
  .btn-primary:hover {
    background: #ffffff;
    border-color: #ffffff;
  }
  .btn-danger {
    width: 100%;
    height: 34px;
    background: var(--err-bg);
    color: var(--err);
    border-color: rgba(239, 68, 68, 0.35);
    margin-top: 8px;
  }
  .btn-danger:hover {
    background: rgba(239, 68, 68, 0.18);
  }
  .metrics-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }
  .metric-cell {
    padding: 12px 16px;
    border-right: 1px solid var(--border);
  }
  .metric-cell:last-child {
    border-right: none;
  }
  .metric-label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
  }
  .metric-value {
    font-family: var(--mono);
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
    font-variant-numeric: tabular-nums;
  }
  .table-container {
    overflow-x: auto;
    max-height: 620px;
    overflow-y: auto;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12.5px;
    font-variant-numeric: tabular-nums;
  }
  th, td {
    padding: 9px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }
  th {
    background: var(--surface);
    color: var(--text-secondary);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    position: sticky;
    top: 0;
    z-index: 1;
    border-bottom: 1px solid var(--border-strong);
  }
  tbody tr:hover td {
    background: var(--surface-hover);
  }
  .ep-code {
    font-family: var(--mono);
    font-weight: 600;
    color: var(--text);
  }
  .ep-title {
    color: var(--text-secondary);
    margin-left: 6px;
  }
  .tag {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 3px;
    font-family: var(--mono);
    font-size: 11px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text-secondary);
  }
  .tag-ok {
    background: var(--ok-bg);
    color: var(--ok);
    border-color: rgba(16, 185, 129, 0.25);
  }
  .tag-warn {
    background: var(--warn-bg);
    color: var(--warn);
    border-color: rgba(245, 158, 11, 0.25);
  }
  .tag-err {
    background: var(--err-bg);
    color: var(--err);
    border-color: rgba(239, 68, 68, 0.25);
  }
  .corr-cell {
    font-family: var(--mono);
    font-size: 12px;
    white-space: nowrap;
  }
  .corr-before {
    color: var(--text-muted);
  }
  .corr-arrow {
    color: var(--text-muted);
    margin: 0 4px;
  }
  .corr-after {
    color: var(--ok);
    font-weight: 600;
  }
  .log-terminal {
    background: #050507;
    border-top: 1px solid var(--border);
    padding: 12px 14px;
    font-family: var(--mono);
    font-size: 11.5px;
    line-height: 1.5;
    color: #d4d4d8;
    height: 220px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
  }
  .status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--text-muted);
    margin-right: 6px;
  }
  .status-dot.running { background: var(--warn); }
  .status-dot.ok { background: var(--ok); }
  .status-dot.err { background: var(--err); }
  .modal-backdrop {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.72);
    z-index: 100;
    align-items: center;
    justify-content: center;
  }
  .modal-card {
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    width: 620px;
    max-width: 92vw;
    overflow: hidden;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
  }
  .dir-browser {
    height: 300px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg);
    margin: 10px 0 14px;
  }
  .dir-row {
    padding: 7px 12px;
    border-bottom: 1px solid #16161a;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 12px;
  }
  .dir-row:hover {
    background: var(--surface-hover);
  }
  .toast {
    position: fixed;
    bottom: 20px;
    right: 24px;
    background: var(--surface-hover);
    border: 1px solid var(--border-strong);
    color: var(--text);
    padding: 8px 14px;
    border-radius: 4px;
    font-size: 12px;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.18s;
    z-index: 200;
  }
  .toast.visible { opacity: 1; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span class="brand-title">subsync</span>
    <span class="brand-sub">Audio-to-subtitle alignment &amp; series pipeline</span>
  </div>
  <div class="toolbar">
    <select id="reportHistory" onchange="loadSelectedReport()" style="width: 310px;">
      <option value="">Select a saved report...</option>
    </select>
    <button onclick="fetchReportsList()">Refresh</button>
    <button onclick="copyMarkdown()">Copy Markdown</button>
  </div>
</header>

<div class="layout">
  <div class="panel">
    <div class="panel-header">
      <h2 class="panel-title">Job Configuration</h2>
      <span style="font-size:11px;color:var(--text-muted);" id="statusLabel">
        <span class="status-dot" id="statusDot"></span>Idle
      </span>
    </div>
    <div class="panel-body">
      <div class="field">
        <label>Mode</label>
        <select id="mode">
          <option value="series">Series (Download &amp; Synchronize)</option>
          <option value="batch">Batch (Synchronize existing local .srt)</option>
        </select>
      </div>

      <div class="field">
        <label>Target Directory</label>
        <div class="input-group">
          <input type="text" class="mono-input" id="targetDir" value="/mnt/f/Séries/The Blacklist" placeholder="/path/to/series">
          <button onclick="openBrowser()">Browse</button>
        </div>
      </div>

      <div class="field grid-2">
        <div>
          <label>Language</label>
          <select id="lang">
            <option value="fr" selected>French (fr)</option>
            <option value="en">English (en)</option>
            <option value="es">Spanish (es)</option>
            <option value="de">German (de)</option>
            <option value="it">Italian (it)</option>
          </select>
        </div>
        <div>
          <label>Provider</label>
          <select id="provider">
            <option value="auto" selected>Auto (Fallback chain)</option>
            <option value="addic7ed">Addic7ed</option>
            <option value="gestdown">Gestdown</option>
            <option value="subdl">SubDL</option>
            <option value="opensubtitles">OpenSubtitles</option>
          </select>
        </div>
      </div>

      <div class="field grid-3">
        <div>
          <label>Season</label>
          <input type="text" class="mono-input" id="seasonFilter" placeholder="All">
        </div>
        <div>
          <label>Episodes</label>
          <input type="text" class="mono-input" id="episodesFilter" placeholder="e.g. 1-5,8">
        </div>
        <div>
          <label>CPU Workers</label>
          <input type="number" class="mono-input" id="workers" value="8" min="1" max="32">
        </div>
      </div>

      <div class="checkbox-list">
        <label><input type="checkbox" id="jellyfin" checked> Jellyfin / Plex naming (<code>VideoName.fr.srt</code>)</label>
        <label><input type="checkbox" id="force" checked> Overwrite existing synchronized files (<code>--force</code>)</label>
        <label><input type="checkbox" id="recursive"> Scan subdirectories recursively (<code>--recursive</code>)</label>
      </div>

      <button class="btn-primary" id="startBtn" onclick="startJob()">Run Synchronization</button>
      <button class="btn-danger" id="stopBtn" style="display:none;" onclick="stopJob()">Stop Job</button>
    </div>

    <div class="panel-header" style="border-top:1px solid var(--border);">
      <h2 class="panel-title">Process Output</h2>
    </div>
    <div class="log-terminal" id="logBox">Ready.</div>
  </div>

  <div class="panel">
    <div class="panel-header">
      <h2 class="panel-title" id="tableTitle">Synchronization Report</h2>
      <span class="mono-input" style="color:var(--text-muted);font-size:11.5px;" id="reportPathDisplay"></span>
    </div>

    <div class="metrics-row">
      <div class="metric-cell">
        <div class="metric-label">Episodes</div>
        <div class="metric-value" id="stTotal">0</div>
      </div>
      <div class="metric-cell">
        <div class="metric-label">Succeeded</div>
        <div class="metric-value" id="stOk" style="color:var(--ok);">0</div>
      </div>
      <div class="metric-cell">
        <div class="metric-label">Recaps Trimmed</div>
        <div class="metric-value" id="stPurge">0</div>
      </div>
      <div class="metric-cell">
        <div class="metric-label">Mean Correlation (Before &rarr; After)</div>
        <div class="metric-value" id="stCorr">-</div>
      </div>
      <div class="metric-cell">
        <div class="metric-label">Elapsed</div>
        <div class="metric-value" id="stTime">0.0s</div>
      </div>
    </div>

    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>Episode</th>
            <th>Release</th>
            <th>Recap</th>
            <th>Offset (Start &rarr; Mid &rarr; End)</th>
            <th>Correlation (Before &rarr; After)</th>
            <th>First Synchronized Line</th>
          </tr>
        </thead>
        <tbody id="recapBody">
          <tr><td colspan="6" style="text-align:center;color:var(--text-muted);padding:36px;">No report loaded.</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<div class="modal-backdrop" id="browserModal">
  <div class="modal-card">
    <div class="panel-header">
      <h2 class="panel-title">Select Directory</h2>
      <button onclick="closeBrowser()" style="height:24px;padding:0 8px;font-size:11px;">Close</button>
    </div>
    <div class="panel-body">
      <div style="display:flex;gap:6px;margin-bottom:8px;">
        <button onclick="browseTo('/mnt/f/Séries')">/mnt/f/Séries</button>
        <button onclick="browseTo('/mnt/f')">/mnt/f</button>
        <button onclick="browseTo('/home/juliench')">/home/juliench</button>
      </div>
      <input type="text" class="mono-input" id="browseCurrent" readonly>
      <div class="dir-browser" id="dirList"></div>
      <div style="display:flex;justify-content:flex-end;gap:8px;">
        <button onclick="closeBrowser()">Cancel</button>
        <button class="btn-primary" style="width:auto;padding:0 16px;" onclick="confirmBrowser()">Select Directory</button>
      </div>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let currentReport = null;
let pollTimer = null;

function showToast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('visible');
  setTimeout(() => el.classList.remove('visible'), 2200);
}

function setStatus(state, text) {
  const dot = document.getElementById('statusDot');
  dot.className = 'status-dot' + (state ? ' ' + state : '');
  document.getElementById('statusLabel').innerHTML = '';
  document.getElementById('statusLabel').appendChild(dot);
  document.getElementById('statusLabel').appendChild(document.createTextNode(text));
}

async function fetchReportsList() {
  const res = await fetch('/api/reports');
  const data = await res.json();
  const sel = document.getElementById('reportHistory');
  const prev = sel.value;
  sel.innerHTML = '<option value="">Select a saved report...</option>';
  for (const item of data.reports) {
    const opt = document.createElement('option');
    opt.value = item.path;
    opt.textContent = item.label;
    sel.appendChild(opt);
  }
  if (prev && data.reports.some(r => r.path === prev)) {
    sel.value = prev;
  } else if (data.reports.length > 0 && !currentReport) {
    sel.value = data.reports[0].path;
    loadSelectedReport();
  }
}

async function loadSelectedReport() {
  const path = document.getElementById('reportHistory').value;
  if (!path) return;
  const res = await fetch('/api/report?path=' + encodeURIComponent(path));
  const data = await res.json();
  if (data && data.results) {
    document.getElementById('reportPathDisplay').textContent = path.split('/').pop();
    renderReport(data);
  }
}

function escapeHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function renderReport(report) {
  currentReport = report;
  const rows = report.results || [];
  const dirLabel = report.series_dir || report.ref_dir || '';
  document.getElementById('tableTitle').textContent = dirLabel ? `Report — ${dirLabel}` : 'Synchronization Report';
  document.getElementById('stTotal').textContent = report.total_episodes || report.total || rows.length;
  document.getElementById('stOk').textContent = report.succeeded || 0;
  document.getElementById('stTime').textContent = ((report.elapsed_seconds || report.total_elapsed_seconds || 0)).toFixed(1) + 's';

  let purgeCount = 0;
  let rawSum = 0, rawN = 0;
  let postSum = 0, postN = 0;
  const tbody = document.getElementById('recapBody');
  tbody.innerHTML = '';

  for (const r of rows) {
    const tr = document.createElement('tr');
    const ep = r.episode_code || (r.season && r.episode ? `S${String(r.season).padStart(2,'0')}E${String(r.episode).padStart(2,'0')}` : (r.video_path || '').split('/').pop());
    const title = r.episode_title ? `<span class="ep-title">${escapeHtml(r.episode_title)}</span>` : '';
    const prov = r.provider_name ? `${r.provider_name} (${r.candidate_release || '-'})` : (r.status || '-');
    const recap = r.recap_status || 'Non';
    if (recap.startsWith('Purge')) purgeCount++;

    const rawVal = r.raw_correlation_score ?? null;
    const postVal = r.post_correlation_score ?? r.correlation_score ?? r.correlation ?? null;
    if (rawVal !== null) { rawSum += rawVal * 100; rawN++; }
    if (postVal !== null) { postSum += postVal * 100; postN++; }

    let corrHtml = '-';
    if (rawVal !== null && postVal !== null) {
      corrHtml = `<span class="corr-before">${(rawVal * 100).toFixed(1)}%</span><span class="corr-arrow">&rarr;</span><span class="corr-after">${(postVal * 100).toFixed(1)}%</span>`;
    } else if (postVal !== null) {
      corrHtml = `<span class="corr-after">${(postVal * 100).toFixed(1)}%</span>`;
    }

    const offEvo = r.offset_evolution || (r.offset_seconds !== undefined && r.offset_seconds !== null ? `${r.offset_seconds >= 0 ? '+' : ''}${Number(r.offset_seconds).toFixed(2)}s` : '-');
    const firstSub = (r.first_sub_time_s !== null && r.first_sub_time_s !== undefined && r.first_sub_text)
      ? `<span class="mono-input" style="color:var(--text-secondary);">${Number(r.first_sub_time_s).toFixed(2)}s</span> &mdash; ${escapeHtml(r.first_sub_text.slice(0, 56))}`
      : escapeHtml(r.error || '-');

    let recapTag = `<span class="tag">${escapeHtml(recap)}</span>`;
    if (recap.startsWith('Purge')) {
      recapTag = `<span class="tag tag-warn">${escapeHtml(recap)}</span>`;
    } else if (recap.startsWith('Inclus')) {
      recapTag = `<span class="tag tag-ok">${escapeHtml(recap)}</span>`;
    }

    tr.innerHTML = `
      <td><span class="ep-code">${escapeHtml(ep)}</span>${title}</td>
      <td><span class="tag">${escapeHtml(prov)}</span></td>
      <td>${recapTag}</td>
      <td class="mono-input">${escapeHtml(offEvo)}</td>
      <td class="corr-cell">${corrHtml}</td>
      <td>${firstSub}</td>
    `;
    tbody.appendChild(tr);
  }
  document.getElementById('stPurge').textContent = purgeCount;
  if (rawN > 0 && postN > 0) {
    document.getElementById('stCorr').textContent = `${(rawSum / rawN).toFixed(1)}% → ${(postSum / postN).toFixed(1)}%`;
  } else if (postN > 0) {
    document.getElementById('stCorr').textContent = `${(postSum / postN).toFixed(1)}%`;
  } else {
    document.getElementById('stCorr').textContent = '-';
  }
}

function copyMarkdown() {
  if (!currentReport || !currentReport.results) return;
  const lines = [
    '| Episode | Release | Recap | Offset (Start -> Mid -> End) | Correlation (Before -> After) | First Line |',
    '| :--- | :--- | :--- | :--- | :---: | :--- |'
  ];
  for (const r of currentReport.results) {
    const ep = r.episode_code ? `**${r.episode_code}**${r.episode_title ? ' — *' + r.episode_title + '*' : ''}` : (r.video_path || '');
    const ver = r.provider_name ? `\`${r.provider_name} (${r.candidate_release || '-'})\`` : `\`${r.status}\``;
    const recap = r.recap_status || 'Non';
    const offEvo = r.offset_evolution ? `\`${r.offset_evolution}\`` : (r.offset_seconds != null ? `\`${r.offset_seconds >= 0 ? '+' : ''}${Number(r.offset_seconds).toFixed(2)}s\`` : '-');
    const rawVal = r.raw_correlation_score;
    const postVal = r.post_correlation_score ?? r.correlation_score ?? r.correlation;
    const corrStr = (rawVal != null && postVal != null)
      ? `\`${(rawVal * 100).toFixed(1)}%\` -> **${(postVal * 100).toFixed(1)}%**`
      : (postVal != null ? `**${(postVal * 100).toFixed(1)}%**` : '-');
    const firstSub = (r.first_sub_time_s != null && r.first_sub_text) ? `\`${Number(r.first_sub_time_s).toFixed(2)}s\` : ${r.first_sub_text.slice(0, 50)}` : '-';
    lines.push(`| ${ep} | ${ver} | ${recap} | ${offEvo} | ${corrStr} | ${firstSub} |`);
  }
  navigator.clipboard.writeText(lines.join('\n'));
  showToast('Markdown copied to clipboard');
}

async function startJob() {
  const payload = {
    mode: document.getElementById('mode').value,
    target_dir: document.getElementById('targetDir').value,
    lang: document.getElementById('lang').value,
    provider: document.getElementById('provider').value,
    season: document.getElementById('seasonFilter').value,
    episodes: document.getElementById('episodesFilter').value,
    workers: parseInt(document.getElementById('workers').value || '8', 10),
    jellyfin: document.getElementById('jellyfin').checked,
    force: document.getElementById('force').checked,
    recursive: document.getElementById('recursive').checked,
  };
  const res = await fetch('/api/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (data.ok) {
    document.getElementById('startBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = 'block';
    setStatus('running', 'Running');
    if (!pollTimer) pollTimer = setInterval(pollStatus, 1000);
  } else {
    showToast(data.error || 'Failed to start job');
  }
}

async function stopJob() {
  await fetch('/api/stop', {method: 'POST'});
  showToast('Stopping job...');
}

async function pollStatus() {
  const res = await fetch('/api/status');
  const data = await res.json();
  const logBox = document.getElementById('logBox');
  if (data.logs) {
    logBox.textContent = data.logs;
    logBox.scrollTop = logBox.scrollHeight;
  }
  if (data.report && data.report.results) {
    renderReport(data.report);
  }
  if (!data.running) {
    clearInterval(pollTimer);
    pollTimer = null;
    document.getElementById('startBtn').style.display = 'block';
    document.getElementById('stopBtn').style.display = 'none';
    setStatus(data.exit_code === 0 ? 'ok' : 'err', data.exit_code === 0 ? 'Completed' : 'Failed');
    fetchReportsList();
  }
}

async function openBrowser() {
  document.getElementById('browserModal').style.display = 'flex';
  await browseTo(document.getElementById('targetDir').value || '/mnt/f/Séries');
}
function closeBrowser() {
  document.getElementById('browserModal').style.display = 'none';
}
function confirmBrowser() {
  document.getElementById('targetDir').value = document.getElementById('browseCurrent').value;
  closeBrowser();
}
async function browseTo(dir) {
  const res = await fetch('/api/browse?dir=' + encodeURIComponent(dir));
  const data = await res.json();
  document.getElementById('browseCurrent').value = data.current;
  const list = document.getElementById('dirList');
  list.innerHTML = '';
  if (data.parent) {
    const up = document.createElement('div');
    up.className = 'dir-row';
    up.innerHTML = '<span>..</span><span style="color:var(--text-muted)">Parent directory</span>';
    up.onclick = () => browseTo(data.parent);
    list.appendChild(up);
  }
  for (const d of data.dirs) {
    const item = document.createElement('div');
    item.className = 'dir-row';
    item.innerHTML = `<span>${escapeHtml(d.name)}/</span><span style="color:var(--text-muted);">${d.mkv_count ? d.mkv_count + ' video files' : ''}</span>`;
    item.onclick = () => browseTo(d.path);
    list.appendChild(item);
  }
}

fetchReportsList();
</script>
</body>
</html>
"""


class _JobManager:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: Optional[subprocess.Popen] = None
        self.logs: List[str] = []
        self.running: bool = False
        self.exit_code: Optional[int] = None
        self.report_path: Optional[Path] = None

    def start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if self.running and self.proc and self.proc.poll() is None:
                return {"ok": False, "error": "A synchronization job is already running."}

            mode = str(payload.get("mode") or "series")
            target_dir = str(payload.get("target_dir") or "").strip()
            if not target_dir:
                return {"ok": False, "error": "Target directory is required."}

            lang = str(payload.get("lang") or "fr").strip()
            provider = str(payload.get("provider") or "auto").strip()
            workers = int(payload.get("workers") or 0)
            reports_dir = Path.home() / ".cache" / "subsync" / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            ts_label = time.strftime("%Y%m%d_%H%M%S")
            folder_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(target_dir).name or "run")
            report_file = reports_dir / f"subsync_{folder_slug}_{ts_label}.json"
            self.report_path = report_file

            cmd = [
                sys.executable,
                str(_RUN_PY),
                mode,
                target_dir,
                "--lang",
                lang,
                "--report-json",
                str(report_file),
            ]
            if mode == "series":
                cmd.extend(["--provider", provider])
                if workers > 0:
                    cmd.extend(["--workers", str(workers)])
                if payload.get("season"):
                    cmd.extend(["--season", str(payload["season"]).strip()])
                if payload.get("episodes"):
                    cmd.extend(["--episodes", str(payload["episodes"]).strip()])
            elif mode == "batch" and workers > 0:
                cmd.extend(["--jobs", str(workers)])

            if payload.get("jellyfin", True):
                cmd.append("--jellyfin")
            if payload.get("force", True):
                cmd.append("--force")
            if payload.get("recursive", False):
                cmd.append("--recursive")

            self.logs = [f"$ {' '.join(cmd)}\n"]
            self.running = True
            self.exit_code = None

            def _runner() -> None:
                try:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(_REPO_ROOT),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    with self.lock:
                        self.proc = proc
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        with self.lock:
                            self.logs.append(line)
                            if len(self.logs) > 600:
                                self.logs = self.logs[-600:]
                    rc = proc.wait()
                    with self.lock:
                        self.exit_code = rc
                        self.running = False
                except Exception as exc:
                    with self.lock:
                        self.logs.append(f"\n[ERROR] {exc}\n")
                        self.exit_code = 1
                        self.running = False

            threading.Thread(target=_runner, daemon=True).start()
            return {"ok": True, "report_path": str(report_file)}

    def stop(self) -> None:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
            self.running = False

    def status(self) -> Dict[str, Any]:
        with self.lock:
            rep_data = None
            if self.report_path and self.report_path.exists():
                try:
                    rep_data = json.loads(self.report_path.read_text(encoding="utf-8"))
                except Exception:
                    rep_data = None
            return {
                "running": self.running,
                "exit_code": self.exit_code,
                "logs": "".join(self.logs[-250:]),
                "report": rep_data,
            }


_JOB_MANAGER = _JobManager()


def discover_saved_reports() -> List[Dict[str, str]]:
    """Find available JSON sync reports in `/tmp` and `~/.cache/subsync/reports`."""
    candidates: List[Path] = []
    tmp_dir = Path("/tmp")
    if tmp_dir.exists():
        for f in sorted(tmp_dir.glob("*summary*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            candidates.append(f)

    rep_dir = Path.home() / ".cache" / "subsync" / "reports"
    if rep_dir.exists():
        for f in sorted(rep_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.name == "last_series_summary.json":
                continue
            candidates.append(f)

    seen_dirs = set()
    items: List[Dict[str, str]] = []
    for c in candidates:
        try:
            data = json.loads(c.read_text(encoding="utf-8"))
            s_dir = data.get("series_dir") or data.get("ref_dir") or c.stem
            tot = data.get("total_episodes") or data.get("total") or len(data.get("results", []))
            ok = data.get("succeeded", 0)
            key = str(s_dir)
            if key in seen_dirs:
                continue
            seen_dirs.add(key)
            label = f"{Path(s_dir).parent.name}/{Path(s_dir).name} ({ok}/{tot})"
            items.append({"path": str(c), "label": label})
        except Exception:
            continue
    return items


class _GUIRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            raw = _HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if parsed.path == "/api/reports":
            self._send_json({"reports": discover_saved_reports()})
            return

        if parsed.path == "/api/report":
            qs = urllib.parse.parse_qs(parsed.query)
            rpath = qs.get("path", [""])[0]
            p = Path(rpath)
            if p.exists() and p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    self._send_json(data)
                    return
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=400)
                    return
            self._send_json({"error": "File not found"}, status=404)
            return

        if parsed.path == "/api/status":
            self._send_json(_JOB_MANAGER.status())
            return

        if parsed.path == "/api/browse":
            qs = urllib.parse.parse_qs(parsed.query)
            req_dir = qs.get("dir", ["/mnt/f/Séries"])[0]
            target = Path(req_dir)
            if not target.exists() or not target.is_dir():
                target = Path("/mnt/f") if Path("/mnt/f").exists() else Path.home()
            dirs = []
            try:
                for entry in sorted(target.iterdir(), key=lambda e: e.name.lower()):
                    if entry.is_dir() and not entry.name.startswith("."):
                        mkv_count = 0
                        try:
                            mkv_count = sum(
                                1
                                for child in entry.iterdir()
                                if child.suffix.lower() in (".mkv", ".mp4", ".avi")
                            )
                        except Exception:
                            pass
                        dirs.append({"name": entry.name, "path": str(entry), "mkv_count": mkv_count})
            except Exception:
                pass
            parent = str(target.parent) if target.parent != target else None
            self._send_json({"current": str(target), "parent": parent, "dirs": dirs})
            return

        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/start":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            payload = json.loads(body)
            self._send_json(_JOB_MANAGER.start(payload))
            return

        if parsed.path == "/api/stop":
            _JOB_MANAGER.stop()
            self._send_json({"ok": True})
            return

        self.send_error(404)


def _open_browser_url(url: str) -> None:
    """Open `url` in the user's desktop browser (supporting WSL2, Linux, Windows, macOS)."""
    if "microsoft" in os.uname().release.lower():
        for cmd in (
            ["cmd.exe", "/c", "start", url],
            ["wslview", url],
            ["xdg-open", url],
        ):
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                continue
    try:
        webbrowser.open(url)
    except Exception:
        pass


def run_web_gui(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> int:
    """Start the local web UI server."""
    server = None
    actual_port = port
    for candidate_port in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, candidate_port), _GUIRequestHandler)
            actual_port = candidate_port
            break
        except OSError:
            continue

    if server is None:
        print(f"Error: could not bind web server on {host}:{port}", file=sys.stderr)
        return 1

    url = f"http://{host}:{actual_port}"
    print(f"subsync web UI listening on {url} (Ctrl+C to stop)")

    if open_browser:
        threading.Timer(0.4, _open_browser_url, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web UI.")
    finally:
        server.server_close()
    return 0
