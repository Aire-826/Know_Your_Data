from __future__ import annotations

import html
import json
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse

from model_engine import process_arbitrary_csv, run_regression_benchmark

app = FastAPI(title="DataForge")

MAX_CSV_BYTES = 2_097_152  # 2MB


# ---------------------------------------------------------------------------
# Session storage
# ---------------------------------------------------------------------------
SESSION_DIR = Path(tempfile.gettempdir()) / "dataforge_sessions"
SESSION_DIR.mkdir(exist_ok=True)
_SESSION_CACHE: dict[str, bytes] = {}


def _save_session(csv_bytes: bytes) -> str:
    sid = uuid.uuid4().hex[:16]
    (SESSION_DIR / f"{sid}.csv").write_bytes(csv_bytes)
    _SESSION_CACHE[sid] = csv_bytes
    return sid


def _load_session(sid: str | None) -> bytes | None:
    if not sid or not sid.isalnum():
        return None
    if sid in _SESSION_CACHE:
        return _SESSION_CACHE[sid]
    p = SESSION_DIR / f"{sid}.csv"
    if p.exists():
        data = p.read_bytes()
        _SESSION_CACHE[sid] = data
        return data
    return None


def _get_bytes(sid: str | None = None, upload: UploadFile | None = None) -> bytes:
    if upload is not None and upload.filename:
        data = upload.file.read(MAX_CSV_BYTES + 1)
        if len(data) > MAX_CSV_BYTES:
            raise ValueError("File exceeds 2MB limit.")
        return data
    if sid:
        data = _load_session(sid)
        if data is not None:
            return data
    return b""


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
PAGE_STYLE = """
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root, html[data-theme="dark"], body[data-theme="dark"]{
  color-scheme:dark;
  --bg:#070a13; --bg-glow-1:rgba(244,63,94,0.14); --bg-glow-2:rgba(56,189,248,0.12);
  --panel:rgba(19,27,46,0.72); --panel-border:rgba(51,65,85,0.55);
  --card:rgba(26,34,52,0.85); --card-border:rgba(42,54,79,0.9);
  --text:#f3f4f6; --text-dim:#94a3b8; --text-mute:#64748b;
  --accent:#fb7185; --accent-2:#38bdf8; --accent-3:#818cf8;
  --sidebar:rgba(19,27,46,0.92); --sidebar-border:rgba(51,65,85,0.55);
  --input-bg:rgba(11,15,25,0.85); --input-border:#334155;
  --hover:rgba(30,41,59,0.85); --chart-bg:rgba(14,21,38,0.8);
  --chart-grid:#1e293b; --chart-font:#94a3b8;
}
html[data-theme="light"], body[data-theme="light"]{
  color-scheme:light;
  --bg:#f8fafc; --bg-glow-1:rgba(244,63,94,0.10); --bg-glow-2:rgba(56,189,248,0.12);
  --panel:rgba(255,255,255,0.85); --panel-border:rgba(203,213,225,0.9);
  --card:rgba(248,250,252,0.95); --card-border:rgba(226,232,240,1);
  --text:#0f172a; --text-dim:#475569; --text-mute:#94a3b8;
  --accent:#e11d48; --accent-2:#0284c7; --accent-3:#6366f1;
  --sidebar:rgba(255,255,255,0.92); --sidebar-border:rgba(203,213,225,0.9);
  --input-bg:#ffffff; --input-border:#cbd5e1;
  --hover:rgba(241,245,249,0.95); --chart-bg:rgba(255,255,255,0.9);
  --chart-grid:#e2e8f0; --chart-font:#475569;
}
*{box-sizing:border-box}
body{
  margin:0;min-height:100vh;display:flex;
  font-family:Inter,system-ui,sans-serif;color:var(--text);
  background:
    radial-gradient(900px 600px at 12% 8%, var(--bg-glow-1), transparent 60%),
    radial-gradient(900px 600px at 92% 92%, var(--bg-glow-2), transparent 60%),
    var(--bg);
  background-attachment:fixed;
  transition:background-color .35s ease, color .35s ease;
}
aside{
  width:260px;background:var(--sidebar);
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  border-right:1px solid var(--sidebar-border);
  padding:24px 14px;display:flex;flex-direction:column;gap:10px;
  position:fixed;height:100vh;overflow-y:auto;z-index:950;
  transition:width .25s ease,padding .25s ease,background .35s ease,border-color .35s ease;
}
aside.collapsed{width:64px;padding:24px 8px}
aside.collapsed .nav-label{display:none}
aside.collapsed a{justify-content:center;padding:12px 0}
.sidebar-toggle{
  background:var(--hover);border:1px solid var(--panel-border);
  color:var(--accent);border-radius:8px;padding:8px 10px;cursor:pointer;
  font-size:1rem;align-self:flex-end;width:auto;margin-bottom:4px;
  transition:background .2s ease;
}
.sidebar-toggle:hover{filter:brightness(1.15)}
.content-area{
  margin-left:260px;flex:1;padding:30px;max-width:1280px;
  transition:margin-left .25s ease;
}
body.sidebar-collapsed .content-area{margin-left:64px}
.menu-toggle{
  display:none;background:none;border:none;color:var(--accent);
  font-size:1.5rem;cursor:pointer;position:fixed;top:15px;left:15px;z-index:1000;
}
.panel{
  background:var(--panel);border:1px solid var(--panel-border);
  border-radius:18px;padding:26px;margin-bottom:22px;
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  box-shadow:0 10px 30px -12px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
  transition:background .35s ease,border-color .35s ease;
}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}
.metric{
  background:var(--card);border:1px solid var(--card-border);
  padding:20px;border-radius:14px;text-align:center;position:relative;overflow:hidden;
  transition:transform .2s ease, background .35s ease, border-color .35s ease;
}
.metric::before{
  content:"";position:absolute;inset:0;
  background:linear-gradient(135deg, rgba(251,113,133,0.10), transparent 55%);
  pointer-events:none;
}
.metric:hover{transform:translateY(-2px)}
.metric small{color:var(--text-dim);text-transform:uppercase;font-size:.72rem;letter-spacing:.08em;font-weight:600}
.metric strong{color:var(--accent);font-size:1.65rem;display:block;margin-top:8px;font-family:'Space Grotesk',sans-serif;letter-spacing:-0.01em}
.hero{text-align:center;padding:40px 0 16px}
.hero h1{
  margin:0;font-family:'Space Grotesk',sans-serif;
  font-size:clamp(3rem, 7vw, 5.2rem);
  font-weight:700;letter-spacing:-0.03em;line-height:1.05;
  background:linear-gradient(135deg,#fb7185 0%,#f43f5e 35%,#818cf8 70%,#38bdf8 100%);
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
  text-decoration:none;cursor:pointer;display:inline-block;
  transition:filter .2s ease;
}
.hero a:hover h1{filter:brightness(1.15) saturate(1.1)}
.hero p{margin:10px 0 0;color:var(--text-dim);font-size:1.05rem;font-weight:500;letter-spacing:.02em}
.theme-toggle{
  position:fixed;top:18px;right:22px;z-index:1001;
  display:flex;align-items:center;gap:8px;
  background:var(--panel);border:1px solid var(--panel-border);
  border-radius:999px;padding:6px;cursor:pointer;
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  transition:background .3s ease,border-color .3s ease;user-select:none;
}
.theme-toggle .thumb{
  width:42px;height:24px;border-radius:999px;position:relative;
  background:linear-gradient(135deg,#38bdf8,#818cf8);transition:background .3s ease;
}
.theme-toggle .thumb::after{
  content:"🌙";position:absolute;top:0;left:0;width:24px;height:24px;border-radius:50%;
  background:#fff;display:flex;align-items:center;justify-content:center;
  font-size:12px;box-shadow:0 2px 6px rgba(0,0,0,0.25);transition:transform .3s ease;
}
body[data-theme="light"] .theme-toggle .thumb{background:linear-gradient(135deg,#fbbf24,#f97316)}
body[data-theme="light"] .theme-toggle .thumb::after{content:"☀️";transform:translateX(18px)}
.theme-toggle .lbl{font-size:.78rem;font-weight:600;color:var(--text-dim);padding:0 8px 0 4px}
input,select{
  background:var(--input-bg);border:1px solid var(--input-border);color:var(--text);
  border-radius:10px;padding:11px;width:100%;margin:6px 0 14px;font-family:inherit;
  transition:border-color .2s ease, background .35s ease;
}
input:focus,select:focus{outline:none;border-color:var(--accent-2)}
button{
  background:linear-gradient(135deg,#f43f5e,#be123c);color:#fff;border:none;
  padding:12px 20px;font-weight:700;border-radius:10px;cursor:pointer;width:100%;
  font-family:inherit;letter-spacing:.02em;transition:filter .2s ease,transform .1s ease;
}
button:hover{filter:brightness(1.1)}
button:active{transform:scale(0.99)}
a{
  color:var(--accent);text-decoration:none;display:flex;align-items:center;gap:10px;
  padding:10px 12px;border-radius:10px;
  transition:background .2s ease,color .35s ease;font-size:.94rem;font-weight:500;
}
a:hover{background:var(--hover)}
.mode-toggle{
  display:flex !important;justify-content:center;text-align:center;
  font-weight:700;color:#0b0f19 !important;
  background:linear-gradient(135deg,#38bdf8,#818cf8) !important;
}
.mode-toggle:hover{filter:brightness(1.1)}
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
.chart-box{
  background:var(--chart-bg);border:1px solid var(--panel-border);
  border-radius:14px;padding:10px;
  transition:background .35s ease,border-color .35s ease;
}
.chart-box h3{margin:6px 6px 0;font-size:.95rem;color:var(--text);font-weight:600}
.chart-box .plot{width:100%;height:320px}
table{width:100%;border-collapse:collapse;margin-top:10px}
th,td{padding:12px;text-align:left;border-bottom:1px solid var(--panel-border);color:var(--text)}
th{color:var(--accent);font-weight:600}
@media(max-width:900px){
  aside{transform:translateX(-100%);width:260px !important;padding:24px 14px !important}
  aside.collapsed{transform:translateX(-100%)}
  aside.open{transform:translateX(0)}
  aside.open .nav-label{display:inline}
  .content-area{margin-left:0 !important;padding:20px 15px;margin-top:56px}
  .menu-toggle{display:block}
  .hero h1{font-size:clamp(2.2rem,10vw,3rem)}
  .chart-grid{grid-template-columns:1fr}
  .theme-toggle{top:14px;right:14px}
}
</style>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
"""


def _escape(val):
    return html.escape(str(val), quote=True)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def _build_charts(df, date_col, target_col):
    if df is None or len(df) == 0 or target_col is None or target_col not in df.columns:
        return """
        <div class="chart-box" style="grid-column:1/-1;text-align:center;padding:40px 20px">
            <h3 style="font-size:1.1rem">📭 No data to visualize yet</h3>
            <p style="color:var(--text-dim);margin:8px 0 0">
                Upload a CSV above to see trend, distribution, cumulative sum, and range charts appear here.
            </p>
        </div>
        """

    has_date = date_col is not None and date_col in df.columns
    try:
        dates = (
            json.dumps(df[date_col].astype(str).tolist())
            if has_date
            else json.dumps(list(range(len(df))))
        )
        values = json.dumps(
            [float(v) for v in pd.to_numeric(df[target_col], errors="coerce").fillna(0).tolist()]
        )
    except Exception:
        return """
        <div class="chart-box" style="grid-column:1/-1;text-align:center;padding:40px 20px">
            <h3>⚠️ Could not interpret the uploaded data</h3>
            <p style="color:var(--text-dim);margin:8px 0 0">Try a CSV with a date column and a numeric target column.</p>
        </div>
        """

    if "rolling_mean_3" in df.columns:
        rolling = json.dumps(
            [float(v) for v in pd.to_numeric(df["rolling_mean_3"], errors="coerce").fillna(0).tolist()]
        )
    else:
        rolling = values

    safe_target = _escape(str(target_col))

    common_layout = (
        "{paper_bgcolor:'transparent',plot_bgcolor:'transparent',"
        "font:{color:'var(--chart-font)',size:11},"
        "xaxis:{gridcolor:'var(--chart-grid)'},yaxis:{gridcolor:'var(--chart-grid)'},"
        "margin:{t:10,r:15,b:35,l:45},showlegend:false}"
    )

    trend = f"""
    <div class="chart-box">
        <h3>📈 Trend of {safe_target}</h3>
        <div id="chartTrend" class="plot"></div>
    </div>
    <script>
    Plotly.newPlot('chartTrend', [
        {{x:{dates},y:{values},type:'scatter',mode:'lines+markers',
          line:{{color:'#f43f5e',width:2}},marker:{{size:5}}}},
        {{x:{dates},y:{rolling},type:'scatter',mode:'lines',
          line:{{color:'#38bdf8',dash:'dash',width:2}}}}
    ], {common_layout}, {{responsive:true}});
    </script>
    """

    dist = f"""
    <div class="chart-box">
        <h3>📊 Distribution</h3>
        <div id="chartDist" class="plot"></div>
    </div>
    <script>
    Plotly.newPlot('chartDist', [{{
        x:{values}, type:'histogram',
        marker:{{color:'#818cf8',line:{{color:'#1e293b',width:1}}}},
        nbinsx:20
    }}], {common_layout}, {{responsive:true}});
    </script>
    """

    cum_script = f"""
    <div class="chart-box">
        <h3>📉 Cumulative {safe_target}</h3>
        <div id="chartCum" class="plot"></div>
    </div>
    <script>
    (function(){{
        const vals = {values};
        let total = 0;
        const cum = vals.map(v => (total += v));
        Plotly.newPlot('chartCum', [{{
            x:{dates}, y:cum, type:'scatter', mode:'lines', fill:'tozeroy',
            line:{{color:'#34d399',width:2}},
            fillcolor:'rgba(52,211,153,0.15)'
        }}], {common_layout}, {{responsive:true}});
    }})();
    </script>
    """

    box = f"""
    <div class="chart-box">
        <h3>📦 Range Overview</h3>
        <div id="chartBox" class="plot"></div>
    </div>
    <script>
    Plotly.newPlot('chartBox', [{{
        y:{values}, type:'box', name:'{safe_target}',
        marker:{{color:'#fbbf24'}},
        boxpoints:'outliers'
    }}], {common_layout}, {{responsive:true}});
    </script>
    """

    return trend + dist + cum_script + box


# ---------------------------------------------------------------------------
# HTML shell (shared by blank + data states)
# ---------------------------------------------------------------------------
def _render_html_shell(nav_html, theme_toggle_html, content_html):
    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DataForge — From CSV to champion model in one click.</title>
<script>
(function(){{
  try {{
    const saved = localStorage.getItem('dataforge-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
  }} catch(e) {{}}
}})();
</script>
{PAGE_STYLE}
</head>
<body data-theme="dark">
<button class="menu-toggle" onclick="toggleSidebar()">☰</button>
{theme_toggle_html}
{nav_html}
<main class="content-area">
{content_html}
</main>
<script>
function toggleSidebar() {{
    document.getElementById('sidebar').classList.toggle('open');
}}
function toggleCollapse() {{
    const s = document.getElementById('sidebar');
    s.classList.toggle('collapsed');
    document.body.classList.toggle('sidebar-collapsed');
    setTimeout(() => {{ window.dispatchEvent(new Event('resize')); }}, 300);
}}
function toggleTheme() {{
    const cur = document.body.getAttribute('data-theme') || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    document.body.setAttribute('data-theme', next);
    document.documentElement.setAttribute('data-theme', next);
    try {{ localStorage.setItem('dataforge-theme', next); }} catch(e) {{}}
    setTimeout(() => {{
        if (window.Plotly) {{
            const fg = getComputedStyle(document.body).getPropertyValue('--chart-font').trim();
            const gr = getComputedStyle(document.body).getPropertyValue('--chart-grid').trim();
            document.querySelectorAll('.js-plotly-plot').forEach(el => {{
                try {{
                    Plotly.relayout(el, {{
                        'font.color': fg,
                        'xaxis.gridcolor': gr,
                        'yaxis.gridcolor': gr,
                    }});
                }} catch(e) {{}}
            }});
        }}
    }}, 60);
}}
(function(){{
    try {{
        const saved = localStorage.getItem('dataforge-theme') || 'dark';
        document.body.setAttribute('data-theme', saved);
        document.documentElement.setAttribute('data-theme', saved);
    }} catch(e) {{}}
}})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------
def render_page(csv_bytes: bytes, mode="simple", active_tab="dashboard", message="", sid=""):
    has_data = csv_bytes is not None and len(csv_bytes) > 0
    df = None
    date_col = None
    target_col = None
    decomp = None

    if has_data:
        try:
            df, date_col, target_col, decomp = process_arbitrary_csv(csv_bytes)
            if df is None or len(df) == 0 or target_col is None:
                has_data = False
                message = "The uploaded file is empty or has no numeric columns. Please upload a valid CSV."
        except Exception as e:
            has_data = False
            message = f"Could not read the uploaded file ({_escape(str(e))}). Please upload a valid CSV."
            df, date_col, target_col, decomp = None, None, None, None

    if not sid and has_data:
        sid = _save_session(csv_bytes)

    if has_data and df is not None and target_col and target_col in df.columns:
        total_records = len(df)
        try:
            target_sum = float(pd.to_numeric(df[target_col], errors="coerce").fillna(0).sum())
            target_avg = float(pd.to_numeric(df[target_col], errors="coerce").fillna(0).mean())
        except Exception:
            target_sum, target_avg = 0.0, 0.0
    else:
        total_records = 0
        target_sum = 0.0
        target_avg = 0.0

    safe_target = _escape(str(target_col)) if target_col else "target"
    safe_date = _escape(str(date_col)) if date_col else "date"

    mode_label = "Switch to Analysis Mode" if mode == "simple" else "Switch to Simple Mode"
    next_mode = "analysis" if mode == "simple" else "simple"

    # ---- Sidebar: ALWAYS identical, regardless of data state ----
    nav_html = f'''
    <aside id="sidebar">
        <button class="sidebar-toggle" onclick="toggleCollapse()" title="Collapse / expand">☰</button>
        <a href="/?mode={mode}&tab=dashboard&sid={sid}">
            <span>📊</span><span class="nav-label">Dashboard Overview</span>
        </a>
        <a href="/?mode={mode}&tab=timeseries&sid={sid}">
            <span>📈</span><span class="nav-label">Time Series Analysis</span>
        </a>
        <a href="/?mode={mode}&tab=regression&sid={sid}">
            <span>🤖</span><span class="nav-label">ML Regression Tournament</span>
        </a>
        <div style="margin-top:auto">
            <a href="/?mode={next_mode}&tab={active_tab}&sid={sid}" class="mode-toggle">
                <span>🔄</span><span class="nav-label">{mode_label}</span>
            </a>
        </div>
    </aside>
    '''

    theme_toggle_html = '''
    <div class="theme-toggle" onclick="toggleTheme()" title="Toggle theme">
        <span class="lbl">Theme</span>
        <div class="thumb"></div>
    </div>
    '''

    hero_html = '''
    <div class="hero">
        <a href="/" style="display:inline-block;padding:0;background:none">
            <h1>DataForge</h1>
        </a>
        <p>From CSV to champion model in one click.</p>
    </div>
    '''

    upload_panel = f'''
    <div class="panel">
        <h2>📁 Upload New CSV</h2>
        <p style="color:var(--text-dim)">Drop any schema-agnostic CSV containing date and numeric metrics.</p>
        <form action="/process" method="post" enctype="multipart/form-data">
            <input type="hidden" name="sid" value="{sid}">
            <input type="hidden" name="mode" value="{mode}">
            <input type="hidden" name="tab" value="dashboard">
            <label>Select CSV File</label>
            <input type="file" name="file" accept=".csv" required>
            <button type="submit">Upload &amp; Analyze</button>
        </form>
        {f'<p style="color:var(--accent);font-weight:600">{_escape(message)}</p>' if message else ''}
    </div>
    '''

    # ---------------------------------------------------------------------
    # Helper: friendly "no data" panel for tabs that need data
    # ---------------------------------------------------------------------
    def _no_data_panel(title: str, subtitle: str = "Upload a CSV to view this section.") -> str:
        return f"""
        {hero_html}
        <div class="panel" style="text-align:center;padding:60px 30px">
            <div style="font-size:3rem;margin-bottom:14px">📭</div>
            <h1 style="margin:0 0 8px">{title}</h1>
            <p style="color:var(--text-dim);margin:0 0 24px">{subtitle}</p>
        </div>
        <div style="max-width:640px;margin:0 auto">
            {upload_panel}
        </div>
        """

    # =====================================================================
    # BLANK STATE — sidebar unchanged, only content area is minimal
    # =====================================================================
    if not has_data:
        if active_tab in ("dashboard", ""):
            content_html = f"""
            {hero_html}
            <div style="max-width:640px;margin:0 auto">
                {upload_panel}
                <div style="text-align:center;padding:30px 20px;color:var(--text-dim);font-size:.9rem">
                    <div style="font-size:2.2rem;margin-bottom:10px">🚀</div>
                    <p style="margin:0">
                        Upload your first CSV to unlock the dashboard,
                        visual insights, time series analysis, and the ML regression tournament.
                    </p>
                </div>
            </div>
            """
        elif active_tab == "timeseries":
            content_html = _no_data_panel(
                "Time Series Analysis",
                "Upload a CSV to see the time series chart and rolling statistics.",
            )
        elif active_tab == "regression":
            content_html = _no_data_panel(
                "ML Regression Tournament",
                "Upload a CSV to run the regression model comparison.",
            )
        else:
            content_html = f"""
            {hero_html}
            <div style="max-width:640px;margin:0 auto">
                {upload_panel}
            </div>
            """
        return _render_html_shell(nav_html, theme_toggle_html, content_html)

    # =====================================================================
    # DATA PRESENT
    # =====================================================================
    content_html = ""

    # ---- SIMPLE MODE ----
    if mode == "simple":
        if active_tab in ("dashboard", ""):
            safe_dates = (
                json.dumps(df[date_col].astype(str).tolist())
                if date_col and date_col in df.columns
                else json.dumps(list(range(len(df))))
            )
            safe_values = json.dumps(
                [float(v) for v in pd.to_numeric(df[target_col], errors="coerce").fillna(0).tolist()]
            )
            simple_chart = f"""
            <div class="panel">
                <h2>📊 Data Overview</h2>
                <div id="chartSimple" style="width:100%;height:340px;"></div>
            </div>
            <script>
            (function(){{
                const dates = {safe_dates};
                const values = {safe_values};
                Plotly.newPlot('chartSimple', [{{
                    x: dates, y: values, type: 'bar',
                    marker: {{color: '#f43f5e'}}
                }}], {{
                    paper_bgcolor:'transparent',plot_bgcolor:'transparent',
                    font:{{color:'var(--chart-font)'}},
                    xaxis:{{gridcolor:'var(--chart-grid)'}},yaxis:{{gridcolor:'var(--chart-grid)'}},
                    margin:{{t:10,r:15,b:35,l:45}}
                }}, {{responsive:true}});
            }})();
            </script>
            """
            content_html = f"""
            {hero_html}
            <div class="panel">
                <h1>Executive Summary</h1>
                <p style="color:var(--text-dim)">Minimalist overview of <b>{safe_target}</b>.</p>
                <div class="grid" style="margin-top:20px">
                    <div class="metric"><small>Total Records</small><strong>{total_records:,}</strong></div>
                    <div class="metric"><small>Total {safe_target}</small><strong>{target_sum:,.2f}</strong></div>
                    <div class="metric"><small>Average {safe_target}</small><strong>{target_avg:,.2f}</strong></div>
                </div>
            </div>
            {upload_panel}
            {simple_chart}
            """
        else:
            content_html = f"""
            {hero_html}
            <div class="panel">
                <h1>Analysis View Restricted</h1>
                <p style="color:var(--text-dim)">You are in <b>Simple Mode</b>. Switch to <b>Analysis Mode</b> from the sidebar for technical charts and ML regression.</p>
            </div>
            """

    # ---- ANALYSIS MODE ----
    else:
        if active_tab == "dashboard":
            charts = _build_charts(df, date_col, target_col)
            content_html = f"""
            {hero_html}
            {upload_panel}
            <div class="panel">
                <h1>Analysis Dashboard</h1>
                <p style="color:var(--text-dim)">
                    Date Column: <b>{safe_date}</b> &nbsp;|&nbsp;
                    Target Column: <b>{safe_target}</b>
                </p>
                <div class="grid" style="margin-top:20px">
                    <div class="metric"><small>Total Rows</small><strong>{total_records:,}</strong></div>
                    <div class="metric"><small>Sum of {safe_target}</small><strong>{target_sum:,.2f}</strong></div>
                    <div class="metric"><small>Mean {safe_target}</small><strong>{target_avg:,.2f}</strong></div>
                </div>
            </div>
            <div class="panel">
                <h2>📊 Visual Insights</h2>
                <div class="chart-grid">
                    {charts}
                </div>
            </div>
            """

        elif active_tab == "timeseries":
            safe_dates = (
                json.dumps(df[date_col].astype(str).tolist())
                if date_col and date_col in df.columns
                else json.dumps(list(range(len(df))))
            )
            safe_values = json.dumps(
                [float(v) for v in pd.to_numeric(df[target_col], errors="coerce").fillna(0).tolist()]
            )
            if "rolling_mean_3" in df.columns:
                safe_rolling = json.dumps(
                    [float(v) for v in pd.to_numeric(df["rolling_mean_3"], errors="coerce").fillna(0).tolist()]
                )
            else:
                safe_rolling = safe_values
            content_html = f"""
            {hero_html}
            <div class="panel">
                <h1>Time Series Analysis &amp; Trend Decomposition</h1>
                <p style="color:var(--text-dim)">Interactive Plotly chart showing rolling statistics.</p>
                <div id="tsChart" style="width:100%;height:450px;"></div>
            </div>
            <script>
                const trace1 = {{ x: {safe_dates}, y: {safe_values}, type: 'scatter', mode: 'lines+markers', name: 'Actual {safe_target}', line: {{color: '#f43f5e'}} }};
                const trace2 = {{ x: {safe_dates}, y: {safe_rolling}, type: 'scatter', mode: 'lines', name: '3-Period Rolling Mean', line: {{color: '#38bdf8', dash: 'dash'}} }};
                const layout = {{ paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {{color: 'var(--chart-font)'}}, xaxis: {{gridcolor: 'var(--chart-grid)'}}, yaxis: {{gridcolor: 'var(--chart-grid)'}}, margin: {{t:20, r:20, b:40, l:50}}, showlegend: true }};
                Plotly.newPlot('tsChart', [trace1, trace2], layout, {{responsive: true}});
            </script>
            """

        elif active_tab == "regression":
            try:
                numeric_feats = [
                    c for c in df.select_dtypes(include=[np.number]).columns if c != target_col
                ]
                best_model, results = run_regression_benchmark(
                    df, target_col, numeric_feats[:5] if numeric_feats else ["year"]
                )
                rows = "".join(
                    f"<tr><td>{_escape(k)}</td><td>{v['mean_r2']}</td><td>{v['std_r2']}</td></tr>"
                    for k, v in results.items()
                )
                content_html = f"""
                {hero_html}
                <div class="panel">
                    <h1>Supervised ML Regression Tournament</h1>
                    <p style="color:var(--text-dim)">Cross-validation performance evaluating regression algorithms.</p>
                    <p style="color:#34d399;font-weight:600">🏆 Champion Model: {_escape(str(best_model))}</p>
                    <table>
                        <thead><tr><th>Algorithm</th><th>Mean R² Score</th><th>Std Dev</th></tr></thead>
                        <tbody>{rows}</tbody>
                    </table>
                </div>
                """
            except Exception as e:
                content_html = f"""
                {hero_html}
                <div class="panel">
                    <h1>ML Regression Tournament</h1>
                    <p style="color:var(--accent);font-weight:600">
                        Could not run regression benchmark: {_escape(str(e))}
                    </p>
                    <p style="color:var(--text-dim)">
                        Ensure your CSV has at least 2 numeric columns and enough rows (≥10).
                    </p>
                </div>
                """

        else:
            content_html = f"""
            {hero_html}
            <div class="panel">
                <h1>Unknown Tab</h1>
                <p style="color:var(--text-dim)">Pick a tab from the sidebar.</p>
            </div>
            """

    return _render_html_shell(nav_html, theme_toggle_html, content_html)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(mode: str = "simple", tab: str = "dashboard", sid: str = ""):
    csv_bytes = b""
    if sid:
        loaded = _load_session(sid)
        if loaded is not None:
            csv_bytes = loaded

    return HTMLResponse(render_page(csv_bytes, mode=mode, active_tab=tab, sid=sid))


@app.post("/process", response_class=HTMLResponse)
async def process_form(
    sid: str = Form(""),
    mode: str = Form("simple"),
    tab: str = Form("dashboard"),
    file: UploadFile | None = File(None),
):
    try:
        csv_bytes = _get_bytes(sid=sid or None, upload=file)
        if not csv_bytes:
            return HTMLResponse(render_page(b"", mode=mode, active_tab=tab, sid=""))

        new_sid = _save_session(csv_bytes)
        return HTMLResponse(
            render_page(csv_bytes, mode=mode, active_tab=tab, sid=new_sid)
        )
    except Exception as e:
        return HTMLResponse(
            render_page(b"", mode=mode, active_tab=tab, sid="", message=str(e))
        )


@app.get("/process", response_class=HTMLResponse)
async def get_process(mode: str = "simple", tab: str = "dashboard", sid: str = ""):
    csv_bytes = b""
    if sid:
        loaded = _load_session(sid)
        if loaded is not None:
            csv_bytes = loaded

    return HTMLResponse(render_page(csv_bytes, mode=mode, active_tab=tab, sid=sid))