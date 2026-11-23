from __future__ import annotations

import html
import io
import json
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from model_engine import process_arbitrary_csv, run_regression_benchmark

app = FastAPI(title="KnowYourData")

MAX_CSV_BYTES = 2_097_152  # 2MB


# ---------------------------------------------------------------------------
# Session storage
# ---------------------------------------------------------------------------
SESSION_DIR = Path(tempfile.gettempdir()) / "KnowYourData_sessions"
SESSION_DIR.mkdir(exist_ok=True)
_SESSION_CACHE: dict[str, bytes] = {}

# NEW: per-session column preferences (persists across tab switches)
_SESSION_PREFS: dict[str, dict] = {}


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


def _save_prefs(sid: str, prefs: dict) -> None:
    _SESSION_PREFS[sid] = prefs


def _load_prefs(sid: str | None) -> dict:
    if not sid:
        return {}
    return _SESSION_PREFS.get(sid, {})


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
# Column classification
# ---------------------------------------------------------------------------
def _classify_columns(df: pd.DataFrame) -> dict:
    date_cols, numeric_cols, cat_cols, text_cols = [], [], [], []
    for c in df.columns:
        s = df[c]
        lc = str(c).lower()
        if any(k in lc for k in ("date", "time", "timestamp", "day")):
            try:
                parsed = pd.to_datetime(s, errors="coerce")
                if parsed.notna().mean() > 0.5:
                    date_cols.append(c)
                    continue
            except Exception:
                pass
        if pd.api.types.is_numeric_dtype(s):
            if "id" in lc and s.nunique() > 20:
                cat_cols.append(c)
            else:
                numeric_cols.append(c)
            continue
        nunique = s.nunique(dropna=True)
        if nunique <= max(30, int(len(df) * 0.05)):
            cat_cols.append(c)
        else:
            text_cols.append(c)
    return {"date": date_cols, "numeric": numeric_cols,
            "categorical": cat_cols, "text": text_cols}


def _filter_df(df: pd.DataFrame, selected_cols: list[str]) -> pd.DataFrame:
    keep = [c for c in selected_cols if c in df.columns]
    return df[keep].copy() if keep else df.copy()


def _fmt_num(v: float) -> str:
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:.1f}K"
    if abs(v) >= 10:
        return f"{v:.0f}"
    return f"{v:.2f}"


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
PAGE_STYLE = """
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root, html[data-theme="dark"], body[data-theme="dark"]{
  color-scheme:dark;
  --bg:#0a0f1a; --bg-glow-1:rgba(99,102,241,0.10); --bg-glow-2:rgba(6,182,212,0.08);
  --panel:rgba(17,24,39,0.78); --panel-border:rgba(51,65,85,0.55);
  --card:rgba(24,32,48,0.9); --card-border:rgba(51,65,85,0.8);
  --text:#e5e7eb; --text-dim:#94a3b8; --text-mute:#64748b;
  --accent:#6366f1; --accent-2:#06b6d4; --accent-3:#8b5cf6;
  --accent-soft:rgba(99,102,241,0.12);
  --sidebar:rgba(12,18,30,0.96); --sidebar-border:rgba(51,65,85,0.55);
  --input-bg:rgba(9,14,24,0.9); --input-border:#334155;
  --hover:rgba(30,41,59,0.85); --chart-bg:rgba(13,19,32,0.85);
  --chart-grid:rgba(148,163,184,0.10); --chart-font:#94a3b8;
  --success:#10b981; --warn:#f59e0b; --danger:#ef4444;
  --insight-bg:rgba(99,102,241,0.10);
  --insight-border:rgba(99,102,241,0.45);

  --robot-head-1:#3b82f6; --robot-head-2:#facc15;
  --robot-body-1:#2563eb; --robot-body-2:#eab308;
  --robot-eye-glow:#fde047; --robot-antenna:#facc15;
  --robot-cloud-bg:rgba(37,99,235,0.18);
  --robot-cloud-border:#3b82f6; --robot-cloud-text:#e5e7eb;
}
html[data-theme="light"], body[data-theme="light"]{
  color-scheme:light;
  --bg:#f5f7fa; --bg-glow-1:rgba(99,102,241,0.06); --bg-glow-2:rgba(6,182,212,0.06);
  --panel:rgba(255,255,255,0.95); --panel-border:rgba(203,213,225,0.9);
  --card:rgba(248,250,252,0.95); --card-border:rgba(226,232,240,1);
  --text:#0f172a; --text-dim:#475569; --text-mute:#94a3b8;
  --accent:#4f46e5; --accent-2:#0891b2; --accent-3:#7c3aed;
  --accent-soft:rgba(79,70,229,0.08);
  --sidebar:rgba(255,255,255,0.96); --sidebar-border:rgba(203,213,225,0.9);
  --input-bg:#ffffff; --input-border:#cbd5e1;
  --hover:rgba(241,245,249,0.95); --chart-bg:rgba(255,255,255,0.95);
  --chart-grid:rgba(71,85,105,0.10); --chart-font:#475569;
  --success:#059669; --warn:#d97706; --danger:#dc2626;
  --insight-bg:rgba(79,70,229,0.06);
  --insight-border:rgba(79,70,229,0.30);

  --robot-head-1:#ef4444; --robot-head-2:#f97316;
  --robot-body-1:#dc2626; --robot-body-2:#fb923c;
  --robot-eye-glow:#fef08a; --robot-antenna:#f97316;
  --robot-cloud-bg:rgba(239,68,68,0.15);
  --robot-cloud-border:#f97316; --robot-cloud-text:#0f172a;
}
*{box-sizing:border-box}
body{
  margin:0;min-height:100vh;display:flex;
  font-family:'IBM Plex Sans',system-ui,-apple-system,sans-serif;
  font-weight:400;letter-spacing:-0.005em;color:var(--text);
  background:
    radial-gradient(900px 600px at 12% 8%, var(--bg-glow-1), transparent 60%),
    radial-gradient(900px 600px at 92% 92%, var(--bg-glow-2), transparent 60%),
    var(--bg);
  background-attachment:fixed;
  transition:background-color .3s ease, color .3s ease;
}
aside{
  width:250px;background:var(--sidebar);backdrop-filter:blur(16px);
  border-right:1px solid var(--sidebar-border);
  padding:20px 14px;display:flex;flex-direction:column;gap:8px;
  position:fixed;height:100vh;overflow-y:auto;z-index:950;
  transition:width .25s ease,padding .25s ease;
}
aside.collapsed{width:64px;padding:20px 8px}
aside.collapsed .nav-label,aside.collapsed .mode-label{display:none}
aside.collapsed a,aside.collapsed .mode-btn{justify-content:center;padding:11px 0}
.sidebar-toggle{
  background:var(--hover);border:1px solid var(--panel-border);
  color:var(--text-dim);border-radius:8px;padding:7px 10px;cursor:pointer;
  font-size:.95rem;align-self:flex-end;width:auto;margin-bottom:8px;
  font-family:inherit;transition:background .2s ease;
}
.sidebar-toggle:hover{background:var(--panel-border);color:var(--text)}
.content-area{
  margin-left:250px;flex:1;padding:32px 34px;max-width:1320px;
  transition:margin-left .25s ease;width:calc(100% - 250px);
}
body.sidebar-collapsed .content-area{margin-left:64px;width:calc(100% - 64px)}
.menu-toggle{
  display:none;background:none;border:none;color:var(--accent);
  font-size:1.4rem;cursor:pointer;position:fixed;top:14px;left:14px;z-index:1000;
}
.panel{
  background:var(--panel);border:1px solid var(--panel-border);
  border-radius:14px;padding:24px;margin-bottom:20px;
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  box-shadow:0 1px 3px rgba(0,0,0,0.06),0 8px 24px -12px rgba(0,0,0,0.25);
}
.panel h1,.panel h2,.panel h3{font-weight:600;letter-spacing:-0.015em;margin-top:0}
.panel h1{font-size:1.5rem}
.panel h2{font-size:1.15rem}
.panel h3{font-size:1rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}
.metric{
  background:var(--card);border:1px solid var(--card-border);
  padding:18px 20px;border-radius:12px;position:relative;overflow:hidden;
}
.metric small{color:var(--text-dim);text-transform:uppercase;font-size:.68rem;letter-spacing:.09em;font-weight:600}
.metric strong{color:var(--text);font-size:1.55rem;display:block;margin-top:6px;font-weight:700;letter-spacing:-0.02em}
.metric.accent strong{color:var(--accent)}
.hero{text-align:center;padding:36px 0 12px}
.hero h1{
  margin:0;font-size:clamp(2.2rem, 5.5vw, 3.6rem);
  font-weight:700;letter-spacing:-0.035em;line-height:1.08;
  color:var(--text);text-decoration:none;cursor:pointer;display:inline-block;
}
.hero a:hover h1{opacity:.85}
.hero p{margin:10px 0 0;color:var(--text-dim);font-size:.98rem;max-width:640px;margin-left:auto;margin-right:auto}
.theme-toggle{
  position:fixed;top:16px;right:20px;z-index:1001;
  display:flex;align-items:center;gap:8px;
  background:var(--panel);border:1px solid var(--panel-border);
  border-radius:999px;padding:5px;cursor:pointer;backdrop-filter:blur(12px);
  user-select:none;
}
.theme-toggle .thumb{
  width:40px;height:22px;border-radius:999px;position:relative;
  background:var(--accent);transition:background .3s ease;
}
.theme-toggle .thumb::after{
  content:"";position:absolute;top:2px;left:2px;width:18px;height:18px;
  border-radius:50%;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,0.3);
  transition:transform .3s ease;
}
body[data-theme="light"] .theme-toggle .thumb{background:var(--accent-2)}
body[data-theme="light"] .theme-toggle .thumb::after{transform:translateX(18px)}
.theme-toggle .lbl{font-size:.74rem;font-weight:600;color:var(--text-dim);padding:0 8px 0 6px;letter-spacing:.02em}
input[type="text"],input[type="file"],select{
  background:var(--input-bg);border:1px solid var(--input-border);color:var(--text);
  border-radius:8px;padding:10px 12px;width:100%;margin:6px 0 12px;
  font-family:inherit;font-size:.94rem;transition:border-color .2s ease;
}
input[type="text"]:focus,select:focus{outline:none;border-color:var(--accent)}
button{
  background:var(--accent);color:#fff;border:none;padding:11px 20px;
  font-weight:600;font-size:.94rem;border-radius:8px;cursor:pointer;
  width:100%;font-family:inherit;transition:filter .15s ease,transform .1s ease;
}
button:hover{filter:brightness(1.08)}
button:active{transform:scale(0.99)}
button.secondary{background:var(--hover);color:var(--text);border:1px solid var(--panel-border)}
a{color:var(--text-dim);text-decoration:none;display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:8px;transition:background .15s ease,color .15s ease;font-size:.92rem;font-weight:500}
a:hover{background:var(--hover);color:var(--text)}
a.active{background:var(--accent-soft);color:var(--accent);font-weight:600}
.mode-wrap{padding:0 0 14px 0;margin-bottom:8px;border-bottom:1px solid var(--sidebar-border)}
.mode-btn{
  display:flex;align-items:center;justify-content:center;gap:8px;width:100%;
  padding:11px 14px;border-radius:9px;background:var(--accent);color:#fff;
  border:none;font-family:inherit;font-size:.88rem;font-weight:600;cursor:pointer;
  transition:filter .15s ease,transform .1s ease;
}
.mode-btn:hover{filter:brightness(1.08)}
.nav-section-label{font-size:.66rem;text-transform:uppercase;letter-spacing:.12em;color:var(--text-mute);font-weight:600;padding:12px 12px 6px}
aside.collapsed .nav-section-label{display:none}

.upload-row{
  display:flex;align-items:center;justify-content:flex-start;
  gap:36px;padding:36px 12px 36px 0;flex-wrap:wrap;min-height:260px;
}
.robot-wrap{position:relative;flex-shrink:0;margin-right:8px;padding-top:56px}
.cloud-bubble{
  position:absolute;top:0;left:20px;
  background:var(--robot-cloud-bg);border:2px solid var(--robot-cloud-border);
  color:var(--robot-cloud-text);padding:14px 22px;border-radius:24px;
  font-size:1rem;font-weight:600;white-space:nowrap;
  box-shadow:0 8px 24px -10px rgba(0,0,0,0.3);
  transition:background .35s ease, border-color .35s ease, color .35s ease;
  animation:bubbleFloat 2.8s ease-in-out infinite;z-index:3;
}
@keyframes bubbleFloat{
  0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)}
}
.cloud-bubble::before{
  content:"";position:absolute;left:-12px;bottom:-4px;
  width:18px;height:18px;border-radius:50%;
  background:var(--robot-cloud-border);
  border:2px solid var(--robot-cloud-border);
}
.cloud-bubble::after{
  content:"";position:absolute;left:-22px;bottom:-16px;
  width:11px;height:11px;border-radius:50%;
  background:var(--robot-cloud-bg);
  border:2px solid var(--robot-cloud-border);
}
.robot{
  position:relative;width:150px;height:190px;
  animation:robotFloat 3.2s ease-in-out infinite;
  filter:drop-shadow(0 12px 24px rgba(0,0,0,0.28));
}
@keyframes robotFloat{
  0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)}
}
.robot .antenna{
  position:absolute;top:0;left:50%;transform:translateX(-50%);
  width:4px;height:22px;background:var(--robot-antenna);border-radius:2px;
}
.robot .antenna::after{
  content:"";position:absolute;top:-10px;left:50%;transform:translateX(-50%);
  width:14px;height:14px;border-radius:50%;background:var(--robot-antenna);
  box-shadow:0 0 12px var(--robot-antenna), 0 0 24px var(--robot-antenna);
  animation:pulseGlow 1.8s ease-in-out infinite;
}
@keyframes pulseGlow{
  0%,100%{opacity:1;transform:translateX(-50%) scale(1)}
  50%{opacity:.6;transform:translateX(-50%) scale(1.15)}
}
.robot .head{
  position:absolute;top:26px;left:50%;transform:translateX(-50%);
  width:120px;height:80px;border-radius:22px;
  background:linear-gradient(135deg,var(--robot-head-1) 0%,var(--robot-head-2) 100%);
  border:3px solid rgba(255,255,255,0.35);
  display:flex;align-items:center;justify-content:center;gap:14px;
  transition:background .35s ease;
}
.robot .head::before,.robot .head::after{
  content:"";position:absolute;top:50%;transform:translateY(-50%);
  width:8px;height:26px;border-radius:4px;background:var(--robot-head-2);
}
.robot .head::before{left:-11px}.robot .head::after{right:-11px}
.robot .eye{
  width:20px;height:20px;border-radius:50%;background:var(--robot-eye-glow);
  box-shadow:0 0 10px var(--robot-eye-glow), inset 0 0 4px rgba(0,0,0,0.4);
  position:relative;animation:blink 4.5s ease-in-out infinite;
}
.robot .eye::after{
  content:"";position:absolute;top:4px;left:4px;
  width:8px;height:8px;border-radius:50%;background:rgba(0,0,0,0.55);
}
@keyframes blink{
  0%,92%,100%{transform:scaleY(1)} 95%{transform:scaleY(0.1)}
}
.robot .mouth{
  position:absolute;bottom:12px;left:50%;transform:translateX(-50%);
  width:34px;height:6px;border-radius:3px;background:rgba(0,0,0,0.35);
}
.robot .body{
  position:absolute;bottom:0;left:50%;transform:translateX(-50%);
  width:96px;height:80px;border-radius:18px 18px 12px 12px;
  background:linear-gradient(135deg,var(--robot-body-1) 0%,var(--robot-body-2) 100%);
  border:3px solid rgba(255,255,255,0.28);transition:background .35s ease;
}
.robot .body::before{
  content:"";position:absolute;top:18px;left:50%;transform:translateX(-50%);
  width:44px;height:26px;border-radius:8px;background:rgba(0,0,0,0.22);
  box-shadow:inset 0 0 6px rgba(0,0,0,0.3);
}
.robot .body::after{
  content:"";position:absolute;bottom:14px;left:50%;transform:translateX(-50%);
  width:10px;height:10px;border-radius:50%;background:var(--robot-antenna);
  box-shadow:0 0 10px var(--robot-antenna);animation:pulseGlow 1.6s ease-in-out infinite;
}
.robot .arm{
  position:absolute;top:66px;width:14px;height:56px;border-radius:8px;
  background:linear-gradient(180deg,var(--robot-body-1) 0%,var(--robot-body-2) 100%);
  transition:background .35s ease;
}
.robot .arm.left{left:-14px;transform-origin:top center;animation:waveLeft 2.4s ease-in-out infinite}
.robot .arm.right{right:-14px;transform-origin:top center;animation:waveRight 2.4s ease-in-out infinite}
@keyframes waveLeft{0%,100%{transform:rotate(0deg)}50%{transform:rotate(18deg)}}
@keyframes waveRight{0%,100%{transform:rotate(0deg)}50%{transform:rotate(-18deg)}}

.upload-form{flex:1 1 520px;min-width:320px;max-width:820px;display:flex;flex-direction:column;gap:12px;padding:0 8px}
.upload-form form{display:flex;flex-direction:row;gap:12px;align-items:stretch;flex-wrap:wrap}
.upload-form input[type="file"]{
  flex:2 1 340px;min-width:240px;margin:0;padding:14px 16px;font-size:.95rem;
  border-radius:10px;border:1.5px solid var(--input-border);background:var(--input-bg);
}
.upload-form button{
  flex:1 1 220px;min-width:180px;width:auto;margin:0;padding:14px 22px;font-size:.95rem;
  border-radius:10px;background:var(--accent);box-shadow:0 4px 14px -6px var(--accent);
}
.upload-form .hint{color:var(--text-dim);font-size:.86rem;margin:0;padding-left:4px}
.upload-form .error{color:var(--danger);font-weight:600;margin:4px 0 0;padding-left:4px}

.robot-mini{
  width:80px;height:100px;margin:0 auto 12px;position:relative;
  animation:robotFloat 3.2s ease-in-out infinite;
}
.robot-mini .antenna{height:14px}
.robot-mini .antenna::after{width:10px;height:10px;top:-8px}
.robot-mini .head{top:18px;width:72px;height:50px;border-radius:14px;border-width:2px}
.robot-mini .head::before,.robot-mini .head::after{width:5px;height:16px}
.robot-mini .head::before{left:-7px}.robot-mini .head::after{right:-7px}
.robot-mini .eye{width:12px;height:12px}
.robot-mini .eye::after{width:4px;height:4px;top:2px;left:2px}
.robot-mini .mouth{width:20px;height:4px;bottom:8px}
.robot-mini .body{bottom:0;width:58px;height:46px;border-radius:12px;border-width:2px}
.robot-mini .body::before{width:26px;height:16px;top:10px}
.robot-mini .body::after{width:6px;height:6px;bottom:8px}
.robot-mini .arm{display:none}

.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:18px}
.chart-card{
  background:var(--chart-bg);border:1px solid var(--panel-border);
  border-radius:14px;padding:18px 18px 14px;display:flex;flex-direction:column;
  overflow:hidden;transition:transform .15s ease,box-shadow .15s ease;
}
.chart-card:hover{transform:translateY(-2px);box-shadow:0 12px 28px -16px rgba(0,0,0,0.35)}
.chart-card .chart-head{margin-bottom:6px}
.chart-card .chart-title{font-size:.98rem;color:var(--text);font-weight:600;margin:0;display:flex;align-items:center;gap:8px}
.chart-card .chart-sub{font-size:.78rem;color:var(--text-mute);margin:4px 0 0}
.chart-card .plot{width:100%;aspect-ratio:16/10;min-height:260px;position:relative;margin-top:4px}
.chart-card .plot.wide{aspect-ratio:21/9;min-height:320px}
.chart-card .plot.square{aspect-ratio:1/1;min-height:280px}
.chart-card .plot > div{width:100%!important;height:100%!important}
.chart-insight{
  margin-top:12px;padding:10px 14px;border-radius:9px;
  background:var(--insight-bg);border-left:3px solid var(--insight-border);
  color:var(--text-dim);font-size:.83rem;line-height:1.55;
}
.chart-insight b{color:var(--text)}

.best-model-card{
  background:linear-gradient(135deg, var(--accent-soft), transparent 70%);
  border:1px solid var(--accent);border-left:4px solid var(--accent);
  border-radius:10px;padding:18px 20px;margin-bottom:20px;
}
.best-model-card .bm-eyebrow{font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:var(--accent);font-weight:700;margin-bottom:4px}
.best-model-card .bm-name{font-size:1.35rem;font-weight:700;color:var(--text);letter-spacing:-0.02em;margin-bottom:10px}
.best-model-card .bm-reason{font-size:.9rem;color:var(--text-dim);line-height:1.55;padding-left:14px;border-left:2px solid var(--accent-soft)}

table{width:100%;border-collapse:collapse;margin-top:12px;font-size:.9rem}
th,td{padding:11px 12px;text-align:left;border-bottom:1px solid var(--panel-border);color:var(--text)}
th{color:var(--text-dim);font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em}
tr:hover td{background:var(--hover)}

.badge{display:inline-block;padding:3px 8px;border-radius:5px;font-size:.72rem;font-weight:600;background:var(--accent-soft);color:var(--accent);margin-right:4px}

.col-picker{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px;margin-top:12px}
.col-group{margin-bottom:22px}
.col-group h3{margin:0 0 8px;font-size:.92rem;color:var(--accent);display:flex;align-items:center;gap:8px}
.col-chip{
  display:flex;align-items:center;gap:10px;padding:10px 12px;
  background:var(--card);border:1.5px solid var(--card-border);
  border-radius:9px;cursor:pointer;transition:all .15s ease;
  font-size:.9rem;user-select:none;
}
.col-chip:hover{border-color:var(--accent)}
.col-chip input[type="checkbox"]{width:16px;height:16px;margin:0;cursor:pointer;accent-color:var(--accent)}
.col-chip input[type="radio"]{width:16px;height:16px;margin:0;cursor:pointer;accent-color:var(--accent-2)}
.col-chip.checked{background:var(--accent-soft);border-color:var(--accent)}
.col-chip .ctag{
  font-size:.65rem;padding:2px 6px;border-radius:4px;
  background:var(--accent-soft);color:var(--accent);font-weight:600;
  text-transform:uppercase;letter-spacing:.05em;
}
.col-chip .ctag.num{background:rgba(6,182,212,0.15);color:var(--accent-2)}
.col-chip .ctag.date{background:rgba(139,92,246,0.15);color:var(--accent-3)}
.col-chip .ctag.cat{background:rgba(16,185,129,0.15);color:var(--success)}
.col-chip .ctag.text{background:rgba(245,158,11,0.15);color:var(--warn)}
.picker-actions{display:flex;gap:10px;margin-top:20px;flex-wrap:wrap}
.picker-actions button{width:auto;flex:1;min-width:140px}

@media(max-width:900px){
  aside{transform:translateX(-100%);width:250px!important;padding:20px 14px!important}
  aside.collapsed{transform:translateX(-100%)}
  aside.open{transform:translateX(0)}
  aside.open .nav-label,aside.open .mode-label,aside.open .nav-section-label{display:inline}
  .content-area{margin-left:0!important;width:100%!important;padding:20px 16px;margin-top:52px}
  .menu-toggle{display:block}
  .hero h1{font-size:clamp(1.8rem,8vw,2.6rem)}
  .theme-toggle{top:12px;right:12px}
  .upload-row{flex-direction:column;align-items:flex-start;gap:14px;padding:24px 0}
  .robot-wrap{padding-top:52px;margin-left:8px}
  .cloud-bubble{left:8px;white-space:normal;max-width:240px}
  .upload-form{width:100%;padding:0}
}
</style>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
"""


def _escape(val):
    return html.escape(str(val), quote=True)


def _plotly_config():
    return "{responsive:true,displayModeBar:false,displaylogo:false}"


def _base_layout(extra: str = "") -> str:
    base = (
        "{autosize:true,paper_bgcolor:'transparent',plot_bgcolor:'transparent',"
        "font:{family:'IBM Plex Sans, sans-serif',color:'var(--chart-font)',size:12},"
        "xaxis:{gridcolor:'var(--chart-grid)',zeroline:false,automargin:true,"
        "tickfont:{size:11},linecolor:'var(--chart-grid)'},"
        "yaxis:{gridcolor:'var(--chart-grid)',zeroline:false,automargin:true,"
        "tickfont:{size:11},linecolor:'var(--chart-grid)'},"
        "margin:{t:20,r:24,b:56,l:60},showlegend:false,"
        "hoverlabel:{bgcolor:'rgba(15,23,42,0.92)',bordercolor:'transparent',"
        "font:{color:'#fff',family:'IBM Plex Sans',size:12}}"
    )
    if extra:
        base += "," + extra
    base += "}"
    return base


# ---------------------------------------------------------------------------
# Robot markup
# ---------------------------------------------------------------------------
def _robot_only_html() -> str:
    return """
    <div class="robot" aria-hidden="true">
        <div class="antenna"></div>
        <div class="arm left"></div>
        <div class="arm right"></div>
        <div class="head">
            <div class="eye"></div>
            <div class="eye"></div>
            <div class="mouth"></div>
        </div>
        <div class="body"></div>
    </div>
    """


def _robot_mini_html() -> str:
    return """
    <div class="robot-mini">
        <div class="antenna"></div>
        <div class="head">
            <div class="eye"></div>
            <div class="eye"></div>
        </div>
        <div class="body"></div>
    </div>
    """


def _robot_cloud_row(message: str = "Upload CSV here") -> str:
    safe_msg = _escape(message)
    return f"""
    <div class="upload-row">
        <div class="robot-wrap">
            <div class="cloud-bubble">{safe_msg}</div>
            {_robot_only_html()}
        </div>
        <div class="upload-form">
            <form action="/process" method="post" enctype="multipart/form-data">
                <input type="hidden" name="mode" value="simple">
                <input type="file" name="file" accept=".csv" required>
                <button type="submit">Upload &amp; Configure</button>
            </form>
            <p class="hint">Any CSV file up to 2 MB. We'll auto-detect columns on the next page.</p>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Charts — 3 charts + time series
# ---------------------------------------------------------------------------
def _build_charts(df, date_col, target_col, selected_cols=None):
    empty = ('<div class="chart-card"><h3 class="chart-title">No data available</h3></div>', "")
    if df is None or len(df) == 0 or target_col is None or target_col not in df.columns:
        return empty

    try:
        y_all = pd.to_numeric(df[target_col], errors="coerce").dropna()
        if len(y_all) == 0:
            return empty

        cfg = _plotly_config()
        safe_target = _escape(str(target_col))

        v_min = float(y_all.min())
        v_max = float(y_all.max())
        v_mean = float(y_all.mean())
        v_med = float(y_all.median())
        n_total = len(y_all)

        values_full = json.dumps([float(v) for v in y_all.tolist()])
        counts, edges = np.histogram(y_all.values, bins=20)
        peak_idx = int(np.argmax(counts))
        peak_lo, peak_hi = float(edges[peak_idx]), float(edges[peak_idx + 1])
        peak_pct = counts[peak_idx] / n_total * 100

        hist_layout = _base_layout(
            "bargap:0.06,xaxis:{" + f"title:{{text:'{safe_target} value',font:{{size:11,color:'var(--chart-font)'}}}}" +
            "},yaxis:{" + "title:{text:'Count',font:{size:11,color:'var(--chart-font)'}}" + "},"
            "shapes:["
            f"{{type:'line',x0:{v_med},x1:{v_med},y0:0,y1:1,yref:'paper',"
            "line:{color:'#8b5cf6',width:2,dash:'dash'}}]"
            ",annotations:["
            f"{{x:{v_med},y:1,yref:'paper',text:'Median {_fmt_num(v_med)}',showarrow:false,"
            "font:{size:10,color:'#8b5cf6'},yshift:8}}]"
        )
        hist_js = f"""
        Plotly.newPlot('chartHist', [{{
            x:{values_full}, type:'histogram', nbinsx:20,
            marker:{{
                color:'rgba(6,182,212,0.75)',
                line:{{color:'rgba(6,182,212,1)',width:1}}
            }},
            hovertemplate:'Range: %{{x}}<br>Sales: %{{y}}<extra></extra>'
        }}], {hist_layout}, {cfg});
        """
        hist_insight = (
            f"💡 Most values fall between <b>{_fmt_num(peak_lo)}</b> and "
            f"<b>{_fmt_num(peak_hi)}</b> ({peak_pct:.0f}% of all rows). "
            f"Median is <b>{_fmt_num(v_med)}</b> — meaning half of your data is above and half below."
        )

        # Categorical bar
        cat_candidates = []
        search_cols = selected_cols if selected_cols else list(df.columns)
        for c in search_cols:
            if c == target_col or c == date_col:
                continue
            if c not in df.columns:
                continue
            s = df[c]
            if s.dtype == object or s.nunique() <= 20:
                if 2 <= s.nunique() <= 20:
                    cat_candidates.append(c)

        cat_col = None
        for c in cat_candidates:
            if any(k in str(c).lower() for k in ("type", "category", "promotion", "weather", "event", "status")):
                cat_col = c
                break
        if cat_col is None and cat_candidates:
            cat_col = cat_candidates[0]

        if cat_col is not None:
            tmp = df[[cat_col, target_col]].copy()
            tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
            tmp = tmp.dropna()
            grp = tmp.groupby(cat_col)[target_col].mean().sort_values(ascending=False).head(10)
            cats = [str(i) for i in grp.index.tolist()]
            vals = [float(v) for v in grp.values.tolist()]
            best_cat = cats[0] if cats else "—"
            worst_cat = cats[-1] if cats else "—"
            best_val = vals[0] if vals else 0
            worst_val = vals[-1] if vals else 0
            ratio = (best_val / worst_val) if worst_val else 0

            cats_json = json.dumps(cats)
            vals_json = json.dumps(vals)

            bar_layout = _base_layout(
                "bargap:0.35,"
                f"xaxis:{{title:{{text:'{_escape(str(cat_col))}',font:{{size:11,color:'var(--chart-font)'}}}},tickangle:-30,automargin:true}},"
                f"yaxis:{{title:{{text:'Avg {safe_target}',font:{{size:11,color:'var(--chart-font)'}}}}}}"
            )
            bar_js = f"""
            (function(){{
                const cats = {cats_json};
                const vals = {vals_json};
                const maxV = Math.max(...vals);
                const colors = vals.map(v => v === maxV
                    ? 'rgba(99,102,241,1)'
                    : 'rgba(99,102,241,0.55)');
                Plotly.newPlot('chartCat', [{{
                    x: cats, y: vals, type:'bar',
                    marker:{{color: colors, line:{{color:'rgba(99,102,241,1)',width:0}}}},
                    text: vals.map(v => v.toFixed(2)),
                    textposition:'outside',
                    textfont:{{size:10,color:'var(--chart-font)'}},
                    hovertemplate:'<b>%{{x}}</b><br>Avg: %{{y:.2f}}<extra></extra>'
                }}], {bar_layout}, {cfg});
            }})();
            """
            cat_insight = (
                f"💡 <b>{_escape(str(best_cat))}</b> has the highest average "
                f"<b>{_fmt_num(best_val)}</b> — about <b>{ratio:.1f}×</b> more than "
                f"<b>{_escape(str(worst_cat))}</b> ({_fmt_num(worst_val)}). "
                f"Grouping by <b>{_escape(str(cat_col))}</b> reveals the biggest differences."
            )
            cat_chart_html = f"""
            <div class="chart-card">
                <div class="chart-head">
                    <h3 class="chart-title">📊 Average {safe_target} by {_escape(str(cat_col))}</h3>
                    <p class="chart-sub">Ranked highest → lowest. The tallest bar stands out in color.</p>
                </div>
                <div id="chartCat" class="plot wide"></div>
                <div class="chart-insight">{cat_insight}</div>
            </div>
            <script>(function(){{{bar_js}}})();</script>
            """
        else:
            top = y_all.nlargest(10).reset_index(drop=True)
            xs = [f"#{i+1}" for i in range(len(top))]
            ys = [float(v) for v in top.tolist()]
            xs_json = json.dumps(xs)
            ys_json = json.dumps(ys)
            bar_layout = _base_layout(
                f"xaxis:{{title:{{text:'Top sale #',font:{{size:11,color:'var(--chart-font)'}}}}}},"
                f"yaxis:{{title:{{text:'{safe_target}',font:{{size:11,color:'var(--chart-font)'}}}}}}"
            )
            bar_js = f"""
            Plotly.newPlot('chartCat', [{{
                x:{xs_json}, y:{ys_json}, type:'bar',
                marker:{{color:'rgba(99,102,241,0.75)',line:{{color:'rgba(99,102,241,1)',width:0}}}},
                text: {ys_json}.map(v => v.toFixed(2)),
                textposition:'outside',
                textfont:{{size:10,color:'var(--chart-font)'}},
                hovertemplate:'<b>%{{x}}</b><br>Value: %{{y:.2f}}<extra></extra>'
            }}], {bar_layout}, {cfg});
            """
            cat_insight = (
                f"💡 Your top 10 sales range from <b>{_fmt_num(min(ys))}</b> "
                f"to <b>{_fmt_num(max(ys))}</b>. No categorical column was picked — "
                f"re-configure columns to see 'avg by category' instead."
            )
            cat_chart_html = f"""
            <div class="chart-card">
                <div class="chart-head">
                    <h3 class="chart-title">📊 Top 10 highest {safe_target}</h3>
                    <p class="chart-sub">The biggest sales in your data.</p>
                </div>
                <div id="chartCat" class="plot wide"></div>
                <div class="chart-insight">{cat_insight}</div>
            </div>
            <script>(function(){{{bar_js}}})();</script>
            """

        hist_chart_html = f"""
        <div class="chart-card">
            <div class="chart-head">
                <h3 class="chart-title">📈 Distribution of {safe_target}</h3>
                <p class="chart-sub">How many rows fall into each value range. Purple line = median.</p>
            </div>
            <div id="chartHist" class="plot square"></div>
            <div class="chart-insight">{hist_insight}</div>
        </div>
        <script>(function(){{{hist_js}}})();</script>
        """

        num_feats = [
            c for c in (selected_cols if selected_cols else df.columns)
            if c in df.columns and c != target_col and c != date_col
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        scatter_html = ""
        if num_feats:
            feat = num_feats[0]
            tmp = df[[feat, target_col]].copy()
            tmp[feat] = pd.to_numeric(tmp[feat], errors="coerce")
            tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
            tmp = tmp.dropna()
            if len(tmp) > 500:
                tmp = tmp.sample(500, random_state=42)
            xs = [float(v) for v in tmp[feat].tolist()]
            ys = [float(v) for v in tmp[target_col].tolist()]
            try:
                corr = float(np.corrcoef(xs, ys)[0, 1])
            except Exception:
                corr = 0.0
            strength = "strong" if abs(corr) >= 0.6 else "moderate" if abs(corr) >= 0.3 else "weak"
            direction = "increases" if corr > 0 else "decreases"

            xs_json = json.dumps(xs)
            ys_json = json.dumps(ys)
            scatter_layout = _base_layout(
                f"xaxis:{{title:{{text:'{_escape(str(feat))}',font:{{size:11,color:'var(--chart-font)'}}}},gridcolor:'var(--chart-grid)',zeroline:false}},"
                f"yaxis:{{title:{{text:'{safe_target}',font:{{size:11,color:'var(--chart-font)'}}}},gridcolor:'var(--chart-grid)',zeroline:false}}"
            )
            scatter_js = f"""
            Plotly.newPlot('chartScatter', [{{
                x:{xs_json}, y:{ys_json}, mode:'markers', type:'scattergl',
                marker:{{color:'rgba(139,92,246,0.75)', size:8,
                    line:{{color:'rgba(139,92,246,1)', width:0.5}}}},
                hovertemplate:'{_escape(str(feat))}: %{{x}}<br>{safe_target}: %{{y:.2f}}<extra></extra>'
            }}], {scatter_layout}, {cfg});
            """
            scatter_html = f"""
            <div class="chart-card">
                <div class="chart-head">
                    <h3 class="chart-title">🎯 {safe_target} vs {_escape(str(feat))}</h3>
                    <p class="chart-sub">Each dot = one row. Look for a trend line shape.</p>
                </div>
                <div id="chartScatter" class="plot square"></div>
                <div class="chart-insight">
                    💡 Correlation is <b>{corr:+.2f}</b> ({strength} {direction}). 
                    {"Higher " + _escape(str(feat)) + " tends to come with higher " + safe_target + "." if corr > 0.15 else "Lower " + _escape(str(feat)) + " tends to come with higher " + safe_target + "." if corr < -0.15 else "There is no clear linear relationship between these two."}
                </div>
            </div>
            <script>(function(){{{scatter_js}}})();</script>
            """

        return hist_chart_html + cat_chart_html + scatter_html, ""
    except Exception as e:
        return f'<div class="chart-card"><h3 class="chart-title">Error: {_escape(str(e))}</h3></div>', ""


def _build_timeseries_chart(df, date_col, target_col):
    if df is None or len(df) == 0 or target_col not in df.columns:
        return '<div class="chart-card"><h3 class="chart-title">No data available</h3></div>'

    MAX_POINTS = 500
    n = len(df)
    if n > MAX_POINTS:
        idx = np.linspace(0, n - 1, MAX_POINTS).astype(int)
        sample = df.iloc[idx]
    else:
        sample = df
        idx = np.arange(n)

    if date_col and date_col in sample.columns:
        safe_dates = json.dumps(sample[date_col].astype(str).tolist())
    else:
        safe_dates = json.dumps([str(int(i) + 1) for i in idx])

    y = pd.to_numeric(sample[target_col], errors="coerce").fillna(0)
    safe_values = json.dumps([float(v) for v in y.tolist()])

    roll = pd.Series([float(v) for v in y.tolist()]).rolling(3, min_periods=1).mean()
    roll = roll.fillna(0).replace([np.inf, -np.inf], 0)
    safe_rolling = json.dumps([float(v) for v in roll.tolist()])

    cfg = _plotly_config()
    safe_target = _escape(str(target_col))
    note = f'<span style="color:var(--text-mute);font-weight:400;font-size:.78rem"> (showing {len(sample)} of {n:,} rows)</span>' if n > MAX_POINTS else ""

    ts_layout = _base_layout(
        "showlegend:true,legend:{orientation:'h',x:0.5,xanchor:'center',y:1.08,font:{size:11}},"
        "xaxis:{gridcolor:'var(--chart-grid)',zeroline:false,tickangle:-30,automargin:true},"
        "yaxis:{gridcolor:'var(--chart-grid)',zeroline:false,automargin:true},"
        "margin:{t:50,r:24,b:60,l:60},"
        "hovermode:'x unified'"
    )
    y_list = [float(v) for v in y.tolist()]
    max_y = max(y_list) if y_list else 0

    ts_js = f"""
    (function(){{
        const actual = {{x:{safe_dates}, y:{safe_values}, type:'scatter', mode:'lines',
            name:'Actual {safe_target}', line:{{color:'#6366f1', width:2.5, shape:'spline', smoothing:0.5}},
            fill:'tozeroy', fillcolor:'rgba(99,102,241,0.14)',
            hovertemplate:'%{{x}}<br>Actual: %{{y:.2f}}<extra></extra>' }};
        const roll = {{x:{safe_dates}, y:{safe_rolling}, type:'scatter', mode:'lines',
            name:'3-period rolling mean', line:{{color:'#06b6d4', dash:'dash', width:2}},
            hovertemplate:'%{{x}}<br>Rolling: %{{y:.2f}}<extra></extra>' }};
        Plotly.newPlot('tsChart', [actual, roll], {ts_layout}, {cfg});
    }})();
    """

    insight = (
        f"💡 The solid line is your actual <b>{safe_target}</b>; "
        f"the dashed line smooths the noise. Peak value is <b>{_fmt_num(max_y)}</b>. "
        f"Look for upward or downward trends over time."
    )

    return f"""
    <div class="chart-card">
        <div class="chart-head">
            <h3 class="chart-title">📈 {safe_target} over time {note}</h3>
            <p class="chart-sub">Solid = actual · dashed = 3-period rolling mean (smooths bumps).</p>
        </div>
        <div id="tsChart" class="plot wide" style="min-height:400px"></div>
        <div class="chart-insight">{insight}</div>
    </div>
    <script>(function(){{{ts_js}}})();</script>
    """


# ---------------------------------------------------------------------------
# Regression block
# ---------------------------------------------------------------------------
def _build_regression_block(df, target_col, best_model, results, extra=None):
    extra = extra or {}
    reasoning = extra.get("reasoning", "")
    pred_vs_actual = extra.get("pred_vs_actual")
    feature_names = extra.get("feature_names", [])
    feature_importance = extra.get("feature_importance")
    three_d = extra.get("three_d")

    cfg = _plotly_config()

    reason_html = f'<div class="bm-reason">{_escape(reasoning)}</div>' if reasoning else ""
    best_card = f"""
    <div class="best-model-card">
        <div class="bm-eyebrow">Champion model</div>
        <div class="bm-name">{_escape(str(best_model))}</div>
        {reason_html}
    </div>
    """

    rows = []
    for k, v in results.items():
        mean_r2 = v.get("mean_r2", v.get("r2", 0))
        std_r2 = v.get("std_r2", v.get("std", 0))
        highlight = ' style="background:var(--accent-soft);font-weight:600"' if k == best_model else ""
        rows.append(
            f"<tr{highlight}><td>{_escape(k)}</td>"
            f"<td>{mean_r2:.4f}</td><td>{std_r2:.4f}</td></tr>"
        )
    table = f"""
    <table>
        <thead><tr><th>Algorithm</th><th>Mean R²</th><th>Std Dev</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    """

    pva_html = ""
    if pred_vs_actual and pred_vs_actual.get("actual") and pred_vs_actual.get("predicted"):
        actual = json.dumps([float(x) for x in pred_vs_actual["actual"]])
        predicted = json.dumps([float(x) for x in pred_vs_actual["predicted"]])
        pva_layout = _base_layout(
            "showlegend:true,legend:{orientation:'h',x:0.5,xanchor:'center',y:1.08,font:{size:11}},"
            "xaxis:{title:{text:'Actual'},gridcolor:'var(--chart-grid)',zeroline:false},"
            "yaxis:{title:{text:'Predicted'},gridcolor:'var(--chart-grid)',zeroline:false}"
        )
        pva_html = f"""
        <div class="chart-card">
            <div class="chart-head">
                <h3 class="chart-title">🎯 Predicted vs Actual — {_escape(str(best_model))}</h3>
                <p class="chart-sub">Dots near the dashed line = good predictions.</p>
            </div>
            <div id="chartPva" class="plot square"></div>
        </div>
        <script>
        (function(){{
            const a = {actual}, p = {predicted};
            const mn = Math.min(...a, ...p), mx = Math.max(...a, ...p);
            Plotly.newPlot('chartPva', [
                {{x:a, y:p, mode:'markers', type:'scatter', name:'Predictions',
                  marker:{{color:'rgba(99,102,241,0.75)',size:8,line:{{color:'#fff',width:1}}}}}},
                {{x:[mn,mx], y:[mn,mx], mode:'lines', type:'scatter', name:'Ideal (y=x)',
                  line:{{color:'#94a3b8',dash:'dash',width:1.5}}}}
            ], {pva_layout}, {cfg});
        }})();
        </script>
        """
    else:
        pva_html = '<div class="chart-card" style="padding:32px;text-align:center;color:var(--text-dim)">Predictions coming soon.</div>'

    fi_html = ""
    if feature_names and feature_importance:
        fn = json.dumps([str(n) for n in feature_names])
        fi = json.dumps([float(v) for v in feature_importance])
        fi_layout = _base_layout(
            "margin:{t:20,r:24,b:60,l:160},"
            "xaxis:{title:{text:'Importance'},gridcolor:'var(--chart-grid)',zeroline:false},"
            "yaxis:{gridcolor:'var(--chart-grid)',zeroline:false,automargin:true}"
        )
        fi_html = f"""
        <div class="chart-card">
            <div class="chart-head">
                <h3 class="chart-title">🏷️ Feature Importance — {_escape(str(best_model))}</h3>
                <p class="chart-sub">Which columns matter most for predicting {_escape(str(target_col))}.</p>
            </div>
            <div id="chartFi" class="plot wide"></div>
        </div>
        <script>
        (function(){{
            Plotly.newPlot('chartFi', [{{
                x:{fi}, y:{fn}, type:'bar', orientation:'h',
                marker:{{color:'rgba(139,92,246,0.8)',line:{{color:'rgba(139,92,246,1)',width:0}}}},
                hovertemplate:'<b>%{{y}}</b><br>Importance: %{{x:.3f}}<extra></extra>'
            }}], {fi_layout}, {cfg});
        }})();
        </script>
        """

    three_d_html = ""
    if three_d and three_d.get("x"):
        x = json.dumps([float(v) for v in three_d["x"]])
        yv = json.dumps([float(v) for v in three_d["y"]])
        z = json.dumps([float(v) for v in three_d["z"]])
        three_d_html = f"""
        <div class="chart-card" style="grid-column:1/-1">
            <div class="chart-head">
                <h3 class="chart-title">🧊 3D view — two features vs target</h3>
                <p class="chart-sub">Rotate with your mouse to explore.</p>
            </div>
            <div id="chart3d" class="plot wide" style="min-height:440px"></div>
        </div>
        <script>
        (function(){{
            Plotly.newPlot('chart3d', [{{
                x:{x}, y:{yv}, z:{z}, mode:'markers', type:'scatter3d',
                marker:{{size:4,color:{z},colorscale:'Viridis',opacity:0.85}}
            }}], {{
                autosize:true, paper_bgcolor:'transparent',
                scene:{{
                    xaxis:{{title:'Feature 1',gridcolor:'var(--chart-grid)',color:'var(--chart-font)'}},
                    yaxis:{{title:'Feature 2',gridcolor:'var(--chart-grid)',color:'var(--chart-font)'}},
                    zaxis:{{title:'{_escape(str(target_col))}',gridcolor:'var(--chart-grid)',color:'var(--chart-font)'}}
                }},
                font:{{family:'IBM Plex Sans',color:'var(--chart-font)'}},
                margin:{{t:10,r:10,b:10,l:10}}
            }}, {cfg});
        }})();
        </script>
        """

    return f"""
    {best_card}
    <div class="panel">
        <h2>Model leaderboard</h2>
        {table}
    </div>
    <div class="panel">
        <h2>Visual diagnosis</h2>
        <div class="chart-grid">
            {pva_html}
            {fi_html}
        </div>
    </div>
    {f'<div class="panel"><h2>Interaction view</h2><div class="chart-grid">{three_d_html}</div></div>' if three_d_html else ''}
    """


# ---------------------------------------------------------------------------
# HTML shell
# ---------------------------------------------------------------------------
def _render_html_shell(nav_html, theme_toggle_html, content_html):
    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KnowYourData — Analytics & Machine Learning</title>
<script>
(function(){{
  try {{
    const saved = localStorage.getItem('KnowYourData-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
  }} catch(e) {{}}
}})();
</script>
{PAGE_STYLE}
</head>
<body data-theme="dark">
<button class="menu-toggle" onclick="toggleSidebar()">&#9776;</button>
{theme_toggle_html}
{nav_html}
<main class="content-area">
{content_html}
</main>
<script>
function toggleSidebar() {{ document.getElementById('sidebar').classList.toggle('open'); }}
function toggleCollapse() {{
    const s = document.getElementById('sidebar');
    s.classList.toggle('collapsed');
    document.body.classList.toggle('sidebar-collapsed');
    setTimeout(() => {{
        window.dispatchEvent(new Event('resize'));
        if (window.Plotly) {{
            document.querySelectorAll('.js-plotly-plot').forEach(el => {{
                try {{ Plotly.Plots.resize(el); }} catch(e) {{}}
            }});
        }}
    }}, 300);
}}
function toggleTheme() {{
    const cur = document.body.getAttribute('data-theme') || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    document.body.setAttribute('data-theme', next);
    document.documentElement.setAttribute('data-theme', next);
    try {{ localStorage.setItem('KnowYourData-theme', next); }} catch(e) {{}}
    setTimeout(() => {{
        if (window.Plotly) {{
            const fg = getComputedStyle(document.body).getPropertyValue('--chart-font').trim();
            const gr = getComputedStyle(document.body).getPropertyValue('--chart-grid').trim();
            document.querySelectorAll('.js-plotly-plot').forEach(el => {{
                try {{
                    Plotly.relayout(el, {{'font.color': fg,'xaxis.gridcolor': gr,'yaxis.gridcolor': gr}});
                    Plotly.Plots.resize(el);
                }} catch(e) {{}}
            }});
        }}
    }}, 60);
}}
(function(){{
    try {{
        const saved = localStorage.getItem('KnowYourData-theme') || 'dark';
        document.body.setAttribute('data-theme', saved);
        document.documentElement.setAttribute('data-theme', saved);
    }} catch(e) {{}}
}})();
let _rs;
window.addEventListener('resize', () => {{
    clearTimeout(_rs);
    _rs = setTimeout(() => {{
        if (window.Plotly) {{
            document.querySelectorAll('.js-plotly-plot').forEach(el => {{
                try {{ Plotly.Plots.resize(el); }} catch(e) {{}}
            }});
        }}
    }}, 150);
}});

document.addEventListener('DOMContentLoaded', function() {{
    document.querySelectorAll('.col-chip').forEach(function(chip) {{
        const input = chip.querySelector('input');
        if (!input) return;
        if (input.checked) chip.classList.add('checked');
        input.addEventListener('change', function() {{
            if (input.type === 'checkbox') {{
                chip.classList.toggle('checked', input.checked);
            }} else if (input.type === 'radio') {{
                document.querySelectorAll('input[name="' + input.name + '"]').forEach(function(r) {{
                    r.closest('.col-chip').classList.toggle('checked', r.checked);
                }});
            }}
        }});
    }});
    document.querySelectorAll('[data-select-all]').forEach(function(btn) {{
        btn.addEventListener('click', function(e) {{
            e.preventDefault();
            const group = btn.getAttribute('data-select-all');
            document.querySelectorAll('input[name="' + group + '"]').forEach(function(inp) {{
                inp.checked = true;
                inp.closest('.col-chip').classList.add('checked');
            }});
        }});
    }});
    document.querySelectorAll('[data-clear-all]').forEach(function(btn) {{
        btn.addEventListener('click', function(e) {{
            e.preventDefault();
            const group = btn.getAttribute('data-clear-all');
            document.querySelectorAll('input[name="' + group + '"]').forEach(function(inp) {{
                inp.checked = false;
                inp.closest('.col-chip').classList.remove('checked');
            }});
        }});
    }});
}});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Nav — FIXED: all links go to /process (GET handler exists)
# ---------------------------------------------------------------------------
def _build_nav(sid: str, mode: str, active_tab: str) -> str:
    mode_label = "Switch to Analysis Mode" if mode == "simple" else "Switch to Simple Mode"
    next_mode = "analysis" if mode == "simple" else "simple"

    def q(tab: str, override_mode: str | None = None) -> str:
        params = {"mode": override_mode or mode, "tab": tab}
        if sid:
            params["sid"] = sid
        return "/process?" + urlencode(params)

    reconfigure_q = ("/configure?" + urlencode({"sid": sid})) if sid else "/"

    return f'''
    <aside id="sidebar">
        <button class="sidebar-toggle" onclick="toggleCollapse()" title="Collapse">&#9776;</button>
        <div class="mode-wrap">
            <a href="{q(active_tab, next_mode)}" class="mode-btn">
                <span class="mode-label">{mode_label}</span>
            </a>
        </div>
        <div class="nav-section-label">Navigation</div>
        <a href="{q('dashboard')}" class="{'active' if active_tab in ('dashboard','') else ''}">
            <span class="nav-label">Dashboard Overview</span>
        </a>
        <a href="{q('timeseries')}" class="{'active' if active_tab == 'timeseries' else ''}">
            <span class="nav-label">Time Series Analysis</span>
        </a>
        <a href="{q('regression')}" class="{'active' if active_tab == 'regression' else ''}">
            <span class="nav-label">ML Regression Tournament</span>
        </a>
        <div class="nav-section-label">Data</div>
        <a href="{reconfigure_q}">
            <span class="nav-label">↺ Re-configure columns</span>
        </a>
    </aside>
    '''


def _theme_toggle_html() -> str:
    return '''
    <div class="theme-toggle" onclick="toggleTheme()" title="Toggle theme">
        <span class="lbl">Theme</span>
        <div class="thumb"></div>
    </div>
    '''


# ---------------------------------------------------------------------------
# Configure page
# ---------------------------------------------------------------------------
def render_configure_page(csv_bytes: bytes, sid: str = "", message: str = "") -> str:
    if not csv_bytes:
        return _render_html_shell(
            _build_nav("", "simple", "dashboard"),
            _theme_toggle_html(),
            '<div class="panel"><h1>No data</h1><p>Please upload a CSV first.</p>'
            '<a href="/" style="color:var(--accent)">← Go to upload</a></div>',
        )

    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as e:
        return _render_html_shell(
            _build_nav("", "simple", "dashboard"),
            _theme_toggle_html(),
            f'<div class="panel"><h1>Could not read CSV</h1><p style="color:var(--danger)">{_escape(str(e))}</p></div>',
        )

    groups = _classify_columns(df)

    def chip(col: str, kind: str, checked: bool = False) -> str:
        tag_class = {"date": "date", "numeric": "num", "categorical": "cat", "text": "text"}[kind]
        tag_label = {"date": "date", "numeric": "number", "categorical": "category", "text": "text"}[kind]
        return f'''
        <label class="col-chip {'checked' if checked else ''}">
            <input type="checkbox" name="cols" value="{_escape(col)}" {'checked' if checked else ''}>
            <span style="flex:1">{_escape(col)}</span>
            <span class="ctag {tag_class}">{tag_label}</span>
        </label>
        '''

    def radio_chip(col: str, group: str, kind: str, checked: bool = False) -> str:
        tag_class = {"date": "date", "numeric": "num"}.get(kind, "cat")
        tag_label = {"date": "date", "numeric": "number"}.get(kind, "col")
        return f'''
        <label class="col-chip {'checked' if checked else ''}">
            <input type="radio" name="{group}" value="{_escape(col)}" {'checked' if checked else ''}>
            <span style="flex:1">{_escape(col)}</span>
            <span class="ctag {tag_class}">{tag_label}</span>
        </label>
        '''

    sections = []

    if groups["date"]:
        chips = "".join(
            radio_chip(c, "date_col", "date", checked=(i == 0))
            for i, c in enumerate(groups["date"])
        )
        sections.append(f'''
        <div class="col-group">
            <h3>📅 Date column <span style="font-weight:400;font-size:.78rem;color:var(--text-mute)">(pick one — used for time series)</span></h3>
            <div class="col-picker">
                {chips}
                <label class="col-chip">
                    <input type="radio" name="date_col" value="">
                    <span style="flex:1"><i>No date column</i></span>
                </label>
            </div>
        </div>
        ''')

    if groups["numeric"]:
        target_chips = "".join(
            radio_chip(c, "target_col", "numeric", checked=(i == 0))
            for i, c in enumerate(groups["numeric"])
        )
        sections.append(f'''
        <div class="col-group">
            <h3>🎯 Target column <span style="font-weight:400;font-size:.78rem;color:var(--text-mute)">(pick one — what you want to study)</span></h3>
            <div class="col-picker">
                {target_chips}
            </div>
        </div>
        ''')

        feat_chips = []
        for c in groups["numeric"]:
            feat_chips.append(chip(c, "numeric", checked=True))
        for c in groups["categorical"]:
            feat_chips.append(chip(c, "categorical", checked=True))
        for c in groups["date"]:
            feat_chips.append(chip(c, "date", checked=False))
        for c in groups["text"]:
            feat_chips.append(chip(c, "text", checked=False))

        sections.append(f'''
        <div class="col-group">
            <h3>🧩 Include columns in analysis
                <span style="font-weight:400;font-size:.78rem;color:var(--text-mute)">(which columns should charts & ML use?)</span>
            </h3>
            <div style="display:flex;gap:8px;margin-bottom:8px;flex-wrap:wrap">
                <button type="button" class="secondary" data-select-all="cols" style="width:auto;padding:6px 12px;font-size:.8rem">Select all</button>
                <button type="button" class="secondary" data-clear-all="cols" style="width:auto;padding:6px 12px;font-size:.8rem">Clear all</button>
            </div>
            <div class="col-picker">
                {''.join(feat_chips)}
            </div>
        </div>
        ''')

    if not groups["numeric"]:
        content = f"""
        <div class="panel">
            <h1>Cannot analyze this file</h1>
            <p style="color:var(--danger)">Your CSV has no numeric column. At least one number column is required for charts and ML.</p>
            <a href="/" class="mode-btn" style="margin-top:14px;text-decoration:none">← Upload a different file</a>
        </div>
        """
        return _render_html_shell(_build_nav(sid, "analysis", "dashboard"), _theme_toggle_html(), content)

    hero = '''
    <div class="hero">
        <a href="/" style="display:inline-block;padding:0;background:none;color:inherit">
            <h1>Configure your analysis</h1>
        </a>
        <p>Tell KnowYourData what to focus on. Everything on the next page depends on your choices.</p>
    </div>
    '''

    content = f"""
    {hero}
    <div class="panel">
        <h1>Step 2 — Choose columns</h1>
        <p style="color:var(--text-dim)">We detected <b>{len(df)} rows</b> and <b>{len(df.columns)} columns</b> in your file.
        Pick one <b>date column</b> and one <b>target column</b>, then check the columns you want included in charts & ML.</p>
        {f'<p style="color:var(--danger);font-weight:600">{_escape(message)}</p>' if message else ''}

        <form action="/analyze" method="post">
            <input type="hidden" name="sid" value="{_escape(sid)}">
            {''.join(sections)}
            <div class="picker-actions">
                <button type="submit">▶ Run analysis with these choices</button>
                <a href="/" class="secondary" style="display:flex;align-items:center;justify-content:center;color:var(--text);background:var(--hover);border:1px solid var(--panel-border);border-radius:8px;padding:11px 20px;font-weight:600;font-size:.94rem;text-decoration:none">← Upload another file</a>
            </div>
        </form>
    </div>
    """

    return _render_html_shell(_build_nav(sid, "analysis", "dashboard"), _theme_toggle_html(), content)


# ---------------------------------------------------------------------------
# Main renderer — reads prefs if not explicitly passed
# ---------------------------------------------------------------------------
def render_page(
    csv_bytes: bytes,
    mode: str = "simple",
    active_tab: str = "dashboard",
    message: str = "",
    sid: str = "",
    selected_cols: list[str] | None = None,
    target_col: str | None = None,
    date_col: str | None = None,
):
    # Load saved prefs for this session if caller didn't pass explicit values
    prefs = _load_prefs(sid)
    if selected_cols is None and prefs.get("selected_cols"):
        selected_cols = prefs["selected_cols"]
    if target_col is None and prefs.get("target_col"):
        target_col = prefs["target_col"]
    if date_col is None and prefs.get("date_col") is not None:
        date_col = prefs["date_col"]

    has_data = csv_bytes is not None and len(csv_bytes) > 0
    df = None

    if has_data:
        try:
            df_full, auto_date, auto_target, decomp = process_arbitrary_csv(csv_bytes)
            if selected_cols:
                df = _filter_df(df_full, selected_cols)
            else:
                df = df_full

            if date_col is not None:
                date_col = date_col if (date_col and date_col in df.columns) else None
            else:
                date_col = auto_date if (auto_date in df.columns) else None

            if target_col and target_col in df.columns:
                pass
            else:
                nums = df.select_dtypes(include=[np.number]).columns.tolist()
                nums = [c for c in nums if "id" not in str(c).lower()]
                target_col = nums[0] if nums else None

            if df is None or len(df) == 0 or target_col is None:
                has_data = False
                message = "The selected columns are empty or have no numeric data."
        except Exception as e:
            has_data = False
            message = f"Could not read file: {_escape(str(e))}"
            df = None

    if not sid and has_data:
        sid = _save_session(csv_bytes)

    if has_data and df is not None and target_col and target_col in df.columns:
        total_records = len(df)
        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
        target_sum = float(y.sum())
        target_avg = float(y.mean())
    else:
        total_records, target_sum, target_avg = 0, 0.0, 0.0

    safe_target = _escape(str(target_col)) if target_col else "target"
    safe_date = _escape(str(date_col)) if date_col else "index"

    nav_html = _build_nav(sid, mode, active_tab)
    theme_toggle_html = _theme_toggle_html()

    robot_upload = _robot_cloud_row("Upload CSV here")
    if message:
        robot_upload = robot_upload.replace(
            "</form>",
            f"</form><p class='error'>{_escape(message)}</p>",
            1,
        )

    hero_html = '''
    <div class="hero">
        <a href="/" style="display:inline-block;padding:0;background:none;color:inherit">
            <h1>KnowYourData</h1>
        </a>
        <p>Upload a CSV to generate statistical charts and benchmark supervised regression models, all in one place.</p>
    </div>
    '''

    def _no_data_panel(title: str, subtitle: str = "Upload a CSV to view this section.") -> str:
        return f"""
        {hero_html}
        <div class="panel" style="text-align:center;padding:56px 30px">
            {_robot_mini_html()}
            <h1 style="margin:0 0 10px">{title}</h1>
            <p style="color:var(--text-dim);margin:0 0 24px">{subtitle}</p>
        </div>
        <div style="max-width:900px;margin:0 auto">{robot_upload}</div>
        """

    if not has_data:
        if active_tab in ("dashboard", ""):
            content_html = f"""
            {hero_html}
            <div style="max-width:1000px;margin:0 auto;padding-left:8px">
                {robot_upload}
                <div style="text-align:center;padding:24px 20px;color:var(--text-dim);font-size:.88rem">
                    <p style="margin:0">Upload your first CSV to unlock the dashboard, visual insights, time series analysis, and the ML regression tournament.</p>
                </div>
            </div>
            """
        elif active_tab == "timeseries":
            content_html = _no_data_panel("Time Series Analysis", "Upload a CSV to see the time series and rolling statistics.")
        elif active_tab == "regression":
            content_html = _no_data_panel("ML Regression Tournament", "Upload a CSV to run the regression model comparison.")
        else:
            content_html = f"{hero_html}<div style='max-width:1000px;margin:0 auto'>{robot_upload}</div>"
        return _render_html_shell(nav_html, theme_toggle_html, content_html)

    content_html = ""
    cfg = _plotly_config()

    if mode == "simple":
        if active_tab in ("dashboard", ""):
            safe_dates = json.dumps(df[date_col].astype(str).tolist()) if date_col and date_col in df.columns else json.dumps([str(i+1) for i in range(len(df))])
            y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
            safe_values = json.dumps([float(v) for v in y.tolist()])
            simple_layout = _base_layout(
                "xaxis:{gridcolor:'var(--chart-grid)',zeroline:false,tickangle:-45,automargin:true,nmaxticks:15},"
                "yaxis:{gridcolor:'var(--chart-grid)',zeroline:false,automargin:true}"
            )
            simple_chart = f"""
            <div class="panel">
                <h2>Data overview</h2>
                <div id="chartSimple" class="plot wide" style="width:100%"></div>
            </div>
            <script>
            (function(){{
                const dates = {safe_dates}, values = {safe_values};
                Plotly.newPlot('chartSimple', [{{
                    x: dates, y: values, type: 'bar',
                    marker: {{color: 'rgba(99,102,241,0.75)', line:{{color:'rgba(99,102,241,1)',width:0}}}},
                    hovertemplate:'<b>%{{x}}</b><br>Value: %{{y:.2f}}<extra></extra>'
                }}], {simple_layout}, {cfg});
            }})();
            </script>
            """
            content_html = f"""
            {hero_html}
            <div class="panel">
                <h1>Executive summary</h1>
                <p style="color:var(--text-dim)">Minimalist overview of <b>{safe_target}</b>.</p>
                <div class="grid" style="margin-top:20px">
                    <div class="metric accent"><small>Total records</small><strong>{total_records:,}</strong></div>
                    <div class="metric"><small>Total {safe_target}</small><strong>{target_sum:,.2f}</strong></div>
                    <div class="metric"><small>Average {safe_target}</small><strong>{target_avg:,.2f}</strong></div>
                </div>
            </div>
            <div style="max-width:1000px;margin:0 auto">{robot_upload}</div>
            {simple_chart}
            """
        else:
            content_html = f"""
            {hero_html}
            <div class="panel">
                <h1>Analysis view restricted</h1>
                <p style="color:var(--text-dim)">You are in <b>Simple Mode</b>. Switch to <b>Analysis Mode</b> from the sidebar for technical charts and ML benchmarking.</p>
            </div>
            """
    else:
        if active_tab == "dashboard":
            charts, _ = _build_charts(df, date_col, target_col, selected_cols)
            selected_info = ""
            if selected_cols:
                badges = "".join(f'<span class="badge">{_escape(c)}</span>' for c in selected_cols)
                selected_info = f'<p style="color:var(--text-dim);margin-top:6px">Included columns: {badges}</p>'
            content_html = f"""
            {hero_html}
            <div class="panel">
                <h1>Analysis dashboard</h1>
                <p style="color:var(--text-dim)">
                    Date column: <b>{safe_date}</b> &nbsp;|&nbsp; Target column: <b>{safe_target}</b>
                </p>
                {selected_info}
                <div class="grid" style="margin-top:20px">
                    <div class="metric accent"><small>Total rows</small><strong>{total_records:,}</strong></div>
                    <div class="metric"><small>Sum of {safe_target}</small><strong>{target_sum:,.2f}</strong></div>
                    <div class="metric"><small>Mean {safe_target}</small><strong>{target_avg:,.2f}</strong></div>
                </div>
            </div>
            <div class="panel">
                <h2>Visual insights</h2>
                <div class="chart-grid">{charts}</div>
            </div>
            """
        elif active_tab == "timeseries":
            content_html = f"""
            {hero_html}
            <div class="panel">
                <h1>Time Series Analysis</h1>
                <p style="color:var(--text-dim)">Actual values against a three-period rolling mean.</p>
                <div class="chart-grid">{_build_timeseries_chart(df, date_col, target_col)}</div>
            </div>
            """
        elif active_tab == "regression":
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_feats = [c for c in numeric_cols if c != target_col]
                if not numeric_feats:
                    numeric_feats = [c for c in df.columns if c != target_col][:6]
                    if not numeric_feats:
                        numeric_feats = ["__index__"]
                result = run_regression_benchmark(df, target_col, numeric_feats[:6])
                if isinstance(result, tuple) and len(result) == 3:
                    best_model, results, extra = result
                else:
                    best_model, results = result
                    extra = {}
                content_html = f"""
                {hero_html}
                <div class="panel">
                    <h1>Supervised ML Regression Tournament</h1>
                    <p style="color:var(--text-dim)">Cross-validated comparison of supervised regression algorithms on your <b>selected columns</b>.</p>
                </div>
                {_build_regression_block(df, target_col, best_model, results, extra)}
                """
            except Exception as e:
                content_html = f"""
                {hero_html}
                <div class="panel">
                    <h1>ML Regression Tournament</h1>
                    <p style="color:var(--danger);font-weight:600">Could not run benchmark: {_escape(str(e))}</p>
                    <p style="color:var(--text-dim)">Ensure your CSV has at least two numeric columns and at least 10 rows.</p>
                </div>
                """
        else:
            content_html = f"{hero_html}<div class='panel'><h1>Unknown tab</h1></div>"

    return _render_html_shell(nav_html, theme_toggle_html, content_html)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(mode: str = "simple", tab: str = "dashboard", sid: str = ""):
    if sid:
        return RedirectResponse(url="/configure?" + urlencode({"sid": sid}), status_code=307)
    return HTMLResponse(render_page(b"", mode=mode, active_tab=tab, sid=""))


@app.post("/process", response_class=HTMLResponse)
async def process_form(
    mode: str = Form("simple"),
    tab: str = Form("dashboard"),
    file: UploadFile | None = File(None),
):
    try:
        csv_bytes = _get_bytes(upload=file)
        if not csv_bytes:
            return HTMLResponse(render_page(b"", mode=mode, active_tab=tab, sid="", message="No file uploaded."))
        new_sid = _save_session(csv_bytes)
        return RedirectResponse(
            url="/configure?" + urlencode({"sid": new_sid}), status_code=303
        )
    except Exception as e:
        return HTMLResponse(render_page(b"", mode=mode, active_tab=tab, sid="", message=str(e)))


@app.get("/configure", response_class=HTMLResponse)
async def configure_page(sid: str = ""):
    csv_bytes = _load_session(sid) or b""
    return HTMLResponse(render_configure_page(csv_bytes, sid=sid))


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    sid: str = Form(""),
    date_col: str = Form(""),
    target_col: str = Form(""),
    cols: list[str] = Form(default=[]),
):
    csv_bytes = _load_session(sid) or b""
    if not csv_bytes:
        return HTMLResponse(
            render_page(b"", mode="analysis", active_tab="dashboard", sid="",
                        message="Session expired. Please re-upload.")
        )
    if not target_col:
        return HTMLResponse(
            render_configure_page(csv_bytes, sid=sid, message="Please pick a target column.")
        )
    if not cols:
        return HTMLResponse(
            render_configure_page(csv_bytes, sid=sid,
                                  message="Please select at least one column to include.")
        )
    if target_col not in cols:
        cols = [target_col] + cols

    # Persist the user's choices for this session
    _save_prefs(sid, {
        "selected_cols": cols,
        "target_col": target_col,
        "date_col": date_col or None,
    })

    return HTMLResponse(
        render_page(
            csv_bytes,
            mode="analysis",
            active_tab="dashboard",
            sid=sid,
            selected_cols=cols,
            target_col=target_col,
            date_col=date_col or None,
        )
    )


# FIXED: this is the GET handler the sidebar links now point to
@app.get("/process", response_class=HTMLResponse)
async def get_process(mode: str = "simple", tab: str = "dashboard", sid: str = ""):
    csv_bytes = _load_session(sid) or b""
    if not csv_bytes:
        return RedirectResponse(url="/", status_code=307)
    return HTMLResponse(render_page(csv_bytes, mode=mode, active_tab=tab, sid=sid))