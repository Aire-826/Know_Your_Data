# charts_v2.py
from __future__ import annotations

import html
import json
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared helpers (mirror main.py style)
# ---------------------------------------------------------------------------
def _escape(val):
    return html.escape(str(val), quote=True)


def _plotly_config():
    return "{responsive:true,displayModeBar:false,displaylogo:false}"


def _layout(extra_opts=""):
    base = (
        "{autosize:true,paper_bgcolor:'transparent',plot_bgcolor:'transparent',"
        "font:{family:'IBM Plex Sans, sans-serif',color:'var(--chart-font)',size:12},"
        "xaxis:{gridcolor:'var(--chart-grid)',zeroline:false,automargin:true,tickfont:{size:10}},"
        "yaxis:{gridcolor:'var(--chart-grid)',zeroline:false,automargin:true,tickfont:{size:11}},"
        "margin:{t:20,r:24,b:60,l:60},showlegend:false"
    )
    if extra_opts:
        base = base[:-1] + "," + extra_opts + "}"
    else:
        base += "}"
    return base


# ---------------------------------------------------------------------------
# a. Time-Series with Shaded Confidence Interval
# ---------------------------------------------------------------------------
def build_timeseries_with_ci(df, date_col, target_col, max_points=500):
    """Line + rolling mean + shaded CI band (adaptive to dataset size)."""
    if df is None or len(df) == 0 or target_col not in df.columns:
        return '<div class="chart-box"><h3>No data available</h3></div>'

    try:
        n = len(df)

        # Downsample preserving order (date order assumed)
        if n > max_points:
            idx = np.linspace(0, n - 1, max_points).astype(int)
            sample = df.iloc[idx].reset_index(drop=True)
        else:
            sample = df.reset_index(drop=True)
            idx = np.arange(n)

        # X axis
        if date_col and date_col in sample.columns:
            x_vals = json.dumps(sample[date_col].astype(str).tolist())
        else:
            x_vals = json.dumps([str(int(i) + 1) for i in idx])

        # Y actual
        y = pd.to_numeric(sample[target_col], errors="coerce").fillna(0)
        y_vals = json.dumps([float(v) for v in y.tolist()])

        # Rolling window: scale with size so band is meaningful
        if n <= 200:
            window, z_mult, band_alpha = 7, 1.0, 0.22
        elif n <= 2000:
            window, z_mult, band_alpha = 14, 1.0, 0.15
        else:
            window, z_mult, band_alpha = 30, 1.96, 0.12

        roll_mean = y.rolling(window, min_periods=1).mean()
        roll_std = y.rolling(window, min_periods=1).std().fillna(0)

        upper = (roll_mean + z_mult * roll_std).fillna(method="bfill").fillna(0)
        lower = (roll_mean - z_mult * roll_std).fillna(method="bfill").fillna(0)

        mean_vals = json.dumps([float(v) for v in roll_mean.tolist()])
        upper_vals = json.dumps([float(v) for v in upper.tolist()])
        lower_vals = json.dumps([float(v) for v in lower.tolist()])

        safe_target = _escape(str(target_col))
        note = (
            f'<span style="color:var(--text-mute);font-weight:400;font-size:.78rem">'
            f' (sample of {len(sample)}/{n:,} rows, {window}-period window)</span>'
            if n > max_points else
            f'<span style="color:var(--text-mute);font-weight:400;font-size:.78rem">'
            f' ({window}-period rolling window)</span>'
        )

        layout = _layout(
            "showlegend:true,legend:{orientation:'h',y:-0.2},"
            "tickangle:-45,nmaxticks:20"
        )
        cfg = _plotly_config()

        return f"""
        <div class="chart-box" style="grid-column:1/-1">
            <h3>{safe_target} over time — actual, rolling mean &amp; confidence band{note}</h3>
            <div id="chartTsCI" class="plot wide" style="min-height:400px;max-height:600px"></div>
        </div>
        <script>
        (function(){{
            const x = {x_vals};
            const yActual = {y_vals};
            const yMean   = {mean_vals};
            const yUpper  = {upper_vals};
            const yLower  = {lower_vals};

            Plotly.newPlot('chartTsCI', [
                {{x:x, y:yLower, type:'scatter', mode:'lines',
                  line:{{width:0}}, showlegend:false, hoverinfo:'skip'}},
                {{x:x, y:yUpper, type:'scatter', mode:'lines',
                  line:{{width:0}}, fill:'tonexty',
                  fillcolor:'rgba(6,182,212,{band_alpha})',
                  showlegend:false, hoverinfo:'skip'}},
                {{x:x, y:yActual, type:'scatter', mode:'lines',
                  name:'Actual', line:{{color:'#6366f1', width:1.8}}}},
                {{x:x, y:yMean, type:'scatter', mode:'lines',
                  name:'Rolling mean', line:{{color:'#06b6d4', dash:'dash', width:2}}}}
            ], {layout}, {cfg});
        }})();
        </script>
        """
    except Exception as e:
        return f'<div class="chart-box"><h3>Time-series error: {_escape(str(e))}</h3></div>'


# ---------------------------------------------------------------------------
# b. Actual vs Predicted (enhanced)
# ---------------------------------------------------------------------------
def build_actual_vs_predicted(actual, predicted, r2=None):
    if not actual or not predicted or len(actual) == 0:
        return '<div class="chart-box" style="padding:32px;text-align:center;color:var(--text-dim)">Predictions coming soon.</div>'

    try:
        a = np.array(actual, dtype=float)
        p = np.array(predicted, dtype=float)
        residuals = a - p

        mn = float(min(a.min(), p.min()))
        mx = float(max(a.max(), p.max()))

        a_json = json.dumps([float(v) for v in a.tolist()])
        p_json = json.dumps([float(v) for v in p.tolist()])
        r_json = json.dumps([float(v) for v in residuals.tolist()])

        r2_badge = (
            f'<span style="background:var(--accent-soft);color:var(--accent);'
            f'padding:2px 10px;border-radius:999px;font-size:.75rem;'
            f'font-weight:600;margin-left:8px">R² = {r2:.3f}</span>'
            if r2 is not None and not (isinstance(r2, float) and np.isnan(r2)) else ""
        )

        layout = _layout(
            "showlegend:true,legend:{orientation:'h',y:-0.2},"
            "xaxis:{title:'Actual',gridcolor:'var(--chart-grid)',zeroline:false},"
            "yaxis:{title:'Predicted',gridcolor:'var(--chart-grid)',zeroline:false}"
        )
        # fix: the helper adds xaxis/yaxis; override via extra opts
        layout = (
            "{autosize:true,paper_bgcolor:'transparent',plot_bgcolor:'transparent',"
            "font:{family:'IBM Plex Sans, sans-serif',color:'var(--chart-font)',size:12},"
            "xaxis:{title:'Actual',gridcolor:'var(--chart-grid)',zeroline:false,automargin:true},"
            "yaxis:{title:'Predicted',gridcolor:'var(--chart-grid)',zeroline:false,automargin:true},"
            "margin:{t:20,r:24,b:60,l:60},showlegend:true,legend:{orientation:'h',y:-0.2}"
            "}"
        )
        cfg = _plotly_config()

        return f"""
        <div class="chart-box">
            <h3>Actual vs predicted {r2_badge}</h3>
            <div id="chartPvaV2" class="plot square"></div>
        </div>
        <script>
        (function(){{
            const a = {a_json}, p = {p_json}, r = {r_json};
            const mn = {mn}, mx = {mx};
            Plotly.newPlot('chartPvaV2', [
                {{x:a, y:p, mode:'markers', type:'scatter', name:'Predictions',
                  marker:{{
                    color: r.map(v => Math.abs(v)),
                    colorscale:'RdYlGn', reversescale:true,
                    size:8, opacity:0.85,
                    colorbar:{{title:'|residual|',thickness:8,len:0.6,outlinewidth:0}}
                  }},
                  hovertemplate:'Actual: %{{x:.2f}}<br>Predicted: %{{y:.2f}}<extra></extra>'}},
                {{x:[mn,mx], y:[mn,mx], mode:'lines', name:'Ideal (y=x)',
                  line:{{color:'#94a3b8', dash:'dash', width:1.5}}, hoverinfo:'skip'}}
            ], {layout}, {cfg});
        }})();
        </script>
        """
    except Exception as e:
        return f'<div class="chart-box"><h3>PvA error: {_escape(str(e))}</h3></div>'


# ---------------------------------------------------------------------------
# c. Feature Importance — sorted horizontal bars
# ---------------------------------------------------------------------------
def build_feature_importance_sorted(feature_names, importances, top_n=10):
    if not feature_names or not importances:
        return ""

    try:
        pairs = list(zip(feature_names, [float(v) for v in importances]))
        pairs = [p for p in pairs if not np.isnan(p[1])]
        pairs.sort(key=lambda x: x[1], reverse=True)
        pairs = pairs[:top_n]

        names = [str(p[0]) for p in pairs][::-1]  # reverse so top is at top of bar chart
        vals = [p[1] for p in pairs][::-1]

        names_json = json.dumps(names)
        vals_json = json.dumps(vals)

        layout = (
            "{autosize:true,paper_bgcolor:'transparent',plot_bgcolor:'transparent',"
            "font:{family:'IBM Plex Sans, sans-serif',color:'var(--chart-font)',size:12},"
            "xaxis:{title:'Importance',gridcolor:'var(--chart-grid)',zeroline:false,automargin:true},"
            "yaxis:{gridcolor:'var(--chart-grid)',zeroline:false,automargin:true,tickfont:{size:11}},"
            "margin:{t:20,r:24,b:60,l:160},showlegend:false"
            "}"
        )
        cfg = _plotly_config()

        return f"""
        <div class="chart-box">
            <h3>Feature importance (top {len(names)})</h3>
            <div id="chartFiV2" class="plot square"></div>
        </div>
        <script>
        (function(){{
            Plotly.newPlot('chartFiV2', [{{
                x:{vals_json}, y:{names_json},
                type:'bar', orientation:'h',
                marker:{{
                    color:{vals_json},
                    colorscale:'Viridis',
                    showscale:false
                }},
                text:{vals_json}.map(v => v.toFixed(3)),
                textposition:'outside',
                hovertemplate:'%{{y}}<br>%{{x:.4f}}<extra></extra>'
            }}], {layout}, {cfg});
        }})();
        </script>
        """
    except Exception as e:
        return f'<div class="chart-box"><h3>Feature importance error: {_escape(str(e))}</h3></div>'


# ---------------------------------------------------------------------------
# d. Model Comparison — grouped bar chart
# ---------------------------------------------------------------------------
def build_model_comparison_chart(model_comparison):
    """model_comparison = {model_name: {r2, mae, rmse}, ...}"""
    if not model_comparison:
        return ""

    try:
        models = list(model_comparison.keys())
        r2_vals = [float(model_comparison[m].get("r2", 0)) for m in models]
        mae_vals = [float(model_comparison[m].get("mae", 0)) for m in models]
        rmse_vals = [float(model_comparison[m].get("rmse", 0)) for m in models]

        # Normalize MAE/RMSE to 0-1 for visual comparison with R²
        def _norm(arr):
            mx = max(arr) if max(arr) > 0 else 1
            return [v / mx for v in arr]

        mae_norm = _norm(mae_vals)
        rmse_norm = _norm(rmse_vals)

        models_json = json.dumps(models)
        r2_json = json.dumps(r2_vals)
        mae_json = json.dumps(mae_norm)
        rmse_json = json.dumps(rmse_norm)
        mae_raw = json.dumps(mae_vals)
        rmse_raw = json.dumps(rmse_vals)

        layout = (
            "{autosize:true,paper_bgcolor:'transparent',plot_bgcolor:'transparent',"
            "font:{family:'IBM Plex Sans, sans-serif',color:'var(--chart-font)',size:12},"
            "xaxis:{gridcolor:'var(--chart-grid)',zeroline:false,automargin:true,tickangle:-30},"
            "yaxis:{title:'Score (R² native; MAE/RMSE scaled)',gridcolor:'var(--chart-grid)',"
            "zeroline:false,automargin:true,range:[0,1.05]},"
            "margin:{t:20,r:24,b:80,l:60},"
            "showlegend:true,legend:{orientation:'h',y:-0.25},barmode:'group'"
            "}"
        )
        cfg = _plotly_config()

        return f"""
        <div class="chart-box" style="grid-column:1/-1">
            <h3>Model comparison — R² vs error metrics (scaled)</h3>
            <div id="chartModelCmp" class="plot wide" style="min-height:360px"></div>
        </div>
        <script>
        (function(){{
            const models = {models_json};
            const r2 = {r2_json};
            const maeNorm = {mae_json};
            const rmseNorm = {rmse_json};
            const maeRaw = {mae_raw};
            const rmseRaw = {rmse_raw};

            Plotly.newPlot('chartModelCmp', [
                {{x:models, y:r2, type:'bar', name:'R² (higher=better)',
                  marker:{{color:'#6366f1'}},
                  hovertemplate:'%{{x}}<br>R²: %{{y:.4f}}<extra></extra>'}},
                {{x:models, y:maeNorm, type:'bar', name:'MAE (scaled, lower=better)',
                  marker:{{color:'#f59e0b'}},
                  customdata:maeRaw,
                  hovertemplate:'%{{x}}<br>MAE: %{{customdata:.3f}}<extra></extra>'}},
                {{x:models, y:rmseNorm, type:'bar', name:'RMSE (scaled, lower=better)',
                  marker:{{color:'#ef4444'}},
                  customdata:rmseRaw,
                  hovertemplate:'%{{x}}<br>RMSE: %{{customdata:.3f}}<extra></extra>'}}
            ], {layout}, {cfg});
        }})();
        </script>
        """

    except Exception as e:
        return f'<div class="chart-box"><h3>Model comparison error: {_escape(str(e))}</h3></div>'


# ---------------------------------------------------------------------------
# e. Correlation Heatmap
# ---------------------------------------------------------------------------
def build_correlation_heatmap(df, max_cols=12):
    if df is None or len(df) == 0:
        return ""

    try:
        numeric = df.select_dtypes(include=[np.number]).copy()

        # Drop ID-like columns and constant columns
        drop = []
        for c in numeric.columns:
            lc = str(c).lower()
            if "id" in lc:
                drop.append(c)
            elif numeric[c].nunique() <= 1:
                drop.append(c)
        numeric = numeric.drop(columns=drop, errors="ignore")

        if numeric.shape[1] < 2:
            return ""

        # Cap columns
        if numeric.shape[1] > max_cols:
            numeric = numeric.iloc[:, :max_cols]

        corr = numeric.corr(numeric_only=True).fillna(0)
        cols = corr.columns.tolist()
        vals = corr.round(3).values.tolist()

        cols_json = json.dumps([str(c) for c in cols])
        vals_json = json.dumps(vals)

        layout = (
            "{autosize:true,paper_bgcolor:'transparent',plot_bgcolor:'transparent',"
            "font:{family:'IBM Plex Sans, sans-serif',color:'var(--chart-font)',size:11},"
            "xaxis:{tickangle:-45,automargin:true},"
            "yaxis:{automargin:true},"
            "margin:{t:20,r:24,b:100,l:140},showlegend:false"
            "}"
        )
        cfg = _plotly_config()

        return f"""
        <div class="chart-box" style="grid-column:1/-1">
            <h3>Correlation heatmap — which numeric columns move together</h3>
            <div id="chartCorr" class="plot wide" style="min-height:400px;max-height:600px"></div>
        </div>
        <script>
        (function(){{
            Plotly.newPlot('chartCorr', [{{
                z:{vals_json},
                x:{cols_json},
                y:{cols_json},
                type:'heatmap',
                colorscale:'RdBu',
                zmid:0, zmin:-1, zmax:1,
                colorbar:{{title:'r',thickness:12,outlinewidth:0}},
                hovertemplate:'%{{y}} ↔ %{{x}}<br>r = %{{z:.2f}}<extra></extra>'
            }}], {layout}, {cfg});
        }})();
        </script>
        """
    except Exception as e:
        return f'<div class="chart-box"><h3>Heatmap error: {_escape(str(e))}</h3></div>'