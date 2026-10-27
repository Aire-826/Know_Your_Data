from __future__ import annotations

import base64
import binascii
import html
import io
import math
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse

from model_engine import (
    load_and_preprocess_data,
    predict_future_client_revenue,
    run_automl_benchmark,
)

MAX_CSV_BYTES = 1_048_576
REQUIRED_COLUMNS = {
    "date",
    "client_id",
    "product_category",
    "unit_price",
    "units_sold",
    "total_revenue",
}
DEFAULT_DATASET = Path(__file__).with_name("client_sales_data.csv")
COLORS = ["#f43f5e", "#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#c084fc"]

app = FastAPI(title="MLPredict AI")

PAGE_STYLE = """
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#0b0f19;color:#f3f4f6}
*{box-sizing:border-box}body{margin:0;background:#0b0f19}
main{max-width:1180px;margin:auto;padding:28px 20px 60px}
.hero,.panel,.metric{background:#131b2e;border:1px solid #3f1d26;border-radius:14px}
.hero{background:linear-gradient(135deg,#2d1319,#131b2e);padding:30px;text-align:center;margin-bottom:20px}
h1{color:#fb7185;margin:0;font-size:2.35rem}h2{font-size:1.2rem;margin-top:0}
.muted{color:#94a3b8}.hero p{margin-bottom:0}
nav{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0 24px}
a{color:#fda4af;text-decoration:none}a:hover{text-decoration:underline}
.panel{padding:22px;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}
.metric{padding:18px;text-align:center}.metric small{display:block;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em}
.metric strong{display:block;color:#fb7185;font-size:1.45rem;margin-top:8px;overflow-wrap:anywhere}
button{border:0;border-radius:8px;background:linear-gradient(135deg,#f43f5e,#be123c);color:white;padding:11px 16px;font-weight:700;cursor:pointer}
button:hover{filter:brightness(1.12)}input,select{background:#0b0f19;border:1px solid #475569;color:#f8fafc;border-radius:7px;padding:10px;width:100%;margin:6px 0 14px}
input[type=file]{padding:8px}.form-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
label{color:#cbd5e1;font-size:.92rem}.notice{padding:12px 15px;border-radius:8px;background:#102d25;border:1px solid #166534;color:#bbf7d0}
.error{background:#3b1720;border-color:#9f1239;color:#fecdd3}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:11px;border-bottom:1px solid #334155}th{color:#fda4af}
svg{width:100%;height:auto}.chart-grid{display:grid;grid-template-columns:2fr 1fr;gap:16px}
.legend{display:flex;gap:10px;align-items:center;margin:8px 0;color:#cbd5e1}.swatch{width:12px;height:12px;border-radius:50%;display:inline-block}
.bar-row{display:grid;grid-template-columns:minmax(90px,180px) 1fr minmax(75px,120px);gap:10px;align-items:center;margin:10px 0;font-size:.85rem}
.bar{height:13px;border-radius:8px;background:#f43f5e}.bar-track{background:#263247;border-radius:8px}
.result{font-size:2rem;color:#fb7185;font-weight:700;text-align:center}
@media(max-width:750px){main{padding:16px 12px}.chart-grid{grid-template-columns:1fr}.hero{padding:24px 12px}h1{font-size:1.9rem}}
</style>
"""


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _read_dataset(csv_bytes: bytes) -> pd.DataFrame:
    if not csv_bytes:
        raise ValueError("The uploaded CSV is empty.")
    if len(csv_bytes) > MAX_CSV_BYTES:
        raise ValueError("CSV files must be 1 MB or smaller.")

    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse the CSV: {exc}") from exc

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if df.empty:
        raise ValueError("The CSV must contain at least five data rows.")
    if len(df) < 5:
        raise ValueError("At least five data rows are required for cross-validation.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in ("unit_price", "units_sold", "total_revenue"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[["date", "client_id", "product_category", "unit_price", "units_sold", "total_revenue"]].isna().any().any():
        raise ValueError("CSV rows must have valid dates, categories, clients, and numeric values.")

    df["client_id"] = df["client_id"].astype(str)
    df["product_category"] = df["product_category"].astype(str)
    return df


def _get_dataset_bytes(dataset: str, upload: UploadFile | None) -> bytes:
    if upload is not None and upload.filename:
        if Path(upload.filename).suffix.lower() != ".csv":
            raise ValueError("Please upload a file with the .csv extension.")
        data = upload.file.read(MAX_CSV_BYTES + 1)
        if len(data) > MAX_CSV_BYTES:
            raise ValueError("CSV files must be 1 MB or smaller.")
        if not data:
            raise ValueError("The uploaded CSV is empty.")
        return data

    if dataset:
        max_encoded_length = ((MAX_CSV_BYTES + 2) // 3) * 4
        if len(dataset) > max_encoded_length:
            raise ValueError("CSV files must be 1 MB or smaller.")
        try:
            data = base64.b64decode(dataset, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("The carried CSV data is invalid. Upload the file again.") from exc
        if len(data) > MAX_CSV_BYTES:
            raise ValueError("CSV files must be 1 MB or smaller.")
        return data

    return DEFAULT_DATASET.read_bytes()


def _dataset_field(dataset_b64: str) -> str:
    return f'<input type="hidden" name="dataset" value="{dataset_b64}">'


def _trend_chart(df: pd.DataFrame) -> str:
    monthly = df.groupby(df["date"].dt.to_period("M"))["total_revenue"].sum()
    values = monthly.to_list()
    width, height, pad = 920, 260, 28
    low, high = min(values), max(values)
    spread = high - low or 1
    points = []
    for index, value in enumerate(values):
        x = pad + index * (width - 2 * pad) / max(len(values) - 1, 1)
        y = height - pad - (value - low) * (height - 2 * pad) / spread
        points.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Monthly revenue trend"><polyline points="{" ".join(points)}" '
        'fill="none" stroke="#f43f5e" stroke-width="4" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def _category_chart(df: pd.DataFrame) -> tuple[str, str]:
    totals = df.groupby("product_category")["total_revenue"].sum().sort_values(ascending=False)
    grand_total = float(totals.sum()) or 1
    radius = 62
    circumference = 2 * math.pi * radius
    offset = 0.0
    circles = []
    legend = []
    for index, (category, amount) in enumerate(totals.items()):
        fraction = float(amount) / grand_total
        segment = fraction * circumference
        color = COLORS[index % len(COLORS)]
        circles.append(
            f'<circle cx="85" cy="85" r="{radius}" fill="none" stroke="{color}" '
            f'stroke-width="22" stroke-dasharray="{segment:.2f} {circumference - segment:.2f}" '
            f'stroke-dashoffset="{-offset:.2f}" transform="rotate(-90 85 85)"/>'
        )
        legend.append(
            f'<div class="legend"><span class="swatch" style="background:{color}"></span>'
            f'{_escape(category)} · ₹ {amount:,.0f}</div>'
        )
        offset += segment
    chart = (
        '<svg viewBox="0 0 170 170" role="img" aria-label="Revenue by product category">'
        + "".join(circles)
        + "</svg>"
    )
    return chart, "".join(legend)


def _client_bars(df: pd.DataFrame) -> str:
    totals = (
        df.groupby("client_id")["total_revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    maximum = float(totals.max()) or 1
    rows = []
    for client, amount in totals.items():
        percent = float(amount) / maximum * 100
        rows.append(
            '<div class="bar-row">'
            f'<span>{_escape(client)}</span>'
            f'<span class="bar-track"><span class="bar" style="display:block;width:{percent:.2f}%"></span></span>'
            f'<span>₹ {amount:,.0f}</span></div>'
        )
    return "".join(rows)


def _render_page(
    df: pd.DataFrame,
    dataset_b64: str,
    *,
    message: str = "",
    is_error: bool = False,
    benchmark: dict[str, Any] | None = None,
    forecast: float | None = None,
    selected_client: str = "",
    selected_category: str = "",
    target_year: int = 2027,
    target_month: int = 6,
    unit_price: float = 500,
    units_sold: int = 150,
) -> str:
    total_revenue = float(df["total_revenue"].sum())
    total_units = int(df["units_sold"].sum())
    clients = sorted(df["client_id"].unique().tolist())
    categories = sorted(df["product_category"].unique().tolist())
    trend = _trend_chart(df)
    donut, legend = _category_chart(df)
    notice = (
        f'<p class="notice{" error" if is_error else ""}">{_escape(message)}</p>'
        if message
        else ""
    )

    options_clients = "".join(
        f'<option value="{_escape(client)}"{" selected" if client == selected_client else ""}>'
        f'{_escape(client)}</option>'
        for client in clients
    )
    options_categories = "".join(
        f'<option value="{_escape(category)}"{" selected" if category == selected_category else ""}>'
        f'{_escape(category)}</option>'
        for category in categories
    )

    benchmark_section = ""
    if benchmark:
        rows = "".join(
            "<tr>"
            f"<td>{_escape(name)}</td>"
            f"<td>{result['mean_r2']:.4f}</td>"
            f"<td>{result['std_r2']:.4f}</td>"
            "</tr>"
            for name, result in sorted(
                benchmark["benchmark_results"].items(),
                key=lambda item: item[1]["mean_r2"],
                reverse=True,
            )
        )
        benchmark_section = (
            '<div class="panel"><h2>Benchmark results</h2>'
            f'<p class="notice">Best model: {_escape(benchmark["best_algorithm"])}</p>'
            "<table><thead><tr><th>Algorithm</th><th>Mean R²</th>"
            f"<th>Standard deviation</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )

    forecast_section = ""
    if forecast is not None:
        forecast_section = (
            '<div class="panel"><h2>Revenue forecast</h2>'
            f'<p class="result">₹ {forecast:,.2f}</p>'
            f'<p class="muted" style="text-align:center">{_escape(selected_client)} · '
            f'{_escape(selected_category)} · {target_year}-{target_month:02d}</p></div>'
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLPredict AI</title>{PAGE_STYLE}</head>
<body><main>
<header class="hero"><h1>⚡ MLPredict AI</h1>
<p class="muted">Client revenue and sales forecasting for XYZ Electricals</p></header>
{notice}
<nav><a href="#dashboard">Executive dashboard</a><a href="#automl">Auto-ML benchmark</a>
<a href="#forecast">Revenue forecast</a></nav>
<section class="panel"><h2>Sales dataset</h2>
<p class="muted">Use the included dataset or upload a CSV (maximum 1 MB). Required columns:
date, client_id, product_category, unit_price, units_sold, total_revenue.</p>
<form method="post" enctype="multipart/form-data">
{_dataset_field(dataset_b64)}
<label for="csv-file">Replace the current dataset</label>
<input id="csv-file" type="file" name="file" accept=".csv,text/csv">
<button type="submit" name="action" value="dashboard">Load dataset</button></form></section>
<section id="dashboard"><h2>Executive dashboard</h2>
<div class="grid">
<div class="metric"><small>Total verified revenue</small><strong>₹ {total_revenue:,.2f}</strong></div>
<div class="metric"><small>Units shipped</small><strong>{total_units:,}</strong></div>
<div class="metric"><small>Active client accounts</small><strong>{len(clients)}</strong></div>
</div>
<div class="chart-grid">
<div class="panel"><h2>Monthly revenue trajectory</h2>{trend}</div>
<div class="panel"><h2>Revenue by product line</h2>{donut}{legend}</div>
</div>
<div class="panel"><h2>Revenue by client (top 10)</h2>{_client_bars(df)}</div>
</section>
<section id="automl" class="panel"><h2>Auto-ML benchmark</h2>
<p class="muted">Compare four regression models using five-fold cross-validation.
The benchmark runs when submitted; models are retrained for forecasting because Vercel
Functions do not guarantee persistent memory.</p>
<form method="post" enctype="multipart/form-data">{_dataset_field(dataset_b64)}
<button type="submit" name="action" value="benchmark">Run Auto-ML benchmark</button></form>
{benchmark_section}</section>
<section id="forecast" class="panel"><h2>Client revenue forecast</h2>
<p class="muted">Forecast a client's revenue using the best-performing benchmark model.</p>
<form method="post" enctype="multipart/form-data">{_dataset_field(dataset_b64)}
<div class="form-grid">
<label>Client<select name="client_id" required>{options_clients}</select></label>
<label>Product line<select name="product_category" required>{options_categories}</select></label>
<label>Projection year<select name="target_year">
{"".join(f'<option value="{year}"{" selected" if year == target_year else ""}>{year}</option>' for year in (2027, 2028, 2029))}
</select></label>
<label>Projection month<input type="number" name="target_month" min="1" max="12" value="{target_month}" required></label>
<label>Unit price (INR)<input type="number" name="unit_price" min="100" step="50" value="{unit_price}" required></label>
<label>Unit volume<input type="number" name="units_sold" min="1" step="1" value="{units_sold}" required></label>
</div>
<button type="submit" name="action" value="forecast">Generate revenue forecast</button></form>
{forecast_section}</section>
</main></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    data = DEFAULT_DATASET.read_bytes()
    df = _read_dataset(data)
    return HTMLResponse(_render_page(df, base64.b64encode(data).decode("ascii")))


@app.post("/", response_class=HTMLResponse)
async def submit(
    action: str = Form(...),
    dataset: str = Form(""),
    file: UploadFile | None = File(default=None),
    client_id: str = Form(""),
    product_category: str = Form(""),
    target_year: int = Form(2027),
    target_month: int = Form(6),
    unit_price: float = Form(500),
    units_sold: int = Form(150),
) -> HTMLResponse:
    try:
        csv_bytes = _get_dataset_bytes(dataset, file)
        df = _read_dataset(csv_bytes)
        dataset_b64 = base64.b64encode(csv_bytes).decode("ascii")

        benchmark = None
        forecast = None
        if action == "benchmark":
            processed, _ = load_and_preprocess_data(io.BytesIO(csv_bytes))
            benchmark = run_automl_benchmark(processed, target_col="total_revenue")
        elif action == "forecast":
            if client_id not in set(df["client_id"]):
                raise ValueError("Select a client present in the current dataset.")
            if product_category not in set(df["product_category"]):
                raise ValueError("Select a product line present in the current dataset.")
            if not 2027 <= target_year <= 2029:
                raise ValueError("Projection year must be 2027, 2028, or 2029.")
            if not 1 <= target_month <= 12:
                raise ValueError("Projection month must be between 1 and 12.")
            if unit_price < 100:
                raise ValueError("Unit price must be at least INR 100.")
            if units_sold < 1:
                raise ValueError("Unit volume must be at least 1.")

            processed, encoders = load_and_preprocess_data(io.BytesIO(csv_bytes))
            benchmark = run_automl_benchmark(processed, target_col="total_revenue")
            forecast_result = predict_future_client_revenue(
                model=benchmark["trained_models"][benchmark["best_algorithm"]],
                encoders=encoders,
                client_id=client_id,
                category=product_category,
                unit_price=unit_price,
                units_sold=units_sold,
                target_year=target_year,
                target_month=target_month,
            )
            if isinstance(forecast_result, str):
                raise ValueError(forecast_result)
            forecast = forecast_result
        elif action != "dashboard":
            raise ValueError("Unknown form action. Reload the page and try again.")

        return HTMLResponse(
            _render_page(
                df,
                dataset_b64,
                message="Dataset loaded successfully." if action == "dashboard" else "",
                benchmark=benchmark,
                forecast=forecast,
                selected_client=client_id,
                selected_category=product_category,
                target_year=target_year,
                target_month=target_month,
                unit_price=unit_price,
                units_sold=units_sold,
            )
        )
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        fallback_data = DEFAULT_DATASET.read_bytes()
        fallback_df = _read_dataset(fallback_data)
        return HTMLResponse(
            _render_page(
                fallback_df,
                base64.b64encode(fallback_data).decode("ascii"),
                message=str(exc),
                is_error=True,
                selected_client=client_id,
                selected_category=product_category,
                target_year=target_year,
                target_month=target_month,
                unit_price=unit_price,
                units_sold=units_sold,
            ),
            status_code=400,
        )
