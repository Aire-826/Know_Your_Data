# model_engine.py
from __future__ import annotations

import io
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def process_arbitrary_csv(csv_bytes: bytes):
    """Return (df, date_col, target_col, decomp_dict)."""
    df = pd.read_csv(io.BytesIO(csv_bytes))

    # Detect date column
    date_col = None
    for c in df.columns:
        lc = c.lower()
        if "date" in lc or "time" in lc or "timestamp" in lc:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() > 0.5:
                    df[c] = parsed
                    date_col = c
                    break
            except Exception:
                pass

    # Detect target = first numeric column that isn't id-like
    target_col = None
    for c in df.select_dtypes(include=[np.number]).columns:
        if "id" not in c.lower():
            target_col = c
            break
    if target_col is None:
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = nums[0] if nums else None

    return df, date_col, target_col, {}


def run_regression_benchmark(df: pd.DataFrame, target_col: str, feature_cols: list[str]):
    """Return (best_model_name, results_dict, extra_dict)."""
    if "__index__" in feature_cols:
        df = df.copy()
        df["__index__"] = np.arange(len(df))

    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce")

    mask = y.notna() & X.notna().any(axis=1)
    X, y = X[mask], y[mask]

    if len(X) < 10:
        raise ValueError("Need at least 10 usable rows.")

    models = {
        "Linear Regression": make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), LinearRegression()),
        "Ridge":             make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), Ridge()),
        "Lasso":             make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), Lasso()),
        "Decision Tree":     make_pipeline(SimpleImputer(strategy="mean"), DecisionTreeRegressor(random_state=42)),
        "Random Forest":     make_pipeline(SimpleImputer(strategy="mean"), RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        "Gradient Boosting": make_pipeline(SimpleImputer(strategy="mean"), GradientBoostingRegressor(random_state=42)),
    }

    cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
    results = {}
    model_comparison = {}   # NEW

    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
            mean_r2 = float(np.nanmean(scores))
            std_r2 = float(np.nanstd(scores))
            results[name] = {"mean_r2": mean_r2, "std_r2": std_r2}

            # NEW: compute MAE + RMSE via CV predictions
            try:
                y_pred_cv = cross_val_predict(model, X, y, cv=cv)
                mae = float(mean_absolute_error(y, y_pred_cv))
                rmse = float(np.sqrt(mean_squared_error(y, y_pred_cv)))
            except Exception:
                mae, rmse = 0.0, 0.0

            model_comparison[name] = {"r2": mean_r2, "mae": mae, "rmse": rmse}
        except Exception:
            results[name] = {"mean_r2": float("nan"), "std_r2": float("nan")}
            model_comparison[name] = {"r2": 0.0, "mae": 0.0, "rmse": 0.0}

    best_model = max(
        results,
        key=lambda k: results[k]["mean_r2"] if not np.isnan(results[k]["mean_r2"]) else -1e9,
    )
    best_r2 = results[best_model]["mean_r2"]

    # Fit best model on full data
    fitted = models[best_model].fit(X, y)
    preds = fitted.predict(X)

    # Subsample pred vs actual
    n = len(y)
    idx = np.linspace(0, n - 1, min(300, n)).astype(int)
    pred_vs_actual = {
        "actual":    [float(v) for v in y.iloc[idx].tolist()],
        "predicted": [float(v) for v in preds[idx].tolist()],
    }

    # Feature importance
    feature_names, feature_importance = [], None
    try:
        last = fitted[-1] if hasattr(fitted, "__getitem__") else fitted
        if hasattr(last, "feature_importances_"):
            feature_importance = [float(v) for v in last.feature_importances_]
            feature_names = list(feature_cols)
        elif hasattr(last, "coef_"):
            coefs = np.ravel(last.coef_)
            feature_importance = [float(abs(v)) for v in coefs]
            feature_names = list(feature_cols)
    except Exception:
        pass

    # 3D view
    three_d = None
    if len(feature_cols) >= 2:
        xx = X[feature_cols[0]].fillna(0).values
        yy = X[feature_cols[1]].fillna(0).values
        idx3 = np.linspace(0, n - 1, min(300, n)).astype(int)
        three_d = {
            "x": [float(v) for v in xx[idx3]],
            "y": [float(v) for v in yy[idx3]],
            "z": [float(v) for v in y.iloc[idx3].tolist()],
        }

    # Correlation matrix (numeric only, no ids)
    correlation_matrix = {"cols": [], "values": []}
    try:
        num = df.select_dtypes(include=[np.number]).copy()
        drop = [c for c in num.columns if "id" in str(c).lower() or num[c].nunique() <= 1]
        num = num.drop(columns=drop, errors="ignore")
        if num.shape[1] >= 2:
            corr = num.corr(numeric_only=True).fillna(0)
            correlation_matrix = {
                "cols": [str(c) for c in corr.columns.tolist()],
                "values": corr.round(3).values.tolist(),
            }
    except Exception:
        pass

    extra = {
        "reasoning": f"{best_model} achieved the highest mean cross-validated R² across folds.",
        "pred_vs_actual": pred_vs_actual,
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "three_d": three_d,
        # NEW keys:
        "model_comparison": model_comparison,
        "best_r2": best_r2,
        "correlation_matrix": correlation_matrix,
    }

    return best_model, results, extra