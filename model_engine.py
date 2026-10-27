import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.seasonal import seasonal_decompose

def auto_detect_columns(df):
    """
    Automatically detects date, target, and feature columns for any random CSV.
    """
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    
    # Detect Date column
    date_col = None
    for col in cols:
        if any(keyword in col.lower() for keyword in ['date', 'time', 'period', 'year', 'day', 'month']):
            try:
                pd.to_datetime(df[col], errors='raise')
                date_col = col
                break
            except Exception:
                continue
    if not date_col:
        # Fallback to first datetime-parseable column
        for col in cols:
            try:
                pd.to_datetime(df[col], errors='raise')
                date_col = col
                break
            except Exception:
                continue
    if not date_col:
        date_col = cols[0] # Fallback default

    # Detect Target column (numeric columns like revenue, sales, amount, total, price)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = None
    for col in numeric_cols:
        if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'total', 'amount', 'price', 'profit', 'demand']):
            target_col = col
            break
    if not target_col and numeric_cols:
        target_col = numeric_cols[-1]
    elif not numeric_cols:
        raise ValueError("The CSV must contain at least one numeric column for forecasting/analysis.")

    return date_col, target_col

def process_arbitrary_csv(csv_bytes, date_col=None, target_col=None):
    """
    Cleans exceptions, parses dates, computes rolling stats and time series decomposition.
    """
    df = pd.read_csv(io_bytes_to_stream(csv_bytes))
    df.columns = df.columns.str.strip()
    
    if df.empty or len(df) < 5:
        raise ValueError("Dataset must contain at least 5 rows of data.")

    # Auto-detect columns if not specified
    detected_date, detected_target = auto_detect_columns(df)
    date_col = date_col if date_col in df.columns else detected_date
    target_col = target_col if target_col in df.columns else detected_target

    # Clean missing values & exceptions
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(by=date_col)
    
    # Convert numeric columns, fill NaNs with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

    # Fill categorical missing values
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype(str)

    # Time Series Feature Engineering & Rolling Statistics
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['rolling_mean_3'] = df[target_col].rolling(window=3, min_periods=1).mean()
    df['rolling_std_3'] = df[target_col].rolling(window=3, min_periods=1).std().fillna(0)

    # Time Series Decomposition (Trend, Seasonal) using statsmodels if enough data
    decomposition_data = {}
    try:
        ts_series = df.set_index(date_col)[target_col].resample('D').mean().interpolate()
        if len(ts_series) >= 14:
            period = min(7, len(ts_series) // 2)
            decomposition = seasonal_decompose(ts_series, model='additive', period=period)
            decomposition_data = {
                "trend": decomposition.trend.dropna().tolist(),
                "seasonal": decomposition.seasonal.dropna().tolist(),
                "dates": [str(d.date()) for d in decomposition.trend.dropna().index]
            }
    except Exception:
        pass

    return df, date_col, target_col, decomposition_data

def io_bytes_to_stream(csv_bytes):
    import io
    return io.BytesIO(csv_bytes)

def run_regression_benchmark(df, target_col, feature_cols):
    """
    Supervised ML Regression Tournament (Random Forest, XGBoost, Ridge, Linear).
    """
    X = df[feature_cols]
    y = df[target_col]
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
    }
    
    cv = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            results[name] = {
                "mean_r2": round(float(np.mean(scores)), 4),
                "std_r2": round(float(np.std(scores)), 4)
            }
            model.fit(X, y)
        except Exception:
            results[name] = {"mean_r2": 0.0, "std_r2": 0.0}
            
    best_model_name = max(results, key=lambda k: results[k]["mean_r2"])
    return best_model_name, results