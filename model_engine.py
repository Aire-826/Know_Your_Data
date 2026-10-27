import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

def load_and_preprocess_data(csv_path="client_sales_data.csv"):
    """
    Loads the CSV dataset, extracts date features, and encodes categorical columns.
    """
    df = pd.read_csv(csv_path)
    
    # 1. Date Feature Engineering
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    
    # 2. Categorical Label Encoding
    client_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    
    df['client_encoded'] = client_encoder.fit_transform(df['client_id'])
    df['category_encoded'] = category_encoder.fit_transform(df['product_category'])
    
    # Store encoders for later use if needed
    encoders = {
        "client": client_encoder,
        "category": category_encoder
    }
    
    return df, encoders

def run_automl_benchmark(df, target_col='total_revenue'):
    """
    Benchmarks multiple supervised ML algorithms using 5-Fold Cross-Validation 
    and automatically selects the best performing model.
    """
    # Define features (X) and target (y)
    feature_cols = ['year', 'month', 'quarter', 'client_encoded', 'category_encoded', 'unit_price', 'units_sold']
    if target_col == 'units_sold':
        feature_cols = [c for c in feature_cols if c != 'units_sold']
        
    X = df[feature_cols]
    y = df[target_col]
    
    # Define candidate supervised models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Evaluate using R-squared cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        mean_r2 = float(np.mean(cv_scores))
        std_r2 = float(np.std(cv_scores))
        
        benchmark_results[name] = {
            "mean_r2": round(mean_r2, 4),
            "std_r2": round(std_r2, 4)
        }
        
        # Fit model on full dataset for future predictions
        model.fit(X, y)
        trained_models[name] = model
        
    # Find the best model based on highest mean R^2 score
    best_algorithm = max(benchmark_results, key=lambda k: benchmark_results[k]["mean_r2"])
    
    return {
        "best_algorithm": best_algorithm,
        "benchmark_results": benchmark_results,
        "trained_models": trained_models,
        "feature_cols": feature_cols
    }

def predict_future_client_revenue(model, encoders, client_id, category, unit_price, units_sold, target_year, target_month):
    """
    Generates a prediction for a specific client and product category in a future month.
    """
    try:
        client_enc = encoders['client'].transform([client_id])[0]
        category_enc = encoders['category'].transform([category])[0]
    except ValueError:
        return "Error: Unknown client or category provided."
    
    target_quarter = (target_month - 1) // 3 + 1
    
    input_data = pd.DataFrame([{
        'year': target_year,
        'month': target_month,
        'quarter': target_quarter,
        'client_encoded': client_enc,
        'category_encoded': category_enc,
        'unit_price': unit_price,
        'units_sold': units_sold
    }])
    
    prediction = model.predict(input_data)[0]
    return max(0.0, float(prediction))