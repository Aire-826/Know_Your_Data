import numpy as np
import pandas as pd

np.random.seed(42)

# 4 years of monthly data: Jan 2023 to Dec 2026 (48 months)
dates = pd.date_range(start="2023-01-01", end="2026-12-01", freq="MS")

clients = ["CUST-101 (Apex Builders)", "CUST-102 (Metro Electricals)", 
           "CUST-103 (Luxe Interiors)", "CUST-104 (Urban Infra Corp)", 
           "CUST-105 (Nova Distributors)"]

products = {
    "Modular Switches": {"base_price": 450.0, "base_units": 150},
    "Home Automation": {"base_price": 8500.0, "base_units": 25},
    "Circuit Protection": {"base_price": 1200.0, "base_units": 80}
}

data = []

for date in dates:
    year_index = date.year - 2023
    macro_growth = 1.0 + (year_index * 0.10)
    seasonality_multiplier = 1.25 if date.month in [10, 11, 12] else (0.9 if date.month in [5, 6] else 1.0)
    
    for client_id in clients:
        client_scale = np.random.uniform(0.8, 1.4)
        
        for category, info in products.items():
            unit_price = round(info["base_price"] * (1.0 + (year_index * 0.03) + np.random.uniform(-0.02, 0.02)), 2)
            random_noise = np.random.normal(1.0, 0.15)
            units_sold = int(info["base_units"] * client_scale * macro_growth * seasonality_multiplier * random_noise)
            units_sold = max(5, units_sold)
            total_revenue = round(unit_price * units_sold, 2)
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "client_id": client_id,
                "product_category": category,
                "unit_price": unit_price,
                "units_sold": units_sold,
                "total_revenue": total_revenue
            })

df_synthetic = pd.DataFrame(data)
df_synthetic.to_csv("client_sales_data.csv", index=False)
print("Successfully generated client_sales_data.csv with last 4 years data")