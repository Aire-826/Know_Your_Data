# MLPredict AI

A Python-only client revenue dashboard and forecasting app, served by FastAPI
and deployable to Vercel Functions.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Open the local URL printed by Uvicorn. The default dashboard data is loaded from
`client_sales_data.csv`.

## Deploy to Vercel

1. Push this repository, including `client_sales_data.csv`, to GitHub.
2. Import the repository in Vercel and leave the project root and build settings
   at their defaults.
3. Deploy. Vercel uses the FastAPI `app` exported from `app.py` and installs
   dependencies from `requirements.txt`.

The app renders its pages and charts on the server and uses ordinary HTML forms;
it has no JavaScript frontend. CSV uploads are limited to 1 MB so data can be
carried between stateless function requests. Uploaded CSV files must contain
`date`, `client_id`, `product_category`, `unit_price`, `units_sold`, and
`total_revenue` columns. Forecast requests retrain the benchmark models because
Vercel Functions do not guarantee persistent memory.