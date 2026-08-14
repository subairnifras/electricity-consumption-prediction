# Sri Lanka Electricity Demand Prediction — Firefly Web Application

This project predicts Sri Lanka's daily national-grid electricity demand using
a Random Forest model optimized by the Firefly Algorithm.

## Dataset

The included file is:

`data/sri_lanka_daily_electricity_demand.csv`

Columns: date, maximum temperature, rainfall, humidity, public-holiday flag,
and daily electricity demand (MWh). The included data is an **educational,
Sri Lanka-calibrated sample dataset**, not an official CEB measurement series.
It follows realistic local ranges and seasonal patterns so the application can
be run and demonstrated immediately. For formal research, replace the rows with
verified CEB observations while keeping the same column names.

## Setup in Visual Studio Code

1. Extract this ZIP.
2. Open the extracted folder in Visual Studio Code.
3. Run these commands in the VS Code terminal:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open:

`http://127.0.0.1:5000`

The application creates lag features from the previous day and previous week,
uses a chronological 80/20 train-test split, and reports MAE, RMSE, and
R-squared. Its output includes five graphs: actual vs predicted demand,
prediction scatter, residual distribution, residual plot, and Random Forest
feature importance. The optimized model and graphs are saved in the project.

## Additional features

- Month-period forecast: choose a year, start month, and end month. A separate
  button creates a daily-demand graph and total, average, and peak forecasts.
- Demand and cost forecast: choose a month/year and enter a tariff in LKR per
  kWh. A separate button predicts demand and estimates total electricity cost.

Train the model at least once before using either forecast. The cost is a simple
planning estimate and is not an official CEB bill calculation.

## Vercel deployment

This version is Vercel-ready. It includes a bundled trained model, embeds
generated charts directly in HTML, and uses Vercel's temporary directory for
runtime training output. The repository-level GitHub Actions workflow runs CI
before production deployment. See `../VERCEL_DEPLOYMENT.md` for setup steps.
