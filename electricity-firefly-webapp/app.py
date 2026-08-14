import os
import base64
from io import BytesIO
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_and_prepare_data
from src.firefly_algorithm import FireflyOptimizer

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, "data", "sri_lanka_daily_electricity_demand.csv")
RESULTS = os.path.join(BASE, "results")
STATIC = os.path.join(BASE, "static")
RUNTIME_RESULTS = os.path.join("/tmp", "electricity-results") if os.environ.get("VERCEL") else RESULTS
os.makedirs(RUNTIME_RESULTS, exist_ok=True)

MONTHS = [(1, "January"), (2, "February"), (3, "March"), (4, "April"),
          (5, "May"), (6, "June"), (7, "July"), (8, "August"),
          (9, "September"), (10, "October"), (11, "November"), (12, "December")]

def page_context(**extra):
    return {"months": MONTHS, "years": range(2020, 2031),
            "model_ready": os.path.exists(model_path("optimized_random_forest.pkl")), **extra}

def model_path(filename):
    runtime_file = os.path.join(RUNTIME_RESULTS, filename)
    bundled_file = os.path.join(RESULTS, filename)
    return runtime_file if os.path.exists(runtime_file) else bundled_file

def figure_data_uri():
    """Return the current Matplotlib figure as an embedded PNG."""
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close()
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return "data:image/png;base64," + encoded

def forecast_period(year, start_month, end_month):
    saved_model = model_path("optimized_random_forest.pkl")
    saved_scaler = model_path("scaler.pkl")
    if not os.path.exists(saved_model) or not os.path.exists(saved_scaler):
        raise ValueError("Please train the model first, then use the prediction features.")
    if start_month > end_month:
        raise ValueError("Start month must be before or equal to end month.")
    model, scaler = joblib.load(saved_model), joblib.load(saved_scaler)
    data = pd.read_csv(DATASET)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")
    weather = data.groupby(data["date"].dt.month)[["max_temperature_c", "rainfall_mm", "humidity_pct"]].mean()
    actual = dict(zip(data["date"], data["daily_demand_mwh"]))
    fallback = float(data["daily_demand_mwh"].median())
    selected_start = pd.Timestamp(year=year, month=start_month, day=1)
    selected_end = pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(1)
    last_known = data["date"].max()
    calculation_start = selected_start if selected_start <= last_known else last_known + pd.Timedelta(days=1)
    predicted = {}

    def predict_date(date):
        climate = weather.loc[date.month]
        lag_1 = actual.get(date - pd.Timedelta(days=1), predicted.get(date - pd.Timedelta(days=1), fallback))
        lag_7 = actual.get(date - pd.Timedelta(days=7), predicted.get(date - pd.Timedelta(days=7), fallback))
        holiday = int(date.dayofweek == 6 or (date.month, date.day) in {(1, 1), (2, 4), (4, 13), (4, 14), (5, 1), (12, 25)})
        features = pd.DataFrame([[
            climate["max_temperature_c"], climate["rainfall_mm"], climate["humidity_pct"],
            holiday, date.dayofweek, date.month, date.dayofyear, lag_1, lag_7
        ]], columns=["max_temperature_c", "rainfall_mm", "humidity_pct", "holiday",
                     "day_of_week", "month", "day_of_year", "lag_1_demand_mwh", "lag_7_demand_mwh"])
        return float(model.predict(scaler.transform(features))[0])

    for date in pd.date_range(calculation_start, selected_end, freq="D"):
        predicted[date] = predict_date(date)
    dates = pd.date_range(selected_start, selected_end, freq="D")
    for date in dates:
        if date not in predicted:
            predicted[date] = predict_date(date)
    return dates, np.array([predicted[date] for date in dates])

@app.route("/")
def index():
    return render_template("index.html", **page_context())

@app.route("/train", methods=["POST"])
def train():
    if not os.path.exists(DATASET):
        return render_template("index.html", **page_context(error="Sri Lankan dataset not found in the data folder."))

    try:
        population = int(request.form.get("population_size", 5))
        generations = int(request.form.get("generations", 3))

        X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(DATASET)
        X_opt, X_val, y_opt, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        optimizer = FireflyOptimizer(population, generations)
        best = optimizer.optimize(X_opt, y_opt, X_val, y_val)

        model = RandomForestRegressor(
            n_estimators=best["n_estimators"],
            max_depth=best["max_depth"],
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        joblib.dump(model, os.path.join(RUNTIME_RESULTS, "optimized_random_forest.pkl"))
        joblib.dump(scaler, os.path.join(RUNTIME_RESULTS, "scaler.pkl"))

        graph_names = []

        sample = min(150, len(y_test))
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.iloc[:sample].values, label="Actual")
        plt.plot(pred[:sample], label="Predicted")
        plt.title("Sri Lanka: Actual vs Predicted Daily Electricity Demand")
        plt.xlabel("Daily test record")
        plt.ylabel("Daily demand (MWh)")
        plt.legend()
        plt.tight_layout()
        graph_names.append(("Actual vs Predicted Demand", figure_data_uri()))

        # Scatter plot: predictions should stay close to the diagonal line.
        low = min(float(y_test.min()), float(pred.min()))
        high = max(float(y_test.max()), float(pred.max()))
        plt.figure(figsize=(7, 6))
        plt.scatter(y_test, pred, alpha=0.55, color="#2563eb")
        plt.plot([low, high], [low, high], "r--", label="Perfect prediction")
        plt.title("Actual Demand vs Predicted Demand")
        plt.xlabel("Actual demand (MWh)")
        plt.ylabel("Predicted demand (MWh)")
        plt.legend()
        plt.tight_layout()
        graph_names.append(("Prediction Scatter Plot", figure_data_uri()))

        residuals = y_test.to_numpy() - pred

        # Residual distribution shows the size and balance of prediction errors.
        plt.figure(figsize=(8, 5))
        plt.hist(residuals, bins=25, color="#0f766e", edgecolor="white")
        plt.axvline(0, color="red", linestyle="--")
        plt.title("Distribution of Prediction Errors")
        plt.xlabel("Residual: actual - predicted (MWh)")
        plt.ylabel("Number of records")
        plt.tight_layout()
        graph_names.append(("Residual Distribution", figure_data_uri()))

        # Residual plot helps reveal patterns that the model did not learn.
        plt.figure(figsize=(8, 5))
        plt.scatter(pred, residuals, alpha=0.55, color="#7c3aed")
        plt.axhline(0, color="red", linestyle="--")
        plt.title("Residual Plot")
        plt.xlabel("Predicted demand (MWh)")
        plt.ylabel("Residual (MWh)")
        plt.tight_layout()
        graph_names.append(("Residual Plot", figure_data_uri()))

        # Feature importance explains which inputs influenced the model most.
        feature_names = [
            "Max temperature", "Rainfall", "Humidity", "Holiday",
            "Day of week", "Month", "Day of year", "Previous day demand",
            "Previous week demand"
        ]
        order = np.argsort(model.feature_importances_)
        plt.figure(figsize=(9, 6))
        plt.barh(np.array(feature_names)[order], model.feature_importances_[order],
                 color="#ea580c")
        plt.title("Random Forest Feature Importance")
        plt.xlabel("Importance score")
        plt.tight_layout()
        graph_names.append(("Feature Importance", figure_data_uri()))

        return render_template("index.html", **page_context(
            trained=True,
            n_estimators=best["n_estimators"],
            max_depth=best["max_depth"],
            best_mse=round(best["best_mse"], 5),
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            r2=round(r2, 4),
            graphs=graph_names, success="Model training and optimization completed successfully."
        ))
    except Exception as exc:
        return render_template("index.html", **page_context(error=str(exc)))

@app.route("/predict-period", methods=["POST"])
def predict_period():
    try:
        year, start_month, end_month = int(request.form["year"]), int(request.form["start_month"]), int(request.form["end_month"])
        dates, predictions = forecast_period(year, start_month, end_month)
        plt.figure(figsize=(12, 5.5)); plt.plot(dates, predictions, color="#3155ff", linewidth=2)
        plt.fill_between(dates, predictions, predictions.min(), color="#3155ff", alpha=0.10)
        plt.title(f"Predicted Daily Electricity Demand — {dates[0]:%B} to {dates[-1]:%B %Y}")
        plt.xlabel("Date"); plt.ylabel("Predicted demand (MWh)"); plt.grid(alpha=0.2); plt.tight_layout()
        graph_data = figure_data_uri()
        return render_template("index.html", **page_context(
            period_result=True, selected_year=year, period_graph=graph_data,
            period_total=round(float(predictions.sum()) / 1000, 2),
            period_average=round(float(predictions.mean()), 2), period_peak=round(float(predictions.max()), 2),
            success="Selected month-period prediction completed."))
    except Exception as exc:
        return render_template("index.html", **page_context(error=str(exc)))

@app.route("/predict-cost", methods=["POST"])
def predict_cost():
    try:
        year, month, tariff = int(request.form["cost_year"]), int(request.form["cost_month"]), float(request.form["tariff"])
        if tariff <= 0:
            raise ValueError("Electricity tariff must be greater than zero.")
        dates, predictions = forecast_period(year, month, month)
        total_mwh = float(predictions.sum()); total_cost = total_mwh * 1000 * tariff
        daily_cost_million = predictions * 1000 * tariff / 1_000_000
        fig, axis_demand = plt.subplots(figsize=(12, 5.5)); axis_cost = axis_demand.twinx()
        axis_demand.plot(dates, predictions, color="#3155ff", linewidth=2); axis_cost.plot(dates, daily_cost_million, color="#ff7a45", linewidth=2)
        axis_demand.set_xlabel("Date"); axis_demand.set_ylabel("Demand (MWh)", color="#3155ff")
        axis_cost.set_ylabel("Estimated cost (LKR million)", color="#ff7a45")
        plt.title(f"Electricity Demand and Cost Forecast — {dates[0]:%B %Y}"); fig.tight_layout()
        graph_data = figure_data_uri()
        return render_template("index.html", **page_context(
            cost_result=True, tariff=round(tariff, 2), demand_gwh=round(total_mwh / 1000, 2),
            average_daily_mwh=round(float(predictions.mean()), 2), estimated_cost_billion=round(total_cost / 1_000_000_000, 3),
            cost_graph=graph_data, success="Demand and electricity-cost prediction completed."))
    except Exception as exc:
        return render_template("index.html", **page_context(error=str(exc)))

if __name__ == "__main__":
    app.run(debug=True)
