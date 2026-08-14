import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(file_path: str):
    data = pd.read_csv(file_path)
    required = {
        "date", "max_temperature_c", "rainfall_mm", "humidity_pct",
        "holiday", "daily_demand_mwh"
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Missing dataset columns: " + ", ".join(sorted(missing)))

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    numeric = list(required - {"date"})
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=list(required)).sort_values("date").reset_index(drop=True)

    # Calendar and past-demand features suitable for Sri Lankan grid forecasting.
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["day_of_year"] = data["date"].dt.dayofyear
    data["lag_1_demand_mwh"] = data["daily_demand_mwh"].shift(1)
    data["lag_7_demand_mwh"] = data["daily_demand_mwh"].shift(7)
    data = data.dropna().reset_index(drop=True)

    features = [
        "max_temperature_c", "rainfall_mm", "humidity_pct", "holiday",
        "day_of_week", "month", "day_of_year", "lag_1_demand_mwh",
        "lag_7_demand_mwh"
    ]
    X = data[features]
    y = data["daily_demand_mwh"]

    # Time-based split prevents future records leaking into model training.
    split = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler
