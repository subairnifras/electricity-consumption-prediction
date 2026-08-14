"""Create the bundled reproducible Sri Lanka-calibrated educational dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


rng = np.random.default_rng(42)
dates = pd.date_range("2020-01-01", "2025-05-31", freq="D")
day = dates.dayofyear.to_numpy()
year_offset = (dates.year.to_numpy() - 2020)
weekend = (dates.dayofweek.to_numpy() >= 5).astype(int)

# A compact public-holiday approximation for demonstration data.
fixed_holidays = {(1, 1), (2, 4), (4, 13), (4, 14), (5, 1), (12, 25)}
holiday = np.array([
    int((d.month, d.day) in fixed_holidays or d.dayofweek == 6) for d in dates
])

temperature = 30.0 + 1.8 * np.sin(2 * np.pi * (day - 35) / 365.25) + rng.normal(0, 0.8, len(dates))
monsoon = 4.0 + 9.0 * (np.sin(2 * np.pi * (day - 110) / 182.625) ** 2)
rainfall = np.maximum(0, rng.gamma(1.3, monsoon / 2.4) - 1.3)
humidity = np.clip(72 + rainfall * 0.65 - (temperature - 30) * 1.5 + rng.normal(0, 3, len(dates)), 55, 96)

trend = 38900 + year_offset * 720
seasonal = 1250 * np.sin(2 * np.pi * (day - 20) / 365.25)
weather_effect = (temperature - 29) * 480 - rainfall * 35
calendar_effect = -2300 * holiday - 900 * weekend
demand = trend + seasonal + weather_effect + calendar_effect + rng.normal(0, 650, len(dates))

df = pd.DataFrame({
    "date": dates.strftime("%Y-%m-%d"),
    "max_temperature_c": np.round(temperature, 1),
    "rainfall_mm": np.round(rainfall, 1),
    "humidity_pct": np.round(humidity, 1),
    "holiday": holiday,
    "daily_demand_mwh": np.round(demand, 1),
})

output = Path(__file__).resolve().parent / "data" / "sri_lanka_daily_electricity_demand.csv"
df.to_csv(output, index=False)
print(f"Created {len(df)} rows at {output}")
