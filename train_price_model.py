import pandas as pd
import numpy as np
from prophet import Prophet
import joblib

# 1. Generate multi-year historical price data (Simulating Agmarknet Mandi trends)
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", end="2026-09-01", freq="D")

# Base price trend with annual seasonality and noise
base_price = 2100
seasonal_component = 300 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
noise = np.random.normal(0, 40, len(dates))
prices = base_price + seasonal_component + noise + (np.arange(len(dates)) * 0.25)

df = pd.DataFrame({"ds": dates, "y": prices})

# 2. Train Meta Prophet Model
model = Prophet(daily_seasonality=True, yearly_seasonality=True)
model.fit(df)

# 3. Save the trained Prophet model
joblib.dump(model, "price_model.pkl")
print("✅ Success: 'price_model.pkl' created and saved!")