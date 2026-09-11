import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib

# 1. Generate/Load Training Data (N, P, K, pH, Rainfall -> Yield)
np.random.seed(42)
num_samples = 1000

N = np.random.uniform(0, 140, num_samples)
P = np.random.uniform(0, 145, num_samples)
K = np.random.uniform(0, 205, num_samples)
ph = np.random.uniform(4.0, 9.0, num_samples)
rainfall = np.random.uniform(50, 400, num_samples)

# Formula simulating non-linear crop response + noise
yield_tons = (0.015 * N + 0.01 * P + 0.008 * K + 0.2 * (7 - abs(6.5 - ph)) + 0.005 * rainfall) + np.random.normal(0, 0.2, num_samples)

df = pd.DataFrame({
    'N': N, 'P': P, 'K': K, 'pH': ph, 'Rainfall': rainfall, 'Yield': yield_tons
})

X = df[['N', 'P', 'K', 'pH', 'Rainfall']]
y = df['Yield']

# 2. Train XGBoost Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.05)
model.fit(X_train, y_train)

# 3. Save the trained model file
joblib.dump(model, 'yield_model.pkl')
print("✅ Success: 'yield_model.pkl' created and saved!")