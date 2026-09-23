import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
n = 5000

N = np.random.uniform(0, 140, n)
P = np.random.uniform(0, 145, n)
K = np.random.uniform(0, 205, n)
ph = np.random.uniform(4.0, 9.0, n)
rainfall = np.random.uniform(50, 400, n)

# More realistic yield formula
ph_factor = np.exp(-0.5 * ((ph - 6.5) / 1.2) ** 2)
rain_factor = np.exp(-0.5 * ((rainfall - 200) / 90) ** 2)
yield_tons = (
    0.6
    + (0.012 * N - 0.00003 * N**2)
    + (0.009 * P - 0.00002 * P**2)
    + (0.007 * K - 0.000015 * K**2)
    + 1.1 * ph_factor
    + 0.9 * rain_factor
    + 0.002 * N * ph_factor
    + np.random.normal(0, 0.15, n)
)
yield_tons = np.clip(yield_tons, 0.3, 6.0)

df = pd.DataFrame({"N": N, "P": P, "K": K, "pH": ph, "Rainfall": rainfall, "Yield": yield_tons})
X = df[["N", "P", "K", "pH", "Rainfall"]]
y = df["Yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [0.5, 1.0, 1.5],
}

model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)

search = RandomizedSearchCV(
    model, param_dist, n_iter=30, scoring="r2", cv=5,
    random_state=42, n_jobs=-1, verbose=1
)
search.fit(X_train, y_train)

best = search.best_estimator_
y_pred = best.predict(X_test)

print("Best params:", search.best_params_)
print(f"R²   : {r2_score(y_test, y_pred):.4f}")
print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE : {mean_squared_error(y_test, y_pred)**0.5:.4f}")

joblib.dump(best, "yield_model.pkl")
print("✅ Optimized yield_model.pkl saved")