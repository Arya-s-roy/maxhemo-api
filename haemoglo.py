import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 1. Load dataset
df = pd.read_csv("tcs34725_hb_data.csv")

# 2. Extract inputs (Removed 'Finger_Width_cm')
features = [
    "Norm_Red",
    "Norm_Green",
    "Norm_Blue",
    "BMI",
    "Subject_Age",
    "Subject_Sex",
]
X = df[features]
y = df["Lab_Total_Hb"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train XGBoost Regressor
model = xgb.XGBRegressor(
    n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
)
model.fit(X_train_scaled, y_train)

# 6. Evaluate Performance
y_pred = model.predict(X_test_scaled)
print("=== Retrained Model Performance ===")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE:      {mean_absolute_error(y_test, y_pred):.4f} g/dL")
print(f"RMSE:     {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} g/dL")

# 7. Overwrite saved artifacts (loaded by main.py at API startup)
joblib.dump(model, "tcs34725_hb_model.pkl")
joblib.dump(scaler, "tcs34725_scaler.pkl")
print("\nUpdated Model and Scaler saved!")