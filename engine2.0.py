import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------
# Step 1: Load and Prepare Data
# ----------------------------------------

df = pd.read_csv(r"C:\Users\janya\OneDrive\Desktop\ML_Project\DATASETS\archive (1)\predictive_maintenance_dataset.csv")

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
df = df.dropna(subset=['date'])  # Drop rows where date conversion failed

# Sort by device and time
df = df.sort_values(by=['device', 'date']).reset_index(drop=True)

# ----------------------------------------
# Step 2: Create RUL Column
# ----------------------------------------

df['RUL'] = np.nan

for device_id in df['device'].unique():
    device_df = df[df['device'] == device_id]
    failure_indices = device_df.index[device_df['failure'] == 1].tolist()

    if not failure_indices:
        continue  # skip devices that never failed

    # Loop over failure events
    for i in range(len(failure_indices)):
        failure_idx = failure_indices[i]
        start_idx = failure_indices[i - 1] + 1 if i > 0 else device_df.index[0]

        for j, row_idx in enumerate(range(start_idx, failure_idx)):
            df.at[row_idx, 'RUL'] = failure_idx - row_idx

# Drop rows without a valid RUL
df = df.dropna(subset=['RUL']).reset_index(drop=True)
df['RUL'] = df['RUL'].astype(int)

# ----------------------------------------
# Step 3: Feature Selection
# ----------------------------------------

features = [col for col in df.columns if col.startswith("metric")]
X = df[features]
y = df['RUL']

# Debug info
print("✅ Dataset size:", X.shape)
print("✅ Sample RULs:\n", y.value_counts().head())

# ----------------------------------------
# Step 4: Train/Test Split
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------
# Step 5: Train Model
# ----------------------------------------

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------
# Step 6: Predict and Evaluate
# ----------------------------------------

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n📊 RMSE: {rmse:.2f}")
print(f"📈 R-squared (R²): {r2:.2f}")

# ----------------------------------------
# Step 7: Plot Actual vs Predicted RUL
# ----------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.4, color='blue', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted RUL")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
