"""
Phiên bản script của ml/train_model.ipynb — chạy local bằng venv.
Train LightGBM dự đoán monthly_electricity_kwh, đánh giá vs baseline,
xuất SHAP plots (PNG) và ghi gold.gold_predictions lên MotherDuck.
"""
import os
import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # không cần GUI
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
MD_TOKEN = os.environ["MOTHERDUCK_TOKEN"]
HERE = os.path.dirname(__file__)

# ── 1. Load Gold ────────────────────────────────────────────────────────────
con = duckdb.connect(f"md:energy_lakehouse?motherduck_token={MD_TOKEN}")
df = con.execute("SELECT * FROM gold.gold_ml_features").fetchdf()
print(f"[1] Loaded gold_ml_features: {len(df):,} dòng, {len(df.columns)} cột")

# ── 2. Feature engineering ──────────────────────────────────────────────────
TARGET = "target_kwh"
y = df[TARGET]
X = df.drop(columns=["household_id", TARGET])

bool_cols = X.select_dtypes(include="bool").columns.tolist()
X[bool_cols] = X[bool_cols].astype(int)
cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category")
print(f"[2] Features: {len(X.columns)} | categorical: {cat_cols}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"    Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── 3. Train ────────────────────────────────────────────────────────────────
model = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, random_state=42, verbose=-1,
)
model.fit(
    X_train, y_train, eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)
print(f"[3] Best iteration: {model.best_iteration_}")

# ── 4. Đánh giá ─────────────────────────────────────────────────────────────
pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, pred)
rmse = root_mean_squared_error(y_test, pred)
r2   = r2_score(y_test, pred)

base = np.full(len(y_test), y_train.mean())
b_mae  = mean_absolute_error(y_test, base)
b_rmse = root_mean_squared_error(y_test, base)
b_r2   = r2_score(y_test, base)

print("=" * 46)
print(f"{'Metric':<10}{'LightGBM':>14}{'Baseline(mean)':>18}")
print("-" * 46)
print(f"{'MAE':<10}{mae:>14.2f}{b_mae:>18.2f}")
print(f"{'RMSE':<10}{rmse:>14.2f}{b_rmse:>18.2f}")
print(f"{'R2':<10}{r2:>14.4f}{b_r2:>18.4f}")
print("=" * 46)
print(f"Cải thiện MAE so baseline: {((b_mae - mae)/b_mae*100):.1f}%")

# Scatter predicted vs actual
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, pred, alpha=0.3, s=10, color="steelblue")
lim = max(y_test.max(), pred.max())
ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect prediction")
ax.set_xlabel("Actual kWh/month"); ax.set_ylabel("Predicted kWh/month")
ax.set_title(f"Predicted vs Actual — R² = {r2:.4f}"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(HERE, "predicted_vs_actual.png"), dpi=150)
plt.close()

# ── 5. SHAP ─────────────────────────────────────────────────────────────────
explainer = shap.TreeExplainer(model)
X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
shap_values = explainer.shap_values(X_sample)

plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Feature Importance — Dự đoán mức tiêu thụ điện")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

idx = 0
print(f"[5] Hộ #{idx}: actual={y_test.iloc[idx]:.1f} kWh, predicted={pred[idx]:.1f} kWh")
shap.plots.force(
    explainer.expected_value, shap_values[idx], X_sample.iloc[idx],
    matplotlib=True, show=False,
)
plt.savefig(os.path.join(HERE, "shap_force_plot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    Đã lưu: predicted_vs_actual.png, shap_summary.png, shap_force_plot.png")

# ── 6. Export gold.gold_predictions ─────────────────────────────────────────
results = X_test.copy()
results["actual_kwh"]    = y_test.values
results["predicted_kwh"] = pred.round(2)
results["error_kwh"]     = (pred - y_test.values).round(2)
results["abs_error_kwh"] = np.abs(pred - y_test.values).round(2)
for c in results.select_dtypes(include="category").columns:
    results[c] = results[c].astype(str)

con.register("_pred_staging", results)
con.execute("CREATE SCHEMA IF NOT EXISTS gold;")
con.execute("CREATE OR REPLACE TABLE gold.gold_predictions AS SELECT * FROM _pred_staging;")
n = con.execute("SELECT count(*) FROM gold.gold_predictions").fetchone()[0]
print(f"[6] Export xong: {n:,} dòng trong gold.gold_predictions")

joblib.dump(model, os.path.join(HERE, "lgbm_kwh_model.pkl"))
con.close()
print("[OK] Hoàn tất ML pipeline.")
