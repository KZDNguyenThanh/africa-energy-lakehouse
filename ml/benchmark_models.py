"""Multi-algorithm benchmark for household electricity-consumption regression.

Compares a mean baseline, two linear models, a bagging model, and four
gradient-boosting models on gold.gold_ml_features, using 5-fold cross-validation
on the training set plus a held-out test set. The three main boosters are tuned
with Optuna. Early stopping uses an internal validation split carved from the
training data only, so the test set never influences model selection. The best
model (by CV MAE) is explained with SHAP, persisted, and its predictions and the
full comparison are written back to MotherDuck.
"""
import os
import warnings

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap
import optuna
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
N_TRIALS = int(os.environ.get("BENCH_N_TRIALS", "30"))
TUNE_TIMEOUT = int(os.environ.get("BENCH_TUNE_TIMEOUT", "240"))
CV_ESTIMATORS = 300


def get_connection():
    load_dotenv(os.path.join(HERE, "..", ".env"))
    token = os.environ["MOTHERDUCK_TOKEN"]
    return duckdb.connect(f"md:energy_lakehouse?motherduck_token={token}")


def load_features(con):
    df = con.execute("SELECT * FROM gold.gold_ml_features").fetchdf()
    print(f"[load] gold_ml_features: {len(df):,} rows, {len(df.columns)} cols")
    return df


def prepare(df):
    target = "target_kwh"
    y = df[target].astype(float)
    X = df.drop(columns=["household_id", target])
    bool_cols = X.select_dtypes(include="bool").columns.tolist()
    X[bool_cols] = X[bool_cols].astype(int)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    for c in cat_cols:
        X[c] = X[c].astype("category")
    print(f"[prep] {X.shape[1]} features | categorical: {cat_cols}")
    return X, y, cat_cols, num_cols


def ohe_preprocessor(cat_cols, num_cols, scale):
    num_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", num_step, num_cols),
        ]
    )


def metrics(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        root_mean_squared_error(y_true, y_pred),
        r2_score(y_true, y_pred),
    )


def cv_ohe(estimator, X, y, kf):
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    res = cross_validate(estimator, X, y, cv=kf, scoring=scoring, n_jobs=-1)
    return {
        "cv_mae": -res["test_mae"].mean(), "cv_mae_std": res["test_mae"].std(),
        "cv_rmse": -res["test_rmse"].mean(), "cv_r2": res["test_r2"].mean(),
    }


def make_native(kind, params, n_estimators):
    if kind == "lgbm":
        return lgb.LGBMRegressor(
            n_estimators=n_estimators, random_state=SEED, verbose=-1, **params)
    if kind == "xgb":
        return xgb.XGBRegressor(
            n_estimators=n_estimators, random_state=SEED, tree_method="hist",
            enable_categorical=True, verbosity=0, **params)
    if kind == "catboost":
        return CatBoostRegressor(
            iterations=n_estimators, random_seed=SEED, verbose=False, **params)
    raise ValueError(kind)


def fit_native(model, kind, X_tr, y_tr, X_val, y_val, cat_cols, early=True):
    if kind == "lgbm":
        cb = [lgb.early_stopping(50, verbose=False)] if early else None
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)] if early else None, callbacks=cb)
    elif kind == "xgb":
        if early:
            model.set_params(early_stopping_rounds=50)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_tr, y_tr, verbose=False)
    elif kind == "catboost":
        tr_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        if early:
            val_pool = Pool(X_val, y_val, cat_features=cat_cols)
            model.fit(tr_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)
        else:
            model.fit(tr_pool, verbose=False)
    return model


def cv_native(kind, params, X, y, cat_cols, kf):
    maes, rmses, r2s = [], [], []
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = make_native(kind, params, CV_ESTIMATORS)
        fit_native(model, kind, X_tr, y_tr, None, None, cat_cols, early=False)
        pred = model.predict(X_va)
        m = metrics(y_va, pred)
        maes.append(m[0]); rmses.append(m[1]); r2s.append(m[2])
    return {
        "cv_mae": float(np.mean(maes)), "cv_mae_std": float(np.std(maes)),
        "cv_rmse": float(np.mean(rmses)), "cv_r2": float(np.mean(r2s)),
    }


def tune_native(kind, X, y, cat_cols):
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)

    def space(trial):
        if kind == "lgbm":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
        if kind == "xgb":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }

    def objective(trial):
        params = space(trial)
        res = cv_native(kind, params, X, y, cat_cols, kf)
        return res["cv_mae"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, timeout=TUNE_TIMEOUT)
    print(f"[tune] {kind}: best CV MAE={study.best_value:.3f} | "
          f"{len(study.trials)} trials | params={study.best_params}")
    return study.best_params


def shap_plots(model, kind, X_sample, cat_cols, y_actual, pred):
    try:
        if kind == "catboost":
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(Pool(X_sample, cat_features=cat_cols))
            expected = explainer.expected_value
        else:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sample)
            expected = explainer.expected_value
        plt.figure(figsize=(9, 6))
        shap.summary_plot(sv, X_sample, show=False)
        plt.title("SHAP Feature Importance — Household Electricity Consumption")
        plt.tight_layout()
        plt.savefig(os.path.join(HERE, "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
        shap.plots.force(expected, sv[0], X_sample.iloc[0], matplotlib=True, show=False)
        plt.savefig(os.path.join(HERE, "shap_force_plot.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("[shap] saved shap_summary.png, shap_force_plot.png")
        return True
    except Exception as e:
        print(f"[shap] skipped ({e})")
        return False


def scatter_plot(y_test, pred, r2):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, pred, alpha=0.3, s=10, color="steelblue")
    lim = max(y_test.max(), pred.max())
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual kWh/month"); ax.set_ylabel("Predicted kWh/month")
    ax.set_title(f"Predicted vs Actual — Best model — R² = {r2:.4f}"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "predicted_vs_actual.png"), dpi=150)
    plt.close()


def comparison_plot(table):
    df = table[table["model"] != "Baseline (mean)"].sort_values("test_mae")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(df["model"], df["test_mae"], color="steelblue")
    axes[0].set_xlabel("Test MAE (kWh) — lower is better"); axes[0].invert_yaxis()
    axes[0].set_title("Test MAE by model")
    axes[1].barh(df["model"], df["test_r2"], color="seagreen")
    axes[1].set_xlabel("Test R² — higher is better"); axes[1].invert_yaxis()
    axes[1].set_title("Test R² by model")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "model_comparison.png"), dpi=150)
    plt.close()
    print("[plot] saved model_comparison.png")


def main():
    con = get_connection()
    df = load_features(con)
    X, y, cat_cols, num_cols = prepare(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    print(f"[split] train={len(X_train):,} test={len(X_test):,}")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    rows = []

    base_pred = np.full(len(y_test), y_train.mean())
    bmae, brmse, br2 = metrics(y_test, base_pred)
    rows.append({"model": "Baseline (mean)", "cv_mae": np.nan, "cv_mae_std": np.nan,
                 "cv_rmse": np.nan, "cv_r2": np.nan,
                 "test_mae": bmae, "test_rmse": brmse, "test_r2": br2})

    ohe_models = {
        "Linear Regression": (LinearRegression(), True),
        "Ridge": (Ridge(alpha=1.0, random_state=SEED), True),
        "Random Forest": (RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=SEED), False),
        "HistGradientBoosting": (HistGradientBoostingRegressor(
            random_state=SEED), False),
    }
    fitted = {}
    for name, (est, scale) in ohe_models.items():
        pipe = Pipeline([("prep", ohe_preprocessor(cat_cols, num_cols, scale)),
                         ("model", est)])
        cvm = cv_ohe(pipe, X_train, y_train, kf)
        pipe.fit(X_train, y_train)
        tm = metrics(y_test, pipe.predict(X_test))
        fitted[name] = ("ohe", pipe)
        rows.append({"model": name, **cvm,
                     "test_mae": tm[0], "test_rmse": tm[1], "test_r2": tm[2]})
        print(f"[ohe] {name:22s} CV MAE={cvm['cv_mae']:.2f}±{cvm['cv_mae_std']:.2f} "
              f"| Test MAE={tm[0]:.2f} R2={tm[2]:.4f}")

    native = {"LightGBM (tuned)": "lgbm", "XGBoost (tuned)": "xgb",
              "CatBoost (tuned)": "catboost"}
    Xtr2, Xval2, ytr2, yval2 = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEED)
    for name, kind in native.items():
        best = tune_native(kind, X_train, y_train, cat_cols)
        cvm = cv_native(kind, best, X_train, y_train, cat_cols, kf)
        model = make_native(kind, best, 2000)
        fit_native(model, kind, Xtr2, ytr2, Xval2, yval2, cat_cols, early=True)
        tm = metrics(y_test, model.predict(X_test))
        fitted[name] = (kind, model)
        rows.append({"model": name, **cvm,
                     "test_mae": tm[0], "test_rmse": tm[1], "test_r2": tm[2]})
        print(f"[native] {name:18s} CV MAE={cvm['cv_mae']:.2f}±{cvm['cv_mae_std']:.2f} "
              f"| Test MAE={tm[0]:.2f} R2={tm[2]:.4f}")

    table = pd.DataFrame(rows)
    ranked = table.dropna(subset=["cv_mae"]).sort_values("cv_mae")
    best_name = ranked.iloc[0]["model"]
    table["is_best"] = table["model"] == best_name
    print("\n" + "=" * 78)
    print(table.to_string(index=False,
          columns=["model", "cv_mae", "cv_mae_std", "cv_r2", "test_mae", "test_rmse",
                   "test_r2", "is_best"], float_format=lambda v: f"{v:.3f}"))
    print("=" * 78)
    print(f"[best] {best_name} (lowest CV MAE)")

    table.to_csv(os.path.join(HERE, "model_comparison.csv"), index=False)
    comparison_plot(table)

    best_kind, best_model = fitted[best_name]
    if best_kind == "ohe":
        best_pred = best_model.predict(X_test)
    else:
        best_pred = best_model.predict(X_test)
    _, _, best_r2 = metrics(y_test, best_pred)
    scatter_plot(y_test, best_pred, best_r2)

    if best_kind in ("lgbm", "xgb", "catboost"):
        X_sample = X_test.sample(n=min(500, len(X_test)), random_state=SEED)
        shap_plots(best_model, best_kind, X_sample, cat_cols, y_test, best_pred)
        joblib.dump(best_model, os.path.join(HERE, "best_model.pkl"))
        if best_kind == "lgbm":
            joblib.dump(best_model, os.path.join(HERE, "lgbm_kwh_model.pkl"))

    results = X_test.copy()
    results["actual_kwh"] = y_test.values
    results["predicted_kwh"] = np.round(best_pred, 2)
    results["error_kwh"] = np.round(best_pred - y_test.values, 2)
    results["abs_error_kwh"] = np.round(np.abs(best_pred - y_test.values), 2)
    for c in results.select_dtypes(include="category").columns:
        results[c] = results[c].astype(str)
    con.register("_pred_staging", results)
    con.execute("CREATE SCHEMA IF NOT EXISTS gold;")
    con.execute("CREATE OR REPLACE TABLE gold.gold_predictions AS SELECT * FROM _pred_staging;")

    metrics_tbl = table[["model", "cv_mae", "cv_mae_std", "cv_rmse", "cv_r2",
                         "test_mae", "test_rmse", "test_r2", "is_best"]].copy()
    con.register("_metrics_staging", metrics_tbl)
    con.execute("CREATE OR REPLACE TABLE gold.gold_model_metrics AS SELECT * FROM _metrics_staging;")
    n = con.execute("SELECT count(*) FROM gold.gold_predictions").fetchone()[0]
    print(f"[export] gold.gold_predictions={n:,} rows | gold.gold_model_metrics="
          f"{len(metrics_tbl)} rows (best={best_name})")
    con.close()
    print("[OK] Benchmark complete.")


if __name__ == "__main__":
    main()
