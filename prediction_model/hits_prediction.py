"""
hits_prediction.py
==================
Trains regression models to predict the number of Hits for each player in the
Muskets Football dataset.

Steps
-----
1. Load & filter Muskets_data_cleaned.csv
2. Feature engineering (frequency-encode Nationality, one-hot categoricals)
3. Impute + scale numeric features
4. Train Ridge, Random Forest, XGBoost, LightGBM on log1p(Hits)
5. 5-fold CV evaluation + hold-out test metrics
6. Save best model, feature importance chart, predictions scatter, Hits distribution
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# --- Paths --------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "Muskets_data_cleaned.csv")
OUT_DIR    = BASE_DIR                              # outputs written here
os.makedirs(OUT_DIR, exist_ok=True)

# --- 1. Load Data -------------------------------------------------------------
print("Loading data ...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# Drop rows where the target is missing
df = df.dropna(subset=["Hits"])
print(f"  Rows after dropping null Hits: {df.shape[0]:,}")

# --- 2. Define Feature Sets ---------------------------------------------------
DROP_COLS = [
    "ID", "Name", "LongName", "photoUrl", "playerUrl",
    "Contract", "Joined", "Loan Date End",
    "Short Passing",        # already captured by Skill composite
    "Hits",                 # target
]

# Categorical features to one-hot encode (low-cardinality)
CAT_OHE = ["Preferred Foot", "A/W", "D/W", "Contract Status", "Best Position"]

# High-cardinality categorical -> frequency encode
HIGH_CAT = ["Nationality", "Positions"]

# Numeric features (everything else minus drops and categoricals)
ALL_CATEGORICAL = CAT_OHE + HIGH_CAT + ["Club"]
NUM_COLS = [
    c for c in df.columns
    if c not in DROP_COLS + ALL_CATEGORICAL
    and df[c].dtype in [np.float64, np.int64, "Int64"]
]

print(f"  Numeric features : {len(NUM_COLS)}")
print(f"  OHE categoricals : {CAT_OHE}")
print(f"  Freq-enc cats    : {HIGH_CAT}")

# --- 3. Frequency Encode High-Cardinality Columns -----------------------------
def frequency_encode(train_df, test_df, col):
    freq = train_df[col].value_counts(normalize=True)
    train_enc = train_df[col].map(freq).fillna(0)
    test_enc  = test_df[col].map(freq).fillna(0)
    return train_enc, test_enc

# --- 4. Prepare X / y ---------------------------------------------------------
TARGET_RAW = df["Hits"].values.astype(float)
y = np.log1p(TARGET_RAW)          # log-transform

feature_cols = NUM_COLS + CAT_OHE + HIGH_CAT
X = df[feature_cols].copy()

# Train / test split (80/20, stratified on log-Hits quantiles for balance)
quantile_strat = pd.qcut(y, q=10, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=quantile_strat
)

print(f"\nSplit -> train: {len(X_train):,}  test: {len(X_test):,}")

# Apply frequency encoding on train, then propagate to test
freq_maps = {}
for col in HIGH_CAT:
    freq_maps[col] = dict(X_train[col].value_counts(normalize=True))
    X_train[col], X_test[col] = frequency_encode(X_train, X_test, col)

# Persist frequency maps so predict.py can reuse them
freq_maps_path = os.path.join(OUT_DIR, "freq_maps.json")
with open(freq_maps_path, "w") as f:
    json.dump(freq_maps, f, indent=2)
print(f"  Saved freq maps -> {freq_maps_path}")

# After freq-enc, these cols are now numeric
FINAL_NUM_COLS = NUM_COLS + HIGH_CAT

# --- 5. Build Sklearn Pipelines -----------------------------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer,    FINAL_NUM_COLS),
    ("cat", categorical_transformer, CAT_OHE),
], remainder="drop")

def make_pipeline(model):
    return Pipeline([
        ("prep",  preprocessor),
        ("model", model),
    ])

# --- 6. Define Models ---------------------------------------------------------
models = {
    "Ridge": make_pipeline(
        Ridge(alpha=10.0)
    ),
    "Random Forest": make_pipeline(
        RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        )
    ),
    "XGBoost": make_pipeline(
        xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )
    ),
    "LightGBM": make_pipeline(
        lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    ),
}

# --- 7. 5-Fold Cross-Validation -----------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

print("\n5-Fold CV on training set ...")
for name, pipe in models.items():
    cv_rmse = cross_val_score(
        pipe, X_train, y_train,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    cv_r2   = cross_val_score(
        pipe, X_train, y_train,
        cv=kf,
        scoring="r2",
        n_jobs=-1,
    )
    cv_results[name] = {
        "cv_rmse_mean": float(-cv_rmse.mean()),
        "cv_rmse_std":  float(cv_rmse.std()),
        "cv_r2_mean":   float(cv_r2.mean()),
        "cv_r2_std":    float(cv_r2.std()),
    }
    print(f"  {name:15s}  CV RMSE = {-cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}   "
          f"R² = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# --- 8. Train & Evaluate on Hold-out Test Set ---------------------------------
print("\nTraining final models on full train set ...")
test_results = {}
fitted_models = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    # Back-transform to raw Hits space for interpretable MAE
    raw_actual = np.expm1(y_test)
    raw_pred   = np.expm1(y_pred)
    raw_mae    = float(mean_absolute_error(raw_actual, raw_pred))

    test_results[name] = {
        "test_rmse_log": rmse,
        "test_mae_log":  mae,
        "test_r2":       r2,
        "test_mae_raw_hits": raw_mae,
    }
    fitted_models[name] = (pipe, y_pred)
    print(f"  {name:15s}  Test RMSE(log)={rmse:.4f}  R²={r2:.4f}  "
          f"MAE(hits)={raw_mae:.1f}")

# --- 9. Pick Best Model -------------------------------------------------------
best_name = min(test_results, key=lambda n: test_results[n]["test_rmse_log"])
best_pipe, best_y_pred = fitted_models[best_name]
print(f"\n* Best model: {best_name}  (lowest test RMSE on log-Hits)")

# --- 10. Save Model & Metrics -------------------------------------------------
model_path = os.path.join(OUT_DIR, "hits_model.pkl")
joblib.dump({"model": best_pipe, "name": best_name, "high_cat": HIGH_CAT}, model_path)
print(f"  Saved model -> {model_path}")

all_results = {
    "best_model": best_name,
    "cv": cv_results,
    "test": test_results,
}
metrics_path = os.path.join(OUT_DIR, "model_results.json")
with open(metrics_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"  Saved metrics -> {metrics_path}")

# --- 11. Charts ---------------------------------------------------------------
# Colour palette
PALETTE = {
    "Ridge":         "#6C8EBF",
    "Random Forest": "#82B366",
    "XGBoost":       "#D6A75B",
    "LightGBM":      "#AE4132",
}

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150})

# --- 11a. Hits distribution (raw vs log) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Hits Distribution", fontsize=14, fontweight="bold")

axes[0].hist(np.expm1(y), bins=80, color="#4A90D9", edgecolor="none", alpha=0.85)
axes[0].set_title("Raw Hits")
axes[0].set_xlabel("Hits")
axes[0].set_ylabel("Count")
axes[0].set_xlim(0, 1000)

axes[1].hist(y, bins=60, color="#50B86C", edgecolor="none", alpha=0.85)
axes[1].set_title("log1p(Hits)")
axes[1].set_xlabel("log1p(Hits)")
axes[1].set_ylabel("Count")

plt.tight_layout()
dist_path = os.path.join(OUT_DIR, "hits_distribution.png")
plt.savefig(dist_path, bbox_inches="tight")
plt.close()
print(f"  Saved -> {dist_path}")

# --- 11b. CV RMSE comparison bar chart ---
fig, ax = plt.subplots(figsize=(8, 4))
names = list(cv_results.keys())
means = [cv_results[n]["cv_rmse_mean"] for n in names]
stds  = [cv_results[n]["cv_rmse_std"]  for n in names]
colors = [PALETTE[n] for n in names]

bars = ax.bar(names, means, yerr=stds, capsize=5,
              color=colors, edgecolor="none", alpha=0.9)
ax.set_title("5-Fold CV RMSE (log-Hits space)", fontsize=13, fontweight="bold")
ax.set_ylabel("RMSE")
ax.set_xlabel("Model")
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
cv_chart_path = os.path.join(OUT_DIR, "cv_rmse_comparison.png")
plt.savefig(cv_chart_path, bbox_inches="tight")
plt.close()
print(f"  Saved -> {cv_chart_path}")

# --- 11c. Predicted vs Actual (best model) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Predicted vs Actual Hits  [{best_name}]",
             fontsize=13, fontweight="bold")

# Log space
ax = axes[0]
ax.scatter(y_test, best_y_pred, alpha=0.25, s=8, color="#4A90D9", rasterized=True)
lims = [min(y_test.min(), best_y_pred.min()),
        max(y_test.max(), best_y_pred.max())]
ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect fit")
ax.set_xlabel("Actual log1p(Hits)")
ax.set_ylabel("Predicted log1p(Hits)")
ax.set_title("Log Space")
ax.legend()

# Raw Hits space (capped at 500 for readability)
ax = axes[1]
raw_actual = np.expm1(y_test)
raw_pred_v = np.expm1(best_y_pred)
mask = raw_actual < 500
ax.scatter(raw_actual[mask], raw_pred_v[mask],
           alpha=0.25, s=8, color="#82B366", rasterized=True)
ax.plot([0, 500], [0, 500], "r--", linewidth=1.2, label="Perfect fit")
ax.set_xlabel("Actual Hits")
ax.set_ylabel("Predicted Hits")
ax.set_title("Raw Space (Hits < 500)")
ax.legend()

plt.tight_layout()
scatter_path = os.path.join(OUT_DIR, "predictions_vs_actual.png")
plt.savefig(scatter_path, bbox_inches="tight")
plt.close()
print(f"  Saved -> {scatter_path}")

# --- 11d. Feature Importance (best tree model) ---
tree_names = ["LightGBM", "XGBoost", "Random Forest"]
fi_model_name = next((n for n in tree_names if n == best_name), None)
if fi_model_name is None:
    fi_model_name = next((n for n in tree_names if n in fitted_models), None)

if fi_model_name:
    fi_pipe = fitted_models[fi_model_name][0]
    inner   = fi_pipe.named_steps["model"]
    prep    = fi_pipe.named_steps["prep"]

    # Build feature names
    ohe_cats = prep.named_transformers_["cat"].named_steps["ohe"]\
                   .get_feature_names_out(CAT_OHE).tolist()
    feat_names = FINAL_NUM_COLS + ohe_cats

    importances = inner.feature_importances_
    # Truncate if mismatch (shouldn't happen, but safety)
    min_len = min(len(feat_names), len(importances))
    fi_series = pd.Series(importances[:min_len], index=feat_names[:min_len])
    fi_top20  = fi_series.nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    colors_fi = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi_top20)))
    fi_top20.plot(kind="barh", ax=ax, color=colors_fi, edgecolor="none")
    ax.set_title(f"Top-20 Feature Importances  [{fi_model_name}]",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fi_path = os.path.join(OUT_DIR, "feature_importance.png")
    plt.savefig(fi_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {fi_path}")

# --- 12. Summary Table --------------------------------------------------------
print("\n" + "="*65)
print("  MODEL PERFORMANCE SUMMARY")
print("="*65)
print(f"  {'Model':<16} {'CV RMSE':>10}  {'Test RMSE':>10}  {'Test R²':>8}  {'MAE (hits)':>12}")
print("-"*65)
for name in models:
    cv_r  = cv_results[name]["cv_rmse_mean"]
    t_r   = test_results[name]["test_rmse_log"]
    t_r2  = test_results[name]["test_r2"]
    t_mae = test_results[name]["test_mae_raw_hits"]
    star  = " *" if name == best_name else ""
    print(f"  {name:<16} {cv_r:>10.4f}  {t_r:>10.4f}  {t_r2:>8.4f}  {t_mae:>12.1f}{star}")
print("="*65)
print(f"\nDone! All outputs saved to: {OUT_DIR}")
