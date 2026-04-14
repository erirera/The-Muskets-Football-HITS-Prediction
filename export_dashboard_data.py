"""
export_dashboard_data.py
========================
Exports model results, feature importances, and player predictions
to JSON files for use by the interactive HTML dashboard.
"""

import os, sys, json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "prediction_model")
DATA_PATH = os.path.join(BASE_DIR, "Muskets_data_cleaned.csv")
DASH_DIR  = os.path.join(BASE_DIR, "dashboard")
os.makedirs(DASH_DIR, exist_ok=True)

# ── Load model & freq maps ────────────────────────────────────────────────────
bundle    = joblib.load(os.path.join(MODEL_DIR, "hits_model.pkl"))
model     = bundle["model"]
best_name = bundle["name"]
high_cat  = bundle["high_cat"]

with open(os.path.join(MODEL_DIR, "freq_maps.json")) as f:
    freq_maps = json.load(f)

with open(os.path.join(MODEL_DIR, "model_results.json")) as f:
    model_results = json.load(f)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df_hits = df.dropna(subset=["Hits"]).copy()

# Frequency-encode
for col in high_cat:
    freq = freq_maps.get(col, {})
    df_hits[col] = df_hits[col].map(freq).fillna(0)

# ── Predict ───────────────────────────────────────────────────────────────────
print("Running predictions ...")
y_log_pred = model.predict(df_hits)
y_pred_raw = np.expm1(y_log_pred)
y_actual   = df_hits["Hits"].values

# ── 1. player_predictions.json (sample 2000 for scatter) ────────────────────
sample_idx = np.random.default_rng(42).choice(len(df_hits), size=min(2000, len(df_hits)), replace=False)
scatter_data = [
    {
        "name":      str(df_hits.iloc[i]["Name"]),
        "actual":    float(y_actual[i]),
        "predicted": round(float(y_pred_raw[i]), 1),
        "ova":       int(df_hits.iloc[i].get("OVA", 0) or 0),
        "position":  str(df_hits.iloc[i].get("Best Position", "?")),
    }
    for i in sample_idx
]
with open(os.path.join(DASH_DIR, "scatter_data.json"), "w") as f:
    json.dump(scatter_data, f)
print(f"  scatter_data.json -> {len(scatter_data)} players")

# ── 2. top_predictions.json (top 50 by actual Hits) ────────────────────────
top_idx  = np.argsort(y_actual)[::-1][:50]
top_data = [
    {
        "rank":      int(r + 1),
        "name":      str(df_hits.iloc[i]["Name"]),
        "actual":    int(y_actual[i]),
        "predicted": round(float(y_pred_raw[i]), 1),
        "ova":       int(df_hits.iloc[i].get("OVA", 0) or 0),
        "position":  str(df_hits.iloc[i].get("Best Position", "?")),
        "nationality": str(df_hits.iloc[i].get("Nationality", "?")),
    }
    for r, i in enumerate(top_idx)
]
with open(os.path.join(DASH_DIR, "top_predictions.json"), "w") as f:
    json.dump(top_data, f)
print(f"  top_predictions.json -> {len(top_data)} players")

# ── 3. feature_importance.json ───────────────────────────────────────────────
prep       = model.named_steps["prep"]
inner      = model.named_steps["model"]

# Re-build feature names
CAT_OHE       = ["Preferred Foot", "A/W", "D/W", "Contract Status", "Best Position"]
ohe_cats      = prep.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(CAT_OHE).tolist()

DROP_COLS     = ["ID","Name","LongName","photoUrl","playerUrl","Contract","Joined","Loan Date End","Short Passing","Hits"]
ALL_CAT       = CAT_OHE + high_cat + ["Club"]
num_cols      = [c for c in df.columns if c not in DROP_COLS + ALL_CAT and df[c].dtype in [np.float64, np.int64, "Int64"]]
FINAL_NUM     = num_cols + high_cat
feat_names    = FINAL_NUM + ohe_cats

importances   = inner.feature_importances_
min_len       = min(len(feat_names), len(importances))
fi_series     = pd.Series(importances[:min_len], index=feat_names[:min_len])
fi_top20      = fi_series.nlargest(20).sort_values(ascending=False)

fi_data = [{"feature": k, "importance": round(float(v), 5)} for k, v in fi_top20.items()]
with open(os.path.join(DASH_DIR, "feature_importance.json"), "w") as f:
    json.dump(fi_data, f)
print(f"  feature_importance.json -> {len(fi_data)} features")

# ── 4. hits_distribution.json (histogram buckets) ────────────────────────────
hits_vals  = y_actual[y_actual <= 500]          # cap at 500 for readability
counts, edges = np.histogram(hits_vals, bins=40)
dist_data  = [{"label": int(edges[i]), "count": int(counts[i])} for i in range(len(counts))]
with open(os.path.join(DASH_DIR, "hits_distribution.json"), "w") as f:
    json.dump(dist_data, f)
print(f"  hits_distribution.json -> {len(dist_data)} buckets")

# ── 5. model_metrics.json (reformatted for dashboard) ────────────────────────
metrics_out = []
for name in ["Ridge", "Random Forest", "XGBoost", "LightGBM"]:
    cv  = model_results["cv"][name]
    tst = model_results["test"][name]
    metrics_out.append({
        "name":         name,
        "best":         name == best_name,
        "cv_rmse":      round(cv["cv_rmse_mean"], 4),
        "cv_rmse_std":  round(cv["cv_rmse_std"], 4),
        "cv_r2":        round(cv["cv_r2_mean"], 4),
        "test_rmse":    round(tst["test_rmse_log"], 4),
        "test_r2":      round(tst["test_r2"], 4),
        "test_mae_raw": round(tst["test_mae_raw_hits"], 1),
    })
with open(os.path.join(DASH_DIR, "model_metrics.json"), "w") as f:
    json.dump(metrics_out, f)
print(f"  model_metrics.json -> {len(metrics_out)} models")

# ── 6. all_player_predictions.json (search feature) ─────────────────────────
all_df = df.copy()
for col in high_cat:
    freq = freq_maps.get(col, {})
    all_df[col] = all_df[col].map(freq).fillna(0)

y_all_log = model.predict(all_df)
y_all_raw = np.round(np.expm1(y_all_log), 1)

def safe_int(v, default=0):
    try:
        val = float(v)
        return int(val) if not np.isnan(val) else default
    except (TypeError, ValueError):
        return default

player_list = [
    {
        "id":          safe_int(df.iloc[i]["ID"], i),
        "name":        str(df.iloc[i]["Name"]),
        "nationality": str(df.iloc[i].get("Nationality", "?")),
        "position":    str(df.iloc[i].get("Best Position", "?")),
        "ova":         safe_int(df.iloc[i].get("OVA", 0)),
        "age":         safe_int(df.iloc[i].get("Age", 0)),
        "club":        str(df.iloc[i].get("Club", "?")),
        "actual_hits": round(float(df.iloc[i]["Hits"]), 0) if pd.notna(df.iloc[i]["Hits"]) else None,
        "pred_hits":   float(y_all_raw[i]),
    }
    for i in range(len(df))
]
with open(os.path.join(DASH_DIR, "all_players.json"), "w") as f:
    json.dump(player_list, f)
print(f"  all_players.json -> {len(player_list)} players")

print("\nDone — all dashboard data exported to:", DASH_DIR)
