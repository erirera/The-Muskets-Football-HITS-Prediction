# 🏆 The Muskets FC — Player Hits Prediction Model

> **Predicting player page-view popularity using machine learning on football data.**  
> An end-to-end ML pipeline: data cleaning → feature engineering → model training → interactive dashboard.

---

## 📋 Project Overview

This project builds a regression model to predict the **number of Hits** (profile page views) for each player in the Muskets Football dataset — a FIFA-style dataset of **19,021 players** with 80+ attributes.

### Why Hits?
Hits represent player popularity and fan engagement. Predicting them from in-game attributes (OVA, skills, position, wage, nationality, etc.) reveals which factors drive a player's public interest — beyond just their rating.

---

## 📁 Project Structure

```
The-Muskets-Football-HITS-Prediction/
│
├── Muskets_data.csv                  # Raw input data (8.3 MB)
├── Muskets_data_cleaned.csv          # Cleaned data (10 MB)
│
├── scratch/
│   └── data_cleaning.py              # Data cleaning script
│
├── prediction_model/
│   ├── hits_prediction.py            # ★ Main training pipeline
│   ├── predict.py                    # Inference script
│   ├── hits_model.pkl                # Saved best model (XGBoost)
│   ├── freq_maps.json                # Frequency encoding maps
│   ├── model_results.json            # Full metrics for all models
│   ├── hits_distribution.png         # Target distribution chart
│   ├── cv_rmse_comparison.png        # Model comparison chart
│   ├── predictions_vs_actual.png     # Scatter plot chart
│   └── feature_importance.png        # Top-20 features chart
│
├── dashboard/
│   ├── index.html                    # ★ Interactive dashboard
│   ├── model_metrics.json
│   ├── feature_importance.json
│   ├── scatter_data.json
│   ├── hits_distribution.json
│   ├── top_predictions.json
│   └── all_players.json              # All 19,021 player predictions
│
└── export_dashboard_data.py          # Generates dashboard JSON files
```

---

## ⚙️ Pipeline

```
Raw Data (Muskets_data.csv)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1 · DATA CLEANING  (scratch/data_cleaning.py)     │
│  • Standardise height (cm) & weight (kg)                │
│  • Parse monetary values (€K / €M → int)                │
│  • Clean star ratings (W/F, SM, IR)                     │
│  • Parse Hits (K suffix expansion)                      │
│  • Split Contract into Start / End / Status             │
└─────────────────────────────────────────────────────────┘
        │
        ▼
Muskets_data_cleaned.csv  (19,021 rows × 80 cols)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2 · FEATURE ENGINEERING                           │
│  • Drop identifiers (ID, Name, URLs, Contract text)     │
│  • Target: log1p(Hits)  ← handles extreme right-skew   │
│  • Numeric (61): skills, physicals, financials, stats   │
│  • OHE (5 cols): Foot, A/W, D/W, Contract Status, Pos  │
│  • Freq-encoded (2): Nationality, Positions             │
│  • Impute median + StandardScaler                       │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3 · MODEL TRAINING  (5-Fold Cross-Validation)     │
│  ┌──────────────────┬────────────┬──────────┬────────┐  │
│  │ Model            │ CV RMSE    │ Test R²  │ MAE    │  │
│  ├──────────────────┼────────────┼──────────┼────────┤  │
│  │ Ridge            │ 0.6985     │ 0.674    │ ±16.9  │  │
│  │ Random Forest    │ 0.5668     │ 0.788    │ ±11.6  │  │
│  │ XGBoost  ★ Best │ 0.5372     │ 0.812    │ ±11.1  │  │
│  │ LightGBM         │ 0.5360     │ 0.812    │ ±11.2  │  │
│  └──────────────────┴────────────┴──────────┴────────┘  │
└─────────────────────────────────────────────────────────┘
        │
        ▼
Best model saved → hits_model.pkl
```

---

## 📊 Results

| Metric | Value |
|---|---|
| **Best Model** | XGBoost |
| **Test R²** | **0.812** — explains 81% of variance |
| **Test RMSE (log)** | **0.536** |
| **MAE (raw Hits)** | **±11.1 hits** per player |
| **CV Folds** | 5-Fold KFold |
| **Training set** | 13,140 players |
| **Test set** | 3,286 players |

### Sample Predictions

| Player | Actual Hits | Predicted |
|---|---|---|
| K. Mbappé | 1,600 | 1,408 ✓ |
| L. Messi | 771 | 626 ✓ |
| Neymar Jr | 595 | 558 ✓ |
| Cristiano Ronaldo | 562 | 543 ✓ |
| V. van Dijk | 321 | 292 ✓ |
| N. Kanté | 202 | 181 ✓ |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn joblib
```

### 2. Clean the Data

```bash
python scratch/data_cleaning.py
```

### 3. Train the Model

```bash
python -X utf8 prediction_model/hits_prediction.py
```

This will:
- Run 5-fold CV for all 4 models
- Save the best model to `hits_model.pkl`
- Generate charts: distribution, CV comparison, scatter, feature importance
- Print a full performance summary

### 4. Make Predictions

```bash
# Demo on first 20 players:
python -X utf8 prediction_model/predict.py --demo

# Batch prediction from CSV:
python -X utf8 prediction_model/predict.py --csv new_players.csv --out results.csv
```

### 5. Launch the Dashboard

```bash
# Export JSON data for the dashboard:
python -X utf8 export_dashboard_data.py

# Open dashboard/index.html in your browser
# Tip: use a local server to avoid CORS issues with JSON:
python -m http.server 8000 --directory dashboard
# Then open: http://localhost:8000
```

---

## 📈 Interactive Dashboard

The dashboard (`dashboard/index.html`) provides 6 interactive views:

| Tab | Content |
|---|---|
| **Overview** | Key metrics cards + Hits distribution chart + pipeline summary |
| **Model Comparison** | Side-by-side metric cards + CV RMSE and R² bar charts |
| **Feature Importance** | Top-20 XGBoost feature importance horizontal bar chart |
| **Predicted vs Actual** | Interactive scatter plot — hover for player names |
| **Player Lookup** | Full-text search across all 19,021 players with predictions |
| **Top 50 Leaderboard** | Ranked table of most-viewed players with accuracy column |

---

## 🤝 Dependencies

| Library | Version | Purpose |
|---|---|---|
| pandas | ≥1.5 | Data loading & manipulation |
| numpy | ≥1.24 | Numerical computing |
| scikit-learn | ≥1.4 | Preprocessing & Ridge / Random Forest |
| xgboost | ≥2.0 | Best-performing model |
| lightgbm | ≥4.0 | Runner-up model |
| matplotlib | ≥3.7 | Static chart generation |
| seaborn | ≥0.12 | Styling for charts |
| joblib | ≥1.3 | Model serialisation |

---

## 📝 Notes

- The target variable `Hits` is **highly right-skewed** (median = 5, max = 8,400).  
  All models are trained on `log1p(Hits)` and predictions are back-transformed via `expm1()`.
- 2,595 players had missing `Hits` values and were excluded from training (but predictions are generated for all 19,021).
- `Nationality` and `Positions` are frequency-encoded due to high cardinality.
- All scripts use `-X utf8` flag on Windows to avoid cp1252 encoding errors.

---

*Project by Dele Falebita · Built with Python & scikit-learn ecosystem in Antigravity AI*
