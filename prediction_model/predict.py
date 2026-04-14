"""
predict.py
==========
Run inference on new player data using the saved hits_model.pkl.

Usage examples
--------------
# Predict for a single player dict:
    python predict.py --player "{'Age': 25, 'OVA': 82, 'Nationality': 'Argentina', ...}"

# Predict for all rows in a CSV (must have same columns as training data):
    python predict.py --csv path/to/new_players.csv --out predictions_output.csv

# Quick demo on cleaned dataset (first 10 rows):
    python predict.py --demo
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hits_model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found at {MODEL_PATH}.\nRun hits_prediction.py first.")
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["name"], bundle["high_cat"]

def frequency_encode_col(df, col, pipe):
    """
    Re-apply frequency encoding learned during training.
    The pipeline stores the fitted preprocessor; we need to recreate
    freq-encoding from the training data embedded in the RF/LGBM trees —
    which is the train-time frequency map learned inside the training script.

    Because we serialised only the sklearn pipeline (which already had
    frequency-encoded values passed in), we look up the saved freq map
    stored alongside the model, or fall back to 0 for unseen categories.
    """
    freq_path = os.path.join(BASE_DIR, "freq_maps.json")
    if os.path.exists(freq_path):
        with open(freq_path) as f:
            maps = json.load(f)
        freq = maps.get(col, {})
        return df[col].map(freq).fillna(0).astype(float)
    else:
        # Fallback: treat all values as unseen → 0
        return pd.Series(0.0, index=df.index)

def predict(df_input: pd.DataFrame):
    model, model_name, high_cat = load_model()

    df = df_input.copy()

    # Frequency-encode high-cardinality columns
    for col in high_cat:
        if col in df.columns:
            df[col] = frequency_encode_col(df, col, model)
        else:
            df[col] = 0.0

    y_log_pred = model.predict(df)
    y_hits     = np.expm1(y_log_pred)
    return y_hits, model_name


def demo():
    DATA_PATH = os.path.join(BASE_DIR, "..", "Muskets_data_cleaned.csv")
    if not os.path.exists(DATA_PATH):
        sys.exit(f"Cleaned data not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False, nrows=20)
    actual_hits = df["Hits"].values

    predictions, model_name = predict(df)

    print(f"\n{'─'*55}")
    print(f"  Demo Predictions using [{model_name}]")
    print(f"{'─'*55}")
    print(f"  {'Player':<25} {'Actual':>8}  {'Predicted':>10}")
    print(f"{'─'*55}")
    for i, (name, actual, pred) in enumerate(zip(df["Name"], actual_hits, predictions)):
        flag = ""
        if pd.notna(actual):
            flag = "✓" if abs(pred - actual) / (actual + 1) < 0.5 else ""
        print(f"  {str(name):<25} {str(actual):>8}  {pred:>10.1f}  {flag}")
    print(f"{'─'*55}\n")


def main():
    parser = argparse.ArgumentParser(description="Predict Hits for football players.")
    parser.add_argument("--demo",   action="store_true",
                        help="Run demo on first 20 rows of cleaned dataset.")
    parser.add_argument("--csv",    type=str, default=None,
                        help="Path to input CSV file for batch prediction.")
    parser.add_argument("--out",    type=str, default="predictions_output.csv",
                        help="Output CSV path for batch prediction results.")
    parser.add_argument("--player", type=str, default=None,
                        help="JSON string of a single player's features.")
    args = parser.parse_args()

    if args.demo:
        demo()

    elif args.csv:
        df = pd.read_csv(args.csv, low_memory=False)
        predictions, model_name = predict(df)
        df["Predicted_Hits"] = np.round(predictions, 1)
        out_path = os.path.join(BASE_DIR, args.out)
        df.to_csv(out_path, index=False)
        print(f"[{model_name}] Predictions saved to: {out_path}")

    elif args.player:
        player_dict = json.loads(args.player)
        df = pd.DataFrame([player_dict])
        predictions, model_name = predict(df)
        print(f"\n[{model_name}] Predicted Hits: {predictions[0]:.1f}\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
