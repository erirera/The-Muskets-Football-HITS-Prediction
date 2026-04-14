# LinkedIn Post — Muskets FC Hits Prediction

---

🏆 **Built a full Machine Learning pipeline to predict football player popularity — here's what I found.**

I've been working on a football analytics project using a FIFA-style dataset of **19,021 players**, and the goal was deceptively interesting: **can we predict how many times a player's profile gets viewed — just from their in-game attributes?**

Here's the full pipeline I built 👇

---

**📊 The Problem**
The dataset had a column called `Hits` — the number of times a player's page was searched. It ranged from 1 to 8,400, was heavily right-skewed, and was missing for ~14% of players. A perfect ML challenge.

---

**🔧 What I built**

✅ **Data Cleaning** — standardised height/weight units, parsed €K/€M financial values, expanded K-suffix Hits, split contract dates into structured columns

✅ **Feature Engineering** — 61 numeric features (skills, physicals, financials), one-hot encoded 5 categorical columns, frequency-encoded Nationality & Positions (high cardinality), log1p-transformed the target to handle skew

✅ **4 Models trained & compared via 5-Fold CV:**
- Ridge Regression → R² = 0.674
- Random Forest → R² = 0.788
- **XGBoost → R² = 0.812 ★ (winner)**
- LightGBM → R² = 0.812

---

**🎯 Key Results (XGBoost)**

The model predicts Hits with:
- **R² = 0.812** — explains 81% of variance
- **Average error of ±11 hits** per player (raw space)

Sample predictions:
- K. Mbappé: actual 1,600 → predicted **1,408** ✓
- L. Messi: actual 771 → predicted **626** ✓
- Cristiano Ronaldo: actual 562 → predicted **543** ✓

---

**💡 What drives player Hits?**

The top XGBoost features point to a clear pattern: **it's not just overall rating**. Value (market price), Wage, Release Clause, and composite stats like Total Stats all rank highly — suggesting that *commercial profile* matters as much as in-game performance when it comes to fan engagement.

---

**📱 Interactive Dashboard**

I also built a **premium dark-mode HTML dashboard** with 6 interactive views:
- Model comparison & metrics
- Feature importance chart
- Predicted vs Actual scatter (hover for player names)
- Full player search across all 19,021 players
- Top-50 most-viewed players leaderboard

---

This was a great reminder that the most interesting ML targets are often the "soft" metrics — popularity, engagement, cultural relevance — not just the obvious performance stats.

Happy to share the code or chat about the approach 👊

#MachineLearning #Football #DataScience #XGBoost #Python #Analytics #FootballAnalytics #SportsTech #AI
