# 🏠 House Price Prediction — Analysis

**Author:** Tirth Shah · **Date:** 21 May 2026 · **Dataset:** `Housing.csv` (Kaggle Housing Prices Dataset)

A regression project predicting house prices from property features (area, rooms, amenities, furnishing status), comparing Linear Regression against Random Forest.

---

## 1. Dataset

| | |
|---|---|
| Records | 545 houses |
| Features | 12 raw → 13 after encoding |
| Target | `price` |
| Missing values | 0 |
| Duplicate rows | 0 |
| Train / test split | 436 / 109 (80/20, `random_state=42`) |

**Cleaning steps:** binary `yes/no` columns (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`) mapped to `1/0`; `furnishingstatus` one‑hot encoded into `furnishingstatus_semi-furnished` and `furnishingstatus_unfurnished` (furnished = baseline).

---

## 2. Model Results

| Metric | Linear Regression | Random Forest |
|---|---|---|
| MAE | ₹970,043 | ₹1,022,560 |
| RMSE | ₹1,324,507 | ₹1,401,497 |
| **R²** | **0.6529** | 0.6114 |

**Winner: Linear Regression.** It edges out Random Forest on every metric here, which — with only 545 rows and 13 mostly-additive features — is a believable result: tree ensembles need more data to earn their extra variance, and price does look close to a linear function of these features (see §4). A ~4-point R² gap on a dataset this size is within the range you'd expect from run-to-run noise, so I wouldn't over-read the "Linear beats RF" story as a deep structural finding — it's a reasonable, mildly interesting result rather than a strong signal.

*Caveat worth stating in the report:* neither model was cross-validated (single 80/20 split) and Random Forest was run with default/near-default hyperparameters (`n_estimators=100`, no tuning of `max_depth`, `min_samples_leaf`, etc.). A tuned RF or 5-fold CV could plausibly close or reverse this gap — worth a one-line caveat rather than a confident claim.

---

## 3. What Actually Drives Price

Cross-checking the correlation heatmap, the Random Forest importance chart, and the LR coefficients — they agree well, which is reassuring:

| Feature | Correlation w/ price | RF importance | LR coefficient |
|---|---|---|---|
| **Area** | 0.54 (highest) | ~0.47 (highest) | positive |
| **Bathrooms** | 0.52 | ~0.15 | ₹1,094,445 |
| **Air conditioning** | 0.45 | ~0.06 | ₹791,427 |
| **Stories** | 0.42 | ~0.06 | — |
| **Parking** | 0.38 | ~0.06 | — |
| **Hot water heating** | 0.09 (weak) | ~0.02 | ₹684,650 |
| **Unfurnished** | −0.28 | ~0.04 | −₹413,645 |
| **Bedrooms** | 0.37 | ~0.05 | — |

**Area dominates** — by a wide margin in both correlation and RF importance. Everything else is second-tier.

**Bathrooms > bedrooms is the most interesting finding**, and it holds up across all three views (0.52 vs 0.37 correlation; 0.15 vs 0.05 importance): buyers seem to pay for bathroom count more than bedroom count. One thing to flag rather than state as fact — bathrooms (r=0.52) and area (r=0.54) are themselves correlated at 0.19, and bathrooms/stories at 0.41, so part of the "bathrooms matter" signal is bathrooms acting as a proxy for overall house size/quality rather than a pure, independent effect. The RF importance (0.15, well above every feature besides area) still supports a real effect, just probably smaller in isolation than the raw coefficient/correlation suggest.

**Hot water heating is the one place the report's original framing overstates things.** The notebook calls it "high coefficient... indicating it's a premium feature," but its correlation with price is only 0.09 — the weakest of any feature discussed. A large LR coefficient on a rare, low-correlation binary feature is a classic sign of an unstable/noisy estimate (few positive cases to estimate the effect from), not strong evidence of a real premium. I'd soften this claim in the write-up rather than presenting it as a confirmed insight.

**Air conditioning** is the most solid "amenity" finding — reasonably high correlation (0.45), consistent RF importance, and a sizeable, plausible coefficient.

---

## 4. Diagnostics from the Charts

- **Chart 1 (price distribution):** right-skewed with a long tail, as claimed — but "most houses ₹3–5 crore" is a units error worth catching before this goes out: the x-axis is in raw rupees (2,000,000–13,000,000), i.e. ₹20 lakh–₹1.3 crore. The bulk of listings sit around **₹30–50 lakh** (3M–5M), not crore. This is a wording bug, not a data or model problem — but it's the kind of thing worth fixing before the report circulates.
- **Chart 2 (heatmap):** no red/negative cells above ~|0.3| except `furnishingstatus_unfurnished` (−0.28 to −0.59) — mild multicollinearity there (semi vs. unfurnished dummies), unsurprising for one-hot encoding, not a real problem for either model here.
- **Chart 3 (actual vs. predicted):** the scatter tracks the diagonal reasonably well in the ₹2M–7M range where most of the data sits, but **the model systematically under-predicts high-end homes** (points above ~₹8M sit well below the red line). This is the clearest visual evidence of the R²=0.65 ceiling: a linear model averages out the luxury segment because it's thin and probably driven by features not well captured here (location/neighborhood isn't in the dataset at all). Worth calling out explicitly — it's a concrete, visible limitation, not just an abstract "R² isn't 1."
- **Chart 4 (RF feature importance):** confirms area's dominance and roughly agrees with the correlation ranking (bathrooms > AC ≈ parking ≈ stories > bedrooms), except `furnishingstatus_unfurnished` shows up higher in RF importance (~0.035) than its correlation rank would suggest — consistent with RF picking up non-linear/threshold effects that correlation misses.

---

## 5. Recommendation for the Business Section

The existing recommendation ("focus on Area + Bathrooms + AC") is directionally reasonable and can stay, but I'd tighten two things:
1. Drop or heavily hedge the hot-water-heating claim (weak evidence, see §3).
2. Add the high-end under-prediction finding (§4) as an explicit limitation: the model is not yet reliable for luxury/high-value listings, so it shouldn't be used to price homes above ~₹8–9M without adjustment.

---

## 6. Suggested Next Steps

- Re-run with 5-fold cross-validation instead of a single split to get a more trustworthy R²/MAE estimate.
- Try `log(price)` as the target — right-skewed targets like this one usually respond well to a log transform and it may narrow the high-end under-prediction gap.
- Light hyperparameter tuning on Random Forest (`max_depth`, `min_samples_leaf`) before concluding LR "wins" — the current comparison uses RF defaults.
- Add a feature for locality/neighborhood if available in a richer version of the dataset — the residual pattern in Chart 3 suggests an omitted variable (likely location) driving the luxury segment.

---

*Files: `analysis.ipynb` (full workflow), `Housing.csv` (source data), `charts/` (chart1–4 PNGs), `House_Price_Prediction_Summary.docx` (original report).*
