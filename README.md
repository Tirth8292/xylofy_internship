# 📁 Tirth Shah — Data Science Internship Projects

A consolidated summary of weekly internship projects: regression, classification, and (pending) time-series forecasting work, each with model comparisons, data-driven fact-checking of the accompanying business write-ups, and next-step recommendations.

---

## Table of Contents
1. Week 1 — House Price Prediction
2. Week 2 — Employee Attrition Prediction
3. Week 3 & 4 — Sales forecasting Intelligence — Forecasting, Anomalies & Product Segmentation

---

## Week 1 — House Price Prediction

**Author:** Tirth Shah · **Date:** 21 May 2026 · **Dataset:** `Housing.csv` (Kaggle Housing Prices Dataset)

A regression project predicting house prices from property features (area, rooms, amenities, furnishing status), comparing Linear Regression against Random Forest.

### 1. Dataset

| | |
|---|---|
| Records | 545 houses |
| Features | 12 raw → 13 after encoding |
| Target | `price` |
| Missing values | 0 |
| Duplicate rows | 0 |
| Train / test split | 436 / 109 (80/20, `random_state=42`) |

**Cleaning steps:** binary `yes/no` columns (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`) mapped to `1/0`; `furnishingstatus` one-hot encoded into `furnishingstatus_semi-furnished` and `furnishingstatus_unfurnished` (furnished = baseline).

### 2. Model Results

| Metric | Linear Regression | Random Forest |
|---|---|---|
| MAE | ₹970,043 | ₹1,022,560 |
| RMSE | ₹1,324,507 | ₹1,401,497 |
| **R²** | **0.6529** | 0.6114 |

**Winner: Linear Regression.** It edges out Random Forest on every metric here, which — with only 545 rows and 13 mostly-additive features — is a believable result: tree ensembles need more data to earn their extra variance, and price does look close to a linear function of these features (see §4). A ~4-point R² gap on a dataset this size is within the range you'd expect from run-to-run noise, so this shouldn't be over-read as a deep structural finding.

*Caveat worth stating in the report:* neither model was cross-validated (single 80/20 split) and Random Forest was run with default/near-default hyperparameters (`n_estimators=100`, no tuning of `max_depth`, `min_samples_leaf`, etc.). A tuned RF or 5-fold CV could plausibly close or reverse this gap.

### 3. What Actually Drives Price

Cross-checking the correlation heatmap, the Random Forest importance chart, and the LR coefficients — they agree well:

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

- **Area dominates** by a wide margin in both correlation and RF importance.
- **Bathrooms > bedrooms** is the most interesting finding, holding up across all three views — though bathrooms (r=0.52) correlate with area (0.19) and stories (0.41), so part of the signal is bathrooms acting as a proxy for overall house size/quality.
- **Hot water heating** is over-claimed in the original notebook ("premium feature" framing) — its correlation is only 0.09, the weakest of any feature discussed. A large coefficient on a rare, low-correlation binary feature is a classic sign of an unstable estimate, not a confirmed premium.
- **Air conditioning** is the most solid "amenity" finding — reasonably high correlation, consistent RF importance, and a sizeable, plausible coefficient.

### 4. Diagnostics from the Charts

- **Chart 1 (price distribution):** right-skewed with a long tail — but "most houses ₹3–5 crore" is a units error; the x-axis is in raw rupees, so the bulk of listings actually sit around **₹30–50 lakh**, not crore.
- **Chart 2 (heatmap):** no notable multicollinearity except `furnishingstatus_unfurnished` (−0.28 to −0.59), which is unsurprising for one-hot encoding.
- **Chart 3 (actual vs. predicted):** tracks well in the ₹2M–7M range, but the model **systematically under-predicts high-end homes** (>₹8M) — likely because location/neighborhood isn't in the dataset.
- **Chart 4 (RF importance):** confirms area's dominance; `furnishingstatus_unfurnished` ranks higher here than its correlation would suggest, consistent with RF picking up non-linear effects.

### 5. Recommendation for the Business Section

The existing recommendation ("focus on Area + Bathrooms + AC") is directionally reasonable, with two tweaks:
1. Drop or heavily hedge the hot-water-heating claim.
2. Add the high-end under-prediction finding as an explicit limitation — the model shouldn't be used to price homes above ~₹8–9M without adjustment.

### 6. Suggested Next Steps

- Re-run with 5-fold cross-validation for a more trustworthy R²/MAE estimate.
- Try `log(price)` as the target to address the right skew and high-end under-prediction.
- Light hyperparameter tuning on Random Forest before concluding LR "wins."
- Add a locality/neighborhood feature if available — likely the omitted variable behind the luxury-segment residuals.

*Files: `analysis.ipynb`, `Housing.csv`, `charts/` (chart1–4 PNGs), `House_Price_Prediction_Summary.docx`.*

---

## Week 2 — Employee Attrition Prediction

**Dataset:** IBM HR Analytics Employee Attrition (`WA_Fn-UseC_-HR-Employee-Attrition.csv`) · 1,470 employees, 35 columns
**Best model:** Logistic Regression (`class_weight='balanced'`) · **Report date:** 28 June 2026

A classification project predicting which employees are likely to leave, comparing Logistic Regression, Random Forest, and Gradient Boosting.

### 1. Dataset & Setup

| | |
|---|---|
| Rows / columns | 1,470 / 35 (45 after one-hot encoding + scaling) |
| Target | `Attrition` — 237 left (16.1%), 1,233 stayed (83.9%) — **imbalanced** |
| Dropped columns | `EmployeeNumber`, `Over18`, `StandardHours`, `EmployeeCount` (IDs/constants) |
| Missing values | 0 |
| Split | 80/20, stratified, `random_state=42` |
| Imbalance handling | `class_weight='balanced'` (LR, RF); `sample_weight` via `compute_sample_weight` (GB) |

### 2. Model Comparison

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | 0.357 | **0.660** | **0.463** | **0.804** |
| Random Forest | 0.333 | 0.511 | 0.403 | 0.772 |
| Gradient Boosting | 0.353 | 0.511 | 0.417 | 0.778 |

**Winner: Logistic Regression**, by F1 and ROC-AUC — a sensible choice given the goal (catch at-risk employees, not just maximize accuracy). Its recall of 0.66 means it catches roughly two-thirds of actual leavers.

**⚠️ Correction worth making before this goes to the HR Director:** the summary document labels 80.4% as **"Model Accuracy,"** but that figure is actually the **ROC-AUC**. From the confusion matrix (TN=191, FP=56, FN=16, TP=31, n=294): Accuracy = (191+31)/294 = **75.5%**. With an 84%/16% class split, a model predicting "everyone stays" would already hit ~84% accuracy while catching zero leavers — so ROC-AUC/recall are the right numbers to lead with, correctly labeled.

### 3. What Drives Attrition

| Feature | Coefficient (abs) |
|---|---|
| OverTime = Yes | 1.626 |
| BusinessTravel = Frequently | 1.597 |
| JobRole = Laboratory Technician | 1.572 |
| JobRole = Sales Representative | 1.264 |
| JobRole = Research Director | 1.112 |
| EducationField = Other | 1.020 |
| BusinessTravel = Rarely | 0.904 |
| MaritalStatus = Single | 0.865 |
| JobRole = Human Resources | 0.671 |
| TotalWorkingYears | 0.612 |

**Overtime and frequent travel are the two strongest, most trustworthy signals** — large-magnitude coefficients on common, well-populated categories. Overtime workers leave at 30.5% vs. 10.4% for everyone else.

**Flag on "Research Director"** (coefficient 1.112, 5th-ranked "driver"): this role actually has the *lowest* attrition rate of any job role (2.5%) — a large coefficient on a near-zero-attrition, sparse category is a sign of an unstable estimate, not a genuine risk driver. Worth a caption clarifying that this and `EducationField=Other` reflect small/sparse groups, unlike overtime and travel.

### 4. Cross-Checking the EDA Claims Against the Data

- ✅ **Sales dept 20.6%, HR 19.0%, R&D 13.8%** — matches raw data.
- ✅ **Sales Representative 39.8–39.9% attrition** — matches raw data.
- ✅ **Income gap: leavers $4,787/mo vs. $6,833/mo stayers** — matches Chart 2.
- ❌ **Laboratory Technician attrition reported as "31.2%"** — actual rate is **23.9%**, confirmed from the chart and raw data. Overstates the problem.
- ❌ **"People who left earned $4,780 less per month"** — conflates leavers' *average income* ($4,787) with the *income gap* between leavers/stayers, which is actually **$2,046**.
- ⚠️ **"Work-Life Balance 1: 25.6% vs Balance 4: 14.3%"** — recomputed as **31.3% vs. 17.6%**. Direction is correct; both magnitudes are off.

None of these change the overall story (overtime and travel dominate, salary is secondary), but the Lab Technician and salary-gap numbers are the most likely to be repeated verbatim in a leadership deck, so worth fixing at the source.

### 5. Business Recommendations — Sanity Check

- **Cap overtime:** 416 employees (28% of workforce) work overtime at a 30.5% attrition rate vs. 10.4% for others. The claimed "60+ employees saved per year" is actually *conservative* — full closure to baseline implies ~84/year.
- **Career paths for Lab Techs / Sales Reps:** the "35% → 15%" target range is worth revisiting given the corrected actual rates (23.9% and 39.8% respectively).

### 6. Model Limitations

Well-documented already: point-in-time snapshot, correlation ≠ causation, single-company dataset may not generalize, 16% base-rate attrition means the model will still miss real leavers. **Add:** precision is low (0.357) — roughly 2 in 3 flagged employees will be false alarms. Acceptable for a "review these people" workflow, but should be stated explicitly.

### 7. Suggested Next Steps

- Fix the numeric discrepancies in §4 before this report circulates further.
- Add a footnote to the coefficient chart flagging sparse-group instability (Research Director, EducationField=Other).
- Tune the decision threshold via precision-recall trade-off analysis rather than relying only on `class_weight='balanced'`.
- Cross-validate rather than relying on a single 80/20 split, given the relatively small number of positive cases (237).

*Files: `analysis.ipynb`, `HR_Attrition.csv`, `chart1–5` PNGs, `summary.docx` (HR Director report).*

---

## Week 3 & 4 —  Sales forecasting Intelligence — Forecasting, Anomalies & Product Segmentation

**Dataset:** Superstore Sales (`train.csv`) · 9,800 orders, 24 columns, 4-year horizon (2015–2018)
**Best forecasting model:** SARIMA(1,1,1)(1,1,1,12) · **Report date:** July 2026

An end-to-end sales intelligence pipeline: time-series decomposition → 3-model forecast bake-off → segment-level forecasting → dual-method anomaly detection → K-Means product clustering → a Streamlit dashboard (`app.py`) that serves it all.

---

## 1. Dataset & Setup

| | |
|---|---|
| Rows / columns | 9,800 / 24 |
| Date range | Jan 2015 – Dec 2018 (48 monthly points, 209 weekly points) |
| Key fields | `Order Date`, `Sales`, `Category`, `Sub-Category`, `Region` |
| Revenue by category | Technology $827,456 · Furniture $728,659 · Office Supplies $705,422 |
| Fulfillment time | 3.96 days average, consistent across regions (3.91–4.07 days) |

---

## 2. Forecast Model Bake-Off (Task 2–3)

Three models were validated on an 80/20 chronological split (last 3 months held out):

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| **SARIMA** | **$19,244.49** | **$19,950.07** | **20.53%** |
| Prophet | $20,296.01 | $22,487.47 | 21.89% |
| XGBoost | $29,086.17 | $29,122.60 | 32.14% |

**SARIMA wins** on all three metrics — this matches the memo's conclusion (§3) to authorize SARIMA for production. ✅ One caveat worth surfacing: a 20.5% MAPE is fairly wide for a "production-authorized" forecasting engine, and the memo doesn't mention this number anywhere, even though it's the model's own validated error rate.

---

## 3. ⚠️ 90-Day Forecast — Numbers Don't Match the Summary Memo

The notebook's actual SARIMA 3-month-ahead forecast (Task 3 output) is:

| | Month 1 | Month 2 | Month 3 |
|---|---|---|---|
| **Notebook (SARIMA, actual)** | $60,331.79 | $91,458.22 | $97,167.57 |
| **Memo (`summary_Tirth_Shah.pdf`, §3)** | $52,450.00 | $58,900.00 | $64,200.00 |

These are not close — the memo's numbers are lower, smoother, and don't reflect the sharp month-2 jump the model actually produces (driven by the strong November/December seasonal peak visible in the monthly seasonal table). This isn't a rounding difference; it looks like the memo's forecast figures were not pulled from this notebook run. **This should be corrected before the memo goes to the CFO** — a 34–52% understatement of projected revenue would directly undersize the procurement and safety-stock plan in §3–4.

---

## 4. Segment-Level Forecasts (Task 4)

Five SARIMA models fit independently by Category and Region:

| Segment | Month 1 | Month 2 | Month 3 |
|---|---|---|---|
| Furniture | $10,526.77 | $9,921.59 | $16,576.87 |
| Technology | $20,100.38 | $18,198.55 | $32,443.12 |
| Office Supplies | $17,978.32 | $15,467.39 | $23,346.41 |
| West | $15,478.12 | $13,405.16 | $28,366.09 |
| East | $11,878.47 | $13,477.98 | $19,848.30 |

Matches `segment_level_forecasts.png` exactly. ✅ Technology leads in absolute revenue, consistent with the memo's §2 claim that Technology is the primary revenue driver.

### West "stability" claim doesn't hold up

The memo (§2) states the **West Region** shows *"an exceptionally stable, upward trajectory... the business unit's most reliable growth source."* The notebook's own **Regional Volatility Index** (year-over-year % change std. dev., lower = more stable) says otherwise:

| Region | Volatility Index |
|---|---|
| **East** | **0.018** (most stable) |
| Central | 0.253 |
| **West** | **0.257** |
| South | 0.371 |

West did grow the most in absolute dollars ($145,908 → $248,131, 2015→2018), so it's a legitimate *growth* leader — but by the notebook's own stability metric, **East is ~14x more stable than West**, not the other way around. The memo's Capital Allocation recommendation (§6) to shift stock toward West is defensible on growth grounds, but the "most reliable/stable" language should be corrected or re-attributed to East.

---

## 5. Anomaly Detection (Task 5)

Weekly sales (209 weeks) run through Isolation Forest (5% contamination) and a rolling 4-week Z-Score (>2σ):

| Method | Anomalies flagged |
|---|---|
| Isolation Forest | **11** |
| Z-Score (rolling, >2σ) | **0** |
| Mutual consensus | 0 |

The Z-Score method flags nothing under this rolling-window setup — the "Z-Score Anomaly" legend entry in `detected_anomalies.png` is effectively decorative (only one faint unflagged point near the huge March 2015 spike). Worth noting in the memo's §4, since it currently presents both methods as if they contributed jointly ("cross-verified with rolling statistical Z-Scores") when in practice only Isolation Forest did the flagging.

---

## 6. Product Demand Clustering (Task 6)

K-Means (k=4, chosen from the elbow curve) on Sub-Category features (order volume, average order value, monthly sales volatility, 2015→2018 growth rate), visualized via PCA:

| Cluster | Members | Memo label (§5) |
|---|---|---|
| 0 | Copiers *(alone)* | High-Volume Core Anchors |
| 1 | Bookcases, Envelopes, Fasteners, Labels, Supplies, Tables | High-Variance Specialized Lines |
| 2 | Accessories, Appliances, Art, Binders, Chairs, Furnishings, Paper, Phones, Storage | Emerging Growth Catalogs |
| 3 | Machines *(alone)* | Low-Velocity / Stagnant Stock |

Labels are consistent with `app.py`'s strategy matrix. ⚠️ One thing worth flagging to the reader: Clusters 0 and 3 each contain a **single** sub-category (Copiers, Machines) — these are high-price, low-frequency items that sit far out on Principal Component 1 in `product_demand_clusters.png`. With only one member each, they're closer to outlier isolation than genuine behavioral segments, so the "stocking protocol" language for those two rows should be read as SKU-specific guidance rather than a generalizable cluster pattern.

The elbow curve itself (`clustering_elbow_curve.png`) doesn't show a sharp elbow — inertia declines smoothly from k=2 to k=7, so k=4 was a reasonable but somewhat judgment-based choice rather than an unambiguous "elbow."

---

## 7. Streamlit App (`app.py`) — one issue worth fixing

The **Forecast Explorer** page (Page 2) displays a "Model Diagnostic Accuracy Footprint" with MAE/RMSE metrics. These are **hardcoded placeholder values** (`mae_bench, rmse_bench = 28450.20, 36100.40`) shown regardless of which segment or horizon the user selects — they don't reflect the actual validated SARIMA error from the notebook ($19,244.49 / $19,950.07 MAE/RMSE, Task 3). As written, a user could reasonably believe these are the real per-segment accuracy numbers. Either compute live validation metrics per segment, or relabel the metric cards as illustrative/placeholder.

Everything else in the app lines up with the notebook: same SARIMA order, same Isolation Forest contamination rate, and the chart path (`charts/product_demand_clusters.png`) matches the notebook's `savefig` path.

---

## 8. Limitations (memo §7 is accurate and worth keeping)

The memo's own caveats section holds up well: the model is historical-precedent-based and can't anticipate black-swan disruptions, trade shocks, or supply shutdowns. Worth adding one more: **the 90-day forecast figures in §3 need to be regenerated from this notebook** before they're used for procurement planning (see §3 above).

---

## 9. Suggested Next Steps

- Regenerate the memo's §3 90-day forecast numbers directly from the notebook's SARIMA output ($60,331.79 / $91,458.22 / $97,167.57) rather than the current figures.
- Correct or re-attribute the "West is most stable" claim in §2 — East is the stability leader by the notebook's own volatility index; West is the growth leader.
- Replace the hardcoded MAE/RMSE benchmark in `app.py` Page 2 with real, segment-specific validation metrics (or clearly label them as illustrative).
- Reword the memo's §4 framing so it doesn't imply Z-Score anomaly detection contributed meaningfully — it flagged zero weeks under the current settings.
- Consider re-running the elbow method with a wider k range or a silhouette score check, since k=4 wasn't a sharp, unambiguous elbow.

---

*Files: `analysis.ipynb` (full workflow) · `train.csv` (source data) · `app.py` + `requirements.txt` (Streamlit dashboard) · `clustering_elbow_curve.png`, `detected_anomalies.png`, `product_demand_clusters.png`, `segment_level_forecasts.png` (exported charts) · `summary_Tirth_Shah.pdf` (executive memo).*

