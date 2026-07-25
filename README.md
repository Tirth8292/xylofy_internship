# 📁 Tirth Shah — Data Science Internship Projects

A consolidated summary of weekly internship projects: regression, classification, and (pending) time-series forecasting work, each with model comparisons, data-driven fact-checking of the accompanying business write-ups, and next-step recommendations.

---

## Table of Contents
1. [Week 1 — House Price Prediction](#week-1--house-price-prediction)
2. [Week 2 — Employee Attrition Prediction](#week-2--employee-attrition-prediction)
3. [Week 3 & 4 — Sales Forecasting Intelligence *(pending correct content)*](#week-3--4--sales-forecasting-intelligence-pending)

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

## Week 3 & 4 —  Employee Attrition Prediction — Analysis

**Dataset:** IBM HR Analytics Employee Attrition (`WA_Fn-UseC_-HR-Employee-Attrition.csv`) · 1,470 employees, 35 columns
**Best model:** Logistic Regression (`class_weight='balanced'`) · **Report date:** 28 June 2026

A classification project predicting which employees are likely to leave, comparing Logistic Regression, Random Forest, and Gradient Boosting.

---

## 1. Dataset & Setup

| | |
|---|---|
| Rows / columns | 1,470 / 35 (45 after one-hot encoding + scaling) |
| Target | `Attrition` — 237 left (16.1%), 1,233 stayed (83.9%) — **imbalanced** |
| Dropped columns | `EmployeeNumber`, `Over18`, `StandardHours`, `EmployeeCount` (IDs/constants) |
| Missing values | 0 |
| Split | 80/20, stratified, `random_state=42` |
| Imbalance handling | `class_weight='balanced'` (LR, RF); `sample_weight` via `compute_sample_weight` (GB) |

---

## 2. Model Comparison

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | 0.357 | **0.660** | **0.463** | **0.804** |
| Random Forest | 0.333 | 0.511 | 0.403 | 0.772 |
| Gradient Boosting | 0.353 | 0.511 | 0.417 | 0.778 |

**Winner: Logistic Regression**, by F1 and ROC-AUC. This is a sensible choice given the goal (catch at-risk employees, not just maximize accuracy) — its recall of 0.66 means it catches roughly two-thirds of actual leavers, well ahead of the other two models. The ROC curve (Chart 5) confirms this: the red LR line sits above the other two for most of the FPR range, especially in the low-FPR region that matters most for a "flag high-risk employees" use case.

### ⚠️ One correction worth making before this goes to the HR Director

The summary document reports **"80.4% accuracy"** and labels the 80.4% figure as **Model Accuracy** in its headline stats table. That 80.4% is actually the **ROC-AUC**, not accuracy — two different things. From the confusion matrix (Chart 3: TN=191, FP=56, FN=16, TP=31, n=294):

Accuracy = (191+31) / 294 = **75.5%**

75.5% is the correct accuracy figure. This isn't just semantics — with a class split of 84%/16%, a model predicting "everyone stays" would already hit ~84% accuracy while catching zero leavers, so accuracy is actually a *misleading* metric here and ROC-AUC/recall are the right ones to lead with. I'd recommend keeping ROC-AUC (0.804) as the headline number, but relabeling it correctly rather than calling it "accuracy" — an HR Director fact-checking this against the confusion matrix would spot the mismatch.

---

## 3. What Drives Attrition (Chart 4 + text)

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

**Overtime and frequent travel are the two strongest, most trustworthy signals** — both are large-magnitude coefficients on common, well-populated categories (416 employees work overtime, plenty travel frequently), so these aren't artifacts of a rare category. The raw rates back this up directly: overtime workers leave at 30.5% vs. 10.4% for everyone else — a real, large gap, verified straight from the data.

**One flag on "Research Director" (coefficient 1.112, 5th-ranked "driver"):** Research Director actually has the *lowest* attrition rate of any job role in the dataset (2.5%), verified directly from the raw data. A large logistic-regression coefficient on a category with almost no leavers is the classic sign of an unstable estimate on a small/sparse group, not a genuine risk driver. As currently presented, the chart could be misread as "being a Research Director increases attrition risk" — nearly the opposite of what's true. Worth a caption clarifying that a few of the mid-table coefficients (Research Director, EducationField=Other) reflect small/sparse groups rather than dependable effects, distinct from the two clearly load-bearing findings (overtime, travel).

---

## 4. Cross-Checking the EDA Claims Against the Data

I recomputed the key EDA numbers directly from the CSV to check them against the notebook's narrative text and the charts. Most hold up; a few don't:

- ✅ **Sales dept 20.6%, HR 19.0%, R&D 13.8%** — matches Chart 1 and the raw data exactly.
- ✅ **Sales Representative 39.8–39.9% attrition** — matches Chart 1 and raw data.
- ✅ **Income gap: leavers average $4,787/mo vs. $6,833/mo for stayers** — matches Chart 2 exactly.
- ❌ **Laboratory Technician attrition is reported as "31.2%"** in the notebook narrative and the HR summary — but the actual rate, confirmed both from Chart 1 and directly from the raw data, is **23.9%**. This is a real inconsistency between the write-up text and the project's own chart (which shows 23.9% right below the same claim). Worth fixing — 31.2% overstates the Lab Technician problem relative to what the data actually shows.
- ❌ **"People who left earned $4,780 less per month"** (from the summary doc's "Salary Is Not the Whole Story" section) — this appears to conflate the leavers' *average income* ($4,787) with the *income gap* between leavers and stayers, which is actually **$2,046** (the number correctly stated earlier in the notebook's own EDA output, and confirmed from the raw data). Worth correcting — $4,780 vs $2,046 meaningfully changes how large the "salary isn't the main driver" argument looks.
- ⚠️ **"Work-Life Balance 1: 25.6% attrition vs Balance 4: 14.3%"** — recomputed from the raw data this is actually **31.3% vs. 17.6%**. Both numbers are off from what's stated, though the *direction* of the finding (low work-life balance → higher attrition) is correct and real.

None of these change the overall story (overtime and travel dominate, salary is secondary), but the Lab Technician and salary-gap numbers specifically are the ones most likely to get repeated verbatim in a leadership deck, so they're worth fixing at the source.

---

## 5. Business Recommendations — Sanity Check

The two recommendations in the summary doc hold up reasonably well against the numbers:

- **Cap overtime:** 416 employees (28% of workforce) work overtime, confirmed directly from the data, at a 30.5% attrition rate vs. 10.4% for others — a genuine ~20-point gap. The claimed "60+ employees saved per year" is actually a *conservative* estimate: if overtime attrition fully closed to the 10.4% baseline, that's roughly 84 employees/year on the current overtime population, so 60+ is a defensible, non-inflated target.
- **Career paths for Lab Techs / Sales Reps:** the "35% → 15%" target range is worth a second look given the *actual* current rates for these two roles (23.9% and 39.8% respectively, per the corrected figures in §4) — the blended starting point is likely somewhat different from 35% depending on how it was weighted.

---

## 6. Model Limitations (already well-documented — worth keeping as-is)

The summary doc's caveats section is genuinely good practice and I'd leave it largely untouched: it correctly notes the model is a point-in-time snapshot, that correlation (e.g. `MaritalStatus_Single`) isn't causation, that the single-company dataset may not generalize, and that with 16% base-rate attrition the model will still miss real leavers. I'd only add one more: **precision is low (0.357)** — of every employee the model flags as high-risk, roughly 2 in 3 flags will be false alarms. That's an acceptable trade-off for a "review these people" workflow (better to over-flag than miss leavers), but it should be stated explicitly alongside the other caveats so HR doesn't over-interpret an individual risk score.

---

## 7. Suggested Next Steps

- Fix the numeric discrepancies in §4 before this report circulates further (Lab Technician rate, salary gap, work-life-balance rates, and the accuracy/ROC-AUC label).
- Add a footnote to Chart 4 flagging that a couple of the mid-ranked coefficients (Research Director, EducationField=Other) come from small/sparse groups and shouldn't be read as dependable drivers the way overtime and travel are.
- Try tuning the decision threshold directly (rather than relying only on `class_weight='balanced'`) via precision-recall trade-off analysis, to see if recall can be pushed higher without a large precision cost.
- Cross-validate rather than relying on a single 80/20 split, given the relatively small number of positive cases (237) available to learn from.

---

