# 👥 Employee Attrition Prediction — Analysis

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

*Files: `analysis.ipynb` (full workflow), `HR_Attrition.csv` (source data), `chart1–5` PNGs, `summary.docx` (HR Director report).*
