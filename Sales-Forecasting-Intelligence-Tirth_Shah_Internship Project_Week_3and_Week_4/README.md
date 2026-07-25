# 📦 Sales Demand Intelligence — Forecasting, Anomalies & Product Segmentation

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
