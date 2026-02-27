# Model Card — Credit Risk PD Model v1

## Overview
A logistic regression probability of default (PD) model trained on 
application-level data from the Home Credit dataset. This is the 
baseline model establishing the performance floor for subsequent 
feature and calibration iterations.

**Scope:** Application-level features only. Behavioural and bureau 
data deferred to later phases.

---

## Dataset
- Source: Home Credit Default Risk (Kaggle)
- Rows: ~307,000 loan applications
- Default rate: ~8% (class imbalance confirmed)
- Train/validation split: 80/20, stratified on target

---

## Feature Set (current: v3)

| Feature | Type | Justification |
|---|---|---|
| YEARS_BIRTH | Numeric | Monotonic bad-rate decrease across deciles in EDA |
| YEARS_EMPLOYED | Numeric | Monotonic bad-rate pattern; sentinel 365243 handled explicitly |
| DAYS_EMPLOYED_MISSING | Binary flag | Sentinel presence correlates with higher default rate |
| CREDIT_INCOME_RATIO | Numeric | Higher leverage predicts higher default; clearer monotone signal than raw amounts |
| ANNUITY_INCOME_RATIO | Numeric | Higher repayment burden predicts higher default |
| GOODS_CREDIT_RATIO | Numeric | Loan-to-value proxy; higher equity predicts lower default |
| EXT_SOURCE_1_MISSING | Binary flag | Missingness correlates with elevated bad rate in EDA |
| EXT_SOURCE_3_MISSING | Binary flag | Missingness correlates with elevated bad rate in EDA |
| LOG_AMT_INCOME_TOTAL | Numeric | Log transform of right-skewed income; compresses long tail for linear model |
| LOG_AMT_CREDIT | Numeric | Log transform of right-skewed loan amount |
| LOG_AMT_GOODS_PRICE | Numeric | Log transform of asset value; appears in top coefficients independently of GOODS_CREDIT_RATIO |
| LOG_AMT_ANNUITY | Numeric | Log transform of monthly repayment amount; consistent with other amount transforms |

---

## Model Architecture
- Algorithm: Logistic Regression (sklearn)
- Preprocessing: Median imputation + StandardScaler (numeric), 
  Mode imputation + OneHotEncoder (categorical)
- Pipeline: ColumnTransformer -> LogisticRegression
- Solver: lbfgs, max_iter=1000

---

## Experiments

| run_id | version | class_weight | calibration | AUC | PR-AUC | KS | Notes |
|---|---|---|---|---|---|---|---|
| 2026-02-26_14-33-36 | v1 | balanced | none | 0.7506 | 0.2320 | 0.3718 | Initial baseline |
| 2026-02-26_15-33-55 | v1 | balanced | none | 0.7506 | 0.2320 | 0.3718 | Added ks_thresh tracking only |
| 2026-02-26_15-07-33 | v1 | none | none | 0.7509 | 0.2355 | 0.3695 | Removed balanced weights |
| 2026-02-26_16-15-40 | v1 | balanced | platt | 0.7595 | 0.2360 | 0.3855 | Restored balanced, added Platt scaling |
| 2026-02-26_19-26-17 | v2 | balanced | platt | 0.7595 | 0.2359 | 0.3862 | Dropped DAYS_BIRTH, DAYS_EMPLOYED |
| 2026-02-26_20-37-47 | v3 | balanced | platt | 0.7612 | 0.2383 | 0.3896 | Log transforms on amount features |

---

## Evaluation — Best Configuration (run_id: 2026-02-26_20-37-47; v3, balanced + Platt scaling)

### Discrimination
- **ROC AUC: 0.7612** 
  - improvement from 0.7506 baseline through 
  feature refinement. Respectable for application-only data; 
  industry benchmark without bureau data is 0.68–0.75.
- **KS: 0.3896** 
  - maximum separation between defaulter and 
  non-defaulter score distributions at predicted PD ≈ 0.077
- **PR-AUC: 0.2383** 
  - approximately 3x better than a random 
  classifier (random baseline = default rate = 0.08)

### Gains
The top 10% of applicants ranked by predicted PD encapsulates 
approximately 33% of all actual defaults. The top 20% captures 
approximately 51%. This means a lender using this model to review 
the riskiest 20% of applications would identify over half of all 
future defaulters.

### Calibration
With class_weight='balanced', the model systematically overpredicts 
default probability — a borrower scored at 0.40 actually defaults at 
roughly 6–7%. This is a known consequence of balanced weighting 
inflating the minority class scores. Discrimination (ranking) is 
unaffected.

Platt scaling (sigmoid method, cv=5) is then applied to correct this 
apparent probability inflation. The calibration curve sits on the 
diagonal across the full, empirical range (0–0.25 predicted PD), 
reassuring us that the model produces meaningful PD estimates that
are appropriate for expected loss calculations

### Score Distribution
With balanced weights, predicted PDs spread across 0.0–1.0, 
providing a wide, working range for threshold selection, yielding a 
ks threshold of 0.5013. With Platt scaling, scores compress 
into the range 0.0–0.25, reflecting the true 8% base rate but making 
threshold selection require more precision. 
However, the ranking is preserved and probabilities are now
interpretable as literal PD estimates.

---

## Feature Refinement History

**v1 → v2: Redundancy removal**
Dropped DAYS_BIRTH and DAYS_EMPLOYED after confirming perfect 
correlation with engineered YEARS_BIRTH and YEARS_EMPLOYED. 
No meaningful performance change (AUC stable), confirming the 
engineered features fully captured the available signal. Retaining 
both would have caused coefficient instability without benefit.

**v2 → v3: Log transforms on amount features**
Applied log1p transforms onto AMT_INCOME_TOTAL, AMT_CREDIT, 
AMT_GOODS_PRICE, and AMT_ANNUITY. These features are heavily 
right-skewed — a small number of very high-value loans compress 
the distribution and limit the model's ability to find linear 
relationships. Financial magnitudes tend to scale multiplicatively 
rather than additively: the difference between a 10k and 20k loan 
is more comparable to the difference between a 100k and 200k loan 
than to the difference between a 100k and 110k loan. 
Log transformation reflects this scaling behaviour.

LOG_AMT_GOODS_PRICE appeared among the top 40 coefficients 
independently of GOODS_CREDIT_RATIO, confirming that absolute 
asset value bears predictive signal beyond the loan-to-value 
ratio. AUC improved from 0.7595 to 0.7612, KS from 0.3855 to 
0.3896 — modest but consistent across both metrics.

---

## Known Limitations
- Application-level features only; bureau and behavioural data would 
  materially improve discrimination
- No held-out test set; all reported metrics are on validation data
- Calibration fitted via cv=5 on training data; a dedicated 
  calibration set would be more rigorous in production
- No stress testing across demographic or income subgroups

---

## Next Steps
- Threshold analysis: approval rate vs default capture rate vs 
  expected loss simulation
- Interaction terms as a candidate for v4