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

## Feature Set (v1)
All features are row-level and computed prior to train/validation 
split (leakage-safe).

| Feature | Type | Justification |
|---|---|---|
| YEARS_BIRTH | Numeric | Monotonic bad-rate decrease across deciles in EDA |
| YEARS_EMPLOYED | Numeric | Monotonic bad-rate pattern; sentinel 365243 handled explicitly |
| DAYS_EMPLOYED_MISSING | Binary flag | Sentinel presence correlates with higher default rate |
| CREDIT_INCOME_RATIO | Numeric | Higher leverage predicts higher default; clearer monotone signal than raw amounts |
| ANNUITY_INCOME_RATIO | Numeric | Higher repayment burden predicts higher default |
| GOODS_CREDIT_RATIO | Numeric | Representative of equity; higher equity predicts lower default |
| EXT_SOURCE_1_MISSING | Binary flag | Missingness correlates with elevated bad rate in EDA |
| EXT_SOURCE_3_MISSING | Binary flag | Missingness correlates with elevated bad rate in EDA |

---

## Model Architecture
- Algorithm: Logistic Regression (sklearn)
- Preprocessing: Median imputation + StandardScaler (numeric), 
  Mode imputation + OneHotEncoder (categorical)
- Pipeline: ColumnTransformer -> LogisticRegression
- Solver: lbfgs, max_iter=1000

---

## Experiments

| run_id | class_weight | calibration | AUC | PR-AUC | KS | KS_THREH | Notes |
|---|---|---|---|---|---|---|
| 2026-02-26_13-33-55 | balanced | none | 0.7506 | 0.2320 | 0.370 | 0.5013 | Initial baseline |
| 2025-02-25_15-10-45 | none | none | 0.7509 | 0.2355 | 0.3695 | 0.0827 | Removed balanced weights |
| 2026-02-26_16-15-40 | balanced | platt | 0.7560 | 0.2360 | 0.3855 | 0.0805 | Restored balanced, added Platt scaling |

---

## Evaluation — Best Configuration (run_id: TBD)

### Discrimination
- **ROC AUC: 0.75** 
    — a random classifer would do 0.50, assigning random scores to everyone
    — respectable for application-only data. 
    Industry benchmark for application-only models is 0.68–0.75 
    without bureau data.
- **KS: 0.37** 
    — the model achieves maximum separation between 
    defaulter and non-defaulter score distributions at threshold ~0.5013
- **PR-AUC: 0.23** 
    — a random classifier would do 0.08, accusing some proportion of the 
    population of being defaulters for some random threshold. 
    Among which, on average, 0.08 would actually be defaulters.
    — approximately 3x better than a random 
    classifier (random baseline = default rate = 0.08).

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

## Known Limitations
- Application-level features only; bureau and behavioural data would 
  materially improve discrimination
- No held-out test set; all reported metrics are on validation data
- Calibration fitted via cv=5 on training data; a dedicated 
  calibration set would be more rigorous in production
- No stress testing across demographic or income subgroups

---

## Next Version (v2)
- Investigate log transforms for skewed amount columns
- VIF analysis to identify and remove multicollinear features
- Assess interaction terms (CREDIT_INCOME × YEARS_EMPLOYED)