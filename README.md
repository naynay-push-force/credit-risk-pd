# Credit Risk  PD Modelling - Home Credit

--- 

## Objective 
Build an end-to-end Probability of Default (PD) model using real-world consumer lending data, with an focus on correctness, reproducability, and professional ML practices. 

### Dataset
Home Credit Default Risk (Kaggle)

Scope (current):
- Application-level data only (`application_train.csv`)
- Behavioural tables deferred to later phases

### Project Roadmap
- Phase 0: Infrastructure & Environment (complete)
- Phase 1: Data plumbing & preprocessing (complete)
- Phase 2: Baseline modelling & evaluation
- Phase 3: Credit-risk decision logic
- Phase 4: Production maturity & deployment

---

## Key Design Decisions (So Far)

### Environment Isolation
- Project-local virtual environment (`.venv`)
- Python 3.12 pinned for stability
- All dependencies installed explicitly

**Rationale:** Prevents hidden global dependencies and ensures reproducibility.

### Git Strategy
- Git introduced after environment stabilisation
- Commits represent coherent engineering decisions
- Data and environment artifacts excluded via `.gitignore`

**Rationale:** Git history doubles as a design log.

### Data Boundary Definition
- Target (`TARGET`) explicitly separated from features
- All preprocessing operates on features only

**Rationale:** Prevents data leakage and mirrors inference-time constraints.

---

## Phase 1 - Data Plumbing & Preprocessing (Complete)
Implemented a leakage-safe preprocessing workflow with the following properties:

- Explicit X / y separation
- Feature typing (numeric vs categorical) based on dataset inspection
- Stratified train/validation split
- Missing value handling via sklearn pipelines consisting of imputation with median and mode
- ColumnTransformer-based preprocessing for reuse at training and inference

All preprocessing steps are fit on training data only and applied consistently to validation data.

---

## Phase 2 - Feature Engineering & Baseline Modelling 

### What was built
- Leakage-safe baseline logistic regression PD model trained on application-level features
- Feature engineering module with EDA-justified ratios, age/employment transforms, 
  and missingness indicators
- Full evaluation pipeline producing ROC, PR, calibration, gains, and coefficient artifacts
- Automated experiment tracking via results/experiments.csv and versioned reports folders
- Platt scaling applied to correct probability inflation from balanced class weighting

### Results (v1, balanced weights + Platt scaling)
- Validation ROC AUC: 0.7595
- Validation KS: 0.3855
- PR-AUC: 0.2360
- Top decile captures 33% of defaults; top two deciles capture 51%
- Calibration curve sits on diagonal across full operating range

### Key decisions
- class_weight='balanced' retained for score spread and threshold flexibility
- Platt scaling (cv=5) applied post-hoc to restore probability interpretability
- See docs/model_card_v1.md for full experiment record and evaluation narrative

---

## Phase 3 - Threshold Analysis & Decision Layer (Complete)

- Evaluated model across thresholds 0.02–0.27
- Recommended operating range: 0.10–0.12 predicted PD
- At threshold 0.10: 73.5% approval rate, 60.8% default capture, 
  4.3% approved portfolio bad rate vs 8.0% population rate
- Expected loss analysis conducted assuming 100% LGD (worst case)
- Beyond threshold 0.12, cost per approval point nearly doubles — 
  diminishing returns on risk-adjusted portfolio growth

## Current State

V1 complete. Application-level PD model with:
- Leakage-safe preprocessing pipeline
- EDA-justified feature engineering (ratios, transforms, missingness flags)
- Calibrated probability outputs via Platt scaling
- Full evaluation suite (ROC, PR, calibration, gains, coefficients)
- Experiment tracking across feature versions
- Threshold analysis with business interpretation

ROC AUC: 0.761 | KS: 0.390 | PR-AUC: 0.238

## Phase 4 - Planned
- Bureau and behavioural data integration
- WoE/IV feature analysis
- Interaction terms
- Stress testing across subgroups
