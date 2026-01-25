# Credit Risk  PD Modelling - Home Credit

## Objective 
Build an end-to-end Probability of Default (PD) model using real-world consumer lending data, with an focus on correctness, reproducability, and professional ML practices. 

## Dataset
Home Credit Default Risk (Kaggle)

Scope (current):
- Application-level data only (`application_train.csv`)
- Behavioural tables deferred to later phases

## Project Roadmap
- Phase 0: Infrastructure & Environment (complete)
- Phase 1: Data plumbing & preprocessing (current)
- Phase 2: Baseline modelling & evaluation
- Phase 3: Credit-risk decision logic
- Phase 4: Production maturity & deployment

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

## Current Status
- X/y separation implemented and validated
- Default rate ≈ 8%, confirming class imbalance
- Next step: feature typing (numeric vs categorical)