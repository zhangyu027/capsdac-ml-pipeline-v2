# CAPSDAC 2.0 Level 2 ML-Enabled Enrollment Forecasting Platform

This repository is an **interview-facing Level 2 package** for a CAPSDAC-style preschool enrollment forecasting system. It builds on the Level 1 Senior Data Engineering package and adds enough ML depth to discuss forecasting design, model comparison, time-series validation, drift monitoring, retraining, and serving.

This public version uses **synthetic, de-identified sample data only**. It is not an official CAPSDAC production system and does not contain confidential child-level records.

## Interview positioning

**Best-fit roles**

- Senior Data Engineer moving into ML Platform Engineering
- ML Platform Engineer
- Applied Machine Learning Engineer with data-platform ownership
- Senior Analytics Engineer for forecasting/data products

**One-line story**

> I extended a governed CAPSDAC-style data platform into a Level 2 ML-enabled forecasting system that validates monthly snapshots, builds time-series features, compares multiple forecasting models, selects the best model through expanding-window validation, monitors feature drift, publishes forecasts, and exposes the outputs through an API layer.

## What changed from Level 1

Level 1 proved the data engineering foundation. Level 2 makes the ML workflow defensible in interviews.

| Area | Level 1 | Level 2 improvement |
|---|---|---|
| Pipeline | Runnable ETL and validation | Same foundation, now drives model selection |
| Features | Lag/rolling/seasonality features | Used consistently across model comparison and monitoring |
| Model | Single baseline model | Multiple candidate models with hyperparameter grids |
| Validation | Time-aware split | Expanding-window time-series cross-validation |
| Metrics | Basic MAE/RMSE/R2 | MAE, RMSE, MAPE, R2, leaderboard, fold-level results |
| Monitoring | Starter monitoring files | PSI-based drift report for features and target |
| Interview story | Senior DE | Senior DE + ML Platform / Level 2 MLE readiness |

## ML modules included

- `forecasting/src/capsdac_ml/feature_engineering.py`  
  Builds site-month lag, rolling-window, seasonality, and trend features.

- `forecasting/src/capsdac_ml/model_selection.py`  
  Runs expanding-window validation across candidate models:
  - seasonal naive / median baseline
  - ridge regression
  - random forest
  - gradient boosting
  - histogram gradient boosting

- `forecasting/src/capsdac_ml/monitoring.py`  
  Produces PSI-based drift diagnostics for feature and target distributions.

- `forecasting/scripts/run_capsdac_pipeline.py`  
  End-to-end runnable pipeline: validate data, build features, compare models, persist best model, generate forecasts, and write monitoring reports.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make validate
```

Equivalent commands without Make:

```bash
export PYTHONPATH=$PWD/forecasting
pytest
python forecasting/scripts/run_capsdac_pipeline.py
python forecasting/scripts/generate_visualization_report.py
```

## Expected outputs

```text
data/processed/monthly_enrollment_features.csv
outputs/metrics/model_metrics.json
outputs/metrics/model_leaderboard.csv
outputs/metrics/time_series_cv_results.json
outputs/reports/data_validation_report.json
outputs/reports/drift_report.json
outputs/forecasts/site_forecast.csv
outputs/forecasts/vendor_forecast.csv
outputs/forecasts/statewide_forecast.csv
models/capsdac_level2_best_forecast_model.joblib
```

## How to answer ML interview questions

### Why not only Random Forest?

Random Forest is a strong tabular baseline, but forecasting should be validated against simpler and boosting-based models. This package compares a median baseline, ridge regression, random forest, gradient boosting, and histogram gradient boosting. The final model is selected by expanding-window cross-validation rather than by preference.

### Why XGBoost vs LightGBM?

This lightweight public package uses scikit-learn boosting so the repo is easy to run without heavy dependencies. In production, I would add XGBoost and LightGBM as optional candidates and choose based on time-series validation metrics, training speed, operational constraints, and interpretability. I would not choose either model by popularity alone.

### How did you tune hyperparameters?

The package uses small, controlled parameter grids so the public repo remains fast and reproducible. In a production environment, I would expand this to MLflow-tracked randomized search or Bayesian optimization with strict time-based validation folds.

### How did you avoid data leakage?

Lag and rolling features are shifted before aggregation, so the current month target is not used to predict itself. Model validation uses expanding-window splits where all training months occur before the validation month.

### How did you evaluate concept drift?

The package creates a PSI-based drift report comparing baseline months with more recent months. Drift is flagged as stable, moderate, or high. In production, drift alerts would trigger investigation, feature review, retraining, or stakeholder validation.

### What is the business impact?

The model supports aggregate planning: enrollment demand, vendor/site forecast review, staffing capacity discussions, and early warning for unusual enrollment changes. The model should not be used for child-level eligibility decisions or official certified counts.

## API demo

```bash
uvicorn serving.api.main:app --reload
```

Then open:

```text
/health
/forecasts/statewide
/forecasts/vendors/top
```

## Responsible use

This repository is for portfolio demonstration and aggregate planning only. Production use would require approved data access, formal governance review, CI/CD secrets, official datasets, documented model monitoring thresholds, stakeholder sign-off, and privacy controls.
