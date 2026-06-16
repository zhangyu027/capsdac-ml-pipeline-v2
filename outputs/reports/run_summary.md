# CAPSDAC 2.0 Level 2 Forecast Run Summary

## Data validation

- Input rows: 216
- Sites: 6
- Vendors: 4
- Snapshot range: 2023-01 to 2025-12
- Missing required columns: []
- Duplicate site-month records: 0
- Negative enrollment rows: 0

## Model selection

- Selected model: ridge_regression
- Validation strategy: expanding-window monthly validation
- MAE: 0.402
- RMSE: 0.452
- MAPE: 0.443%
- R²: 0.999

## Leaderboard snapshot

- ridge_regression: RMSE 0.452, MAE 0.402, MAPE 0.443%
- ridge_regression: RMSE 0.618, MAE 0.563, MAPE 0.622%
- gradient_boosting: RMSE 1.086, MAE 0.792, MAPE 0.839%
- gradient_boosting: RMSE 1.109, MAE 0.797, MAPE 0.835%
- gradient_boosting: RMSE 1.176, MAE 0.843, MAPE 0.868%

## Drift monitoring

- Drift method: population_stability_index
- Overall drift status: high

## Forecast outputs

- Statewide forecast rows: 1
- Vendor forecast rows: 4
- Top vendor forecast: BrightStart Education with 207 predicted enrollments

## Interview note

This report is generated from synthetic public data. In a production CAPSDAC environment, this same workflow would be connected to official data quality thresholds, MLflow or Vertex AI model registry metadata, automated retraining, monitoring alerts, and stakeholder sign-off.
