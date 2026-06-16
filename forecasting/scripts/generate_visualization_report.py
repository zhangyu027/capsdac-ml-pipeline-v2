from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    metrics_path = ROOT / "outputs/metrics/model_metrics.json"
    validation_path = ROOT / "outputs/reports/data_validation_report.json"
    drift_path = ROOT / "outputs/reports/drift_report.json"
    statewide_path = ROOT / "outputs/forecasts/statewide_forecast.csv"
    vendor_path = ROOT / "outputs/forecasts/vendor_forecast.csv"
    leaderboard_path = ROOT / "outputs/metrics/model_leaderboard.csv"

    metrics = json.loads(metrics_path.read_text())
    validation = json.loads(validation_path.read_text())
    drift = json.loads(drift_path.read_text()) if drift_path.exists() else {"overall_status": "not_generated"}
    statewide = pd.read_csv(statewide_path)
    vendors = pd.read_csv(vendor_path)
    leaderboard = pd.read_csv(leaderboard_path) if leaderboard_path.exists() else pd.DataFrame()

    selected_model = metrics.get("selected_model_name", metrics.get("model_type", "unknown"))
    avg = metrics.get("time_series_cv_avg_metrics", metrics)
    top_rows = ""
    if not leaderboard.empty:
        top_rows = "\n".join(
            f"- {row.model_name}: RMSE {row.rmse:.3f}, MAE {row.mae:.3f}, MAPE {row.mape:.3f}%"
            for row in leaderboard.head(5).itertuples()
        )

    report = f"""# CAPSDAC 2.0 Level 2 Forecast Run Summary

## Data validation

- Input rows: {validation['row_count']}
- Sites: {validation['site_count']}
- Vendors: {validation['vendor_count']}
- Snapshot range: {validation['snapshot_month_min']} to {validation['snapshot_month_max']}
- Missing required columns: {validation['missing_required_columns']}
- Duplicate site-month records: {validation['duplicate_grain_count']}
- Negative enrollment rows: {validation['negative_enrollment_count']}

## Model selection

- Selected model: {selected_model}
- Validation strategy: {metrics.get('cv_strategy', 'time-aware validation')}
- MAE: {avg.get('mae', 0):.3f}
- RMSE: {avg.get('rmse', 0):.3f}
- MAPE: {avg.get('mape', 0):.3f}%
- R²: {avg.get('r2', 0):.3f}

## Leaderboard snapshot

{top_rows or '- Leaderboard not available'}

## Drift monitoring

- Drift method: {drift.get('method', 'not_generated')}
- Overall drift status: {drift.get('overall_status', 'not_generated')}

## Forecast outputs

- Statewide forecast rows: {len(statewide)}
- Vendor forecast rows: {len(vendors)}
- Top vendor forecast: {vendors.iloc[0]['VendorName']} with {int(vendors.iloc[0]['PredictedEnrollment'])} predicted enrollments

## Interview note

This report is generated from synthetic public data. In a production CAPSDAC environment, this same workflow would be connected to official data quality thresholds, MLflow or Vertex AI model registry metadata, automated retraining, monitoring alerts, and stakeholder sign-off.
"""
    output_path = ROOT / "outputs/reports/run_summary.md"
    output_path.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
