from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd

from src.capsdac_ml.contribution_analysis import statewide_forecast, vendor_contribution
from src.capsdac_ml.data_contracts import validate_enrollment_snapshot
from src.capsdac_ml.feature_engineering import build_monthly_features
from src.capsdac_ml.forecasting import generate_site_forecast
from src.capsdac_ml.model_selection import run_model_selection
from src.capsdac_ml.monitoring import drift_report

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    for folder in [
        ROOT / "data/processed",
        ROOT / "outputs/metrics",
        ROOT / "outputs/reports",
        ROOT / "outputs/forecasts",
        ROOT / "models",
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    raw_path = ROOT / "data/raw/capsdac_child_enrollment_sample.csv"
    raw = pd.read_csv(raw_path, dtype={"VendorNumber": str, "PreschoolCDSCode": str})

    validation_report = validate_enrollment_snapshot(raw)
    write_json(ROOT / "outputs/reports/data_validation_report.json", validation_report)

    features = build_monthly_features(raw)
    features.to_csv(ROOT / "data/processed/monthly_enrollment_features.csv", index=False)

    best_model, model_selection_report, leaderboard = run_model_selection(features)
    model_path = ROOT / "models/capsdac_level2_best_forecast_model.joblib"
    joblib.dump(best_model, model_path)

    leaderboard.to_csv(ROOT / "outputs/metrics/model_leaderboard.csv", index=False)
    selected = model_selection_report["selected_model"]
    write_json(
        ROOT / "outputs/metrics/model_metrics.json",
        {
            "package_level": "Level 2 ML-enabled data platform",
            "selected_model_name": selected["model_name"],
            "selected_params": selected["params"],
            "time_series_cv_avg_metrics": selected["avg_metrics"],
            "cv_strategy": "expanding-window monthly validation",
            "train_target": "EnrollmentCount",
            "model_artifact": str(model_path.relative_to(ROOT)),
        },
    )
    write_json(ROOT / "outputs/metrics/time_series_cv_results.json", model_selection_report)
    write_json(ROOT / "outputs/reports/drift_report.json", drift_report(features))

    site = generate_site_forecast(best_model, features)
    site.to_csv(ROOT / "outputs/forecasts/site_forecast.csv", index=False)
    vendor_contribution(site).to_csv(ROOT / "outputs/forecasts/vendor_forecast.csv", index=False)
    statewide_forecast(site).to_csv(ROOT / "outputs/forecasts/statewide_forecast.csv", index=False)

    run_summary = {
        "validation": validation_report,
        "selected_model": selected["model_name"],
        "selected_params": selected["params"],
        "metrics": selected["avg_metrics"],
        "outputs": [
            "outputs/metrics/model_leaderboard.csv",
            "outputs/metrics/time_series_cv_results.json",
            "outputs/reports/drift_report.json",
            "outputs/forecasts/site_forecast.csv",
        ],
    }
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
