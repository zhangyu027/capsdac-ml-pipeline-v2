from __future__ import annotations

from fastapi import FastAPI
from pathlib import Path
from typing import Any
import json
import pandas as pd

app = FastAPI(title="CAPSDAC Level 2 Forecast Serving API", version="2.2")
ROOT = Path(__file__).resolve().parents[2]

DEMO_STATEWIDE = [
    {"ForecastMonth": "2026-01", "PredictedEnrollment": 612, "Scope": "synthetic_statewide_demo"},
]


def read_csv_or_demo(path: Path, demo: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if path.exists():
        return pd.read_csv(path).to_dict(orient="records")
    return demo


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "capsdac-level2-forecast-api", "version": "2.2"}


@app.get("/forecasts/statewide")
def statewide_forecast() -> list[dict[str, Any]]:
    return read_csv_or_demo(ROOT / "outputs" / "forecasts" / "statewide_forecast.csv", DEMO_STATEWIDE)


@app.get("/forecasts/vendors/top")
def top_vendor_forecast(limit: int = 10) -> list[dict[str, Any]]:
    path = ROOT / "outputs" / "forecasts" / "vendor_forecast.csv"
    if path.exists():
        df = pd.read_csv(path)
        value_col = "PredictedEnrollment" if "PredictedEnrollment" in df.columns else df.columns[-1]
        return df.sort_values(value_col, ascending=False).head(limit).to_dict(orient="records")
    return []


@app.get("/metrics/model")
def model_metrics() -> dict[str, Any]:
    path = ROOT / "outputs" / "metrics" / "model_metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"status": "not_generated", "hint": "Run `make run` first."}


@app.get("/metrics/leaderboard")
def model_leaderboard(limit: int = 10) -> list[dict[str, Any]]:
    path = ROOT / "outputs" / "metrics" / "model_leaderboard.csv"
    if path.exists():
        return pd.read_csv(path).head(limit).to_dict(orient="records")
    return []


@app.get("/monitoring/drift")
def monitoring_drift() -> dict[str, Any]:
    path = ROOT / "outputs" / "reports" / "drift_report.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"status": "not_generated", "hint": "Run `make run` first."}
