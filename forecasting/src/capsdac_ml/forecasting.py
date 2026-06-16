from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .feature_engineering import FEATURES


def train_random_forest_forecast(feature_df: pd.DataFrame, cutoff: str = "2025-07-01") -> tuple[RandomForestRegressor, dict[str, Any]]:
    """Train a transparent baseline forecasting model using a time-aware split."""
    train_mask = feature_df["MonthDate"] < pd.Timestamp(cutoff)
    X_train = feature_df.loc[train_mask, FEATURES]
    y_train = feature_df.loc[train_mask, "EnrollmentCount"]
    X_test = feature_df.loc[~train_mask, FEATURES]
    y_test = feature_df.loc[~train_mask, "EnrollmentCount"]

    if X_train.empty or X_test.empty:
        raise ValueError("Time split produced empty train or test data. Adjust the cutoff or sample data range.")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=2026,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "model_type": "RandomForestRegressor",
        "cutoff": cutoff,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
    }
    return model, metrics


def generate_site_forecast(model: RandomForestRegressor, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Generate one-step-ahead demo forecasts from the latest feature row for each site."""
    latest = (
        feature_df.sort_values(["PreschoolCDSCode", "MonthDate"])
        .groupby("PreschoolCDSCode", as_index=False)
        .tail(1)
        .copy()
    )
    latest["ForecastMonth"] = (latest["MonthDate"] + pd.DateOffset(months=1)).dt.strftime("%Y-%m")
    latest["PredictedEnrollment"] = model.predict(latest[FEATURES]).round().astype(int)
    cols = [
        "ForecastMonth",
        "VendorNumber",
        "VendorName",
        "PreschoolCDSCode",
        "SiteName",
        "County",
        "FundingType",
        "PredictedEnrollment",
    ]
    return latest[cols].sort_values(["ForecastMonth", "VendorNumber", "PreschoolCDSCode"]).reset_index(drop=True)


def save_model(model: RandomForestRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
