from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "lag_1",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_3",
    "rolling_6",
    "month_sin",
    "month_cos",
    "trend_index",
]


def build_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lag, rolling, seasonality, and trend features at site-month grain."""
    out = df.copy()
    out["MonthDate"] = pd.to_datetime(out["SnapshotMonth"] + "-01")
    out = out.sort_values(["PreschoolCDSCode", "MonthDate"]).reset_index(drop=True)

    site_group = out.groupby("PreschoolCDSCode", group_keys=False)
    for lag in [1, 3, 6, 12]:
        out[f"lag_{lag}"] = site_group["EnrollmentCount"].shift(lag)

    for window in [3, 6]:
        out[f"rolling_{window}"] = site_group["EnrollmentCount"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=window).mean()
        )

    out["month"] = out["MonthDate"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["trend_index"] = (out["MonthDate"].dt.year - out["MonthDate"].dt.year.min()) * 12 + out["month"]

    return out.dropna(subset=FEATURES).reset_index(drop=True)
