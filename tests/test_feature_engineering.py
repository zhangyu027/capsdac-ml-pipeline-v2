import pandas as pd
from src.capsdac_ml.feature_engineering import build_monthly_features, FEATURES

def test_build_monthly_features_has_lags():
    months = pd.date_range("2020-01-01", periods=14, freq="MS")
    df = pd.DataFrame({
        "SnapshotMonth": [m.strftime("%Y-%m") for m in months],
        "VendorNumber": [1]*14,
        "VendorName": ["Vendor"]*14,
        "PreschoolCDSCode": ["A"]*14,
        "SiteName": ["Site"]*14,
        "County": ["California"]*14,
        "EnrollmentCount": list(range(14)),
        "FundingType": ["CSPP"]*14,
        "DataSource": ["test"]*14,
    })
    out = build_monthly_features(df)
    assert all(c in out.columns for c in FEATURES)
    assert len(out) == 2
