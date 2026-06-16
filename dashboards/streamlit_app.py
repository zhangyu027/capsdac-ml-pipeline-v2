import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="CAPSDAC Forecast Platform", layout="wide")
st.title("CAPSDAC Enrollment Forecast Platform")
st.caption("Privacy-conscious portfolio dashboard using de-identified sample data")

metrics = json.loads((ROOT / "outputs/metrics/model_metrics.json").read_text())
c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{metrics['mae']:.2f}")
c2.metric("RMSE", f"{metrics['rmse']:.2f}")
c3.metric("R2", f"{metrics['r2']:.2f}")

st.subheader("Statewide Forecast")
st.dataframe(pd.read_csv(ROOT / "outputs/forecasts/statewide_forecast.csv"))
st.image(str(ROOT / "outputs/figures/statewide_forecast_trend.png"))

st.subheader("Top Vendor Forecast")
st.image(str(ROOT / "outputs/figures/top_vendor_forecast.png"))

st.subheader("Top Site Heatmap")
st.image(str(ROOT / "outputs/figures/top_site_heatmap.png"))
