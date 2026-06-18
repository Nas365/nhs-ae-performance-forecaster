"""
Feature builder for the production XGBoost model.

Reproduces the exact features used during training so the app's inference
path is identical to what the model was fitted on:

    lag_1, lag_2, lag_3   — recent monthly performance (1–3 months back)
    roll_3                 — 3-month rolling mean of those lags
    trend_1                — change from the month before last to last month
    lag_12                 — same calendar month last year
    calendar_month         — month number of the forecast target (1–12)

All features are computed relative to the latest available month M; the
forecast target is M+1.  lag_12 uses the hospital's history where available
and is left as NaN otherwise — XGBoost routes NaN values to a learned default
branch, so a missing lag_12 degrades accuracy slightly but does not break the
prediction.
"""
import numpy as np
import pandas as pd

FEATURE_COLS = ["lag_1", "lag_2", "lag_3", "roll_3", "trend_1", "lag_12", "calendar_month"]


def build_forecast_features(hospital_df: pd.DataFrame) -> dict | None:
    """
    Build the feature vector for the next-month forecast.

    Parameters
    ----------
    hospital_df : DataFrame with at least columns [month, within_4hrs],
                  covering one hospital only.  Must have >= 3 rows.

    Returns
    -------
    dict with keys: lag_1 … calendar_month (model inputs), plus
        forecast_month  — the pd.Timestamp being predicted
        latest_month    — the most recent month in hospital_df
    Returns None if hospital_df has fewer than 3 rows.
    """
    df = hospital_df[["month", "within_4hrs"]].copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

    if len(df) < 3:
        return None

    perf = df["within_4hrs"].to_numpy(dtype=float)
    months = df["month"].tolist()

    latest_month = months[-1]
    forecast_month = latest_month + pd.DateOffset(months=1)

    lag_1 = perf[-1]
    lag_2 = perf[-2]
    lag_3 = perf[-3]
    roll_3 = float(np.mean(perf[-3:]))
    trend_1 = lag_1 - lag_2

    # lag_12: performance at forecast_month - 12 months = latest_month - 11 months
    target_month = pd.Timestamp(forecast_month) - pd.DateOffset(months=12)
    match = df[df["month"] == target_month]
    lag_12 = float(match["within_4hrs"].iloc[0]) if len(match) == 1 else float("nan")

    return {
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "roll_3": roll_3,
        "trend_1": trend_1,
        "lag_12": lag_12,
        "calendar_month": forecast_month.month,
        "forecast_month": pd.Timestamp(forecast_month),
        "latest_month": pd.Timestamp(latest_month),
    }
