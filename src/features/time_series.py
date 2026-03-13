from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.seasonal import STL
except ImportError:  # pragma: no cover - optional fallback
    STL = None


class TimeSeriesDecomposer:
    def __init__(self, freq: int = 7):
        self.freq = freq

    def decompose_stl(self, series: pd.Series) -> dict:
        ts = pd.Series(series).astype(float).sort_index()

        if STL is not None and len(ts) >= max(self.freq * 2, 8):
            result = STL(ts, period=self.freq, robust=True).fit()
            trend = result.trend
            seasonal = result.seasonal
            resid = result.resid
        else:
            trend = ts.rolling(window=min(self.freq, len(ts)), min_periods=1).mean()
            seasonal = ts - trend
            resid = ts - trend - seasonal

        strength_denom = seasonal.abs().mean() + resid.abs().mean() or 1.0
        return {
            "trend": trend,
            "seasonal": seasonal,
            "resid": resid,
            "seasonality_strength": float(seasonal.abs().mean() / strength_denom),
        }

    def compute_rolling_features(self, series: pd.Series) -> pd.DataFrame:
        ts = pd.Series(series).astype(float).sort_index()
        features = pd.DataFrame(index=ts.index)
        features["value"] = ts
        for window in (7, 30, 90):
            actual_window = min(window, len(ts))
            features[f"rolling_mean_{window}"] = ts.rolling(actual_window, min_periods=1).mean()
            features[f"rolling_std_{window}"] = ts.rolling(actual_window, min_periods=1).std().fillna(0)
        features["pct_change_1"] = ts.pct_change().fillna(0)
        return features
