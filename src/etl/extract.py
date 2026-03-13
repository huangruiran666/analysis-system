from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

try:
    from sklearn.ensemble import IsolationForest
except ImportError:  # pragma: no cover - optional fallback
    IsolationForest = None


class DataExtractor:
    """Minimal ETL helper for the demo pipeline."""

    def handle_missing_values(self, df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
        cleaned = df.copy()
        numeric_cols = cleaned.select_dtypes(include=["number"]).columns

        if method == "forward_fill":
            cleaned = cleaned.ffill().bfill()
        elif method == "median":
            cleaned[numeric_cols] = cleaned[numeric_cols].fillna(cleaned[numeric_cols].median())
        else:
            cleaned[numeric_cols] = cleaned[numeric_cols].fillna(cleaned[numeric_cols].mean())

        for column in cleaned.columns.difference(numeric_cols):
            if cleaned[column].isna().any():
                cleaned[column] = cleaned[column].fillna("unknown")

        return cleaned

    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: str = "isolation_forest",
        columns: list[str] | None = None,
        contamination: float = 0.02,
    ) -> tuple[pd.DataFrame, pd.Series]:
        numeric_cols = list(columns or df.select_dtypes(include=["number"]).columns)
        if not numeric_cols:
            mask = pd.Series(False, index=df.index)
            return df.copy(), mask

        numeric_frame = df[numeric_cols].fillna(df[numeric_cols].median())

        if method == "isolation_forest" and IsolationForest is not None and len(df) > 10:
            detector = IsolationForest(contamination=contamination, random_state=42)
            mask = pd.Series(detector.fit_predict(numeric_frame) == -1, index=df.index)
        else:
            std = numeric_frame.std(ddof=0).replace(0, 1)
            z_scores = ((numeric_frame - numeric_frame.mean()) / std).abs()
            mask = z_scores.gt(3).any(axis=1)

        filtered = df.loc[~mask].reset_index(drop=True)
        return filtered, mask

    def create_sparse_matrix(
        self,
        df: pd.DataFrame,
        user_col: str,
        item_col: str,
        value_col: str,
    ) -> tuple[csr_matrix, dict]:
        user_codes = pd.Categorical(df[user_col])
        item_codes = pd.Categorical(df[item_col])
        values = df[value_col].fillna(0).astype(float).to_numpy()

        matrix = csr_matrix(
            (values, (user_codes.codes, item_codes.codes)),
            shape=(len(user_codes.categories), len(item_codes.categories)),
        )

        total = matrix.shape[0] * matrix.shape[1] or 1
        metadata = {
            "n_users": matrix.shape[0],
            "n_items": matrix.shape[1],
            "sparsity": 1 - (matrix.nnz / total),
            "memory_mb": (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / (1024**2),
            "user_index": list(user_codes.categories),
            "item_index": list(item_codes.categories),
        }
        return matrix, metadata
