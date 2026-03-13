from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import TruncatedSVD
except ImportError:  # pragma: no cover - optional fallback
    TruncatedSVD = None


class ALSFactorization:
    """SVD-backed stand-in for ALS-style latent factor generation."""

    def __init__(self, n_factors: int = 20, iterations: int = 5):
        self.n_factors = n_factors
        self.iterations = iterations
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, sparse_matrix, verbose: bool = False):
        if sparse_matrix.shape[0] == 0 or sparse_matrix.shape[1] == 0:
            self.user_factors = np.zeros((sparse_matrix.shape[0], self.n_factors))
            self.item_factors = np.zeros((sparse_matrix.shape[1], self.n_factors))
            return self.user_factors, self.item_factors

        components = max(1, min(self.n_factors, min(sparse_matrix.shape) - 1))
        if TruncatedSVD is None or components <= 0:
            dense = sparse_matrix.toarray()
            self.user_factors = dense[:, : self.n_factors]
            self.item_factors = dense.T[: self.n_factors].T
            return self.user_factors, self.item_factors

        model = TruncatedSVD(
            n_components=components,
            n_iter=max(self.iterations, 5),
            random_state=42,
        )
        user_factors = model.fit_transform(sparse_matrix)
        item_factors = model.components_.T

        if components < self.n_factors:
            user_factors = np.pad(user_factors, ((0, 0), (0, self.n_factors - components)))
            item_factors = np.pad(item_factors, ((0, 0), (0, self.n_factors - components)))

        self.user_factors = user_factors
        self.item_factors = item_factors
        if verbose:
            print(f"Latent factors generated with {self.n_factors} columns")
        return self.user_factors, self.item_factors

    def generate_features(self) -> pd.DataFrame:
        if self.user_factors is None:
            raise RuntimeError("fit() must be called before generate_features().")
        columns = [f"latent_factor_{index + 1}" for index in range(self.user_factors.shape[1])]
        return pd.DataFrame(self.user_factors, columns=columns)
