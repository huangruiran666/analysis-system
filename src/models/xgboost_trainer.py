from __future__ import annotations

import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional fallback
    xgb = None


class XGBoostTrainer:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        cv_strategy: str = "kfold",
        n_splits: int = 3,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.model = None

    def _build_model(self):
        if xgb is not None:
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                objective="reg:squarederror",
                random_state=42,
            )
        return HistGradientBoostingRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            max_iter=self.n_estimators,
            random_state=42,
        )

    def train(self, X, y, early_stopping_rounds: int | None = None, verbose: bool = False):
        self.model = self._build_model()
        self.model.fit(np.asarray(X), np.asarray(y))
        return self.model, self.evaluate(X, y)

    def evaluate(self, X, y) -> dict:
        predictions = self.model.predict(np.asarray(X))
        y_true = np.asarray(y)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
            "mae": float(mean_absolute_error(y_true, predictions)),
            "r2": float(r2_score(y_true, predictions)),
        }

    def save_model(self, path: str):
        with open(path, "wb") as handle:
            pickle.dump(self.model, handle)
