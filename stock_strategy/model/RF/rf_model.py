"""Random Forest model implementing the BaseModel interface."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from model.base import BaseModel


class RFModel(BaseModel):
    """Random Forest classifier for stock ranking.

    Wraps sklearn's RandomForestClassifier behind the BaseModel interface.
    predict_proba() returns the probability of label=1, which is used as
    a ranking score — higher probability = stronger buy signal.
    """

    def __init__(self, config: dict):
        """Args:
            config: dict with keys:
              rf_params (dict): RandomForestClassifier kwargs
              forward_days (int): prediction horizon
              return_threshold (float): label threshold
        """
        self.config = config
        self.clf: RandomForestClassifier | None = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train RF on feature matrix and binary labels."""
        rf_params = self.config.get("rf_params", {})
        self.clf = RandomForestClassifier(**rf_params)
        self.clf.fit(X.values, y.values)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return probability of positive class (label=1) as ranking score."""
        if self.clf is None:
            raise RuntimeError("Model not trained. Call train() first.")
        probs = self.clf.predict_proba(X.values)[:, 1]
        return pd.Series(probs, index=X.index, name="y_prob", dtype="float32")

    def save(self, path: Path) -> None:
        """Save trained RF to disk via joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path)

    def load(self, path: Path) -> None:
        """Load trained RF from disk."""
        self.clf = joblib.load(path)

    @property
    def name(self) -> str:
        return "RandomForest"

    def get_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """Compute permutation importance on a sample of the test set."""
        if self.clf is None:
            raise RuntimeError("Model not trained.")
        n = min(50_000, len(X))
        idx = np.random.RandomState(42).choice(len(X), n, replace=False)
        X_sub = X.iloc[idx]
        y_sub = y.iloc[idx]
        result = permutation_importance(
            self.clf, X_sub.values, y_sub.values,
            n_repeats=5, random_state=42, n_jobs=-1,
        )
        return pd.DataFrame({
            "feature": feature_names,
            "importance": result.importances_mean,
            "std": result.importances_std,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
