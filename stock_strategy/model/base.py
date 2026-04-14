"""Abstract base class for all stock prediction models.

Any model under model/ must implement this interface.
The test/ module interacts only through this API — it never imports
model-specific code, enabling plug-and-play model swapping.
"""
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseModel(ABC):

    @abstractmethod
    def __init__(self, config: dict):
        """Initialize model with configuration dict.

        The config dict must include at minimum:
          forward_days (int): prediction horizon in trading days
          return_threshold (float): label threshold (e.g. 0.02 for >2%)
        """

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on labeled data.

        Args:
            X: Feature DataFrame (rows = samples, cols = features)
            y: Binary labels Series (0 or 1)
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return predicted probability/score for the positive class.

        Higher score = stronger buy signal. Used for ranking.

        Args:
            X: Feature DataFrame (same schema as train)

        Returns:
            Series of float scores in [0, 1], same index as X.
        """

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """Binary prediction derived from predict_proba. Default impl provided."""
        return (self.predict_proba(X) >= threshold).astype("int8")

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist trained model to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load trained model from disk."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name (e.g. 'RandomForest', 'XGBoost')."""

    @abstractmethod
    def get_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """Return feature importance as DataFrame with columns:
        ['feature', 'importance', 'std'].

        The implementation is model-specific (RF = permutation importance,
        XGBoost = gain, etc.).
        """
