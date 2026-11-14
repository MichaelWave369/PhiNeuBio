"""Small deterministic logistic readout for binary reservoir classification."""

from dataclasses import dataclass, field
import math
from typing import Sequence


@dataclass
class LogisticReadout:
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0

    def fit(
        self,
        features: Sequence[Sequence[float]],
        targets: Sequence[int],
        *,
        epochs: int = 20,
        learning_rate: float = 0.035,
    ) -> None:
        if not features or len(features) != len(targets):
            raise ValueError("nonempty features and targets must have equal lengths")
        width = len(features[0])
        if width == 0 or any(len(row) != width for row in features):
            raise ValueError("all feature vectors must have the same positive width")
        if any(target not in (0, 1) for target in targets):
            raise ValueError("targets must be binary")
        if epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")

        self.weights = [0.0] * width
        self.bias = 0.0
        for _ in range(epochs):
            for row, target in zip(features, targets):
                error = target - self.predict_proba(row)
                self.bias += learning_rate * error
                for index, value in enumerate(row):
                    self.weights[index] += learning_rate * error * value

    def predict_proba(self, features: Sequence[float]) -> float:
        if not self.weights or len(features) != len(self.weights):
            raise ValueError("readout is unfitted or feature width is incorrect")
        logit = self.bias + sum(w * x for w, x in zip(self.weights, features))
        logit = max(-40.0, min(40.0, logit))
        return 1.0 / (1.0 + math.exp(-logit))

    def predict(self, features: Sequence[float]) -> int:
        return int(self.predict_proba(features) >= 0.5)

    def score(self, features: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
        if not targets or len(features) != len(targets):
            raise ValueError("nonempty features and targets must have equal lengths")
        return sum(self.predict(row) == target for row, target in zip(features, targets)) / len(targets)
