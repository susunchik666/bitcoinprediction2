from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def effort_to_compress(sequence: np.ndarray) -> int:
    seq = list(map(int, sequence))
    if len(seq) <= 1 or len(set(seq)) <= 1:
        return 0

    effort = 0
    next_symbol = max(seq) + 1

    while len(seq) > 1 and len(set(seq)) > 1:
        counts: dict[tuple[int, int], int] = {}
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            counts[pair] = counts.get(pair, 0) + 1

        max_count = max(counts.values())
        best_pair = None
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            if counts[pair] == max_count:
                best_pair = pair
                break

        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                new_seq.append(next_symbol)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1

        seq = new_seq
        next_symbol += 1
        effort += 1

    return effort


@dataclass
class PDTNode:
    prediction: int
    probability: float
    depth: int
    feature_index: int | None = None
    threshold: float | None = None
    left: Any | None = None
    right: Any | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class PermutationDecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int = 40,
        n_thresholds: int = 10,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds = n_thresholds
        self.random_state = random_state
        self.root_: PDTNode | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X, y, feature_names: list[str] | None = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.feature_names_ = feature_names or [f"x{i}" for i in range(X.shape[1])]
        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def _node_prediction(self, y: np.ndarray) -> tuple[int, float]:
        proba = float(np.mean(y)) if len(y) else 0.5
        return int(proba >= 0.5), proba

    def _candidate_thresholds(self, values: np.ndarray) -> np.ndarray:
        values = values[np.isfinite(values)]
        if len(np.unique(values)) <= 1:
            return np.array([])
        quantiles = np.linspace(0.1, 0.9, self.n_thresholds)
        return np.unique(np.quantile(values, quantiles))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None, float]:
        parent_etc = effort_to_compress(y)
        if parent_etc == 0:
            return None, None, 0.0

        best_feature = None
        best_threshold = None
        best_gain = 0.0
        n = len(y)

        for feature_idx in range(X.shape[1]):
            values = X[:, feature_idx]
            for threshold in self._candidate_thresholds(values):
                left_mask = values <= threshold
                right_mask = ~left_mask
                left_count = int(left_mask.sum())
                right_count = int(right_mask.sum())

                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue

                left_etc = effort_to_compress(y[left_mask])
                right_etc = effort_to_compress(y[right_mask])
                gain = parent_etc - (left_count / n) * left_etc - (right_count / n) * right_etc

                if gain > best_gain:
                    best_feature = feature_idx
                    best_threshold = float(threshold)
                    best_gain = float(gain)

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> PDTNode:
        prediction, probability = self._node_prediction(y)
        node = PDTNode(prediction=prediction, probability=probability, depth=depth)

        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or len(np.unique(y)) == 1:
            return node

        feature_idx, threshold, gain = self._best_split(X, y)
        if feature_idx is None or threshold is None or gain <= 0:
            return node

        mask = X[:, feature_idx] <= threshold
        node.feature_index = feature_idx
        node.threshold = threshold
        node.left = self._build_tree(X[mask], y[mask], depth + 1)
        node.right = self._build_tree(X[~mask], y[~mask], depth + 1)
        return node

    def _predict_one_proba(self, row: np.ndarray) -> float:
        if self.root_ is None:
            raise RuntimeError("Model is not fitted")
        node = self.root_
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.probability

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        p1 = np.array([self._predict_one_proba(row) for row in X])
        return np.column_stack([1 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def rules(self, max_rules: int = 12) -> list[str]:
        if self.root_ is None:
            return []
        names = self.feature_names_ or []
        result: list[str] = []

        def walk(node: PDTNode, prefix: list[str]):
            if len(result) >= max_rules:
                return
            if node.is_leaf:
                direction = "UP" if node.prediction == 1 else "DOWN"
                result.append(f"IF {' AND '.join(prefix) if prefix else 'all rows'} THEN {direction} (p={node.probability:.2f})")
                return
            name = names[node.feature_index] if node.feature_index is not None and node.feature_index < len(names) else f"x{node.feature_index}"
            walk(node.left, prefix + [f"{name} <= {node.threshold:.4f}"])
            walk(node.right, prefix + [f"{name} > {node.threshold:.4f}"])

        walk(self.root_, [])
        return result
