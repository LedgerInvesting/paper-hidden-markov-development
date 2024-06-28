from typing import Optional, Callable, Dict, Any
import numpy as np
from dataclasses import dataclass
import json

@dataclass
class Score(object):
    scores: np.ndarray
    function: Callable
    periods: np.ndarray
    lags: np.ndarray

    def _filter(self, filter: Optional[Callable] = None) -> np.ndarray:
        if filter is None:
            return self.scores
        keep = [[idx for idx, v in enumerate(zip(*t)) if all(filter(vv) for vv in v)] for t in self.scores]
        new_scores = np.array([[[e if idx in keep[i] else 0 for idx, e in enumerate(t)] for t in scores] for i, scores in enumerate(self.scores)])
        return new_scores

    def by_model(self, filter: Optional[Callable] = None) -> np.ndarray:
        score = self.function(self._filter(filter), axis=2)
        return np.array(list(zip(*(score.mean(axis=0), score.std(axis=0)))))

    def by_ultimate(self, filter: Optional[Callable] = None) -> np.ndarray:
        score = np.array([[e for i, j, e in zip(self.periods, self.lags, t) if j == 10] for t in self._filter(filter).mean(axis=0)])
        std = np.array([[e for i, j, e in zip(self.periods, self.lags, t) if j == 10] for t in self._filter(filter).std(axis=0)])
        return np.array(list(zip(*(score, std))))

    def summary(self, filter: Optional[Callable] = None) -> Dict[str, np.ndarray]:
        return {
            "scores": self._filter(filter),
            "scores_model": self.by_model(filter),
            "scores_ultimate": self.by_ultimate(filter),
            "scores_ultimate_total": self.by_ultimate(filter).mean(axis=2),
            "proportion_included": 1 - (self._filter(filter) == 0.0).mean().round(4),
        }

    def write(self, file: str, filter: Optional[Callable] = None) -> None:
        with open(file, "w") as f:
            json.dump(self._serialize(self.summary(filter)), f)

    def _serialize(self, x: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in x.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out

