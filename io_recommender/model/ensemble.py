from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from io_recommender.model.labels import gains_to_relevance
from io_recommender.types import Observation

lgb = None

class _NumpyLinearRegressor:
    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyLinearRegressor":
        Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=float)])
        self.coef_, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("model is not fitted")
        Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=float)])
        return Xb @ self.coef_


@dataclass
class EnsembleConfig:
    mode: str = "ranking"  # ranking | regression
    ensemble_size: int = 6
    seed: int = 7
    subsample_ratio: float = 0.9
    relevance_levels: int = 5
    use_lightgbm: bool = True


class EnsembleModel:
    def __init__(self, cfg: EnsembleConfig):
        self.cfg = cfg
        self.models: List[object] = []
        self.feature_dim: int | None = None
        self.use_lightgbm = bool(cfg.use_lightgbm and self._lightgbm_module() is not None)

    @staticmethod
    def _lightgbm_module():
        global lgb
        if lgb is not None:
            return lgb
        try:
            import lightgbm as lightgbm_module
        except Exception:
            return None
        lgb = lightgbm_module
        return lgb

    @staticmethod
    def _obs_to_xy(observations: Sequence[Observation]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        sorted_obs = sorted(observations, key=lambda o: o.pattern_id)
        X = np.vstack([np.concatenate([o.workload_vec, o.config_vec]) for o in sorted_obs])
        y = np.array([o.gain for o in sorted_obs], dtype=float)
        groups = [o.pattern_id for o in sorted_obs]
        return X, y, groups

    @staticmethod
    def _group_sizes(sorted_groups: List[str]) -> List[int]:
        out: List[int] = []
        last = None
        count = 0
        for g in sorted_groups:
            if g != last:
                if count > 0:
                    out.append(count)
                last = g
                count = 1
            else:
                count += 1
        if count > 0:
            out.append(count)
        return out

    def _bootstrap_indices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        k = max(1, int(n * self.cfg.subsample_ratio))
        return rng.choice(np.arange(n), size=k, replace=True)

    def fit(self, observations: Sequence[Observation]) -> "EnsembleModel":
        if not observations:
            raise ValueError("observations cannot be empty")

        X, y, groups = self._obs_to_xy(observations)
        self.feature_dim = X.shape[1]
        self.models = []

        for m_idx in range(self.cfg.ensemble_size):
            rng = np.random.default_rng(self.cfg.seed + m_idx)
            idx = self._bootstrap_indices(len(y), rng)
            Xb = X[idx]
            yb = y[idx]
            gb = [groups[i] for i in idx]

            if self.cfg.mode == "ranking" and self.use_lightgbm:
                rel = gains_to_relevance(yb, levels=self.cfg.relevance_levels)
                order = np.argsort(gb)
                Xr = Xb[order]
                yr = rel[order]
                gr = [gb[i] for i in order]
                group_sizes = self._group_sizes(gr)
                lgb_module = self._lightgbm_module()
                if lgb_module is None:
                    raise RuntimeError("LightGBM unavailable for ranking mode; disable use_lightgbm or install lightgbm")
                model = lgb_module.LGBMRanker(
                    objective="lambdarank",
                    metric="ndcg",
                    ndcg_eval_at=[3],
                    n_estimators=120,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=self.cfg.seed + m_idx,
                    n_jobs=1,
                    verbosity=-1,
                )
                model.fit(Xr, yr, group=group_sizes)
            else:
                if self.use_lightgbm:
                    lgb_module = self._lightgbm_module()
                    if lgb_module is None:
                        raise RuntimeError("LightGBM unavailable for regression mode with use_lightgbm=true")
                    model = lgb_module.LGBMRegressor(
                        objective="regression",
                        n_estimators=150,
                        learning_rate=0.05,
                        num_leaves=31,
                        random_state=self.cfg.seed + m_idx,
                        n_jobs=1,
                        verbosity=-1,
                    )
                else:  # pragma: no cover
                    model = _NumpyLinearRegressor()
                model.fit(Xb, yb)
            self.models.append(model)
        return self

    def predict_mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.models:
            raise ValueError("model ensemble is not trained")
        preds = np.vstack([np.asarray(m.predict(X), dtype=float) for m in self.models])
        return preds.mean(axis=0), preds.std(axis=0)
