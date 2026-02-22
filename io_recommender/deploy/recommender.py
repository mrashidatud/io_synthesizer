from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import numpy as np

from io_recommender.model import ConfigEncoder, EnsembleModel, WorkloadEncoder
from io_recommender.types import Observation, WorkloadPattern


@dataclass
class RecommendationMatrix:
    topk_by_pattern: Dict[str, List[str]]
    config_by_id: Dict[str, Dict[str, object]]
    config_vec_by_id: Dict[str, np.ndarray]


class DeploymentRecommender:
    def __init__(
        self,
        workload_encoder: WorkloadEncoder,
        config_encoder: ConfigEncoder,
        ensemble: EnsembleModel,
        patterns: Sequence[WorkloadPattern],
        rec_matrix: RecommendationMatrix,
        knn_k: int = 5,
    ):
        self.workload_encoder = workload_encoder
        self.config_encoder = config_encoder
        self.ensemble = ensemble
        self.patterns = list(patterns)
        self.rec_matrix = rec_matrix
        self.knn_k = knn_k

        Xw = self.workload_encoder.encode_many(self.patterns)
        self._workload_matrix = Xw

    def recommend(self, workload_features: Mapping[str, float] | np.ndarray, top_k: int = 3) -> List[Dict[str, object]]:
        wvec = self.workload_encoder.encode_workload(workload_features)
        distances = np.linalg.norm(self._workload_matrix - wvec.reshape(1, -1), axis=1)
        n = min(self.knn_k, len(self.patterns))
        idxs = np.argsort(distances)[:n]

        candidate_ids = []
        for idx in idxs:
            pid = self.patterns[int(idx)].pattern_id
            candidate_ids.extend(self.rec_matrix.topk_by_pattern.get(pid, []))

        # stable dedup
        seen = set()
        dedup_ids = []
        for cid in candidate_ids:
            if cid not in seen:
                seen.add(cid)
                dedup_ids.append(cid)

        if not dedup_ids:
            return []

        X = np.vstack([
            np.concatenate([wvec, self.rec_matrix.config_vec_by_id[cid]])
            for cid in dedup_ids
        ])
        mu, _ = self.ensemble.predict_mean_std(X)
        order = np.argsort(mu)[::-1][:top_k]
        out = []
        for i in order:
            cid = dedup_ids[int(i)]
            out.append(
                {
                    "config_id": cid,
                    "config": self.rec_matrix.config_by_id[cid],
                    "score": float(mu[int(i)]),
                }
            )
        return out


def materialize_recommendation_matrix(observations: Sequence[Observation], top_k: int = 20) -> RecommendationMatrix:
    by_pattern: Dict[str, List[Observation]] = {}
    for obs in observations:
        by_pattern.setdefault(obs.pattern_id, []).append(obs)

    topk_by_pattern: Dict[str, List[str]] = {}
    config_by_id: Dict[str, Dict[str, object]] = {}
    config_vec_by_id: Dict[str, np.ndarray] = {}

    for pid, obs_list in by_pattern.items():
        best_by_cfg = {}
        for obs in obs_list:
            prev = best_by_cfg.get(obs.config_id)
            if prev is None or obs.gain > prev.gain:
                best_by_cfg[obs.config_id] = obs
        ranked = sorted(best_by_cfg.values(), key=lambda x: x.gain, reverse=True)
        topk = ranked[:top_k]
        topk_by_pattern[pid] = [o.config_id for o in topk]
        for o in topk:
            config_by_id[o.config_id] = dict(o.config_params)
            config_vec_by_id[o.config_id] = o.config_vec

    return RecommendationMatrix(
        topk_by_pattern=topk_by_pattern,
        config_by_id=config_by_id,
        config_vec_by_id=config_vec_by_id,
    )
