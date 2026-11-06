from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from utils import sigmoid

from .ours import OursSelection


class TopKSelection(OursSelection):
    """Budgeted Top-K baseline with the same learned score as OursSelection.

    Uses the identical frozen features, logistic probe, initialization, and
    selected-sample gradient update as EDIT-REP, but drops the alpha threshold
    and the cost-adjusted ranking: items are ranked by eta alone and added
    greedily while the per-round budget allows.
    """

    @staticmethod
    def _budgeted_topk_selection(eta: np.ndarray, costs: np.ndarray, budget: float, rng: np.random.Generator) -> np.ndarray:
        n = len(eta)
        selected = np.zeros(n, dtype=int)
        if n == 0 or budget <= 0:
            return selected
        idx = np.arange(n)
        rng.shuffle(idx)
        order = idx[np.argsort(-eta[idx], kind="mergesort")]
        spent = 0.0
        for i in order:
            c = float(costs[i])
            if c > 0 and spent + c <= budget + 1e-12:
                selected[i] = 1
                spent += c
        return selected

    def process_batch(self, batch_df: pd.DataFrame, t: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        features = self.score_model.encode_rows(batch_df.to_dict("records"))
        self._ensure_theta(features.shape[1])

        costs = batch_df["cost"].to_numpy(dtype=float)
        A = batch_df["A"].to_numpy(dtype=float)
        budget = float(self.policy["budget_per_batch"])
        theta_lr = float(self.policy["theta_step_size"])
        l2_reg = float(self.policy.get("l2_reg", 0.0))

        rng = np.random.default_rng(self.seed + 1000003 * int(t))
        eta = sigmoid(features @ self.theta).astype(np.float32)

        selected = self._budgeted_topk_selection(eta=eta.astype(float), costs=costs, budget=budget, rng=rng)
        n_sel = float(selected.sum())
        if n_sel > 0:
            grad = ((selected * (eta - A))[:, None] * features).sum(axis=0) / n_sel
            if l2_reg > 0:
                grad = grad + l2_reg * self.theta
            self.theta = (self.theta - theta_lr * grad).astype(np.float32)

        out = batch_df.copy()
        out["eta"] = eta
        out["alpha"] = np.nan
        out["beta"] = np.nan
        out["selected"] = selected
        out["selection_mode"] = "topk"
        return out, self.state_dict()
