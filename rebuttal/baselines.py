"""Literature baselines on the same frozen features / logistic probe as EDIT-REP.

gollapudi : Gollapudi et al. 2026, boundary-rule primal-dual ACI over threshold arms (bandit feedback).
            success Y_t = 1 - roundwise Type-I (observed), target phi = 1 - epsilon,
            efficiency cost = budget - edits found (observed). Arms: thresholds on eta, plus
            "select nothing" (guaranteed success, max cost) and tau=0 (most aggressive).
conformal : Conformal Alignment (Gui, Jin & Ren, NeurIPS 2024). Warm-up of K rounds fully labelled,
            first half trains the probe, second half is the calibration set. Afterwards each batch:
            conformal p-values for H_j: "needs edit", BH at level q -> trusted (not reviewed);
            the rest is reviewed (top eta within budget). The probe is frozen after the warm-up.
zliobaite : Variable-uncertainty stream active learning (Zliobaite et al., TNNLS 2014),
            per-item query if certainty max(eta,1-eta) < theta_u; theta_u adapts with step s;
            randomized=1 is the randomized variant. Per-round budget cap.
thompson  : Thompson sampling with a Laplace posterior over the logistic probe (Chapelle & Li 2011);
            top-K by sampled eta within budget. v scales the posterior.
"""
from __future__ import annotations

import numpy as np

from methods.ours import OursSelection
from methods.topk import TopKSelection
from utils import sigmoid


def _topk(eta, costs, budget, rng):
    return TopKSelection._budgeted_topk_selection(eta=np.asarray(eta, float), costs=costs, budget=budget, rng=rng)


class _Base(OursSelection):
    def _sgd(self, features, A, eta, mask):
        n = float(mask.sum())
        if n > 0:
            grad = ((mask * (eta - A))[:, None] * features).sum(axis=0) / n
            self.theta = (self.theta - float(self.policy["theta_step_size"]) * grad).astype(np.float32)

    def _out(self, batch_df, eta, selected):
        out = batch_df.copy()
        out["eta"] = eta
        out["selected"] = selected
        return out, {}


class Gollapudi(_Base):
    TAUS = np.round(np.arange(0.0, 0.951, 0.05), 2)

    def process_batch(self, batch_df, t):
        X = self.score_model.encode_rows(batch_df.to_dict("records"))
        self._ensure_theta(X.shape[1])
        costs = batch_df["cost"].to_numpy(float)
        A = batch_df["A"].to_numpy(float)
        B = float(self.policy["budget_per_batch"])
        phi = 1.0 - float(self.policy["epsilon"])
        step = float(self.policy["dual_step"])
        T = int(self.policy.get("horizon", 250))
        rng = np.random.default_rng(self.seed + 1000003 * int(t))
        eta = sigmoid(X @ self.theta).astype(np.float32)
        n_arms = len(self.TAUS) + 1
        nothing = n_arms - 1
        if not hasattr(self, "lam"):
            self.lam, self.init = 0.0, True
            self.cnt = np.zeros(n_arms)
            self.rsum = np.zeros(n_arms)
            self.csum = np.zeros(n_arms)
        Lam = B / max(1.0 - phi, 1e-6)
        played = [i for i in range(n_arms - 1) if self.cnt[i] == 0]
        if played:
            arm = played[0]
        elif self.lam >= Lam:
            arm = nothing
        elif self.lam <= 0:
            arm = 0
        else:
            d = np.sqrt(2 * np.log(n_arms * T) / np.maximum(self.cnt, 1))
            R = self.rsum / np.maximum(self.cnt, 1) + d
            C = self.csum / np.maximum(self.cnt, 1) - B * d
            R[nothing], C[nothing] = 1.0, B
            arm = int(np.argmin(C - self.lam * R))
        if arm == nothing:
            selected = np.zeros(len(eta), dtype=int)
        else:
            tau = self.TAUS[arm]
            selected = _topk(np.where(eta >= tau, eta, -np.inf), np.where(eta >= tau, costs, 0.0), B, rng)
        n = selected.sum()
        Y = 1.0 if n == 0 else 1.0 - float((selected * (1 - A)).sum()) / n
        c = B - float((selected * A).sum())
        self.cnt[arm] += 1
        self.rsum[arm] += Y
        self.csum[arm] += c
        if not played:
            self.lam += step * (phi - Y)
        elif all(self.cnt[:-1] > 0):
            self.lam = 0.0
        self._sgd(X, A, eta, selected)
        return self._out(batch_df, eta, selected)


class ConformalAlignment(_Base):
    def process_batch(self, batch_df, t):
        X = self.score_model.encode_rows(batch_df.to_dict("records"))
        self._ensure_theta(X.shape[1])
        costs = batch_df["cost"].to_numpy(float)
        A = batch_df["A"].to_numpy(float)
        B = float(self.policy["budget_per_batch"])
        K = int(self.policy["warmup"])
        q = float(self.policy["q"])
        rng = np.random.default_rng(self.seed + 1000003 * int(t))
        eta = sigmoid(X @ self.theta).astype(np.float32)
        if not hasattr(self, "calX"):
            self.calX, self.calA = [], []
        if t < K:
            selected = np.ones(len(eta), dtype=int)
            if t < K // 2:
                self._sgd(X, A, eta, selected)
            else:
                self.calX.append(X)
                self.calA.append(A)
            return self._out(batch_df, eta, selected)
        if not hasattr(self, "cX"):
            self.cX, self.cA = np.concatenate(self.calX), np.concatenate(self.calA)
            self.null_eta = np.sort(sigmoid(self.cX[self.cA == 1] @ self.theta))
        n_cal = len(self.cA)
        # H_j: unit needs an edit. Evidence against: low eta. p_j = (1 + #{null cal: eta_i <= eta_j}) / (n_cal + 1)
        p = (1.0 + np.searchsorted(self.null_eta, eta, side="right")) / (n_cal + 1.0)
        m = len(p)
        order = np.argsort(p, kind="mergesort")
        ok = np.nonzero(p[order] <= q * np.arange(1, m + 1) / m)[0]
        trusted = np.zeros(m, dtype=bool)
        if len(ok):
            trusted[order[: ok[-1] + 1]] = True
        selected = _topk(np.where(trusted, -np.inf, eta), np.where(trusted, 0.0, costs), B, rng)
        return self._out(batch_df, eta, selected)


class Zliobaite(_Base):
    def process_batch(self, batch_df, t):
        X = self.score_model.encode_rows(batch_df.to_dict("records"))
        self._ensure_theta(X.shape[1])
        costs = batch_df["cost"].to_numpy(float)
        A = batch_df["A"].to_numpy(float)
        B = float(self.policy["budget_per_batch"])
        s = float(self.policy.get("adj_step", 0.01))
        randomized = int(self.policy.get("randomized", 0))
        rng = np.random.default_rng(self.seed + 1000003 * int(t))
        eta = sigmoid(X @ self.theta).astype(np.float32)
        if not hasattr(self, "thr"):
            self.thr = 1.0
        selected = np.zeros(len(eta), dtype=int)
        spent = 0.0
        for i in rng.permutation(len(eta)):
            if spent + costs[i] > B + 1e-12:
                continue
            thr = self.thr * rng.normal(1.0, 1.0) if randomized else self.thr
            if max(eta[i], 1 - eta[i]) < thr:
                selected[i] = 1
                spent += costs[i]
                self.thr *= 1 - s
            else:
                self.thr *= 1 + s
        self._sgd(X, A, eta, selected)
        return self._out(batch_df, eta, selected)


class Thompson(_Base):
    def process_batch(self, batch_df, t):
        X = self.score_model.encode_rows(batch_df.to_dict("records")).astype(np.float64)
        self._ensure_theta(X.shape[1])
        costs = batch_df["cost"].to_numpy(float)
        A = batch_df["A"].to_numpy(float)
        B = float(self.policy["budget_per_batch"])
        v = float(self.policy["v"])
        rng = np.random.default_rng(self.seed + 1000003 * int(t))
        if not hasattr(self, "H"):
            self.H = float(self.policy.get("prior_precision", 1.0)) * np.eye(X.shape[1])
        Lc = np.linalg.cholesky(self.H)
        z = rng.standard_normal(X.shape[1])
        th = self.theta + v * np.linalg.solve(Lc.T, z)
        eta_s = sigmoid(X @ th)
        eta = sigmoid(X @ self.theta).astype(np.float32)
        selected = _topk(eta_s, costs, B, rng)
        sel = selected.astype(bool)
        if sel.any():
            w = eta[sel] * (1 - eta[sel])
            self.H += (X[sel] * w[:, None]).T @ X[sel]
        self._sgd(X.astype(np.float32), A, eta, selected)
        return self._out(batch_df, eta, selected)


CLASSES = {"gollapudi": Gollapudi, "conformal": ConformalAlignment, "zliobaite": Zliobaite, "thompson": Thompson}

