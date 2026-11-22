"""RAGTruth frontier using the shared baseline grid and frozen Qwen3-4B features.

Five permutations of the held-out test set, no repeats; score rounds >=20.
The probe learning rate is fixed from main-figure training (0.5), with the
same initialization for every method. Frontier envelopes follow the existing
baseline protocol and are descriptive test-set sweeps, not train-selected estimates.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from engine import Setting
from run_baselines import grid, SEEDS, SCORE_FROM_ROUND
from run_ragtruth import RESULTS, simulate_rep


class RAGTruthSetting(Setting):
    def __init__(self):
        train = pd.read_csv(RESULTS / 'data_train.csv', usecols=['example_id'])
        self.pool = pd.read_csv(RESULTS / 'data_test.csv', usecols=['example_id', 'A', 'cost'])
        meta = json.loads((RESULTS / 'features_qwen4b_2048.json').read_text())
        X = np.load(RESULTS / 'features_qwen4b_2048.npy')
        if meta['completed_rows'] != len(train) + len(self.pool) or len(X) != meta['completed_rows']:
            raise ValueError('Incomplete or misaligned feature cache')
        if len(self.pool) % 10 or len(self.pool) <= 10 * SCORE_FROM_ROUND:
            raise ValueError('Test set must contain full batches and exceed warm-up')
        if self.pool.example_id.duplicated().any() or not self.pool.cost.eq(1).all():
            raise ValueError('Expected unique test IDs and unit costs')
        self.features = X[len(train):].astype(np.float32)
        if not np.isfinite(self.features).all():
            raise ValueError('Nonfinite features')
        self.A = self.pool.A.to_numpy(int)
        self.costs = np.ones(len(self.pool))
        chosen = json.loads((RESULTS / 'main_chosen.json').read_text())['Qwen3-4B']
        self.cfg = {'seed': 7, 'policy': {
            'budget_per_batch': 5, 'epsilon': 0.45, 'alpha_step_size': chosen['gamma'],
            'theta_step_size': chosen['lr'], 'theta_init_scale': 0.1,
            'initial_alpha': 0.0, 'warm_start_batches': chosen['warm'], 'l2_reg': 0.0,
            'horizon': len(self.pool) // 10,
        }}

    def stream(self, seed):
        return np.random.default_rng(seed).permutation(len(self.pool)).reshape(-1, 10)

    def validate(self):
        policy = self.cfg['policy']
        for seed in SEEDS:
            stream, selected = self.run('edit', seed)
            idx = stream.ravel()
            reference = simulate_rep(self.A[idx], self.features[idx], policy['alpha_step_size'],
                                     policy['theta_step_size'], policy['epsilon'],
                                     policy['warm_start_batches'], True)
            np.testing.assert_array_equal(selected, np.stack([r[3] for r in reference]))
            expected = np.array([r[:3] for r in reference])[SCORE_FROM_ROUND:].mean(axis=0)
            actual = self.metrics(stream, selected, t0=SCORE_FROM_ROUND)
            np.testing.assert_allclose([actual['type_i'], actual['type_ii'], actual['mean_review_cost']], expected)
        print(f'[check] exact EDIT replay for all five seeds; {len(self.pool)} test rows; '
              f'{self.features.shape[1]} dimensions', flush=True)


_setting = None


def work(job):
    global _setting
    if _setting is None:
        _setting = RAGTruthSetting()
    method, params, seed = job
    stream, selected = _setting.run(method, seed=seed, **params)
    metrics = _setting.metrics(stream, selected, t0=SCORE_FROM_ROUND)
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError('Nonfinite metrics')
    return dict(setting='RAGTruth', method=method, seed=seed,
                **{f'p_{key}': value for key, value in params.items()},
                type_i=metrics['type_i'], type_ii=metrics['type_ii'], cost=metrics['mean_review_cost'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    setting = RAGTruthSetting()
    setting.validate()
    if args.check_only:
        for method in dict(grid()):
            params = next(p for m, p in grid() if m == method)
            result = work((method, params, 7))
            print(f'[smoke] {result}', flush=True)
        return
    out = RESULTS / 'edit_search_large' / 'inputs'
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(m, p, seed) for m, p in grid() if m != 'edit' for seed in SEEDS]
    print(f'[run] {len(jobs)} runs; workers={args.workers}', flush=True)
    rows, started = [], time.time()
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(work, jobs, chunksize=1), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(jobs):
                elapsed = time.time() - started
                temporary = out / 'baselines_raw.partial.csv'
                pd.DataFrame(rows).to_csv(temporary, index=False)
                print(f'[run] {i}/{len(jobs)} elapsed={elapsed/60:.1f}m '
                      f'eta={elapsed/i*(len(jobs)-i)/60:.1f}m', flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / 'baselines_raw.csv.gz', index=False)
    params = [c for c in raw if c.startswith('p_')]
    raw[params] = raw[params].fillna(-1)
    mean = raw.groupby(['setting', 'method'] + params).agg(type_i=('type_i', 'mean'), type_ii=('type_ii', 'mean'), cost=('cost', 'mean'), n_streams=('seed', 'count')).reset_index()
    mean.to_csv(out / 'baselines_mean.csv.gz', index=False)
    (out / 'baselines_raw.partial.csv').unlink(missing_ok=True)
    print('[done] retained frontier baseline inputs', flush=True)


if __name__ == '__main__':
    main()
