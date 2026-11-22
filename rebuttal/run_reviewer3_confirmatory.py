"""Validation-selected controller diagnostics and all-round RAGTruth baselines.

Previously examined benchmark pools are reused: this is a locked-protocol
reanalysis, not a claim of newly untouched data. Order seeds quantify order
robustness, not independent population samples.
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from engine import Setting
from run_baselines import grid
from run_baselines_norepeat import NoRepeatSetting, SETTINGS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/reviewer3_confirmatory'
EPS = [0., .01, .02, .05, .1, .2, .3, .45]
GAMMA = [.02, .05, .1, .2, .5, 1.]

def evaluate(y, scores, method, eps, param, seed, shift=False):
    order = np.random.default_rng(seed).permutation(len(y))
    y, scores = y[order], scores[order].copy()
    if shift:
        scores[len(scores)//2:] = np.clip(scores[len(scores)//2:] * .6, 0, 1)
    alpha = 0.
    rows = []
    for t, start in enumerate(range(0, len(y)-9, 10)):
        a, s = y[start:start+10], scores[start:start+10]
        threshold = param if method == 'fixed' else alpha
        idx = np.argsort(-s, kind='stable')[:5]
        if method != 'topk':
            idx = idx[s[idx] >= threshold]
        n, fp = len(idx), int((1-a[idx]).sum())
        missed = int(a.sum()-a[idx].sum())
        ti = fp/max(n, 1)
        rows.append([ti, missed/(len(a)-n), n, fp, missed, int(a.sum()), alpha])
        if method in ('edit', 'pooled'):
            violation = ti-eps if method == 'edit' else (fp-eps*n)/5
            alpha = max(0., alpha + param*violation)
    v = np.array(rows)
    return dict(type_i=v[:,0].mean(), type_ii=v[:,1].mean(), cost=v[:,2].mean(),
                pooled_type_i=v[:,3].sum()/max(v[:,2].sum(),1),
                pooled_type_ii=v[:,4].sum()/(10*len(v)-v[:,2].sum()),
                missed_edit_fraction=v[:,4].sum()/max(v[:,5].sum(),1),
                empty_fraction=float((v[:,2]==0).mean()), final_alpha=alpha,
                early_type_ii=v[:20,1].mean(), late_type_ii=v[20:,1].mean() if len(v)>20 else None)

def diagnostic(name, train_y, train_x, test_y, test_x):
    n = len(train_y)
    o = np.random.default_rng(20261115).permutation(n)
    fit, val = o[:n//2], o[n//2:]
    model = LogisticRegression(C=1., max_iter=1000).fit(train_x[fit], train_y[fit])
    vs, ts = model.predict_proba(train_x[val])[:,1], model.predict_proba(test_x)[:,1]
    selected, tuning, results = [], [], []
    for shift in [False, True]:
        for eps in EPS:
            for method in ['edit', 'pooled', 'fixed', 'topk']:
                params = GAMMA if method in ('edit','pooled') else (np.linspace(0,1,31) if method=='fixed' else [0.])
                candidates = []
                for p in params:
                    metrics = [evaluate(train_y[val],vs,method,eps,float(p),s,shift) for s in [101,102,103]]
                    r = dict(name=name,shift=shift,epsilon=eps,method=method,param=float(p),
                             **{k:np.mean([m[k] for m in metrics]) for k in ['type_i','type_ii','cost','pooled_type_i']})
                    candidates.append(r)
                tuning.extend(candidates)
                feasible = [r for r in candidates if r['pooled_type_i'] <= eps]
                best = min(feasible or candidates,key=lambda r: (0 if feasible else r['pooled_type_i'],r['type_ii'],r['cost']))
                best = dict(best, validation_feasible=bool(feasible))
                selected.append(best)
                for seed in range(1000,1030):
                    results.append(dict(name=name,shift=shift,epsilon=eps,method=method,param=best['param'],seed=seed,validation_feasible=bool(feasible),
                                        **evaluate(test_y,ts,method,eps,best['param'],seed,shift)))
    safe = name.replace('/','_').replace(' ','_')
    for suffix, rows in [('tuning',tuning),('selected',selected),('test',results)]:
        pd.DataFrame(rows).to_csv(OUT/f'{safe}_{suffix}.csv',index=False)
    print('DONE controller',name,flush=True)

class SplitSetting(Setting):
    def __init__(self, frame, x):
        self.pool=frame.reset_index(drop=True); self.A=frame.A.to_numpy(int)
        self.features=x; self.costs=np.ones(len(frame))
        self.cfg={'seed':7,'policy':{'budget_per_batch':5,'epsilon':.2,'alpha_step_size':.1,
                 'theta_step_size':1.,'theta_init_scale':.1,'initial_alpha':0.,'warm_start_batches':0,'l2_reg':0.,'horizon':len(frame)//10}}
    def stream(self,seed):
        o=np.random.default_rng(seed).permutation(len(self.A))
        return o[:len(o)//10*10].reshape(-1,10)

def frontier(train,test,x):
    tr=SplitSetting(train,x[:len(train)]); te=SplitSetting(test,x[len(train):])
    candidates=[]
    for method in sorted(set(m for m,p in grid())):
        choices=[p for m,p in grid() if m==method]
        # Equal maximum of 24 configurations per method, chosen before evaluation.
        choices=[choices[i] for i in np.unique(np.linspace(0,len(choices)-1,min(24,len(choices))).astype(int))]
        for p in choices:
            vals=[tr.metrics(*tr.run(method,seed=s,**p)) for s in [101,102,103]]
            candidates.append(dict(method=method,params=p,**{k:np.mean([v[k] for v in vals]) for k in ['type_i','type_ii','mean_review_cost']}))
        print('TUNED frontier',method,flush=True)
        (OUT/'frontier_validation.json').write_text(json.dumps(candidates,indent=2))
    rows=[]; locked=[]
    for eps in EPS:
        for method in sorted(set(r['method'] for r in candidates)):
            group=[r for r in candidates if r['method']==method]
            feasible=[r for r in group if r['type_i']<=eps]
            best=min(feasible or group,key=lambda r:(0 if feasible else r['type_i'],r['type_ii'],r['mean_review_cost']))
            locked.append(dict(epsilon=eps,**best))
    (OUT/'frontier_locked.json').write_text(json.dumps(locked,indent=2))
    for best in locked:
        for seed in range(1000,1030):
            stream,sel=te.run(best['method'],seed=seed,**best['params'])
            rows.append(dict(epsilon=best['epsilon'],method=best['method'],seed=seed,**te.metrics(stream,sel)))
        pd.DataFrame(rows).to_csv(OUT/'frontier_test.csv',index=False)
        print('TESTED frontier',best['method'],best['epsilon'],flush=True)

def main(diagnostics_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'protocol.json').write_text(json.dumps(dict(seed=20261115,epsilon=EPS,gamma=GAMMA,test_seeds=list(range(1000,1030)),
        all_rounds=True,shift='synthetic score compression by 0.6 halfway through; mechanism stress test',
        limitation='Previously inspected benchmark data; order replicates are not independent datasets',
        controller='Frozen logistic scorer; C=1; train split used for fitting and validation only'),indent=2))
    audit=pd.read_csv(ROOT.parent/'human_audit_app/data/audit_samples_50.csv')
    for encoder in ['minilm','1.7B']:
        import run_baselines_norepeat as nr
        nr.FEATURE_ROOT=ROOT/'results/norepeat_gamma'/encoder
        for name in SETTINGS:
            s=NoRepeatSetting(name)
            keep=~s.pool.question.isin(audit.question)
            # Keep identical questions together and use identical splits across encoders/models.
            group=pd.util.hash_pandas_object(s.pool.question,index=False).to_numpy()%10
            train=keep.to_numpy() & (group<6); test=keep.to_numpy() & (group>=6)
            diagnostic(name+' '+encoder,s.A[train],s.features[train],s.A[test],s.features[test])
    r=ROOT/'results/ragtruth'
    train=pd.read_csv(r/'data_train.csv'); test=pd.read_csv(r/'data_test.csv')
    x=np.load(r/'features_qwen4b_2048.npy')
    assert len(x)==len(train)+len(test)
    assert not set(train.source_id)&set(test.source_id)
    diagnostic('RAGTruth Qwen4B',train.A.to_numpy(int),x[:len(train)],test.A.to_numpy(int),x[len(train):])
    if not diagnostics_only:
        frontier(train,test,x)
    print('ALL CACHED EXPERIMENTS COMPLETE',flush=True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--diagnostics-only',action='store_true')
    args=ap.parse_args()
    if args.diagnostics_only:
        OUT=ROOT/'results/reviewer3_diagnostics_corrected'
    main(args.diagnostics_only)
