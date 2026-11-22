"""Budget-comparable score-space literature baselines; keep prior study immutable.
Thompson is explicitly a 1D score-space exploration variant, not the original
full-representation posterior. LLM-Select requires a separate textual evaluation.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json
from pathlib import Path
from multiprocessing import Pool
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.special import logit
from baselines import CLASSES
from paper_analysis.common import compute_metrics
from run_review_quality import paired_ci

SRC=Path(__file__).resolve().parent/'results/review_quality_llama_qwen17_v1'
OUT=SRC.parent/'review_quality_all_baselines_v1'
EPS=[.2,.4,.6]
METHODS=['gollapudi','conformal','zliobaite','thompson_score_space']

class Frozen:
    def __init__(self,score):self.x=logit(np.clip(score,1e-6,1-1e-6)).astype(np.float32)[:,None]
    def encode_rows(self,rows):return self.x[[r['pos'] for r in rows]]

def replay(y,score,cal_y,cal_score,method,params,seed):
    rng=np.random.default_rng(seed);idx=rng.permutation(len(y));encoder=Frozen(score)
    policy=dict(budget_per_batch=5,epsilon=.4,theta_step_size=0.,theta_init_scale=0.,
                initial_alpha=0.,warmup=0,horizon=int(np.ceil(len(y)/10)))
    policy.update(params)
    cfg={'seed':7,'policy':policy}
    obj=None
    if method!='random':
        cls=CLASSES['thompson' if method=='thompson_score_space' else method]
        obj=cls(cfg,encoder,state={'theta':[1.]})
        if method=='conformal':
            obj.cX=np.empty((len(cal_y),1));obj.cA=cal_y
            obj.null_eta=np.sort(cal_score[cal_y==1])
    rows=[]
    for t,start in enumerate(range(0,len(y),10)):
        j=idx[start:start+10]
        batch=pd.DataFrame(dict(t=t,pos=j,A=y[j],cost=1.))
        if method=='random':
            sel=np.zeros(len(j),int);sel[np.random.default_rng(7+1000003*t).permutation(len(j))[:5]]=1
        else:
            res,_=obj.process_batch(batch,t);sel=res.selected.to_numpy(int)
            np.testing.assert_array_equal(obj.theta,[1.])
        assert sel.sum()<=5
        batch['selected']=sel;rows.append(batch)
    frame=pd.concat(rows,ignore_index=True);m=compute_metrics(frame)
    sel=frame.selected.to_numpy();a=frame.A.to_numpy();count=sel.sum();fp=(sel*(1-a)).sum()
    return dict(type_i=m['type_i'],type_ii=m['type_ii'],cost=m['mean_review_cost'],
        selected=float(count),false_reviews=float(fp),records=len(y),rounds=len(rows),
        pooled_precision=float((count-fp)/count) if count else np.nan,coverage=count/len(y))

def grid(method,eps):
    if method=='gollapudi':return [dict(epsilon=eps,dual_step=s) for s in [.063,.25,1.]]
    if method=='conformal':return [dict(q=float(q)) for q in np.round(np.arange(.02,.801,.02),2)]
    if method=='zliobaite':return [dict(randomized=r,adj_step=s) for r in [0,1] for s in [.01,.05]]
    if method=='thompson_score_space':return [dict(v=v) for v in [.3,1.,3.]]
    return [{}]

def load(job):
    setting,fold,probe=job;folder=SRC/setting;prefix=f'fold{fold}_probe{probe}'
    split=json.loads((folder/f'split_{fold}.json').read_text())
    y=np.load(folder/'labels.npy');score=np.load(folder/f'{prefix}_scores.npy')
    return folder,prefix,split,y,score

def tune(job):
    folder,prefix,split,y,s=load(job);dest=OUT/job[0];dest.mkdir(exist_ok=True)
    existing=dest/f'{prefix}_lock.json'
    if existing.exists():
        locks=json.loads(existing.read_text())
        if all(any(r['method']==m and r['epsilon']==e for r in locks) for m in METHODS for e in EPS):
            existing.write_text(json.dumps([r for r in locks if r['method'] in METHODS],indent=2))
            return job
    v=np.array(split['validation']);c=np.array(split['calibration']);locks=[];search=[]
    for eps in EPS:
        for method in METHODS:
            candidates=[]
            for params in grid(method,eps):
                r=[replay(y[v],s[v],y[c],s[c],method,params,seed) for seed in [101,102,103]]
                candidates.append(dict(params=params,**{k:float(np.mean([a[k] for a in r])) for k in ['type_i','type_ii','cost']}))
            feasible=[r for r in candidates if r['type_i']<=eps]
            best=min(feasible or candidates,key=lambda r:(0 if feasible else r['type_i'],r['type_ii'],r['cost']))
            locks.append(dict(epsilon=eps,method=method,validation_feasible=bool(feasible),**best))
            search.extend([dict(epsilon=eps,method=method,**r) for r in candidates])
    (dest/f'{prefix}_lock.json').write_text(json.dumps(locks,indent=2))
    (dest/f'{prefix}_search.json').write_text(json.dumps(search,indent=2))
    return job

def test(job):
    folder,prefix,split,y,s=load(job);dest=OUT/job[0];c=np.array(split['calibration']);ii=np.array(split['test'])
    locks=[r for r in json.loads((dest/f'{prefix}_lock.json').read_text()) if r['method'] in METHODS];rows=[]
    # Match the original study's record bootstrap and paired order seeds exactly.
    group=np.load(folder/'groups.npy')[ii];unique=np.unique(group);members={g:np.flatnonzero(group==g) for g in unique}
    scenarios=[('order',seed,np.arange(len(ii))) for seed in range(1000,1030)]
    if job[2]==0:
        for seed in range(2000,2200):
            selected=np.random.default_rng(seed).choice(unique,len(unique),replace=True)
            scenarios.append(('records',seed,np.concatenate([members[g] for g in selected])))
    cache={}
    for lock in locks:
        for axis,seed,ix in scenarios:
            k=(lock['method'],json.dumps(lock['params'],sort_keys=True),axis,seed)
            if k not in cache:cache[k]=replay(y[ii][ix],s[ii][ix],y[c],s[c],lock['method'],lock['params'],seed)
            rows.append(dict(setting=job[0],fold=job[1],probe=job[2],epsilon=lock['epsilon'],
                method=lock['method'],axis=axis,seed=seed,validation_feasible=lock['validation_feasible'],**cache[k]))
    pd.DataFrame(rows).to_csv(dest/f'{prefix}_test.csv',index=False)
    return job

def summarize():
    extra=pd.concat([pd.read_csv(p) for p in OUT.glob('*/*_test.csv')],ignore_index=True)
    original=pd.concat([pd.read_csv(p) for p in SRC.glob('*/*_test.csv')],ignore_index=True)
    raw=pd.concat([original,extra],ignore_index=True)
    def agg(g):
        n=g.selected.sum();fp=g.false_reviews.sum()
        return pd.Series(dict(type_i=np.average(g.type_i,weights=g['rounds']),type_ii=np.average(g.type_ii,weights=g['rounds']),
            cost=n/g['rounds'].sum(),pooled_precision=(n-fp)/n if n else np.nan,coverage=n/g.records.sum(),
            validation_feasible=g.validation_feasible.all()))
    a=raw.groupby(['setting','probe','epsilon','axis','seed','method']).apply(agg,include_groups=False).reset_index()
    a.to_csv(OUT/'all_methods_by_replicate.csv',index=False)
    a.groupby(['setting','probe','epsilon','axis','method']).agg(type_i=('type_i','mean'),type_ii=('type_ii','mean'),
        cost=('cost','mean'),precision=('pooled_precision','mean'),coverage=('coverage','mean'),
        validation_feasible=('validation_feasible','all')).to_csv(OUT/'all_methods_summary.csv')
    rows=[]
    for (setting,eps),g in a[(a.probe==0)&(a.axis=='order')].groupby(['setting','epsilon']):
        adaptive=g[g.method=='adaptive'].set_index('seed').sort_index()
        for method,h in g.groupby('method'):
            if method=='adaptive':continue
            h=h.set_index('seed').reindex(adaptive.index)
            rows.append(dict(setting=setting,epsilon=eps,baseline=method,
                adaptive_type_i=adaptive.type_i.mean(),baseline_type_i=h.type_i.mean(),
                adaptive_cost=adaptive.cost.mean(),baseline_cost=h.cost.mean(),
                **paired_ci((h.type_ii-adaptive.type_ii).to_numpy())))
    effects=pd.DataFrame(rows);effects['p_holm']=np.nan
    ix=effects[effects.epsilon==.4].sort_values('p').index
    effects.loc[ix,'p_holm']=np.minimum(1,np.maximum.accumulate(effects.loc[ix,'p'].to_numpy()*np.arange(len(ix),0,-1)))
    effects.to_csv(OUT/'paired_all_baselines.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps(dict(complete=True,rows=len(extra))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if (OUT/'complete.json').exists():raise RuntimeError('Completed output exists; do not overwrite')
    (OUT/'protocol.json').write_text(json.dumps(dict(source=str(SRC),methods=METHODS,
        epsilons=EPS,budget=5,all_rounds=True,all_records=True,frozen_calibrated_scores=True,
        conformal='existing code; offline common calibration; warmup=0; no new test-label calibration',
        thompson='explicit 1D posterior exploration on logit of common calibrated score; frozen mean; selected-feedback covariance updates',
        limitation='score-space adaptations, not exact reproductions of original full-feature baselines; LLM-Select not included',
        historical_data=True,grids='existing published grids restricted to identical budget 5'),indent=2))
    jobs=json.loads((SRC/'jobs.json').read_text())
    for phase,fn in [('tune',tune),('test',test)]:
        with Pool(args.workers) as pool:
            for i,job in enumerate(pool.imap_unordered(fn,jobs),1):
                print(phase,i,len(jobs),job,flush=True)
                (OUT/'status.json').write_text(json.dumps(dict(phase=phase,completed=i,total=len(jobs))))
    summarize();print('ALL_BASELINE_EXTENSION_COMPLETE',flush=True)

if __name__=='__main__':main()
