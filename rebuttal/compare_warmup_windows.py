"""Descriptive replay of original Llama frontier at Type-I target .45.
Configurations are selected from existing post-warmup results, not retuned.
This preserves the original exploratory selection and is not held-out tuning.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import json
from multiprocessing import Pool
import pandas as pd
from engine import Setting, RESULTS

def work(job):
    setting,method,params,seed,feasible=job
    s=Setting(setting,'llama')
    stream,sel=s.run(method,seed=seed,**params)
    row=dict(setting=setting,method=method,seed=seed,feasible=feasible)
    for name,t0 in [('all',0),('late',20)]:
        m=s.metrics(stream,sel,t0=t0)
        row.update({name+'_'+k:m[k] for k in ['type_i','type_ii','mean_review_cost']})
    return row

if __name__=='__main__':
    raw=pd.read_csv(RESULTS/'baselines_raw.csv')
    pc=[c for c in raw if c.startswith('p_')]
    raw[pc]=raw[pc].fillna(-1)
    mean=raw.groupby(['setting','method']+pc,dropna=False)[['type_i','type_ii','cost']].mean().reset_index()
    jobs=[]; selected=[]
    for (setting,method),g in mean.groupby(['setting','method']):
        f=g[g.type_i<=.45+1e-9]; feasible=len(f)>0
        b=(f if feasible else g).sort_values(['type_ii','cost'] if feasible else ['type_i','type_ii','cost']).iloc[0]
        params={c[2:]:float(b[c]) for c in pc if b[c]!=-1}
        selected.append(dict(setting=setting,method=method,feasible=feasible,params=params))
        for seed in sorted(raw.seed.unique()):
            jobs.append((setting,method,params,int(seed),feasible))
    (RESULTS/'warmup_comparison_selected.json').write_text(json.dumps(selected,indent=2))
    rows=[]
    with Pool(4) as p:
        for row in p.imap_unordered(work,jobs):
            rows.append(row)
            pd.DataFrame(rows).to_csv(RESULTS/'warmup_comparison_raw.csv',index=False)
            print(f'{len(rows)}/{len(jobs)} {row["setting"]} {row["method"]}',flush=True)
    result=pd.DataFrame(rows).groupby(['setting','method','feasible']).mean(numeric_only=True).drop(columns='seed').reset_index()
    result.to_csv(RESULTS/'warmup_comparison.csv',index=False)
    print(result.round(4).to_string(index=False),flush=True)
