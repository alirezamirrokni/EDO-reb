"""Learning-rate sensitivity check for the strongest fixed-grid competitors."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
from multiprocessing import Pool
import pandas as pd
from run_ragtruth_frontier import work, RESULTS, SEEDS
from run_baselines import grid

if __name__ == '__main__':
    out = RESULTS / 'edit_search_large' / 'inputs'
    out.mkdir(parents=True, exist_ok=True)
    lrs=[.05,.1,.25,.5,1.,2.,4.,8.,16.,32.,64.]
    jobs=[(m,dict(p,theta_step_size=lr),seed) for m,p in grid()
          if m in ['conformal','topk'] for lr in lrs for seed in SEEDS]
    rows=[]; start=time.time()
    print(f'[run] {len(jobs)} sensitivity runs: Conformal Alignment and Top-K',flush=True)
    with Pool(4) as pool:
        for i,r in enumerate(pool.imap_unordered(work,jobs,chunksize=2),1):
            rows.append(r)
            if i%250==0 or i==len(jobs):
                print(f'[run] {i}/{len(jobs)} elapsed={(time.time()-start)/60:.1f}m',flush=True)
    raw=pd.DataFrame(rows)
    raw.to_csv(out/'baseline_lr_audit.csv.gz',index=False)
    params=[c for c in raw if c.startswith('p_')]
    raw[params]=raw[params].fillna(-1)
    mean=raw.groupby(['method']+params).agg(type_i=('type_i','mean'),type_ii=('type_ii','mean'),cost=('cost','mean'),n=('seed','count')).reset_index()
    assert mean.n.eq(5).all()
    mean.to_csv(out/'baseline_lr_audit_mean.csv.gz',index=False)
    print('[done] BASELINE_LR_AUDIT_DONE',flush=True)
