"""Restore rounds 0-19 to saved round-mean frontier metrics, without retuning.
Replay startup exactly and weight by round counts. Calibration remains 20 rounds.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
from multiprocessing import Pool
import subprocess
import sys
import numpy as np
import pandas as pd
from engine import Setting, results_dir

CACHE={}
def work(job):
    model,row=job
    key=(model,row['setting'])
    if key not in CACHE:
        CACHE[key]=Setting(row['setting'],model)
    s=CACHE[key]
    params={k[2:]:v for k,v in row.items() if k.startswith('p_') and pd.notna(v) and v!=-1}
    seed=int(row['seed'])
    stream,sel=s.run(row['method'],seed=seed,rounds=20,**params)
    early=s.metrics(stream,sel)
    total=len(s.stream(seed))
    for k,m in [('type_i','type_i'),('type_ii','type_ii'),('cost','mean_review_cost')]:
        row[k]=(20*early[m]+(total-20)*row[k])/total
    # Verify one full trajectory per method/setting against reconstructed means.
    if seed==7 and job[1].get('_verify',False):
        full=s.metrics(*s.run(row['method'],seed=seed,**params))
        np.testing.assert_allclose([row['type_i'],row['type_ii'],row['cost']],
                                   [full['type_i'],full['type_ii'],full['mean_review_cost']],atol=1e-10)
    row.pop('_verify',None)
    return row

if __name__=='__main__':
    for model in ['llama','qwen']:
        src=results_dir(model)
        out=src/'all_rounds'; out.mkdir(exist_ok=True)
        raw=pd.read_csv(src/'baselines_raw.csv')
        seen=set(); jobs=[]
        for row in raw.to_dict('records'):
            key=(row['setting'],row['method'])
            if row['seed']==7 and key not in seen:
                row['_verify']=True; seen.add(key)
            jobs.append((model,row))
        rows=[]
        with Pool(12) as pool:
            for row in pool.imap_unordered(work,jobs,chunksize=8):
                rows.append(row)
                if len(rows)%500==0: print(model,len(rows),'/',len(jobs),flush=True)
        pd.DataFrame(rows).to_csv(out/'baselines_raw.csv',index=False)
        subprocess.run([sys.executable,str(Path(__file__).with_name('plot_baselines.py')),
                        '--model',model,'--results-dir',str(out)],check=True)
        print('SAVED',out/'baselines_frontier.pdf',flush=True)
