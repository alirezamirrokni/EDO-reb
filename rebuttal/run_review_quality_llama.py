"""Llama-only four-task Qwen1.7B study plus unchanged Qwen4B RAGTruth."""
import argparse,json,time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
import run_review_quality as base
from engine import Setting,SETTINGS

def load():
    rows=[]
    for dataset in SETTINGS:
        s=Setting(dataset,'llama')
        path=base.ROOT/'features/qwen17_primary'/f'llama_{dataset}.npz'
        z=np.load(path)
        np.testing.assert_array_equal(z['example_id'],s.pool.example_id.to_numpy())
        assert len(z['features'])==len(s.pool) and np.isfinite(z['features']).all()
        rows.append((f'llama_{dataset}_Qwen1.7B',s.pool,z['features'],None))
    r=base.ROOT/'results/ragtruth'
    tr=pd.read_csv(r/'data_train.csv');te=pd.read_csv(r/'data_test.csv')
    assert not set(tr.source_id)&set(te.source_id)
    x=np.load(r/'features_qwen4b_2048.npy')
    assert len(x)==len(tr)+len(te)
    rows.append(('RAGTruth_Qwen4B',pd.concat([tr,te],ignore_index=True),x,len(tr)))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(exist_ok=True,parents=True)
    base.OUT=out;base.load_all=load
    path=out/'protocol.json'
    if not path.exists():
        base.save(path,dict(primary_epsilon=base.PRIMARY,sensitivity=base.EPS,alpha_grid=base.ALPHAS,gamma_grid=base.GAMMAS,
            primary_settings='Llama only: MATH-500, MMLU-Pro, PopQA, GPQA; Qwen3-1.7B representations',
            ragtruth='official split and existing Qwen4B representations unchanged',
            crossfit='five grouped folds, disjoint probe fit / Platt calibration / policy validation / test',
            order_seeds=base.ORDER_SEEDS,validation_seeds=[101,102,103],record_seeds=list(range(2000,2200)),
            probe_fits=5,split_seed=20261116,all_records=True,all_rounds=True,budget=5,
            original_protocol='rebuttal/REVIEW_QUALITY_PROTOCOL.md; scope overridden by this file',
            historical_data_caveat=True,base_code_hash=base.hashfile(base.__file__),wrapper_hash=base.hashfile(__file__)))
        base.validate();jobs=base.prepare()
    else:
        p=json.loads(path.read_text());assert p['base_code_hash']==base.hashfile(base.__file__)
        assert p['wrapper_hash']==base.hashfile(__file__)
        jobs=json.loads((out/'jobs.json').read_text())
    for phase,fn in [('tune',base.tune),('test',base.test)]:
        with Pool(args.workers) as pool:
            for i,job in enumerate(pool.imap_unordered(fn,jobs),1):
                print(phase,i,len(jobs),job,flush=True)
                base.save(out/'status.json',dict(phase=phase,completed=i,total=len(jobs),time=time.time()))
        if phase=='tune':base.save(out/'all_locks.json',{str(p.relative_to(out)):base.hashfile(p) for p in out.glob('*/*_lock.json')})
    base.summarize()

if __name__=='__main__':main()
