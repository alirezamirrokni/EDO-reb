"""Human-post-edit evaluation: official MLQE-PE archives, development-only tuning."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import io,json,tarfile,hashlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
import requests
from engine import Setting

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/mlqe_postedit'
EPS=[.05,.1,.2,.3,.45]
SEEDS=list(range(1000,1030))
BASE='https://raw.githubusercontent.com/sheffieldnlp/mlqe-pe/master/data/post-editing'

def download(pair,split):
    dest=OUT/f'{pair}-{split}.tar.gz'
    if not dest.exists():
        r=requests.get(f'{BASE}/{split}/{pair}-{split}.tar.gz',timeout=120)
        r.raise_for_status(); dest.write_bytes(r.content)
    with tarfile.open(fileobj=io.BytesIO(dest.read_bytes())) as t:
        fields={}
        for suffix in ['src','mt','pe','hter']:
            matches=[m for m in t.getmembers() if m.isfile() and m.name.endswith('.'+suffix)]
            if len(matches)!=1: raise ValueError((pair,split,suffix,'missing/ambiguous field'))
            fields[suffix]=t.extractfile(matches[0]).read().decode('utf-8').splitlines()
    assert len(set(map(len,fields.values())))==1
    d=pd.DataFrame(fields)
    d['hter']=pd.to_numeric(d.hter,errors='raise')
    assert np.isfinite(d.hter).all() and d.hter.ge(0).all()
    norm=lambda s:' '.join(str(s).split())
    d['A']=(d.mt.map(norm)!=d.pe.map(norm)).astype(int)
    d['source_key']=d.src.map(norm)
    d['example_id']=[f'{pair}-{split}-{i}' for i in range(len(d))]
    d['question']=d.src; d['model_answer']=d.mt; d['cost']=1.
    d.to_csv(OUT/f'{pair}_{split}_original.csv',index=False)
    return d

def encode(pair,split,d,enc):
    path=OUT/f'{pair}_{split}_qwen4b.npy'; meta=path.with_suffix('.json')
    signature=hashlib.sha256('\n'.join(d.example_id).encode()).hexdigest()
    if meta.exists():
        old=json.loads(meta.read_text()); assert old['ids_sha256']==signature
        done=old['completed']
    else: done=0
    x=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype='float32',shape=(len(d),enc.hidden_size))
    for start in range(done,len(d),32):
        end=min(start+32,len(d))
        x[start:end]=enc.encode_rows(d.iloc[start:end][['question','model_answer']].to_dict('records'))
        x.flush();meta.write_text(json.dumps(dict(completed=end,ids_sha256=signature)))
        if end%320==0 or end==len(d):print('EMBED',pair,split,end,len(d),flush=True)
    return np.asarray(x)

class PostEditSetting(Setting):
    def __init__(self,d,x):
        self.pool=d;self.A=d.A.to_numpy(int);self.features=x;self.costs=np.ones(len(d))
        self.cfg={'seed':7,'policy':dict(budget_per_batch=5,epsilon=.2,alpha_step_size=.1,
            theta_step_size=1.,theta_init_scale=.1,initial_alpha=0.,warm_start_batches=0,l2_reg=0.,horizon=len(d)//10)}
    def stream(self,seed):
        o=np.random.default_rng(seed).permutation(len(self.A))
        return o[:len(o)//10*10].reshape(-1,10)
    def assess(self,method,params,seed):
        if method=='random':
            stream=self.stream(seed);sel=np.zeros(stream.shape,dtype=int);rng=np.random.default_rng(seed+771)
            for row in sel:row[rng.choice(10,size=int(params['budget_per_batch']),replace=False)]=1
        else: stream,sel=self.run(method,seed=seed,**params)
        m=self.metrics(stream,sel)
        severity=self.pool.hter.to_numpy()[stream]
        total=severity.sum()
        m['severity_captured']=float((severity*sel).sum()/total) if total else 0.
        m['severity_per_review']=float((severity*sel).sum()/max(sel.sum(),1))
        return m

CURRENT=None
def tune_job(job):
    method,params=job
    rows=[CURRENT.assess(method,params,s) for s in [101,102,103]]
    return dict(method=method,params=params,**{k:float(np.mean([r[k] for r in rows])) for k in rows[0]})

def experiment(pair,dev,test,xd,xt):
    global CURRENT
    CURRENT=PostEditSetting(dev,xd)
    jobs=[]
    for e in EPS:
        for gamma in [.02,.1,.5]:
            for lr in [.1,1.,4.]:
                jobs.append(('edit',dict(epsilon=e,alpha_step_size=gamma,theta_step_size=lr)))
    for lr in [.1,1.,4.]:
        for tau in np.linspace(0,1,21):jobs.append(('tau',dict(tau=float(tau),theta_step_size=lr)))
        for budget in [1,2,3,4,5]:jobs.append(('topk',dict(budget_per_batch=budget,theta_step_size=lr)))
    jobs += [('random',dict(budget_per_batch=b)) for b in [1,2,3,4,5]]
    tuning=[]
    with Pool(4) as p:
        for r in p.imap_unordered(tune_job,jobs):
            tuning.append(r)
            if len(tuning)%10==0:print('TUNING',pair,len(tuning),len(jobs),flush=True)
    (OUT/f'{pair}_development_search.json').write_text(json.dumps(tuning,indent=2))
    locked=[]
    for eps in EPS:
        for method in ['edit','tau','topk','random']:
            group=[r for r in tuning if r['method']==method and (method!='edit' or r['params']['epsilon']==eps)]
            feasible=[r for r in group if r['type_i']<=eps]
            best=min(feasible or group,key=lambda r:(0 if feasible else r['type_i'],r['type_ii'],r['mean_review_cost']))
            locked.append(dict(target_epsilon=eps,validation_feasible=bool(feasible),**best))
    (OUT/f'{pair}_locked.json').write_text(json.dumps(locked,indent=2))
    s=PostEditSetting(test,xt);rows=[]
    cache={}
    for r in locked:
        for seed in SEEDS:
            key=(r['method'],json.dumps(r['params'],sort_keys=True),seed)
            if key not in cache:cache[key]=s.assess(r['method'],r['params'],seed)
            rows.append(dict(pair=pair,method=r['method'],epsilon=r['target_epsilon'],
                validation_feasible=r['validation_feasible'],seed=seed,**cache[key]))
        pd.DataFrame(rows).to_csv(OUT/f'{pair}_test.csv',index=False)
        print('TESTED',pair,r['method'],r['target_epsilon'],flush=True)
    pd.DataFrame(rows).groupby(['pair','method','epsilon','validation_feasible']).mean(numeric_only=True).to_csv(OUT/f'{pair}_summary.csv')

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'protocol.json').write_text(json.dumps(dict(pairs=['en-de','en-zh'],epsilon=EPS,seeds=SEEDS,
        label='whitespace-normalized MT differs from human post-edit',severity='provided HTER',
        encoder='Qwen3-4B last-token L2 normalized; max_length=512',
        training='cold-start online probes; official train archived but not used for fitting',
        selection='official dev only; all rounds; min Type-II subject to roundwise Type-I target',
        limitation='HTER is an edit-effort proxy, not editing time; ordering seeds are not independent datasets'),indent=2))
    from models.qwen3_embedding import Qwen3ScoreModel
    paths=[Path('/home/mll/ali/models/Qwen3-4B')]
    assert (paths[0]/'config.json').exists(),'Missing local Qwen-4B'
    enc=None
    for pair in ['en-de','en-zh']:
        train,dev,test=[download(pair,s) for s in ['train','dev','test']]
        # Preserve official test set; remove repeated sources within it and across dev/test.
        test=test.drop_duplicates('source_key').reset_index(drop=True)
        dev=dev[~dev.source_key.isin(test.source_key)].drop_duplicates('source_key').reset_index(drop=True)
        audit=dict(train=len(train),development=len(dev),test=len(test),
            dev_edit_rate=float(dev.A.mean()),test_edit_rate=float(test.A.mean()),
            test_label_hter_disagreement=int(((test.hter>0).astype(int)!=test.A).sum()))
        (OUT/f'{pair}_audit.json').write_text(json.dumps(audit,indent=2));print('DATA',pair,audit,flush=True)
        dev.to_csv(OUT/f'{pair}_dev.csv',index=False);test.to_csv(OUT/f'{pair}_test_data.csv',index=False)
        if enc is None:enc=Qwen3ScoreModel(dict(model_name=str(paths[0]),max_length=512,encode_batch_size=8,
            pooling='last',normalize_features=True,device='cuda:0',torch_dtype='bfloat16',cache_embeddings=False,
            prompt_template='Source sentence:\n{question}\n\nMachine translation:\n{model_answer}'))
        xd=encode(pair,'dev',dev,enc);xt=encode(pair,'test',test,enc)
        experiment(pair,dev,test,xd,xt)
    print('MLQE_POSTEDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
