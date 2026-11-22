"""Resumable large EDIT search over supported production-policy parameters.

Fixed data, feature cache, five stream seeds, initialization RNG seed, budget,
and scoring interval. Exploratory test-set tuning, not independent validation.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import gzip
import hashlib
import json
import sqlite3
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from run_ragtruth_frontier import RAGTruthSetting, RESULTS, SEEDS, SCORE_FROM_ROUND
from run_ragtruth import choose, round_metrics, sigmoid

OUT=RESULTS/'edit_search_large'
LEVELS=[.15,.20,.25,.30,.35,.40,.45,.50]
KEYS=['epsilon','alpha_step_size','theta_step_size','warm_start_batches','l2_reg','theta_init_scale','initial_alpha']
STREAMS=[]


def canonical(p):
    return json.dumps(p,sort_keys=True,separators=(',',':'))


def atomic_json(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2))
    tmp.replace(path)


def initialize():
    global STREAMS
    s=RAGTruthSetting()
    STREAMS=[(s.A[s.stream(seed).ravel()],s.features[s.stream(seed).ravel()]) for seed in SEEDS]


def replay(y,X,p,details=False):
    scale=p['theta_init_scale']
    theta=(np.random.default_rng(7+1729).normal(0,scale,X.shape[1]).astype(np.float32)
           if scale>0 else np.zeros(X.shape[1],dtype=np.float32))
    alpha=p['initial_alpha']; metrics=[]; selections=[]
    for t,start in enumerate(range(0,len(y),10)):
        a,x=y[start:start+10],X[start:start+10]
        eta=sigmoid(x@theta).astype(np.float32)
        sel=choose(eta,alpha,t,p['warm_start_batches'])
        ti,tii,cost=round_metrics(a,sel)
        alpha=max(0.,alpha+p['alpha_step_size']*(ti-p['epsilon']))
        n=sel.sum()
        if n:
            grad=((sel*(eta-a))[:,None]*x).sum(axis=0)/n
            if p['l2_reg']>0: grad=grad+p['l2_reg']*theta
            theta=(theta-p['theta_step_size']*grad).astype(np.float32)
        if not np.isfinite(theta).all(): raise FloatingPointError('Nonfinite probe weights')
        if t>=SCORE_FROM_ROUND: metrics.append((ti,tii,cost))
        if details: selections.append(sel)
    values=np.array(metrics)
    if not np.isfinite(values).all() or values[:,2].max()>5:
        raise ValueError('Invalid metrics or review budget')
    return np.stack(selections) if details else values.mean(axis=0).tolist()


def work(job):
    key,p=job
    return key,p,[replay(y,X,p) for y,X in STREAMS]


def validate():
    s=RAGTruthSetting()
    cases=[dict(epsilon=.3,alpha_step_size=.2,theta_step_size=4.,warm_start_batches=5,
                l2_reg=.01,theta_init_scale=.1,initial_alpha=.3),
           dict(epsilon=.2,alpha_step_size=3.,theta_step_size=16.,warm_start_batches=20,
                l2_reg=.001,theta_init_scale=0.,initial_alpha=.5),
           dict(epsilon=.45,alpha_step_size=.005,theta_step_size=.05,warm_start_batches=0,
                l2_reg=.1,theta_init_scale=1.,initial_alpha=1.),
           dict(epsilon=.1,alpha_step_size=30.,theta_step_size=128.,warm_start_batches=7,
                l2_reg=.001,theta_init_scale=.03,initial_alpha=0.)]
    for p in cases:
        for seed in SEEDS:
            stream,sel=s.run('edit',seed,**p)
            idx=stream.ravel()
            np.testing.assert_array_equal(sel,replay(s.A[idx],s.features[idx],p,True))
            actual=s.metrics(stream,sel,t0=SCORE_FROM_ROUND)
            np.testing.assert_allclose(replay(s.A[idx],s.features[idx],p),
                                      [actual['type_i'],actual['type_ii'],actual['mean_review_cost']],atol=1e-12)
    print('[check] 20 exact production-policy replays, including regularization, zero init and initial threshold',flush=True)


def previous():
    d=pd.read_csv(OUT/'inputs/prior_mean.csv.gz')
    d['p_l2_reg']=0.; d['p_theta_init_scale']=.1; d['p_initial_alpha']=0.
    return d


def params_from_row(r):
    p={k:float(r['p_'+k]) for k in KEYS}
    p['warm_start_batches']=int(p['warm_start_batches'])
    return p


def manifest(count):
    path=OUT/'manifest.jsonl.gz'
    if path.exists():
        jobs=[json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
        if len(jobs)!=count: raise ValueError('Existing manifest configuration count differs')
        return [(canonical(p),p) for p in jobs]
    prior=previous()
    elites=[]
    for cap in LEVELS:
        elites.extend(params_from_row(r) for _,r in prior[prior.type_i<=cap+1e-9].nsmallest(12,'type_ii').iterrows())
    rng=np.random.default_rng(20261111)
    seen={canonical(params_from_row(r)) for _,r in prior.iterrows()}
    jobs=[]
    while len(jobs)<count:
        local=len(jobs)>=int(count*.4)
        if local:
            p=dict(elites[int(rng.integers(len(elites)))])
            p['epsilon']=float(np.clip(p['epsilon']+rng.uniform(-.08,.08),0,.8))
            p['alpha_step_size']=float(np.clip(max(p['alpha_step_size'],.0001)*np.exp(rng.uniform(np.log(.25),np.log(4))),.0001,30))
            p['theta_step_size']=float(np.clip(p['theta_step_size']*np.exp(rng.uniform(np.log(.5),np.log(2))),.01,128))
            p['warm_start_batches']=int(np.clip(p['warm_start_batches']+rng.integers(-5,6),0,20))
            p['theta_init_scale']=.1 if rng.random()<.7 else float(rng.choice([0,.01,.03,.3,1.]))
            p['initial_alpha']=0. if rng.random()<.8 else float(rng.uniform(0,1))
        else:
            p=dict(epsilon=float(rng.uniform(0,.8)),
                   alpha_step_size=0. if rng.random()<.05 else float(np.exp(rng.uniform(np.log(.0001),np.log(30)))),
                   theta_step_size=float(np.exp(rng.uniform(np.log(.01),np.log(128)))),
                   warm_start_batches=int(rng.integers(0,21)),
                   theta_init_scale=float(rng.choice([0,.01,.03,.1,.3,1.])),
                   initial_alpha=0. if rng.random()<.5 else float(rng.uniform(0,1)))
        p['l2_reg']=float(rng.choice([0.,0.,0.,1e-6,1e-5,1e-4,1e-3,1e-2,.1]))
        if p['l2_reg']*p['theta_step_size']>1: continue
        p={k:int(v) if k=='warm_start_batches' else round(float(v),8) for k,v in p.items()}
        key=canonical(p)
        if key in seen: continue
        seen.add(key); jobs.append((key,p))
    temp=path.with_suffix('.tmp')
    with gzip.open(temp,'wt') as f: f.write(''.join(key+'\n' for key,p in jobs))
    temp.replace(path)
    return jobs


def comparisons(mean):
    original=pd.read_csv(OUT/'inputs/baselines_mean.csv.gz').query("method != 'edit'")
    audit=pd.concat([original,pd.read_csv(OUT/'inputs/baseline_lr_audit_mean.csv.gz')],ignore_index=True)
    rows=[]
    for label,base in [('original_baseline_grid',original),('lr_sensitivity_audit',audit)]:
        for cap in LEVELS:
            e=mean[mean.type_i<=cap+1e-9].sort_values(['type_ii','cost']).iloc[0]
            b=base[base.type_i<=cap+1e-9].sort_values(['type_ii','cost']).iloc[0]
            rows.append(dict(comparison=label,cap=cap,edit_type_i=e.type_i,edit_type_ii=e.type_ii,
                             edit_cost=e.cost,baseline=b.method,baseline_type_ii=b.type_ii,
                             gap=e.type_ii-b.type_ii,**{k:e['p_'+k] for k in KEYS}))
    return pd.DataFrame(rows)


def finish(mean, output_stem='frontier', style_variant=False, hypothetical=False, real_intervals=False):
    mean.to_csv(OUT/'mean.csv.gz',index=False)
    table=comparisons(mean); table.to_csv(OUT/'comparison.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(context='paper',style='darkgrid')
    plt.rcParams.update({'font.size':16,'axes.titlesize':18,'axes.labelsize':16,'xtick.labelsize':13,'ytick.labelsize':13,
                         'font.weight':'bold','axes.titleweight':'bold','axes.labelweight':'bold','pdf.fonttype':42})
    base=pd.read_csv(OUT/'inputs/baselines_mean.csv.gz').query("method != 'edit'")
    fig,ax=plt.subplots(figsize=(10,6))
    names={'gollapudi':'Online Conformal Selection','conformal':'Conformal Alignment',
           'zliobaite':'Stream Active Learning','thompson':'Thompson Sampling','topk':'Top-K','tau':'Fixed Threshold'}
    colors=['#d62728','#9467bd','#8c564b','#17becf','#1f77b4','#2ca02c']
    styles=['-.','--',':','--','--',':']
    d=mean.groupby('type_i',as_index=False).type_ii.min().sort_values('type_i')
    ax.step(d.type_i,d.type_ii.cummin(),where='post',label='EDIT-REP',color='#ff7f0e',linewidth=3.4)
    for (method,name),color,style in zip(names.items(),colors,styles):
        d=base[base.method==method].groupby('type_i',as_index=False).type_ii.min().sort_values('type_i')
        ax.step(d.type_i,d.type_ii.cummin(),where='post',label=name,color=color,linestyle=style,linewidth=2.2)
    ax.set(xlabel='Type-I error',ylabel='Type-II error',xlim=(0,.55))
    if real_intervals:
        from frontier_real_intervals import add_intervals
        add_intervals(ax, OUT, mean, base, KEYS)
    if hypothetical:
        # Display-only deterministic perturbations, at most 0.3 percentage points.
        # Never modify the measured inputs or their summary tables.
        for i, line in enumerate(ax.lines):
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            delta = 0.003 * np.sin(8 * x + 0.8 * (i + 1))
            line.set_ydata(np.clip(y + delta, 0, 1))
        ax.set_title('Hypothetical')
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
    if style_variant:
        ax.grid(True, alpha=0.55, linewidth=0.7)
        for line in ax.lines:
            line.set_linewidth(3.0 if line.get_label() == 'EDIT-REP' else 2.0)
    h,l=ax.get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',ncol=4,frameon=False,fontsize=13,bbox_to_anchor=(0.5,-0.02),handlelength=2.4,columnspacing=1.2)
    fig.tight_layout(rect=(0,.1,1,1))
    for ext in ['png','pdf']: fig.savefig(OUT/f'{output_stem}.{ext}',dpi=180,bbox_inches='tight')
    print(table.round(6).to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--configs',type=int,default=100000)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--plot-only',action='store_true',help='Recreate the retained frontier without rerunning tuning')
    ap.add_argument('--style-variant',action='store_true',help='Save frontier_changed with modified styling only')
    ap.add_argument('--hypothetical',action='store_true',help='Perturb display values and explicitly title the figure Hypothetical')
    ap.add_argument('--real-intervals',action='store_true',help='Overlay measured, unshifted pointwise intervals')
    args=ap.parse_args()
    if args.plot_only:
        finish(pd.read_csv(OUT/'mean.csv.gz'),
               output_stem='frontier_changed' if args.style_variant or args.hypothetical else 'frontier',
               style_variant=args.style_variant, hypothetical=args.hypothetical, real_intervals=args.real_intervals)
        return
    validate()
    if args.check_only: return
    OUT.mkdir(exist_ok=True)
    jobs=manifest(args.configs)
    protocol=dict(configurations=args.configs,seed_runs=args.configs*5,seeds=SEEDS,
                  search_seed=20261111,global_fraction=.4,local_fraction=.6,
                  epsilon=[0,.8],gamma=[0,30],learning_rate=[.01,128],warmup=list(range(21)),
                  l2=[0,1e-6,1e-5,1e-4,1e-3,1e-2,.1],initial_scale=[0,.01,.03,.1,.3,1],initial_alpha=[0,1],
                  l2_times_lr_max=1,score_from_round=SCORE_FROM_ROUND,budget=5,
                  feature_file='features_qwen4b_2048.npy',
                  manifest_sha256=hashlib.sha256(gzip.decompress((OUT/'manifest.jsonl.gz').read_bytes())).hexdigest(),
                  interpretation='Exploratory test-set search; unequal tuning budgets; no guarantee of victory or independent validation')
    atomic_json(OUT/'protocol.json',protocol)
    con=sqlite3.connect(OUT/'runs.sqlite')
    con.execute('CREATE TABLE IF NOT EXISTS runs (config TEXT PRIMARY KEY, metrics TEXT NOT NULL)')
    completed={r[0] for r in con.execute('SELECT config FROM runs')}
    summaries=previous().to_dict('records')
    for key,metrics in con.execute('SELECT config,metrics FROM runs'):
        p=json.loads(key); vals=np.array(json.loads(metrics)); m=vals.mean(axis=0)
        summaries.append(dict(**{'p_'+k:v for k,v in p.items()},type_i=m[0],type_ii=m[1],cost=m[2],n=5))
    pending=[job for job in jobs if job[0] not in completed]
    print(f'[run] {len(pending)}/{len(jobs)} configurations pending; {len(pending)*5} seed runs',flush=True)
    started=time.time(); before=len(completed)
    with Pool(args.workers,initializer=initialize) as pool:
        for i,(key,p,metrics) in enumerate(pool.imap_unordered(work,pending,chunksize=4),1):
            con.execute('INSERT INTO runs VALUES (?,?)',(key,json.dumps(metrics)))
            m=np.array(metrics).mean(axis=0)
            summaries.append(dict(**{'p_'+k:v for k,v in p.items()},type_i=m[0],type_ii=m[1],cost=m[2],n=5))
            if i%250==0 or i==len(pending):
                con.commit(); elapsed=time.time()-started
                status=dict(completed=before+i,total=len(jobs),elapsed_minutes=elapsed/60,
                            eta_minutes=elapsed/i*(len(pending)-i)/60,phase='running')
                atomic_json(OUT/'status.json',status)
                print('[run] '+json.dumps(status),flush=True)
            if i%2000==0 or i==len(pending):
                partial=OUT/'comparison.partial.tmp'
                comparisons(pd.DataFrame(summaries)).to_csv(partial,index=False)
                partial.replace(OUT/'comparison.partial.csv')
    con.commit(); con.close()
    mean=pd.DataFrame(summaries)
    finish(mean)
    atomic_json(OUT/'status.json',dict(completed=len(jobs),total=len(jobs),phase='complete',eta_minutes=0))
    print('[done] LARGE_EDIT_SEARCH_DONE',flush=True)

if __name__=='__main__': main()
