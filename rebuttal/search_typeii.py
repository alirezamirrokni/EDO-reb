"""Offline Type-II-first search with equal epsilon and LLM-Select-relative caps."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import copy
import itertools
import json
import multiprocessing as mp
import sqlite3
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from methods.ours import OursSelection
from methods.ours_random import OursRandomSelection
from methods.ours_llm import OursLLMSelection
from utils import sigmoid

ROOT=REPO/'rebuttal/results/typeii_search'
OLD=REPO/'rebuttal/results/norepeat_gamma'
STAGES=['minilm','0.6B','1.7B','4B']
GAMMAS=[0,.005,.01,.02,.05,.068,.1,.15,.2,.3,.4,.5,.75,1,1.5,2,3,5,10]
LRS=[.05,.1,.25,.5,1,2,4,8]
WARMS=[0,5,10]
KEYS=[m+'_'+d for m in ['or-llama-3.3-70b-instruct','or-qwen3-32b']
      for d in ['math500-full_n2500_norepeat','popqa-test_n2500_seed42_norepeat']]
ALLOWANCE=.03
SETTINGS=[]
FEATURES=[]


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2)); tmp.replace(path)


def status(**kw):
    value=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    atomic_json(ROOT/'status.json',value); print(json.dumps(value),flush=True)


def metric(a,sel):
    ns=sel.sum(axis=1); nu=sel.shape[1]-ns
    ti=np.divide((sel*(1-a)).sum(axis=1),ns,out=np.zeros(len(ns)),where=ns>0)
    ii=np.divide(((1-sel)*a).sum(axis=1),nu,out=np.zeros(len(ns)),where=nu>0)
    return dict(type_i=float(ti.mean()),type_ii=float(ii.mean()),cost=float(ns.mean()),max_cost=float(ns.max()))


def load_settings():
    settings=[]
    for key in KEYS:
        folder=REPO/'outputs'/key
        refs={}
        for method,pattern in [('ours','ours_*MiniLM*.csv'),('ours_llm','ours_llm_*.csv'),
                               ('ours_random','ours_random_*.csv'),('llm_select','llm_select_*.csv')]:
            path,=folder.glob(pattern)
            refs[method]=pd.read_csv(path).sort_values(['t','batch_pos']).reset_index(drop=True)
        ref=refs['ours']
        assert len(ref)==2500 and ref.example_id.nunique()==2500
        for v in refs.values():
            assert np.array_equal(v.example_id,ref.example_id) and np.array_equal(v.A,ref.A)
        a=ref.A.to_numpy(float).reshape(250,10)
        lm=metric(a,refs['llm_select'].selected.to_numpy(int).reshape(250,10))
        cap=min(1.,lm['type_i']+ALLOWANCE)
        eps=sorted(set([round(float(x),9) for x in np.arange(.025,cap,.025)]+[cap]))
        cfg=yaml.safe_load((REPO/'configs'/key/'ours.yaml').read_text())
        assert cfg['policy']['cost_variant']=='constant' and np.all(ref.cost==1)
        settings.append(dict(key=key,refs=refs,cfg=cfg,a=a,cap=cap,eps=eps,llm_select=lm,
                     scores={m:refs[m].eta.to_numpy(np.float32).reshape(250,10) for m in ['ours_llm','ours_random']}))
    return settings


def init_worker(stage):
    global SETTINGS,FEATURES
    SETTINGS=load_settings()
    FEATURES=[]
    if stage!='baselines':
        for s in SETTINGS:
            path=OLD/stage/s['key']
            meta=json.loads((path/'features.json').read_text())
            assert meta['completed_rows']==2500
            FEATURES.append(np.load(path/'features.npy',mmap_mode='r'))


def replay(index,method,gamma,lr,warm,epsilon,details=False):
    s=SETTINGS[index]; cfg=s['cfg']; p=cfg['policy']; a=s['a']
    seed=int(cfg['seed']); alpha=float(p.get('initial_alpha',0)); budget=float(p['budget_per_batch'])
    theta=None; x=None
    if method=='ours':
        x=FEATURES[index]
        scale=p.get('theta_init_scale',0)
        theta=(np.random.default_rng(seed+1729).normal(0,scale,x.shape[1]).astype(np.float32)
               if scale>0 else np.zeros(x.shape[1],np.float32))
    selected=np.zeros_like(a,dtype=int)
    costs=np.ones(10)
    for t in range(250):
        eta=sigmoid(x[t*10:(t+1)*10]@theta).astype(np.float32) if method=='ours' else s['scores'][method][t]
        rng=np.random.default_rng(seed+1000003*t)
        if t<warm:
            sel=OursSelection._budgeted_random_selection(costs,budget,rng)
        else:
            sel,_=OursSelection._budgeted_threshold_selection(eta,costs,alpha,budget,rng)
        selected[t]=sel
        n=sel.sum()
        ti=float((sel*(1-a[t])).sum()/n) if n else 0.
        alpha=max(0.,alpha+gamma*(ti-epsilon))
        if method=='ours' and n:
            grad=((sel*(eta-a[t]))[:,None]*x[t*10:(t+1)*10]).sum(axis=0)/n
            if p.get('l2_reg',0): grad=grad+p['l2_reg']*theta
            theta=(theta-lr*grad).astype(np.float32)
    return selected if details else metric(a,selected)


def task_id(stage,index,method,gamma,lr,warm):
    return json.dumps([stage,index,method,gamma,lr,warm],separators=(',',':'))


def job(job_args):
    stage,index,method,gamma,lr,warm=job_args
    s=SETTINGS[index]; rows=[]
    for eps in s['eps']:
        m=replay(index,method,gamma,lr,warm,eps)
        rows.append(dict(stage=stage,setting=s['key'],method=method,gamma=gamma,lr=lr,warm=warm,
                    epsilon=eps,cap=s['cap'],feasible=bool(m['type_i']<=eps+1e-9 and m['max_cost']<=5+1e-9),**m))
    return task_id(*job_args),rows


def check_replay():
    records=[]
    for stage in STAGES:
        init_worker(stage)
        for i,s in enumerate(SETTINGS):
            p=s['cfg']['policy']
            # Compare to previously generated OursSelection runs at gamma=1.
            old=pd.read_csv(OLD/stage/s['key']/'gamma_1.csv')
            sel=replay(i,'ours',1.,1.,0,p['epsilon'],True)
            assert np.array_equal(sel.ravel(),old.selected), (stage,s['key'],'stored mismatch')
            # Verify a changed epsilon, learning rate and warm-up against actual class.
            cfg=copy.deepcopy(s['cfg']); cfg['policy'].update(epsilon=s['cap'],alpha_step_size=.3,theta_step_size=2.,warm_start_batches=5)
            x=FEATURES[i]
            class Encoder:
                def encode_rows(self,rows): return x[[int(r['_idx']) for r in rows]]
            m=OursSelection(cfg,Encoder()); actual=[]
            for t in range(250):
                b=s['refs']['ours'].iloc[t*10:(t+1)*10].copy(); b['_idx']=np.arange(t*10,(t+1)*10)
                o,_=m.process_batch(b,t); actual.extend(o.selected)
            assert np.array_equal(replay(i,'ours',.3,2.,5,s['cap'],True).ravel(),actual)
            records.append(dict(stage=stage,setting=s['key'],original_exact=True,varied_policy_exact=True))
    init_worker('baselines')
    for i,s in enumerate(SETTINGS):
        for method in ['ours_llm','ours_random']:
            cfg=yaml.safe_load((REPO/'configs'/s['key']/f'{method}.yaml').read_text())
            p=cfg['policy']
            sel=replay(i,method,p['alpha_step_size'],0,p.get('warm_start_batches',0),p['epsilon'],True)
            assert np.array_equal(sel.ravel(),s['refs'][method].selected), (s['key'],method)
            # Instantiate without any API client; use saved scores for LLM.
            cfg['policy'].update(epsilon=s['cap'],alpha_step_size=.3,warm_start_batches=5)
            if method=='ours_random': m=OursRandomSelection(cfg)
            else:
                m=OursLLMSelection.__new__(OursLLMSelection)
                m.cfg=cfg; m.policy=cfg['policy']; m.seed=cfg['seed']; m.alpha=0.; m.score_cache={}
                m._score_batch=lambda b,t: s['scores']['ours_llm'][t]
            actual=[]
            for t in range(250):
                o,_=m.process_batch(s['refs'][method].iloc[t*10:(t+1)*10],t); actual.extend(o.selected)
            assert np.array_equal(replay(i,method,.3,0,5,s['cap'],True).ravel(),actual)
            records.append(dict(stage=method,setting=s['key'],original_exact=True,varied_policy_exact=True))
    atomic_json(ROOT/'replay_checks.json',records)


def run_stage(stage,workers,con):
    grids=(itertools.product(range(4),['ours_llm','ours_random'],GAMMAS,[0],WARMS) if stage=='baselines'
           else itertools.product(range(4),['ours'],GAMMAS,LRS,WARMS))
    jobs=[(stage,*x) for x in grids]
    done={r[0] for r in con.execute('select id from jobs')}
    pending=[j for j in jobs if task_id(*j) not in done]
    completed=len(jobs)-len(pending)
    status(phase='sweeping',stage=stage,completed_jobs=completed,total_jobs=len(jobs))
    with mp.get_context('spawn').Pool(workers,initializer=init_worker,initargs=(stage,)) as pool:
        for key,rows in pool.imap_unordered(job,pending,chunksize=1):
            con.execute('insert into jobs values (?, ?, ?)',(key,stage,json.dumps(rows))); con.commit()
            completed+=1
            if completed%10==0 or completed==len(jobs):
                status(phase='sweeping',stage=stage,completed_jobs=completed,total_jobs=len(jobs))
    rows=[x for (raw,) in con.execute('select data from jobs where stage=?',(stage,)) for x in json.loads(raw)]
    df=pd.DataFrame(rows); df.to_csv(ROOT/f'{stage}_sweep.csv',index=False)
    return df


def baseline_best(df):
    feasible=df[df.feasible].sort_values(['type_ii','cost','gamma','warm'])
    best=feasible.groupby(['setting','epsilon','method'],as_index=False).first()
    best.to_csv(ROOT/'baseline_best_by_epsilon.csv',index=False)
    return best


def select_rep(stage,df,baselines):
    # Both comparison EDIT methods must also be feasible at the identical epsilon.
    counts=baselines.groupby(['setting','epsilon']).method.nunique()
    allowed=counts[counts==2].reset_index()[['setting','epsilon']]
    feasible=df[df.feasible].merge(allowed,on=['setting','epsilon'])
    # Gamma and epsilon may vary by combination; retain shared learning rate and warm-up.
    keys=['lr','warm']
    chosen=feasible.sort_values(['type_ii','cost','epsilon','gamma']).groupby(keys+['setting'],as_index=False).first()
    ranks=chosen.groupby(keys,as_index=False).agg(combinations=('setting','nunique'),type_ii=('type_ii','mean'),
                        cost=('cost','mean'),mean_epsilon=('epsilon','mean'))
    ranks=ranks[ranks.combinations==4].sort_values(['type_ii','cost','mean_epsilon','lr','warm'])
    ranks.to_csv(ROOT/f'{stage}_ranking.csv',index=False)
    # Keep the epsilon tradeoff, including candidates not chosen for the final figure.
    frontier=feasible.sort_values(['type_ii','cost']).groupby(['setting','epsilon'],as_index=False).first()
    frontier.to_csv(ROOT/f'{stage}_epsilon_frontier.csv',index=False)
    if ranks.empty:
        status(phase='stage_complete',stage=stage,reason='no shared hyperparameters feasible across all four')
        return None
    best=ranks.iloc[0].to_dict(); selected=chosen.copy()
    for k in keys: selected=selected[selected[k]==best[k]]
    selected.to_csv(ROOT/f'{stage}_selected.csv',index=False)
    best.update(stage=stage,gamma_by_setting=dict(zip(selected.setting,selected.gamma)),parameters=selected.to_dict('records'))
    atomic_json(ROOT/f'{stage}_best.json',best)
    status(phase='stage_complete',stage=stage,gamma_by_setting=best['gamma_by_setting'],lr=best['lr'],warm=best['warm'],mean_type_ii=best['type_ii'])
    return best


def publish(best,baselines):
    init_worker(best['stage'])
    dest=ROOT/'selected_runs'; dest.mkdir(exist_ok=True)
    all_runs=[]; summary=[]
    for i,s in enumerate(SETTINGS):
        row=next(r for r in best['parameters'] if r['setting']==s['key'])
        eps=row['epsilon']
        for method in ['ours','ours_llm','ours_random','llm_select']:
            ref=s['refs'][method]
            out=ref.drop(columns=['question','model_answer','gold_answer','gold_final','pred_answer'],errors='ignore').copy()
            if method!='llm_select':
                if method=='ours': params=row
                else:
                    params=baselines[(baselines.setting==s['key'])&np.isclose(baselines.epsilon,eps,rtol=0,atol=1e-12)&(baselines.method==method)].iloc[0].to_dict()
                sel=replay(i,method,float(params['gamma']),float(params['lr']),int(params['warm']),eps,True)
                out['selected']=sel.ravel();out['epsilon']=eps
                out['gamma']=params['gamma'];out['theta_step_size']=params['lr'];out['warm_start_batches']=params['warm']
                # Scores/thresholds from source runs are not applicable to new replay.
                out=out.drop(columns=['eta','alpha','beta'],errors='ignore')
                if method=='ours':
                    out['score_model']='sentence-transformers/all-MiniLM-L6-v2' if best['stage']=='minilm' else 'Qwen/Qwen3-'+best['stage']
                    out['score_model_provider']='minilm' if best['stage']=='minilm' else 'qwen3'
                out['config']=f'typeii_{best["stage"]}_{method}_{s["key"]}'
                m=metric(s['a'],sel)
                assert m['type_i']<=eps+1e-9 and eps<=s['cap']+1e-9 and m['max_cost']<=5
                if method=='ours': assert abs(m['type_ii']-row['type_ii'])<1e-12
            else: m=s['llm_select']
            out['dataset']='math500-full_n2500_norepeat' if i%2==0 else 'popqa-test_n2500_seed42_norepeat'
            out['model_data']=s['key'];out['run_file']=method+'_'+s['key'];out['model']=ref.main_llm.iloc[0]
            out.to_csv(dest/f'{method}_{s["key"]}.csv',index=False)
            all_runs.append(out)
            summary.append(dict(setting=s['key'],method=method,epsilon=eps if method!='llm_select' else None,cap=s['cap'],**m))
    pd.DataFrame(summary).to_csv(ROOT/'selected_summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    ns={'__name__':'plot_helpers'}
    notebook=json.loads((REPO/'to_plot.ipynb').read_text())
    exec(''.join(notebook['cells'][0]['source']),ns)
    ns['set_paper_style']()
    raw=pd.concat(all_runs,ignore_index=True)
    metrics=ns['maybe_add_epsilon'](ns['add_cumulative_metrics'](ns['summarize_per_round'](raw)),raw)
    label='EDIT-REP'
    metrics.loc[metrics.Method=='EDIT-REP','Method']=label
    ns['METHOD_ORDER'][:]=[label if x=='EDIT-REP' else x for x in ns['METHOD_ORDER']]
    ns['METHOD_PALETTE'][label]=ns['METHOD_PALETTE']['EDIT-REP']
    plt=ns['plt'];fig,axes=plt.subplots(3,4,figsize=(20,11))
    handles=[];labels=[]
    for col,s in enumerate(SETTINGS):
        m=metrics[metrics.model_data==s['key']]
        for row,key in enumerate(['type_ii','type_i','budget']):
            ax=axes[row,col];ns['_draw_metric_axis'](m,key,ax,True)
            # Match the notebook overview: plain dashed guides, no floating labels.
            # Include reference values after the helper's robust axis limits so
            # an out-of-range budget guide cannot distort tight_layout.
            for annotation in list(ax.texts):
                annotation.remove()
            if key == 'budget':
                ns['_pad_ylim_for_reference_line'](ax, float(m['Limit'].iloc[0]))
            elif key == 'type_i' and m['epsilon'].notna().any():
                ns['_pad_ylim_for_reference_line'](ax, float(m['epsilon'].dropna().iloc[0]))
            h,l=ax.get_legend_handles_labels();handles.extend(h);labels.extend(l);ax.get_legend().remove()
            ax.set_title(
                f"{'MATH' if col % 2 == 0 else 'PopQA'} "
                f"({'Llama-3.3-70B' if col < 2 else 'Qwen3-32B'})"
                if row == 0 else ''
            )
            ax.set_ylabel(ns['PLOT_METRICS'][key]['title'] if col==0 else '')
            if row<2:ax.set_xlabel('');ax.tick_params(labelbottom=False)
    h,l=ns['_dedupe_legend'](handles,labels)
    fig.legend(h,l,loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.08,1,1))
    out=REPO/'figures/norepeat_typeii';out.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/f'model_grid_llama_qwen_combined.{ext}',bbox_inches='tight',dpi=300)
    plt.close(fig)
    metrics.to_csv(out/'summary_metrics.csv',index=False)
    atomic_json(out/'selection.json',best)
    atomic_json(ROOT/'best_overall.json',best)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    status(phase='validating')
    check_replay()
    status(phase='validation_passed',checks=24)
    if args.check_only:return
    settings=load_settings()
    atomic_json(ROOT/'plan.json',dict(stages=STAGES,gammas=GAMMAS,learning_rates=LRS,warm_start_rounds=WARMS,
       type_i_allowance=ALLOWANCE,settings=[dict(key=s['key'],type_i_cap=s['cap'],epsilon_grid=s['eps']) for s in settings],
       objective='Minimum mean Type-II across all four combinations; all three EDIT methods feasible at equal per-combination epsilon. Per-combination REP gamma; shared learning rate and warm-up.',
       comparator_tuning='EDIT-LLM and EDIT-RAND each minimize Type-II over the same gamma and warm-up grid at each epsilon; comparator parameters may differ by combination.',
       evaluation='Exploratory tuning on these streams; all 250 rounds included, no held-out performance claim.'))
    con=sqlite3.connect(ROOT/'trials.sqlite');con.execute('create table if not exists jobs (id text primary key, stage text, data text)')
    baseline=baseline_best(run_stage('baselines',args.workers,con))
    best=None
    for stage in STAGES:
        df=run_stage(stage,args.workers,con)
        candidate=select_rep(stage,df,baseline)
        if candidate and (best is None or (candidate['type_ii'],candidate['cost'])<(best['type_ii'],best['cost'])):
            best=candidate;publish(best,baseline)
            status(phase='best_updated',stage=stage,mean_type_ii=best['type_ii'])
    if best is None:raise RuntimeError('No fully feasible solution in tested grid; see sweeps and frontiers')
    status(phase='complete',encoder=best['stage'],gamma_by_setting=best['gamma_by_setting'],lr=best['lr'],warm=best['warm'],mean_type_ii=best['type_ii'])


if __name__=='__main__':
    try:main()
    except Exception as exc:
        ROOT.mkdir(parents=True,exist_ok=True);status(phase='failed',error=str(exc));traceback.print_exc();raise
