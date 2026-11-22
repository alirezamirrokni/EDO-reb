"""Locked full-task review-quality study. No historical result files are modified."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,hashlib,time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from scipy.stats import binomtest
from engine import Setting,SETTINGS
import run_baselines_norepeat as nr
from run_alpha_controlled import GAMMAS,ALPHAS,validate
from methods.ours import OursSelection
from paper_analysis.common import compute_metrics

ROOT=Path(__file__).resolve().parent
EPS=[.2,.4,.6];PRIMARY=.4;ORDER_SEEDS=list(range(1000,1030))
METHODS=['adaptive','frozen','fixed','topk','dev_cost_matched']
OUT=None

def save(path,value):path.write_text(json.dumps(value,indent=2))
def hashfile(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def groups(text):return np.array([hashlib.sha256(str(x).strip().encode()).hexdigest() for x in text])
def simulate(y,score,seed,alpha,gamma,eps,budget=5,check=False):
    # Retain every item; final partial batch has the same budget ceiling.
    idx=np.random.default_rng(seed).permutation(len(y));rows=[];picks=[]
    for t,start in enumerate(range(0,len(y),10)):
        j=idx[start:start+10];a=y[j];s=score[j]
        sel,_=OursSelection._budgeted_threshold_selection(s,np.ones(len(j)),alpha,budget,
            np.random.default_rng(7+1000003*t))
        n=sel.sum();fp=float((sel*(1-a)).sum());miss=float(((1-sel)*a).sum())
        ti=fp/n if n else 0.;ii=miss/(len(j)-n) if len(j)>n else 0.
        rows.append([ti,ii,n,fp,miss,len(j),float(a.sum())]);picks.extend(sel)
        alpha=max(0.,alpha+gamma*(ti-eps))
    v=np.array(rows);selected=v[:,2].sum();false=v[:,3].sum()
    m=dict(type_i=v[:,0].mean(),type_ii=v[:,1].mean(),cost=v[:,2].mean(),
        selected=selected,false_reviews=false,missed=v[:,4].sum(),records=v[:,5].sum(),
        rounds=len(v),coverage=selected/len(y),pooled_precision=(selected-false)/selected if selected else np.nan)
    if check:
        df=pd.DataFrame(dict(t=np.arange(len(y))//10,A=y[idx],selected=picks,cost=1.))
        ref=compute_metrics(df)
        np.testing.assert_allclose([m['type_i'],m['type_ii'],m['cost']],
            [ref['type_i'],ref['type_ii'],ref['mean_review_cost']],atol=1e-12)
    return m

def scores_fit(x,y,fit,cal,rep):
    rng=np.random.default_rng(20261116+rep)
    use=fit if rep==0 else rng.choice(fit,len(fit),replace=True)
    if len(np.unique(y[use]))<2:raise ValueError('Single-class training resample')
    model=LogisticRegression(C=1.,max_iter=2000,random_state=7).fit(x[use],y[use])
    if model.n_iter_.max()>=2000:raise ValueError('Probe failed to converge')
    z=model.decision_function(x)
    calibration=LogisticRegression(C=1.,max_iter=2000).fit(z[cal,None],y[cal])
    return calibration.predict_proba(z[:,None])[:,1].astype(np.float32),model,calibration

def configs(y,score):
    def avg(a,g,e,b=5):
        rows=[simulate(y,score,s,a,g,e,b) for s in [101,102,103]]
        return {k:float(np.mean([r[k] for r in rows])) for k in ['type_i','type_ii','cost','coverage']}
    fixed=[dict(alpha=a,gamma=0.,budget=5,**avg(a,0,.4)) for a in ALPHAS]
    tops=[dict(alpha=0.,gamma=0.,budget=b,**avg(0,0,.4,b)) for b in [1,2,3,4,5]]
    def pick(rows,e):
        feasible=[r for r in rows if r['type_i']<=e]
        r=min(feasible or rows,key=lambda r:(0 if feasible else r['type_i'],r['type_ii'],r['cost'],r.get('gamma',0),r['alpha']))
        return dict(r,validation_feasible=bool(feasible))
    result=[];grid=[]
    for eps in EPS:
        initial=pick(fixed,eps)
        adaptive=[dict(alpha=initial['alpha'],gamma=g,budget=5,**avg(initial['alpha'],g,eps)) for g in GAMMAS]
        chosen=pick(adaptive,eps)
        match=min(fixed,key=lambda r:(abs(r['cost']-chosen['cost']),r['type_ii'],r['alpha']))
        result.append(dict(epsilon=eps,policies={'adaptive':chosen,'frozen':initial,'fixed':pick(fixed,eps),
                         'topk':dict(tops[-1],validation_feasible=tops[-1]['type_i']<=eps),
                         'dev_cost_matched':dict(match,validation_feasible=match['type_i']<=eps)}))
        grid.extend([dict(epsilon=eps,kind='adaptive',**r) for r in adaptive])
    grid.extend([dict(kind='fixed',**r) for r in fixed])
    return result,grid

def load_all():
    data=[]
    for model in ['llama','qwen']:
        for name in SETTINGS:
            s=Setting(name,model)
            data.append((f'{model}_{name}_MiniLM',s.pool,s.features,None))
    for enc in ['1.7B','minilm']:
        nr.FEATURE_ROOT=ROOT/'results/norepeat_gamma'/enc
        for name in nr.SETTINGS:
            s=nr.NoRepeatSetting(name)
            data.append((s.key+'_'+enc,s.pool,s.features,None))
    r=ROOT/'results/ragtruth'
    tr=pd.read_csv(r/'data_train.csv');te=pd.read_csv(r/'data_test.csv')
    assert not set(tr.source_id)&set(te.source_id)
    data.append(('RAGTruth_Qwen4B',pd.concat([tr,te],ignore_index=True),np.load(r/'features_qwen4b_2048.npy'),len(tr)))
    return data

def prepare():
    jobs=[];inventory=[]
    for key,d,x,official in load_all():
        folder=OUT/key;folder.mkdir(exist_ok=True)
        y=d.A.to_numpy(int);g=groups(d.source_id if official else d.question)
        unique=np.unique(g);rng=np.random.default_rng(20261116);rng.shuffle(unique)
        mapping={v:i%5 for i,v in enumerate(unique)};fold=np.array([mapping[v] for v in g])
        np.save(folder/'features.npy',x)
        np.save(folder/'labels.npy',y);np.save(folder/'groups.npy',g)
        d[['example_id','A']].to_csv(folder/'records.csv',index=False)
        folds=[]
        for f in ([0] if official else range(5)):
            if official:
                eligible=np.arange(len(d))<official
                test=np.flatnonzero(~eligible)
                fit=np.flatnonzero(eligible&(fold<3));cal=np.flatnonzero(eligible&(fold==3));val=np.flatnonzero(eligible&(fold==4))
            else:
                test=np.flatnonzero(fold==f);cal=np.flatnonzero(fold==(f+1)%5);val=np.flatnonzero(fold==(f+2)%5)
                fit=np.flatnonzero(~np.isin(fold,[f,(f+1)%5,(f+2)%5]))
            partitions=[fit,cal,val,test]
            for i,a in enumerate(partitions):
                for b in partitions[i+1:]:assert not set(g[a])&set(g[b])
            split=dict(fit=fit.tolist(),calibration=cal.tolist(),validation=val.tolist(),test=test.tolist())
            save(folder/f'split_{f}.json',split)
            folds.append(dict(fold=f,**{k:len(v) for k,v in split.items()}))
            for rep in range(5):jobs.append((key,f,rep))
        inventory.append(dict(setting=key,records=len(d),encoder='Qwen4B' if official else key.rsplit('_',1)[-1],
            split='official RAGTruth' if official else 'five-fold grouped cross-fitting',folds=folds,
            feature_hash=hashfile(folder/'features.npy')))
    save(OUT/'inventory.json',inventory)
    save(OUT/'jobs.json',jobs)
    return jobs

def tune(job):
    key,fold,rep=job;folder=OUT/key;prefix=f'fold{fold}_probe{rep}'
    path=folder/f'{prefix}_lock.json'
    if path.exists():return job
    split=json.loads((folder/f'split_{fold}.json').read_text());fit=np.array(split['fit']);cal=np.array(split['calibration']);val=np.array(split['validation'])
    x=np.load(folder/'features.npy',mmap_mode='r');y=np.load(folder/'labels.npy')
    scores,model,calibrator=scores_fit(x,y,fit,cal,rep)
    np.save(folder/f'{prefix}_scores.npy',scores)
    np.savez(folder/f'{prefix}_probe.npz',coef=model.coef_,intercept=model.intercept_,
             cal_coef=calibrator.coef_,cal_intercept=calibrator.intercept_)
    chosen,grid=configs(y[val],scores[val])
    pd.DataFrame(grid).to_csv(folder/f'{prefix}_validation.csv',index=False)
    save(path,dict(job=job,configs=chosen,feedback_labels_fit=len(fit),feedback_labels_calibration=len(cal),
        feedback_labels_validation=len(val),scores_hash=hashfile(folder/f'{prefix}_scores.npy')))
    return job

def test(job):
    key,fold,rep=job;folder=OUT/key;prefix=f'fold{fold}_probe{rep}'
    output=folder/f'{prefix}_test.csv'
    if output.exists():return job
    split=json.loads((folder/f'split_{fold}.json').read_text());idx=np.array(split['test'])
    y=np.load(folder/'labels.npy')[idx];s=np.load(folder/f'{prefix}_scores.npy')[idx]
    group=np.load(folder/'groups.npy')[idx]
    lock=json.loads((folder/f'{prefix}_lock.json').read_text());rows=[]
    scenarios=[('order',seed,np.arange(len(y))) for seed in ORDER_SEEDS]
    # Cluster bootstrap of test sources, full policy replay. Not a no-repeat deployment claim.
    if rep==0:
        unique=np.unique(group);members={g:np.flatnonzero(group==g) for g in unique}
        for seed in range(2000,2200):
            sampled=np.random.default_rng(seed).choice(unique,len(unique),replace=True)
            ii=np.concatenate([members[g] for g in sampled]);scenarios.append(('records',seed,ii))
    for cfg in lock['configs']:
        eps=cfg['epsilon']
        for axis,seed,ii in scenarios:
            for method,p in cfg['policies'].items():
                m=simulate(y[ii],s[ii],seed,p['alpha'],p['gamma'],eps,p['budget'],check=(seed==1000))
                rows.append(dict(setting=key,fold=fold,probe=rep,epsilon=eps,axis=axis,seed=seed,
                    method=method,validation_feasible=p['validation_feasible'],
                    offline_labels=lock['feedback_labels_fit']+lock['feedback_labels_calibration']+lock['feedback_labels_validation'],**m))
    pd.DataFrame(rows).to_csv(output,index=False)
    return job

def paired_ci(delta,seed=20261116):
    rng=np.random.default_rng(seed);n=len(delta)
    boot=delta[rng.integers(n,size=(20000,n))].mean(axis=1)
    null=(rng.choice([-1,1],size=(50000,n))*delta).mean(axis=1)
    nz=delta[abs(delta)>1e-12]
    return dict(effect=delta.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),
        p=(1+(abs(null)>=abs(delta.mean())-1e-15).sum())/50001,
        sign_p=binomtest(int((nz>0).sum()),len(nz)).pvalue if len(nz) else 1.,wins=(delta>1e-12).mean())

def summarize():
    raw=pd.concat([pd.read_csv(p) for p in OUT.glob('*/fold*_probe*_test.csv')],ignore_index=True)
    # Aggregate held-out folds by number of rounds; no treating folds as independent datasets.
    def aggregate(g):
        rounds=g['rounds'].sum();sel=g.selected.sum();fp=g.false_reviews.sum();n=g.records.sum()
        return pd.Series(dict(type_i=np.average(g.type_i,weights=g['rounds']),type_ii=np.average(g.type_ii,weights=g['rounds']),
            cost=sel/rounds,coverage=sel/n,pooled_precision=(sel-fp)/sel if sel else np.nan,
            pooled_type_i=fp/sel if sel else np.nan,validation_feasible=g.validation_feasible.all(),
            offline_labels=g.offline_labels.sum(),selected=sel,records=n))
    keys=['setting','probe','epsilon','axis','seed','method']
    agg=raw.groupby(keys).apply(aggregate,include_groups=False).reset_index()
    agg.to_csv(OUT/'aggregated_by_replicate.csv',index=False)
    effects=[];quality=[]
    for (setting,eps),g in agg[(agg.probe==0)&(agg.axis=='order')].groupby(['setting','epsilon']):
        a=g[g.method=='adaptive'].set_index('seed')
        for method in ['frozen','fixed','topk','dev_cost_matched']:
            b=g[g.method==method].set_index('seed').reindex(a.index)
            effects.append(dict(setting=setting,epsilon=eps,comparison=method,
                cost_difference=(a.cost-b.cost).mean(),type_i_difference=(a.type_i-b.type_i).mean(),
                **paired_ci((b.type_ii-a.type_ii).to_numpy())))
        for method,h in g.groupby('method'):
            rng=np.random.default_rng(20261116);v=h.type_i.to_numpy();boots=v[rng.integers(len(v),size=(20000,len(v)))].mean(axis=1)
            quality.append(dict(setting=setting,epsilon=eps,method=method,type_i=h.type_i.mean(),
                type_i_upper95=np.quantile(boots,.95),order_feasible=np.quantile(boots,.95)<=eps,
                validation_feasible=h.validation_feasible.all(),type_ii=h.type_ii.mean(),cost=h.cost.mean(),
                pooled_precision=h.pooled_precision.mean(),coverage=h.coverage.mean(),offline_labels=h.offline_labels.iloc[0]))
    eff=pd.DataFrame(effects);eff['p_holm']=np.nan
    ix=eff[(eff.epsilon==PRIMARY)&(eff.comparison=='frozen')].sort_values('p').index
    eff.loc[ix,'p_holm']=np.minimum(1,np.maximum.accumulate(eff.loc[ix,'p'].to_numpy()*np.arange(len(ix),0,-1)))
    eff.to_csv(OUT/'paired_effects.csv',index=False)
    q=pd.DataFrame(quality)
    records=[]
    for (setting,eps),g in agg[(agg.probe==0)&(agg.axis=='records')].groupby(['setting','epsilon']):
        a=g[g.method=='adaptive'].set_index('seed');b=g[g.method=='frozen'].set_index('seed').reindex(a.index)
        delta=(b.type_ii-a.type_ii).to_numpy()
        records.append(dict(setting=setting,epsilon=eps,effect_boot_mean=delta.mean(),
            effect_lower=np.quantile(delta,.025),effect_upper=np.quantile(delta,.975),
            adaptive_type_i_upper95=np.quantile(a.type_i,.95),record_feasible=np.quantile(a.type_i,.95)<=eps))
    rec=pd.DataFrame(records);rec.to_csv(OUT/'record_bootstrap.csv',index=False)
    probe=agg[agg.axis=='order'].groupby(['setting','probe','epsilon','method'])[['type_i','type_ii','cost']].mean().reset_index()
    probe.to_csv(OUT/'probe_training_sensitivity.csv',index=False)
    q.to_csv(OUT/'quality_cost_precision_coverage.csv',index=False)
    primary=eff[(eff.epsilon==PRIMARY)&(eff.comparison=='frozen')].merge(
        q[(q.epsilon==PRIMARY)&(q.method=='adaptive')],on=['setting','epsilon']).merge(rec,on=['setting','epsilon'])
    primary['joint_order_supported']=(primary.ci_low>0)&(primary.p_holm<.05)&primary.order_feasible&primary.validation_feasible
    primary['joint_record_supported']=(primary.effect_lower>0)&primary.record_feasible
    primary.to_csv(OUT/'primary_joint_results.csv',index=False)
    tex=[r'\begin{tabular}{lrrrrrrr}',r'\toprule',r'Setting & $\Delta$ [95\% CI] (pp) & $p_{Holm}$ & Type-I & Upper95 & Cost & Precision & Coverage \\',r'\midrule']
    for _,r in primary.iterrows():
        label=r.setting.replace('_',r'\_')
        tex.append(f'{label} & {100*r.effect:.2f} [{100*r.ci_low:.2f},{100*r.ci_high:.2f}] & {r.p_holm:.4g} & {100*r.type_i:.2f} & {100*r.type_i_upper95:.2f} & {r.cost:.2f} & {100*r.pooled_precision:.2f} & {100*r.coverage:.2f}'+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}']
    (OUT/'primary_table.tex').write_text('\n'.join(tex))
    n=int(primary.joint_order_supported.sum());nrec=int(primary.joint_record_supported.sum());total=len(primary)
    paragraph=(f'At the prespecified tolerance 0.40, {n}/{total} settings jointly showed positive paired Type-II effects, Holm-adjusted significance, development feasibility and an ordering-bootstrap upper Type-I bound below the tolerance. '
        f'The separate source-bootstrap diagnostic supported positive effects and Type-I feasibility in {nrec}/{total} settings. '
        'A shared frozen calibrated probe, initial threshold and review budget isolated the threshold update. '
        'All held-out examples and rounds were scored; separate tables report cost, pooled precision, coverage, five probe-training variants and tolerance sensitivities. '
        'Ordering intervals are conditional on fixed records and training; source bootstrap reruns policies and includes ordering variation; probe bootstrap is reported separately. '
        'Historical benchmark reuse and cross-fitting prevent an untouched-test confirmation claim. Unequal realized costs preclude interpreting a Type-II gain alone as improved efficiency.')
    (OUT/'academic_paragraph.txt').write_text(paragraph)
    response=(f'We isolated the alpha update using identical frozen calibrated probes, initial thresholds, budgets and tie-breaking. Configurations were locked on separate validation data before full-task evaluation. At epsilon=0.40, {n}/{total} settings jointly show a positive Type-II effect, Holm-adjusted significance and validation/order-based Type-I feasibility. Source-bootstrap diagnostics support positive effects and feasibility in {nrec}/{total}. We report paired effect sizes, CIs, costs, pooled precision and coverage; all rounds and feedback are accounted for. Uncertainty from 30 orders, 200 source resamples and five probe fits is reported separately, with epsilon=0.20/0.60 sensitivity. These are locked reanalyses of historical pools, not untouched-test confirmation. Lower Type-II alone is not a cost-efficiency claim.')
    assert len(response)<1000
    (OUT/'openreview_response.md').write_text(response)
    save(OUT/'complete.json',dict(status='complete',rows=len(raw),settings=total))
    print('COMPLETE',n,total,flush=True)

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args();OUT=a.output.resolve()
    OUT.mkdir(exist_ok=True,parents=True)
    if not (OUT/'protocol.json').exists():
        save(OUT/'protocol.json',dict(primary_epsilon=PRIMARY,sensitivity=EPS,alpha_grid=ALPHAS,gamma_grid=GAMMAS,
            order_seeds=ORDER_SEEDS,development_seeds=[101,102,103],source_bootstrap_seeds=list(range(2000,2200)),
            probe_bootstrap_replicates=5,split_seed=20261116,all_rounds=True,all_records=True,
            budget=5,primary_comparison='adaptive vs same-alpha frozen',
            calibration='disjoint Platt logistic C=1; frozen at test',
            accounting='equal offline fit/calibration/validation labels; test feedback only selected items; all costs reported',
            conditional_inference='order bootstrap/sign-flip; source bootstrap with policy replay; separate probe-training bootstrap',
            caveat='Historical data reuse; locked reanalysis not untouched test. Existing grids used; fit probe and Platt C fixed at 1.',
            feature_coverage='MiniLM all four primary tasks both generators; Qwen1.7B and MiniLM no-repeat MATH/PopQA both; Qwen4B RAGTruth',
            code_hash=hashfile(__file__)))
        validate();jobs=prepare()
    else:
        protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['code_hash']==hashfile(__file__),'Code changed after lock'
        jobs=json.loads((OUT/'jobs.json').read_text())
    # Barrier: every fit/calibration/configuration is locked before any test metrics are evaluated.
    for phase,fn in [('tune',tune),('test',test)]:
        with Pool(a.workers) as pool:
            for i,job in enumerate(pool.imap_unordered(fn,jobs),1):
                print(phase,i,len(jobs),job,flush=True)
                save(OUT/'status.json',dict(phase=phase,completed=i,total=len(jobs),time=time.time()))
        if phase=='tune':save(OUT/'all_locks.json',{str(p.relative_to(OUT)):hashfile(p) for p in OUT.glob('*/*_lock.json')})
    summarize()

if __name__=='__main__':main()
