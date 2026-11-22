"""Locked same-initial-alpha ablation; historical data, ordering-conditional inference.

Run: python rebuttal/run_alpha_controlled.py --output rebuttal/results/alpha_controlled_v1
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import binomtest
import run_baselines_norepeat as nr
from methods.ours import OursSelection
from paper_analysis.common import compute_metrics
from utils import sigmoid

ROOT=Path(__file__).resolve().parent
ALPHAS=np.round(np.arange(0,.981,.02),2).tolist()
GAMMAS=[0.,.005,.01,.02,.05,.068,.1,.15,.2,.3,.4,.5,.75,1.,1.5,2.,3.,5.,10.]
EPS=[.2,.4,.6]
DEV_SEEDS=[101,102,103]
TEST_SEEDS=list(range(1000,1030))
NAMES=['Llama-3.3-70B / MATH','Qwen3-32B / MATH']

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False))

def order(n,seed):
    # Drop a fixed trailing remainder BEFORE permutation: all seeds use identical examples.
    n=n//10*10
    return np.random.default_rng(seed).permutation(n).reshape(-1,10)

def replay(y,scores,seed,alpha0,gamma,epsilon,details=False,budget=5):
    stream=order(len(y),seed); a=y[stream]; eta=scores[stream]
    selections=[]; alphas=[]; alpha=float(alpha0)
    for t in range(len(stream)):
        alphas.append(alpha)
        sel,_=OursSelection._budgeted_threshold_selection(eta[t],np.ones(10),alpha,budget,
                np.random.default_rng(7+1000003*t))
        selections.append(sel)
        rate=float((sel*(1-a[t])).sum()/sel.sum()) if sel.sum() else 0.
        alpha=max(0.,alpha+gamma*(rate-epsilon))
    sel=np.asarray(selections); ns=sel.sum(axis=1)
    ti=np.divide((sel*(1-a)).sum(axis=1),ns,out=np.zeros(len(ns)),where=ns>0)
    ii=np.divide(((1-sel)*a).sum(axis=1),10-ns,out=np.zeros(len(ns)),where=ns<10)
    metrics=dict(type_i=float(ti.mean()),type_ii=float(ii.mean()),cost=float(ns.mean()))
    if details:
        df=pd.DataFrame(dict(t=np.repeat(np.arange(len(stream)),10),A=a.ravel(),selected=sel.ravel(),cost=1.))
        ref=compute_metrics(df)
        np.testing.assert_allclose([metrics['type_i'],metrics['type_ii'],metrics['cost']],
                                   [ref['type_i'],ref['type_ii'],ref['mean_review_cost']],atol=1e-12)
        return metrics,stream,sel,np.asarray(alphas),np.stack([ti,ii,ns],axis=1)
    return metrics

def validate():
    # Production controller replay with theta_lr=0: ensures exact update and ties.
    x=np.array([[0.],[0.],[1.],[-1.],[.5]]*6,dtype=np.float32)
    y=np.array([1,0,1,0,1,0]*5);score=sigmoid(x[:,0]).astype(np.float32)
    class Encoder:
        def encode_rows(self,rows):return x[[r['pos'] for r in rows]]
    cfg={'seed':7,'policy':dict(budget_per_batch=5,epsilon=.4,alpha_step_size=.3,
          theta_step_size=0.,initial_alpha=.4,warm_start_batches=0,l2_reg=0.)}
    obj=OursSelection(cfg,Encoder(),state={'theta':[1.],'alpha':.4})
    m,st,sel,aa,_=replay(y,score,1000,.4,.3,.4,True)
    for t,idx in enumerate(st):
        frame=pd.DataFrame({'pos':idx,'A':y[idx],'cost':1.})
        out,_=obj.process_batch(frame,t)
        np.testing.assert_array_equal(sel[t],out.selected)
        np.testing.assert_allclose(aa[t],out.alpha.iloc[0])
    a=replay(y,score,1000,.4,0,.4,True)
    b=replay(y,score,1000,.4,0,.6,True)
    np.testing.assert_array_equal(a[2],b[2])
    assert max(a[2].sum(axis=1))<=5
    print('PASS production update, threshold tie-breaking, paper metrics, frozen-alpha invariance',flush=True)

def average(y,score,alpha,gamma,eps,budget=5):
    rows=[replay(y,score,s,alpha,gamma,eps,budget=budget) for s in DEV_SEEDS]
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}

def choose(rows,eps):
    feasible=[r for r in rows if r['type_i']<=eps]
    chosen=min(feasible or rows,key=lambda r:(0 if feasible else r['type_i'],r['type_ii'],r['cost'],r.get('gamma',0),r.get('alpha',0)))
    return dict(chosen,validation_feasible=bool(feasible))

def prepare(out):
    audit=pd.read_csv(ROOT.parent/'human_audit_app/data/audit_samples_50.csv')
    bundles=[]; locked=[]; searches=[]; manifest=[]
    for name in NAMES:
        for encoder in ['1.7B','minilm']:
            nr.FEATURE_ROOT=ROOT/'results/norepeat_gamma'/encoder
            s=nr.NoRepeatSetting(name)
            key=s.key+'_'+encoder
            # Existing diagnostic outer split. Within development group duplicate question text.
            group=pd.util.hash_pandas_object(s.pool.question,index=False).to_numpy()%10
            keep=~s.pool.question.isin(audit.question).to_numpy()
            dev=np.flatnonzero(keep&(group<6));test=np.flatnonzero(keep&(group>=6))
            questions=s.pool.iloc[dev].question
            unique=questions.drop_duplicates().to_numpy()
            shuffle=np.random.default_rng(20261115).permutation(len(unique))
            fitq=set(unique[shuffle[:len(unique)//2]]);fit=dev[questions.isin(fitq).to_numpy()]
            val=dev[~questions.isin(fitq).to_numpy()]
            assert not set(s.pool.iloc[fit].question)&set(s.pool.iloc[val].question)
            assert not set(s.pool.iloc[dev].question)&set(s.pool.iloc[test].question)
            val=val[:len(val)//10*10]; test=test[:len(test)//10*10]
            model=LogisticRegression(C=1.,max_iter=1000,random_state=20261115).fit(s.features[fit],s.A[fit])
            assert model.n_iter_.max()<1000,'Probe did not converge'
            # Frozen once: same scores used by every policy.
            vs=model.predict_proba(s.features[val])[:,1].astype(np.float32)
            ts=model.predict_proba(s.features[test])[:,1].astype(np.float32)
            np.savez_compressed(out/f'{key}_probe.npz',coef=model.coef_,intercept=model.intercept_)
            pd.DataFrame(dict(example_id=s.ids[test],score=ts)).to_csv(out/f'{key}_test_scores.csv',index=False)
            write_json(out/f'{key}_split.json',dict(fit=s.ids[fit].tolist(),validation=s.ids[val].tolist(),test=s.ids[test].tolist()))
            manifest.append(dict(key=key,name=name,encoder=encoder,n_fit=len(fit),n_validation=len(val),n_test=len(test),
                features_sha256=digest(nr.FEATURE_ROOT/s.key/'features.npy'),
                split_sha256=digest(out/f'{key}_split.json'),score_sha256=digest(out/f'{key}_test_scores.csv')))
            fixed=[dict(alpha=alpha,gamma=0.,**average(s.A[val],vs,alpha,0,.4)) for alpha in ALPHAS]
            for eps in EPS:
                # Initial threshold selected for non-adaptive performance, not adaptive advantage.
                initial=choose(fixed,eps)
                adaptive=[dict(alpha=initial['alpha'],gamma=g,**average(s.A[val],vs,initial['alpha'],g,eps)) for g in GAMMAS]
                best=choose(adaptive,eps)
                # Independent secondary fixed threshold minimizes development Type-II (same grid).
                # Therefore it legitimately coincides with Frozen-alpha under this protocol.
                match=min(fixed,key=lambda r:(abs(r['cost']-best['cost']),r['type_ii'],r['alpha']))
                locked.append(dict(key=key,epsilon=eps,alpha0=initial['alpha'],gamma=best['gamma'],
                    frozen_validation=initial,adaptive_validation=best,matched_validation=match))
                searches.extend([dict(key=key,epsilon=eps,search='gamma',**r) for r in adaptive])
            searches.extend([dict(key=key,search='initial_threshold',**r) for r in fixed])
            bundles.append((key,s.A[test],ts))
            print('LOCKED',name,encoder,[(r['epsilon'],r['alpha0'],r['gamma']) for r in locked if r['key']==key],flush=True)
    pd.DataFrame(searches).to_csv(out/'validation_search.csv',index=False)
    write_json(out/'manifest.json',manifest)
    write_json(out/'locked_configurations.json',locked)
    print('ALL FOUR CONFIGURATIONS LOCKED; beginning test evaluation',flush=True)
    return bundles,locked,manifest

def statistics(delta,seed):
    rng=np.random.default_rng(seed);n=len(delta)
    means=np.concatenate([delta[rng.integers(n,size=(1000,n))].mean(axis=1) for _ in range(100)])
    ci=np.quantile(means,[.025,.975]);obs=abs(delta.mean());extreme=0
    for _ in range(200):
        null=(rng.choice([-1,1],size=(1000,n))*delta).mean(axis=1)
        extreme+=int((np.abs(null)>=obs-1e-15).sum())
    nz=delta[np.abs(delta)>1e-12]
    return dict(delta=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),
                p_signflip=(extreme+1)/200001,
                p_sign=binomtest(int((nz>0).sum()),len(nz)).pvalue if len(nz) else 1.,
                fraction_improved=float((delta>1e-12).mean()),fraction_tied=float((abs(delta)<=1e-12).mean()))

def report(out,raw,manifest):
    means=raw.groupby(['key','epsilon','method'])[['type_i','type_ii','cost']].mean().reset_index()
    means.to_csv(out/'means.csv',index=False)
    effects=[]
    for (key,eps),g in raw.groupby(['key','epsilon']):
        a=g[g.method=='adaptive'].set_index('seed').sort_index();f=g[g.method=='frozen'].set_index('seed').sort_index()
        np.testing.assert_array_equal(a.index,f.index)
        effects.append(dict(key=key,epsilon=eps,**statistics((f.type_ii-a.type_ii).to_numpy(),20261115),
            adaptive_type_i=a.type_i.mean(),frozen_type_i=f.type_i.mean(),adaptive_type_ii=a.type_ii.mean(),
            frozen_type_ii=f.type_ii.mean(),adaptive_cost=a.cost.mean(),frozen_cost=f.cost.mean()))
    eff=pd.DataFrame(effects);eff['p_holm_primary']=np.nan
    ix=eff[eff.epsilon==.4].sort_values('p_signflip').index
    adj=np.minimum(1,np.maximum.accumulate(eff.loc[ix,'p_signflip'].to_numpy()*np.arange(4,0,-1)))
    eff.loc[ix,'p_holm_primary']=adj
    eff.to_csv(out/'paired_effects.csv',index=False)
    primary=eff[eff.epsilon==.4]
    lookup={m['key']:m for m in manifest}
    lines=[r'\begin{table*}[t]',r'\centering',r'\small',
      r'\begin{tabular}{llrrrrrrrrr}',r'\toprule',
      r'Model & Encoder & I$_A$ & I$_F$ & II$_A$ & II$_F$ & Cost A/F & $\Delta$ [95\% CI] & $p$ & $p_{\rm Holm}$ & Wins \\',r'\midrule']
    md=['# Controlled alpha update: primary target epsilon = 0.40','',
        '| Model | Encoder | Type-I A/F (%) | Type-II A/F (%) | Cost A/F | Delta pp [95% CI] | p | Holm p | Wins |',
        '|---|---|---|---|---|---|---|---|---|']
    for _,r in primary.iterrows():
        m=lookup[r.key];name=m['name'].split(' / ')[0];enc='Qwen3-1.7B' if m['encoder']=='1.7B' else 'MiniLM'
        delta=f'{100*r.delta:.2f} [{100*r.ci_low:.2f}, {100*r.ci_high:.2f}]'
        lines.append(f'{name} & {enc} & {100*r.adaptive_type_i:.2f} & {100*r.frozen_type_i:.2f} & {100*r.adaptive_type_ii:.2f} & {100*r.frozen_type_ii:.2f} & {r.adaptive_cost:.2f}/{r.frozen_cost:.2f} & {delta} & {r.p_signflip:.4f} & {r.p_holm_primary:.4f} & {100*r.fraction_improved:.1f}'+r'\% \\')
        md.append(f'| {name} | {enc} | {100*r.adaptive_type_i:.2f}/{100*r.frozen_type_i:.2f} | {100*r.adaptive_type_ii:.2f}/{100*r.frozen_type_ii:.2f} | {r.adaptive_cost:.2f}/{r.frozen_cost:.2f} | {delta} | {r.p_signflip:.4f} | {r.p_holm_primary:.4f} | {100*r.fraction_improved:.1f}% |')
    lines += [r'\bottomrule',r'\end{tabular}',
       r'\caption{Same-initial-threshold ablation on MATH at prespecified $\epsilon=0.40$. A: adaptive; F: frozen. Errors and effects are percentage points; cost is reviews per round. Positive $\Delta=\mathrm{II}_F-\mathrm{II}_A$ favors adaptation. Percentile paired-bootstrap CIs and two-sided sign-flip tests use 30 stream permutations; Holm correction covers four primary comparisons. Inference is conditional on the fixed historical test pools and concerns stream ordering. Training feasibility and secondary comparisons are reported separately.}',r'\label{tab:alpha-controlled}',r'\end{table*}']
    (out/'primary_table.tex').write_text('\n'.join(lines)+'\n')
    (out/'primary_table.md').write_text('\n'.join(md)+'\n')
    sig=primary[(primary.p_holm_primary<.05)&(primary.delta>0)]
    negative=primary[(primary.p_holm_primary<.05)&(primary.delta<0)]
    text=(f'With a common pretrained frozen logistic probe and validation-selected initial threshold, enabling the original EDIT threshold update yielded positive ordering-conditional effects in {len(sig)} of four primary MATH comparisons after Holm correction; {len(negative)} showed significant effects in the opposite direction. '
          'The paired effect sizes, 95% bootstrap intervals, two-sided sign-flip p-values, and review costs are reported in the table. '
          'These comparisons hold scores, examples, budget, initialization, and tie-breaking fixed; only the threshold update differs. '
          'Cost differences are part of the intervention and do not establish improved cost efficiency. '
          'Sensitivity analyses at epsilon 0.20 and 0.60 are reported without selecting the most favorable target. '
          'Because the same historical examples recur across independent orders, inference concerns ordering robustness, not population-level generalization or an untouched-test confirmation.')
    (out/'academic_paragraph.txt').write_text(text+'\n')
    response=(f'We ran a controlled ablation on four MATH model/encoder settings (Llama-70B and Qwen-32B; Qwen-1.7B and MiniLM). Both policies use identical frozen logistic scores, initial threshold, budget and tie-breaking; only the alpha update is disabled in the control. Settings were locked using development data before evaluation on 30 paired test orders, including all rounds. At prespecified epsilon=0.40, {len(sig)}/4 comparisons favor adaptation and {len(negative)}/4 favor freezing after Holm correction. The table reports effect sizes, paired 95% bootstrap CIs, two-sided sign-flip p-values and costs. We also report all epsilon=0.20/0.60 sensitivities and a development-cost-matched baseline. Inference describes ordering robustness on previously inspected pools, not independent-dataset generalization; lower Type-II alone does not imply greater cost efficiency.')
    assert len(response)<1000
    (out/'openreview_response.md').write_text(response+'\n')
    print('\n'.join(md),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise RuntimeError('Use a new output directory; do not overwrite a prior run')
    protocol=dict(primary_epsilon=.4,sensitivity_epsilons=[.2,.6],alpha_grid=ALPHAS,gamma_grid=GAMMAS,
        development_seeds=DEV_SEEDS,test_seeds=TEST_SEEDS,split_seed=20261115,inference_seed=20261115,
        bootstrap_resamples=100000,signflip_resamples=200000,all_rounds=True,budget=5,batch_size=10,
        probe='sklearn LogisticRegression C=1 max_iter=1000; existing diagnostic specification; fit partition only',
        selection='alpha0: best frozen development Type-II subject to Type-I<=epsilon; gamma: same constrained criterion at locked alpha0; zero gamma allowed',
        fallback='least Type-I then Type-II then cost; explicitly flagged as infeasible',
        split='existing question hash outer 60/40 split; unique development questions split in half into fit/validation with seed 20261115; fixed tail discarded for full batches',
        matched_cost='secondary threshold chosen closest to adaptive development cost; report test cost mismatch, no exact matching claim',
        caveat='Previously inspected benchmark pools; prospective locked reanalysis, not untouched-test confirmation',
        signflip_assumption='joint sign exchangeability/symmetry of paired order effects under null; sign-test sensitivity also saved',
        bootstrap_assumption='independent pseudorandom orders conditional on fixed data and fitted probe',
        script_sha256=digest(__file__),python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    write_json(out/'protocol.json',protocol)
    validate()
    bundles,locked,manifest=prepare(out)
    rows=[];trajectories=[]
    for key,y,scores in bundles:
        for cfg in [r for r in locked if r['key']==key]:
            eps=cfg['epsilon'];alpha=cfg['alpha0'];gamma=cfg['gamma']
            policies={'adaptive':(alpha,gamma),'frozen':(alpha,0.),'fixed_threshold':(alpha,0.),
                      'topk':(0.,0.),'dev_cost_matched':(cfg['matched_validation']['alpha'],0.)}
            for seed in TEST_SEEDS:
                runs={}
                for method,(a,g) in policies.items():
                    m,st,sel,alphas,rounds=replay(y,scores,seed,a,g,eps,True)
                    rows.append(dict(key=key,epsilon=eps,seed=seed,method=method,**m))
                    runs[method]=(sel,alphas)
                    for t,v in enumerate(rounds):
                        trajectories.append(dict(key=key,epsilon=eps,seed=seed,method=method,t=t,
                            alpha=float(alphas[t]),type_i=float(v[0]),type_ii=float(v[1]),cost=float(v[2])))
                np.testing.assert_array_equal(runs['adaptive'][0][0],runs['frozen'][0][0])
                np.testing.assert_array_equal(runs['frozen'][0],runs['fixed_threshold'][0])
                if gamma==0:np.testing.assert_array_equal(runs['adaptive'][0],runs['frozen'][0])
            print('TESTED',key,eps,flush=True)
    raw=pd.DataFrame(rows);raw.to_csv(out/'test_by_order.csv',index=False)
    pd.DataFrame(trajectories).to_csv(out/'round_trajectories.csv.gz',index=False)
    report(out,raw,manifest)
    write_json(out/'complete.json',dict(status='complete',test_runs=len(rows),locked_sha256=digest(out/'locked_configurations.json')))
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
