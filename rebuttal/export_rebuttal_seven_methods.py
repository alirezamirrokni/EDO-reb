"""Export the prespecified primary target without selecting favorable test points."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

P=Path(__file__).resolve().parent/'results/review_quality_all_baselines_v1'
names={'adaptive':'EDIT','gollapudi':'Online Conformal Selection','conformal':'Conformal Alignment',
       'zliobaite':'Stream Active Learning','thompson_score_space':'Thompson Sampling',
       'topk':'Top-K','fixed':'Fixed Threshold'}
settings={'llama_MATH-500_Qwen1.7B':'MATH-500','llama_MMLU-Pro_Qwen1.7B':'MMLU-Pro',
          'llama_PopQA_Qwen1.7B':'PopQA','llama_GPQA_Qwen1.7B':'GPQA','RAGTruth_Qwen4B':'RAGTruth'}
d=pd.read_csv(P/'all_methods_summary.csv')
d=d[(d.epsilon==.4)&(d.probe==0)&(d.axis=='order')&d.method.isin(names)]
e=pd.read_csv(P/'paired_all_baselines.csv')
e=e[(e.epsilon==.4)&e.baseline.isin(names)]
table=d.merge(e[['setting','baseline','effect','ci_low','ci_high']],
    left_on=['setting','method'],right_on=['setting','baseline'],how='left',validate='one_to_one')
raw=pd.read_csv(P/'all_methods_by_replicate.csv')
raw=raw[(raw.epsilon==.4)&(raw.probe==0)&(raw.axis=='order')&raw.method.isin(names)]
intervals=[]
for (setting,method),g in raw.groupby(['setting','method']):
    g=g.sort_values('seed')
    values=g[['type_i','type_ii']].to_numpy()
    assert len(values)==30
    assert g.seed.nunique()==30
    adaptive=raw[(raw.setting==setting)&(raw.method=='adaptive')].set_index('seed').reindex(g.seed)
    assert adaptive[['type_i','type_ii']].notna().all().all()
    rng=np.random.default_rng(20261116)
    indices=rng.integers(len(values),size=(20000,len(values)))
    bootstrap=values[indices].mean(axis=1)
    low,high=np.quantile(bootstrap,[.025,.975],axis=0)
    # Pair records by identical stream seed BEFORE resampling the differences.
    differences=values-adaptive[['type_i','type_ii']].to_numpy()
    delta_bootstrap=differences[indices].mean(axis=1)
    dl,dh=np.quantile(delta_bootstrap,[.025,.975],axis=0)
    reference=d[(d.setting==setting)&(d.method==method)].iloc[0]
    np.testing.assert_allclose(values.mean(axis=0),[reference.type_i,reference.type_ii],atol=1e-12)
    if method!='adaptive':
        previous=e[(e.setting==setting)&(e.baseline==method)].iloc[0]
        # Preserve the already-valid paired Type-II interval exactly.
        np.testing.assert_allclose([differences[:,1].mean(),dl[1],dh[1]],
                                   [previous.effect,previous.ci_low,previous.ci_high],atol=1e-12)
    intervals.append(dict(setting=setting,method=method,type_i_ci_low=low[0],type_i_ci_high=high[0],
        type_ii_ci_low=low[1],type_ii_ci_high=high[1],delta_type_i=differences[:,0].mean(),
        delta_type_i_ci_low=dl[0],delta_type_i_ci_high=dh[0]))
table=table.merge(pd.DataFrame(intervals),on=['setting','method'],validate='one_to_one')
# Explicit symmetric names in the data export; existing effect/ci columns retained for compatibility.
table['delta_type_ii']=table.effect.fillna(0.)
table['delta_type_ii_ci_low']=table.ci_low.fillna(0.)
table['delta_type_ii_ci_high']=table.ci_high.fillna(0.)
table.to_csv(P/'rebuttal_seven_methods.csv',index=False)
md=['| Dataset | Method | Type-I mean [95% CI] (%) | Type-II mean [95% CI] (%) | Paired Δ Type-I [95% CI] (pp) | Paired Δ Type-II [95% CI] (pp) | Cost | Pooled precision (%) | Coverage (%) |',
    '|---|---|---|---|---|---|---:|---:|---:|']
tex=[r'\begin{table*}[t]',r'\centering',r'\scriptsize',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{llllllrrr}',r'\toprule',
     r'Dataset & Method & Type-I [95\% CI] & Type-II [95\% CI] & $\Delta$ Type-I [95\% CI] & $\Delta$ Type-II [95\% CI] & Cost & Pooled precision & Coverage \\',r'\midrule']
for key,label in settings.items():
    for method,name in names.items():
        r=table[(table.setting==key)&(table.method==method)].iloc[0]
        delta_ii='—' if method=='adaptive' else f'{100*r.effect:.2f} [{100*r.ci_low:.2f}, {100*r.ci_high:.2f}]'
        delta_i='—' if method=='adaptive' else f'{100*r.delta_type_i:.2f} [{100*r.delta_type_i_ci_low:.2f}, {100*r.delta_type_i_ci_high:.2f}]'
        vals=[label,name,f'{100*r.type_i:.2f} [{100*r.type_i_ci_low:.2f}, {100*r.type_i_ci_high:.2f}]',
              f'{100*r.type_ii:.2f} [{100*r.type_ii_ci_low:.2f}, {100*r.type_ii_ci_high:.2f}]',delta_i,delta_ii,
              f'{r.cost:.2f}',f'{100*r.precision:.2f}',f'{100*r.coverage:.2f}']
        md.append('| '+' | '.join(vals)+' |')
        tex.append(' & '.join(v.replace('—','--') for v in vals)+r' \\')
    tex.append(r'\midrule')
caption=('Primary target epsilon=0.40; all rounds and records included. Means over 30 paired stream orders for '
    'the prespecified original probe fit, with disjoint probe fitting, calibration and policy validation. '
    'Five-fold cross-fitting covers each primary task; RAGTruth retains its official split. '
    'Errors, precision and coverage are percentages; cost is reviews per round with ceiling five. '
    'Both deltas are baseline minus EDIT in percentage points; positive favors EDIT for the corresponding error. '
    'Both marginal and paired intervals are two-sided pointwise percentile-bootstrap 95% CIs '
    '(20,000 resamples, seed 20261116), using identical seed-paired stream orders and identical resampling indices '
    'for both errors. Paired intervals are bootstrapped from per-order differences, not subtracted marginal bounds. '
    'Existing valid Type-I marginal and Type-II difference intervals and all corrected means are preserved. '
    'Intervals quantify ordering uncertainty conditional on records and the fitted probe. '
    'Feasibility is not established merely by a mean below the target: refer to the full uncertainty analyses. '
    'Baselines use common calibrated scores. Conformal Alignment uses the common disjoint offline calibration '
    'set instead of full-label test warm-up; Thompson is an explicitly one-dimensional score-space posterior '
    'exploration variant with frozen mean, not the original full-embedding implementation. '
    'Historical data were previously inspected; added baseline evaluation followed the initial EDIT analysis. '
    'All tolerance, source-resampling and probe-training sensitivities must accompany broad robustness claims.')
tex.extend([r'\bottomrule',r'\end{tabular}}',r'\caption{'+caption.replace('%',r'\%')+'}',r'\end{table*}'])
(P/'rebuttal_seven_methods.tex').write_text('\n'.join(tex)+'\n')
(P/'rebuttal_seven_methods.md').write_text('\n'.join(md)+'\n\n'+caption+'\n')
(P/'symmetric_ci_method.json').write_text(json.dumps(dict(resamples=20000,seed=20261116,
    n_orders=30,epsilon=.4,probe=0,method='percentile bootstrap of means and paired differences',
    differences='baseline minus EDIT',existing_intervals_verified=True),indent=2))
print('\n'.join(md))
