"""Document existing runs only: no retuning and no new performance selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT=Path(__file__).resolve().parent/'results'
SRC=ROOT/'review_quality_llama_qwen17_v1'
OUT=ROOT/'review_quality_all_baselines_v1'
METHODS=['fixed','topk','gollapudi','conformal','zliobaite','thompson_score_space']

def main():
    d=pd.read_csv(OUT/'all_methods_by_replicate.csv');d=d[d.epsilon==.4]
    rows=[]
    for setting,g in d.groupby('setting'):
        for method in METHODS:
            r=dict(setting=setting,baseline=method)
            for axis in ['order','records']:
                h=g[(g.axis==axis)&(g.probe==0)]
                a=h[h.method=='adaptive'].set_index('seed');b=h[h.method==method].set_index('seed').reindex(a.index)
                delta=(b.type_ii-a.type_ii).to_numpy()*100
                if axis=='records':
                    r.update(record_effect_low=np.quantile(delta,.025),record_effect_high=np.quantile(delta,.975))
            h=g[g.axis=='order'].groupby(['probe','method'])[['type_ii','type_i','cost']].mean().reset_index()
            a=h[h.method=='adaptive'].set_index('probe');b=h[h.method==method].set_index('probe').reindex(a.index)
            delta=(b.type_ii-a.type_ii)*100
            r.update(probe_effect_min=delta.min(),probe_effect_max=delta.max(),probe_improved=int((delta>1e-12).sum()),
                     probe_fits=len(delta),adaptive_probe_type_i_min=100*a.type_i.min(),adaptive_probe_type_i_max=100*a.type_i.max())
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT/'uncertainty_sources.csv',index=False)
    cutoffs=[];calibration=[]
    for folder in SRC.iterdir():
        if not folder.is_dir():continue
        ys=[];raws=[];calibrated=[]
        for lockpath in sorted(folder.glob('fold*_probe0_lock.json')):
            prefix=lockpath.name.removesuffix('_lock.json');fold=int(prefix.split('_')[0].replace('fold',''))
            split=json.loads((folder/f'split_{fold}.json').read_text());val=np.array(split['validation']);test=np.array(split['test'])
            weights=np.load(folder/f'{prefix}_probe.npz');x=np.load(folder/'features.npy',mmap_mode='r');y=np.load(folder/'labels.npy')
            score=np.load(folder/f'{prefix}_scores.npy')
            raw=expit(np.asarray(x[test])@weights['coef'].reshape(-1)+weights['intercept'].item())
            ys.extend(y[test]);raws.extend(raw);calibrated.extend(score[test])
            cfg=next(c for c in json.loads(lockpath.read_text())['configs'] if c['epsilon']==.4)
            f=cfg['policies']['fixed'];a=cfg['policies']['adaptive']
            cutoffs.append(dict(setting=folder.name,fold=fold,alpha0=f['alpha'],gamma=a['gamma'],
                fixed_val_type_i=100*f['type_i'],fixed_val_type_ii=100*f['type_ii'],fixed_val_cost=f['cost'],
                fixed_val_coverage=100*f['coverage'],validation_zero_reviews=f['cost']==0,
                val_mean_score=float(score[val].mean()),val_edit_prevalence=float(y[val].mean()),
                calibration_slope=float(weights['cal_coef'].item()),calibration_intercept=float(weights['cal_intercept'].item())))
        if not ys:continue
        y=np.asarray(ys)
        for label,values in [('raw',raws),('platt',calibrated)]:
            s=np.asarray(values);bins=np.minimum((s*10).astype(int),9)
            ece=sum(np.mean(bins==i)*abs(s[bins==i].mean()-y[bins==i].mean()) for i in range(10) if (bins==i).any())
            calibration.append(dict(setting=folder.name,score=label,n=len(y),brier=float(np.mean((s-y)**2)),
                logloss=float(-np.mean(y*np.log(np.clip(s,1e-8,1))+(1-y)*np.log(np.clip(1-s,1e-8,1)))),ece10=ece))
    pd.DataFrame(cutoffs).to_csv(OUT/'cutoff_validation_details.csv',index=False)
    pd.DataFrame(calibration).to_csv(OUT/'calibration_diagnostics.csv',index=False)
    notes='''# Suggested additions to the rebuttal

Uncertainty: report the 30-order paired intervals in the main table, then add the source-bootstrap intervals and the range of effects over all five probe fits. The source bootstrap resamples held-out question/source clusters and reruns the complete policy (200 replicates). It includes ordering variation and is not a pure record variance component. The five probe variants bootstrap the fitting records and reselect policies on the same validation partition; their range measures pipeline sensitivity, not a 95% population CI. Neither sensitivity analysis resamples the full outer split or all calibration records. Cross-fit folds overlap in training data. These limits preclude a claim of complete variance decomposition.

Cutoffs: the logistic probe is trained on a disjoint fitting partition. A sigmoid/Platt map is fitted on a separate calibration partition, with both regularization constants fixed at C=1. A third, validation partition selects alpha0 from 0:0.02:0.98 to minimize paper-defined Type-II subject to mean roundwise Type-I<=epsilon; gamma is selected from the existing grid at that same alpha0. Cross-fitting gives fold-specific cutoffs, which must be reported as such. The score is an estimated edit probability; alpha0 is a probability threshold, not a guarantee on review precision. A cutoff of 1-epsilon is not equivalent to a constraint on mean roundwise Type-I, especially with budget truncation and imperfect calibration.

Report the actual alpha0 and gamma per fold, validation Type-I, cost and coverage. If validation selects no reviews, disclose it as a degenerate operating point. Calibration fitting alone does not prove improved calibration: report held-out Brier score, log loss and a descriptive 10-bin ECE for both raw and calibrated probabilities; do not select a new calibrator based on these test diagnostics. Improvements may be absent. Source/probe diagnostics and calibration diagnostics are descriptive additions to already evaluated data, not an independent confirmatory experiment.
'''
    (OUT/'UNCERTAINTY_AND_CUTOFFS.md').write_text(notes)
    print(pd.DataFrame(rows).query("baseline == 'fixed'").to_string(index=False))
    print('\nCUTOFF SUMMARY')
    print(pd.DataFrame(cutoffs).groupby('setting').agg(alpha_min=('alpha0','min'),alpha_max=('alpha0','max'),val_cost_min=('fixed_val_cost','min'),val_cost_max=('fixed_val_cost','max'),zero=('validation_zero_reviews','sum')).to_string())
    print('\nCALIBRATION');print(pd.DataFrame(calibration).to_string(index=False))

if __name__=='__main__':main()
