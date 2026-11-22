"""Pointwise standard errors across five stored stream orders, without shifting centers."""
import json
import sqlite3
import numpy as np
import pandas as pd

def add_intervals(ax, out, edit_mean, baseline_mean, keys):
    prior=pd.read_csv(out/'inputs/prior_raw.csv.gz')
    baseline=pd.read_csv(out/'inputs/baselines_raw.csv.gz')
    con=sqlite3.connect('file:'+str(out/'runs.sqlite')+'?mode=ro',uri=True)
    methods=['edit','gollapudi','conformal','zliobaite','thompson','topk','tau']
    records=[]
    for method,line in zip(methods,ax.lines[:7]):
        frame=edit_mean if method=='edit' else baseline_mean[baseline_mean.method==method]
        frame=frame.sort_values(['type_i','type_ii']).drop_duplicates('type_i')
        envelope=frame[frame.type_ii < frame.type_ii.cummin().shift(1,fill_value=np.inf)-1e-12]
        bands=[]
        for _,r in envelope.iterrows():
            vals=None
            if method=='edit':
                params={k:int(r['p_'+k]) if k=='warm_start_batches' else float(r['p_'+k]) for k in keys}
                key=json.dumps(params,sort_keys=True,separators=(',',':'))
                row=con.execute('select metrics from runs where config=?',(key,)).fetchone()
                if row: vals=np.asarray(json.loads(row[0]))[:,1]
                source=prior
            else:
                source=baseline[baseline.method==method]
            if vals is None:
                match=source
                for k in [c for c in source if c.startswith('p_')]:
                    if k in r and pd.notna(r[k]) and r[k]!=-1:
                        match=match[np.isclose(match[k],r[k],rtol=1e-9,atol=1e-12)]
                vals=match.type_ii.to_numpy()
            if len(vals)!=5:
                raise ValueError(f'{method}: expected five seed values, got {len(vals)}')
            np.testing.assert_allclose(vals.mean(),r.type_ii,atol=1e-10)
            half=vals.std(ddof=1)/np.sqrt(5)
            low,high=max(0,r.type_ii-half),min(1,r.type_ii+half)
            bands.append((r.type_i,low,high))
            records.append(dict(method=method,type_i=r.type_i,measured_type_ii=r.type_ii,
                                se=half,se_low=low,se_high=high,n=5))
        b=np.asarray(bands)
        if len(b):
            # Extend the final real interval over the displayed step plateau.
            b=np.vstack([b,[frame.type_i.max(),b[-1,1],b[-1,2]]])
            ax.fill_between(b[:,0],b[:,1],b[:,2],step='post',color=line.get_color(),alpha=.12)
    con.close()
    pd.DataFrame(records).to_csv(out/'frontier_changed_real_intervals.csv',index=False)
    (out/'frontier_changed_intervals_notes.md').write_text(
        'Title: Hypothetical. Lines have synthetic display perturbations; shaded bands remain centered '
        'on the unmodified measured means. Bands show plus/minus one standard error across five '
        'stream permutations (sample standard deviation divided by sqrt(5)). They quantify ordering variability, '
        'not uncertainty from independently sampled datasets. They do not adjust for frontier '
        'configuration selection on the same test streams and are not 95% confidence intervals for '
        'the hypothetical lines.\n')
