"""Figure 22 replacement: locked train-selected points, all test rounds."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'results/reviewer3_confirmatory'
OUT=ROOT.parent/'figures/ragtruth'
names={'edit':'EDIT-REP','gollapudi':'Online Conformal Selection','conformal':'Conformal Alignment',
       'zliobaite':'Stream Active Learning','thompson':'Thompson Sampling','topk':'Top-K','tau':'Fixed Threshold'}
colors=['#ff7f0e','#d62728','#9467bd','#8c564b','#17becf','#1f77b4','#2ca02c']
styles=['-','-.','--',':','--','--',':']
raw=pd.read_csv(SRC/'frontier_test.csv')
assert len(raw)==1680
assert raw.groupby(['epsilon','method']).size().eq(30).all()
locked=pd.DataFrame(json.loads((SRC/'frontier_locked.json').read_text()))
locked['validation_feasible']=locked.type_i<=locked.epsilon+1e-12
locked['config_key']=locked.params.map(lambda p:json.dumps(p,sort_keys=True))
mean=raw.groupby(['epsilon','method']).agg(type_i=('type_i','mean'),type_ii=('type_ii','mean'),
    cost=('mean_review_cost','mean'),type_i_sd=('type_i','std'),type_ii_sd=('type_ii','std')).reset_index()
mean=mean.merge(locked[['epsilon','method','validation_feasible','config_key']],on=['epsilon','method'],validate='one_to_one')
OUT.mkdir(exist_ok=True,parents=True)
mean.to_csv(OUT/'ragtruth_frontier_train_selected.csv',index=False)
sns.set_theme(context='paper',style='darkgrid')
plt.rcParams.update({'font.size':16,'axes.labelsize':16,'xtick.labelsize':13,'ytick.labelsize':13,
                     'font.weight':'bold','axes.labelweight':'bold','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(10,6))
for (method,name),color,style in zip(names.items(),colors,styles):
    d=mean[mean.method==method].sort_values('epsilon')
    # Same configuration can be selected at several targets; plot it once.
    points=d.groupby('config_key',sort=False).agg(type_i=('type_i','first'),type_ii=('type_ii','first'),
            feasible=('validation_feasible','any')).reset_index()
    ax.plot(points.type_i,points.type_ii,label=name,color=color,linestyle=style,
            linewidth=3.4 if method=='edit' else 2.2)
    for feasible,part in points.groupby('feasible'):
        ax.scatter(part.type_i,part.type_ii,color=color,marker='o' if feasible else 'x',s=35,zorder=4)
ax.set(xlabel='Type-I error',ylabel='Type-II error',xlim=(0,.57))
h,l=ax.get_legend_handles_labels()
fig.legend(h,l,loc='lower center',ncol=4,frameon=False,fontsize=13,
           bbox_to_anchor=(.5,-.02),handlelength=2.4,columnspacing=1.2)
fig.tight_layout(rect=(0,.1,1,1))
for ext in ['pdf','png']:
    fig.savefig(OUT/f'ragtruth_frontier_train_selected.{ext}',dpi=220,bbox_inches='tight')
(OUT/'ragtruth_frontier_train_selected_caption.md').write_text(
    'RAGTruth: train-selected operating points evaluated on all 45 test rounds, averaged over 30 orders. '
    'EDIT-REP and representation baselines use Qwen-4B features. Lines connect distinct configurations '
    'in target-tolerance order; no test-set minimum envelope is applied. Circles mark configurations '
    'feasible for at least one training target selecting them; crosses mark fallback configurations '
    'that met none of their selecting targets. Training feasibility does not guarantee test feasibility. '
    'Conformal Alignment reviews all ten items during its 20 calibration rounds; other methods have '
    'a maximum budget of five. Costs and across-order standard deviations are provided in the companion CSV. '
    'This reuses previously inspected test data.\n')
print(OUT/'ragtruth_frontier_train_selected.pdf')
