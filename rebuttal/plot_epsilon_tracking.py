"""Requested epsilon versus achieved Type-I, from saved no-repeat EDIT runs.

Fixed gamma curves; no epsilon-specific configuration selection or new runs.
Shading is one sample standard deviation over the five stream orderings.
"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS = Path(__file__).resolve().parent / 'results' / 'norepeat'
SOURCE = RESULTS / 'baselines_edit_qwen17_raw.csv'
SETTINGS = ['Llama-3.3-70B / MATH', 'Llama-3.3-70B / PopQA',
            'Qwen3-32B / MATH', 'Qwen3-32B / PopQA']
GAMMAS = [.02, .5, 5., 10.]
SEEDS = [7, 17, 27, 37, 47]
EPSILONS = np.round(np.arange(0., .801, .02), 2)
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']


def main():
    raw = pd.read_csv(SOURCE)
    data = raw[raw.p_alpha_step_size.isin(GAMMAS)].copy()
    assert set(data.setting) == set(SETTINGS)
    assert data.method.eq('edit').all()
    assert data.p_theta_step_size.eq(4.).all()
    assert data.p_warm_start_batches.eq(0).all()
    assert len(data) == 4 * 4 * 41 * 5
    keys = ['setting', 'p_alpha_step_size', 'p_epsilon']
    assert not data.duplicated(keys + ['seed']).any()
    for _, group in data.groupby(keys):
        assert sorted(group.seed.tolist()) == SEEDS
    for _, group in data.groupby(keys[:2]):
        np.testing.assert_allclose(np.sort(group.p_epsilon.unique()), EPSILONS, atol=1e-12)
    assert np.isfinite(data[['type_i', 'type_ii', 'cost']]).all().all()
    assert data.type_i.between(0, 1).all()
    mean = data.groupby(keys).agg(
        type_i_mean=('type_i', 'mean'), type_i_sd=('type_i', 'std'),
        type_i_min=('type_i', 'min'), type_i_max=('type_i', 'max'),
        type_ii_mean=('type_ii', 'mean'), review_cost_mean=('cost', 'mean'),
        n_streams=('seed', 'count')).reset_index()
    mean['type_i_minus_epsilon'] = mean.type_i_mean - mean.p_epsilon
    mean.to_csv(RESULTS / 'epsilon_tracking.csv', index=False)
    summary = mean.groupby(keys[:2]).agg(
        max_mean_excess=('type_i_minus_epsilon', 'max'),
        mean_excess=('type_i_minus_epsilon', 'mean'),
        targets_exceeded=('type_i_minus_epsilon', lambda x: int((x > 1e-12).sum()))).reset_index()
    summary.to_csv(RESULTS / 'epsilon_tracking_summary.csv', index=False)
    sns.set_theme(context='paper', style='darkgrid')
    plt.rcParams.update({'font.size':16, 'axes.titlesize':18, 'axes.labelsize':16,
                         'xtick.labelsize':13, 'ytick.labelsize':13, 'font.weight':'bold',
                         'axes.titleweight':'bold', 'axes.labelweight':'bold', 'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.4), sharex=True, sharey=True)
    for ax, setting in zip(axes, SETTINGS):
        for gamma, color in zip(GAMMAS, COLORS):
            d = mean[(mean.setting == setting) & (mean.p_alpha_step_size == gamma)].sort_values('p_epsilon')
            x, y, sd = d.p_epsilon.to_numpy(), d.type_i_mean.to_numpy(), d.type_i_sd.to_numpy()
            ax.plot(x, y, color=color, lw=2.7, label=rf'$\gamma={gamma:g}$', zorder=3)
            ax.fill_between(x, np.maximum(0, y-sd), np.minimum(1, y+sd), color=color, alpha=.13, linewidth=0)
        ax.plot([0,.8], [0,.8], color='#444444', lw=1.8, ls='--', zorder=2)
        model, dataset = setting.split(' / ')
        ax.set_title(f'{dataset} ({model})')
        ax.set_xlabel(r'Requested tolerance $\epsilon$')
        ax.set_xlim(0,.8); ax.set_ylim(0,.85)
        ax.set_xticks(np.arange(0,.81,.2)); ax.set_yticks(np.arange(0,.81,.2))
    axes[0].set_ylabel('Achieved Type-I error')
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0],[0],color='#444444',lw=1.8,ls='--'))
    labels.append(r'Target: Type-I $=\epsilon$')
    fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,fontsize=15,bbox_to_anchor=(.5,.005))
    fig.tight_layout(rect=(0,.12,1,1))
    for ext in ['png','pdf']:
        fig.savefig(RESULTS / f'epsilon_tracking.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    caption = '''**Requested tolerance versus achieved Type-I error.** EDIT-REP uses frozen
Qwen3-1.7B representations, probe learning rate 4.0, and no warm start on the
four no-repeat model–dataset settings. Each curve fixes the controller step
size γ while sweeping ε from 0 to 0.80 in increments of 0.02. Lines show the
mean and shaded bands show ±1 sample standard deviation over five orderings
(seeds 7, 17, 27, 37, and 47) of the same 2,500 examples. Bands are clipped to
[0, 1]. Type-I is the average per-round fraction of reviewed items not requiring
edits; rounds 0–19 are excluded and a round with no reviews contributes zero.
All runs use batches of 10 and a review budget of 5. The dashed diagonal marks
achieved Type-I equal to ε; points below it are conservative relative to the
requested tolerance, and points above it exceed that tolerance. ε is specified
independently of LLM-Select, and no configuration is selected separately at
each ε. This is a descriptive replay of saved runs, not independent held-out
validation or a claim that every finite-run average lies below ε.

The four γ values were chosen for a compact post-hoc diagnostic; the original
frontier experiments used the full 19-value grid. Shading describes variation
across orderings, not uncertainty from independently sampled datasets.
'''
    (RESULTS / 'epsilon_tracking_caption.md').write_text(caption)
    metadata = dict(source=SOURCE.name, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    gammas=GAMMAS,seeds=SEEDS,epsilons=EPSILONS.tolist(),selected_seed_rows=len(data),
                    plotted_means=len(mean),shade='mean +/- 1 sample SD over stream orderings',
                    lr=4,warm_start=0,score_from_round=20,configuration_selection='none')
    (RESULTS / 'epsilon_tracking_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Validated {len(data)} seed rows and {len(mean)} plotted means; wrote PNG, PDF, tables, caption, and metadata.')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
