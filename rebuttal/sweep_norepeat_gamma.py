"""Resumable, ordered EDIT-REP gamma search on the four stored no-repeat streams."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from methods.ours import OursSelection
from models import build_score_model

ROOT = REPO / 'rebuttal/results/norepeat_gamma'
ROOT.mkdir(parents=True, exist_ok=True)
GAMMAS = [0, .001, .002, .005, .01, .02, .03, .05, .068, .075, .1, .125, .15, .2, .25, .3, .4, .5, .75, 1, 1.5, 2, 3, 5, 10]
MODELS = ['or-llama-3.3-70b-instruct', 'or-qwen3-32b']
DATASETS = ['math500-full_n2500_norepeat', 'popqa-test_n2500_seed42_norepeat']
STAGES = ['minilm', '0.6B', '1.7B', '4B']
TOL = .005


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def status(**kw):
    value = dict(time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    save_json(ROOT / 'status.json', value)
    print(json.dumps(value), flush=True)


def metrics(df):
    values = []
    for _, b in df.groupby('t', sort=True):
        s = b.selected.to_numpy(float)
        a = b.A.to_numpy(float)
        ns, nu = s.sum(), (1-s).sum()
        values.append((float((s*(1-a)).sum()/ns) if ns else 0.,
                       float(((1-s)*a).sum()/nu) if nu else 0.,
                       float((s*b.cost.to_numpy(float)).sum())))
    v = np.array(values)
    return dict(type_i=float(v[:,0].mean()), type_ii=float(v[:,1].mean()),
                cost=float(v[:,2].mean()), max_cost=float(v[:,2].max()))


def load_settings():
    settings, baselines = [], []
    for model in MODELS:
        for dataset in DATASETS:
            key = f'{model}_{dataset}'
            cfg = yaml.safe_load((REPO/'configs'/key/'ours.yaml').read_text())
            folder = REPO/'outputs'/key
            refpath, = folder.glob('ours_*MiniLM*.csv')
            ref = pd.read_csv(refpath).sort_values(['t','batch_pos']).reset_index(drop=True)
            assert len(ref) == 2500 and ref.t.nunique() == 250 and ref.example_id.nunique() == 2500
            baseline = []
            for p in folder.glob('*.csv'):
                if p.name.startswith('gen_'): continue
                d = pd.read_csv(p)
                if 'selected' not in d or d.method.iloc[0] == 'random': continue
                m = metrics(d)
                m.update(setting=key, method=d.method.iloc[0], epsilon=cfg['policy']['epsilon'])
                m['feasible'] = m['type_i'] <= m['epsilon']+1e-9 and m['max_cost'] <= cfg['policy']['budget_per_batch']+1e-9
                baseline.append(m)
            target = min([x['type_ii'] for x in baseline if x['feasible']], default=None)
            baselines.extend(baseline)
            settings.append(dict(key=key, cfg=cfg, ref=ref, refpath=refpath, target=target))
    pd.DataFrame(baselines).to_csv(ROOT/'baselines.csv', index=False)
    return settings


class CachedEncoder:
    def __init__(self, features): self.features = features
    def encode_rows(self, rows): return self.features[[r['_feature_index'] for r in rows]]


def encoder_config(setting, stage):
    sc = copy.deepcopy(setting['cfg']['score_model'])
    sc['dataset_name'] = setting['cfg']['data']['name']
    if stage != 'minilm':
        sc.update(provider='qwen3', model_name=f'Qwen/Qwen3-{stage}', pooling='last',
                  encode_batch_size={'0.6B':8, '1.7B':4, '4B':2}[stage])
        # Prefer existing cache. Download missing weights into this workspace.
        from huggingface_hub import snapshot_download
        def complete_snapshot(path):
            path = Path(path)
            if (path / 'model.safetensors').is_file():
                return True
            index = path / 'model.safetensors.index.json'
            if index.is_file():
                shards = set(json.loads(index.read_text())['weight_map'].values())
                return bool(shards) and all((path / shard).is_file() for shard in shards)
            return False
        local_checkpoint = Path('/home/mll/ali/models') / f'Qwen3-{stage}'
        if complete_snapshot(local_checkpoint):
            sc['model_name'] = str(local_checkpoint)
            return sc
        found = None
        for cache_dir in [None, str(ROOT/'model_cache')]:
            try:
                snapshot = snapshot_download(f'Qwen/Qwen3-{stage}', cache_dir=cache_dir, local_files_only=True)
                if complete_snapshot(snapshot):
                    found = snapshot
                    break
            except Exception:
                pass
        if found is None:
            import shutil
            required_gb = {'0.6B': 1.5, '1.7B': 4.0, '4B': 8.5}[stage]
            free_gb = shutil.disk_usage(ROOT).free / 1e9
            if free_gb < required_gb + 0.5:
                raise RuntimeError(f'Insufficient disk for missing Qwen {stage} weights: {free_gb:.1f} GB free; need about {required_gb + 0.5:.1f} GB')
            status(phase='downloading_encoder', stage=stage)
            found = snapshot_download(f'Qwen/Qwen3-{stage}', cache_dir=str(ROOT/'model_cache'),
                                      allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])
            if not complete_snapshot(found):
                raise RuntimeError(f'Incomplete weights after download: {found}')
        sc['model_name'] = found
    return sc


def get_features(setting, stage):
    folder = ROOT/stage/setting['key']
    folder.mkdir(parents=True, exist_ok=True)
    sc = encoder_config(setting, stage)
    fingerprint = hashlib.sha256(setting['refpath'].read_bytes() + json.dumps(sc, sort_keys=True).encode() +
        (REPO/'models/minilm.py').read_bytes() + (REPO/'models/qwen3_embedding.py').read_bytes()).hexdigest()
    meta_path, feat_path = folder/'features.json', folder/'features.npy'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get('fingerprint') == fingerprint and meta.get('completed_rows') == 2500:
        return np.load(feat_path, mmap_mode='r'), folder
    encoder = build_score_model(sc)
    if meta.get('fingerprint') != fingerprint or not feat_path.exists():
        features = np.lib.format.open_memmap(feat_path, mode='w+', dtype='float32', shape=(2500,encoder.hidden_size))
        meta = dict(fingerprint=fingerprint, completed_rows=0, encoder=sc)
    else:
        features = np.load(feat_path, mmap_mode='r+')
    # Preserve the original batch boundaries and padding for exact replay.
    for start in range(meta['completed_rows'], 2500, 10):
        features[start:start+10] = encoder.encode_rows(setting['ref'].iloc[start:start+10].to_dict('records'))
        features.flush()
        meta['completed_rows'] = start+10
        save_json(meta_path, meta)
        if (start+10)%250 == 0:
            status(phase='encoding', stage=stage, setting=setting['key'], completed=start+10, total=2500)
    del encoder
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return features, folder


def replay(setting, features, gamma):
    cfg = copy.deepcopy(setting['cfg'])
    cfg['policy']['alpha_step_size'] = gamma
    method = OursSelection(cfg, CachedEncoder(features))
    source = setting['ref'].copy()
    source['_feature_index'] = np.arange(len(source))
    batches = []
    for t, b in source.groupby('t', sort=True):
        result, _ = method.process_batch(b, int(t))
        batches.append(result)
    return pd.concat(batches, ignore_index=True).drop(columns='_feature_index')


def plot_winner(stage, gamma, settings):
    import matplotlib
    matplotlib.use('Agg')
    notebook = json.loads((REPO/'to_plot.ipynb').read_text())
    ns = {'__name__':'plot_helpers'}
    exec(''.join(notebook['cells'][0]['source']), ns)
    label = f'EDIT-REP ({stage}, γ={gamma:g})'
    ns['METHOD_ORDER'].append(label)
    ns['METHOD_PALETTE'][label] = '#17becf'
    ns['set_paper_style']()
    plt = ns['plt']
    fig, axes = plt.subplots(3,4,figsize=(20,11))
    handles, labels = [], []
    for col, s in enumerate(settings):
        raw = ns['collect_run_csvs']([str(s['refpath'].parent)])
        candidate = pd.read_csv(ROOT/stage/s['key']/f'gamma_{gamma:g}.csv')
        candidate['run_file'] = 'candidate'
        candidate['config'] = 'candidate'
        candidate['dataset'] = raw.dataset.iloc[0]
        candidate['model_data'] = raw.model_data.iloc[0]
        raw = pd.concat([raw,candidate], ignore_index=True)
        m = ns['maybe_add_epsilon'](ns['add_cumulative_metrics'](ns['summarize_per_round'](raw)),raw)
        m.loc[m.run_file=='candidate','Method'] = label
        m = m[m.Method.isin(ns['METHOD_ORDER'])]
        for row, key in enumerate(['type_ii','type_i','budget']):
            ax=axes[row,col]
            ns['_draw_metric_axis'](m,key,ax,True)
            h,l=ax.get_legend_handles_labels(); handles.extend(h); labels.extend(l)
            ax.get_legend().remove()
            ax.set_title(
                f"{'MATH' if col % 2 == 0 else 'PopQA'} "
                f"({'Llama-3.3-70B' if col < 2 else 'Qwen3-32B'})"
                if row == 0 else ''
            )
            ax.set_ylabel(ns['PLOT_METRICS'][key]['title'] if col==0 else '')
            if row<2: ax.set_xlabel('')
    h,l=ns['_dedupe_legend'](handles,labels)
    fig.legend(h,l,loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.12,1,1))
    for ext in ['png','pdf']: fig.savefig(ROOT/stage/f'best_shared_gamma.{ext}',bbox_inches='tight',dpi=180)
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    settings=load_settings()
    plan = dict(stages=STAGES, gammas=GAMMAS, co_best_type_ii_tolerance=TOL,
                epsilon_tolerance=1e-9, success='4/4 combinations feasible and best/co-best against feasible plotted baselines',
                ranking='co-best count, feasible count, mean epsilon excess, mean Type-II, mean cost',
                metric='mean per-round ratios over all 250 rounds, matching notebook',
                caveat='Exploratory tuning on these same streams; not held-out evaluation.',
                fixed='stream, seed, epsilon, theta learning rate, initialization, budget and encoder max length')
    save_json(ROOT/'plan.json',plan)
    for stage in STAGES:
        completed = ROOT/stage/'best.json'
        if completed.exists() and (ROOT/stage/'best_shared_gamma.pdf').exists():
            best = json.loads(completed.read_text())
            if best['co_best'] == 4:
                status(phase='complete', reason='success', stage=stage, best=best)
                return
            print(f'Resuming past completed stage {stage}', flush=True)
            continue
        status(phase='stage_started', stage=stage)
        rows=[]
        for s in settings:
            features, folder=get_features(s,stage)
            if stage=='minilm':
                check=replay(s,features,s['cfg']['policy']['alpha_step_size'])
                agreement=float(np.mean(check.selected.to_numpy()==s['ref'].selected.to_numpy()))
                save_json(folder/'replay_check.json',dict(selection_agreement=agreement))
                if agreement != 1.: raise RuntimeError(f"Reference replay mismatch: {s['key']} {agreement}")
            for gamma in GAMMAS:
                dest=folder/f'gamma_{gamma:g}.csv'
                if dest.exists(): run=pd.read_csv(dest)
                else:
                    run=replay(s,features,gamma)
                    run['config']=f'sweep_{stage}_gamma{gamma:g}'
                    run['score_model']='sentence-transformers/all-MiniLM-L6-v2' if stage=='minilm' else f'Qwen/Qwen3-{stage}'
                    run['score_model_provider']='minilm' if stage=='minilm' else 'qwen3'
                    run['gamma']=gamma
                    run = run.drop(columns=['question', 'gold_answer', 'gold_final', 'model_answer', 'pred_answer'], errors='ignore')
                    tmp=dest.with_suffix('.tmp'); run.to_csv(tmp,index=False); tmp.replace(dest)
                m=metrics(run)
                eps=s['cfg']['policy']['epsilon']
                feasible=m['type_i']<=eps+1e-9 and m['max_cost']<=s['cfg']['policy']['budget_per_batch']+1e-9
                rows.append(dict(stage=stage,setting=s['key'],gamma=gamma,epsilon=eps,feasible=feasible,
                                 epsilon_excess=max(0,m['type_i']-eps),target=s['target'],
                                 co_best=feasible and (s['target'] is None or m['type_ii']<=s['target']+TOL),**m))
                pd.DataFrame(rows).to_csv(ROOT/stage/'sweep.csv',index=False)
                status(phase='sweeping',stage=stage,setting=s['key'],gamma=gamma,completed=len(rows),total=4*len(GAMMAS))
        table=pd.DataFrame(rows)
        ranked=table.groupby('gamma',as_index=False).agg(co_best=('co_best','sum'),feasible=('feasible','sum'),
                  epsilon_excess=('epsilon_excess','mean'),type_ii=('type_ii','mean'),cost=('cost','mean'))
        ranked=ranked.sort_values(['co_best','feasible','epsilon_excess','type_ii','cost'],ascending=[False,False,True,True,True])
        ranked.to_csv(ROOT/stage/'ranking.csv',index=False)
        best=ranked.iloc[0].to_dict()
        save_json(ROOT/stage/'best.json',best)
        plot_winner(stage,float(best['gamma']),settings)
        status(phase='stage_complete',stage=stage,best=best)
        if best['co_best']==4:
            status(phase='complete',reason='success',stage=stage,best=best)
            return
    status(phase='complete',reason='grid_exhausted; inspect rankings, no candidate satisfied all four combinations')


if __name__=='__main__':
    try: main()
    except Exception as exc:
        status(phase='failed',error=str(exc))
        traceback.print_exc()
        raise
