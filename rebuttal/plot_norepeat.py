"""Plot the four OpenRouter no-repeat runs using to_plot.ipynb's plotting code.

Run from any directory with the project's plotting environment. Outputs are
written to figures/norepeat_n2500; source runs and the notebook are read-only.
"""
import ast
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--best-minilm', action='store_true', help='Replace EDIT-REP with the best shared-gamma MiniLM sweep candidate')
    args = parser.parse_args()
    notebook = json.loads((REPO / "to_plot.ipynb").read_text())
    scope = {"__name__": "notebook_plotting"}
    exec(compile("".join(notebook["cells"][0]["source"]), "to_plot.ipynb:cell0", "exec"), scope)

    # Load the overview definitions without executing its default all-run plots.
    source = "".join(notebook["cells"][1]["source"])
    source = source.replace("ncols=4,", "ncols=len(dataset_order),")
    source = source.replace("figsize=(20.0, 11.0),", "figsize=(5.0 * len(dataset_order), 11.0),")
    tree = ast.parse(source)
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "metrics" for t in node.targets):
            tree.body = tree.body[:i]
            break
    exec(compile(tree, "to_plot.ipynb:cell1", "exec"), scope)

    datasets = ["math500-full_n2500_norepeat", "popqa-test_n2500_seed42_norepeat"]
    scope["DATASET_ORDER"] = datasets
    scope["DATASET_LABELS"].update({datasets[0]: "MATH", datasets[1]: "PopQA"})
    # The notebook's original MATH regex does not accept the new suffix.
    scope["KNOWN_DATASET_PATTERNS"] = (r"(math500-full_n2500_norepeat)(?=_budget|[/\\]|$)",) + scope["KNOWN_DATASET_PATTERNS"]
    runs = [REPO / "outputs" / f"{model}_{dataset}"
            for model in ["or-llama-3.3-70b-instruct", "or-qwen3-32b"]
            for dataset in datasets]
    for run in runs:
        if not run.is_dir():
            raise FileNotFoundError(run)
    gamma = None
    if args.best_minilm:
        sweep = REPO / 'rebuttal/results/norepeat_gamma/minilm'
        gamma = float(json.loads((sweep / 'best.json').read_text())['gamma'])
        original_collect = scope['collect_run_csvs']

        def collect_with_best_minilm(paths):
            raw = original_collect(paths)
            replacements = []
            for run in runs:
                original = raw[(raw['model_data'] == run.name) & (raw['method'] == 'ours')]
                if len(original) != 2500:
                    raise ValueError(f'Expected one original EDIT-REP run for {run.name}')
                candidate_path = sweep / run.name / f'gamma_{gamma:g}.csv'
                candidate = scope['pd'].read_csv(candidate_path)
                if not candidate['gamma'].eq(gamma).all():
                    raise ValueError(f'Gamma mismatch: {candidate_path}')
                candidate['dataset'] = original['dataset'].iloc[0]
                candidate['model_data'] = run.name
                candidate['run_file'] = f'best_minilm_gamma_{gamma:g}'
                candidate['run_path'] = str(candidate_path)
                replacements.append(candidate)
            return scope['pd'].concat([raw[raw['method'] != 'ours'], *replacements], ignore_index=True)

        scope['collect_run_csvs'] = collect_with_best_minilm
    metrics = scope["build_model_overview_metrics"](runs=runs)
    if gamma is not None:
        label = f'EDIT-REP (γ={gamma:g})'
        metrics.loc[metrics['Method'] == 'EDIT-REP', 'Method'] = label
        for name in ['METHOD_ORDER', 'METHODS_TO_PLOT']:
            scope[name][:] = [label if m == 'EDIT-REP' else m for m in scope[name]]
        scope['METHOD_PALETTE'][label] = scope['METHOD_PALETTE']['EDIT-REP']
    curves = metrics.groupby("curve_id")["Round"].agg(["size", "min", "max"])
    if len(curves) != 16 or not ((curves["size"] == 250) & (curves["min"] == 0) & (curves["max"] == 249)).all():
        raise ValueError("Expected 16 plotted curves with all 250 rounds")
    out = REPO / "figures" / "norepeat_n2500"
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "summary_metrics.csv", index=False)
    (out / 'plot_settings.json').write_text(json.dumps({
        'edit_rep': 'best shared-gamma MiniLM sweep candidate' if gamma is not None else 'original runs',
        'gamma': gamma,
        'exploratory_tuning': gamma is not None,
    }, indent=2))
    # Use one distinct panel key per model/dataset pair, retaining curve IDs.
    combined = metrics.copy()
    panel_order = []
    for model, label in [("meta-llama/llama-3.3-70b-instruct", "Llama-3.3-70B"),
                         ("qwen/qwen3-32b", "Qwen3-32B")]:
        for dataset in datasets:
            key = model + "|" + dataset
            panel_order.append(key)
            scope["DATASET_LABELS"][key] = label + "\n" + scope["DATASET_LABELS"][dataset]
    combined["dataset"] = combined["model"] + "|" + combined["dataset"]
    if set(combined["dataset"]) != set(panel_order):
        raise ValueError("Unexpected model/dataset panels")
    combined["model"] = "llama_qwen_combined"
    fig, _ = scope["plot_one_model_grid"](
        combined, "llama_qwen_combined", out_dir=out, dataset_order=panel_order
    )
    scope["plt"].close(fig)
    print(f"Verified {len(curves)} curves, each with 250 rounds. Output: {out}")


if __name__ == "__main__":
    main()
