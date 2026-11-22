"""Resumable RAGTruth experiment: Llama-70B selectors and Qwen3-1.7B EDIT-REP."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from methods.ours_llm import OursLLMSelection  # noqa: E402
from models.openrouter import OpenRouterLLM  # noqa: E402
from models.qwen3_embedding import Qwen3ScoreModel  # noqa: E402
from utils import sigmoid  # noqa: E402

DATA = REPO / "rebuttal/data/ragtruth"
RESULTS = REPO / "rebuttal/results/ragtruth"
FIGURES = REPO / "figures/ragtruth"
RESPONSE_MODEL = "llama-2-70b-chat"
SELECTOR_MODEL = "meta-llama/llama-3.3-70b-instruct"
SEEDS = [7, 17, 27, 37, 47]
BATCH_SIZE = 10
BUDGET = 5
ALLOWANCE = 0.03
GAMMAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.068, 0.1, 0.15, 0.2, 0.3,
          0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
LRS = [0.5, 1.0, 2.0, 4.0, 8.0]
WARMS = [0, 5, 10]

SCORE_SYSTEM = """You are an expert evaluator of retrieval-augmented model responses.
Estimate whether a human editor would need to correct the response because it contains information unsupported by or contradictory to the supplied evidence.
Return only valid JSON. Do not include explanations, markdown, or extra text."""

SCORE_PROMPT = """Estimate the probability that the model response contains at least one substantive hallucination requiring human correction.

Judge claims against the retrieved evidence in the original prompt. Correct information absent from the evidence is not automatically an error unless it conflicts with the task's requirement to rely on the evidence. Treat unsupported or contradictory claims, incorrect refusals, and material factual distortions as requiring correction.

Return only: {{"score": 0.73}}

Original RAG prompt and evidence:
{question}

Model response:
{model_answer}"""

SELECT_SYSTEM = """You are an expert review-selection system for retrieval-augmented generation.
Privately verify each response against its supplied evidence, rank the items by probability of requiring substantive correction, and use the available review budget.
Return only valid JSON and do not reveal your analysis."""

SELECT_PROMPT = """Select the model responses most likely to contain information unsupported by or contradictory to their retrieved evidence.

Rules:
- Independently verify every response against the evidence in its original prompt.
- Rank all items by probability of requiring substantive human correction.
- Spend as much of the budget {budget} as possible without exceeding it; costs are listed per item.
- Use no external gold answer.
- Return only valid JSON. With unit costs and budget 5, return exactly five indices, for example:
  {{"selected_indices": [0, 3, 4, 7, 9]}}

Batch:
{items}"""

_TLS = threading.local()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float))
    tmp.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def status(phase: str, **kwargs) -> None:
    value = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "phase": phase, **kwargs}
    atomic_json(RESULTS / "status.json", value)
    print(json.dumps(value, default=float), flush=True)


def compact(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    return text[:head] + "\n...[middle truncated]...\n" + text[-tail:]


def effective_label(labels) -> int:
    if not isinstance(labels, list):
        return 0
    return int(any(not bool((label or {}).get("implicit_true", False)) for label in labels))


def prepare() -> None:
    response_path = DATA / "response.jsonl"
    source_path = DATA / "source_info.jsonl"
    if not response_path.exists() or not source_path.exists():
        raise FileNotFoundError("Download official RAGTruth response.jsonl and source_info.jsonl first")
    responses = pd.read_json(response_path, lines=True)
    sources = pd.read_json(source_path, lines=True)
    responses = responses[responses["model"] == RESPONSE_MODEL].copy()
    responses["A"] = responses["labels"].map(effective_label).astype(int)
    source_columns = sources[["source_id", "task_type", "source", "prompt"]]
    frame = responses.merge(source_columns, on="source_id", validate="one_to_one")
    if frame["source_id"].duplicated().any():
        raise AssertionError("Expected exactly one fixed-model response per source")
    frame = frame.rename(columns={"id": "example_id", "prompt": "question",
                                  "response": "model_answer"})
    keep = ["example_id", "source_id", "split", "task_type", "source", "model",
            "question", "model_answer", "A", "quality"]
    frame = frame[keep].sort_values(["split", "task_type", "source_id"]).reset_index(drop=True)
    frame["cost"] = 1.0
    for split in ["train", "test"]:
        part = frame[frame.split == split].reset_index(drop=True)
        atomic_csv(part, RESULTS / f"data_{split}.csv")
    audit = (frame.groupby(["split", "task_type"])["A"]
             .agg(rows="size", edits="sum", edit_rate="mean").reset_index())
    atomic_csv(audit, RESULTS / "data_audit.csv")
    status("prepared", train=int((frame.split == "train").sum()),
           test=int((frame.split == "test").sum()), test_edit_rate=float(
               frame.loc[frame.split == "test", "A"].mean()))


def llm_config(system_prompt: str, usage_name: str) -> dict:
    return {
        "provider": "openrouter",
        "model_name": SELECTOR_MODEL,
        "temperature": 0.0,
        "max_output_tokens": 64 if usage_name == "scores" else 256,
        "request_timeout": 180,
        "retry_attempts": 10,
        "retry_sleep": 2.0,
        "min_seconds_between_calls": 0.0,
        "system_prompt": system_prompt,
        "provider_routing": {"only": ["CoreWeave"], "allow_fallbacks": False},
        "usage_log": str(RESULTS / f"openrouter_usage_{usage_name}.jsonl"),
    }


def thread_llm(kind: str) -> OpenRouterLLM:
    attr = f"llm_{kind}"
    client = getattr(_TLS, attr, None)
    if client is None:
        system = SCORE_SYSTEM if kind == "score" else SELECT_SYSTEM
        client = OpenRouterLLM(llm_config(system, "scores" if kind == "score" else "select"))
        setattr(_TLS, attr, client)
    return client


def score_one(row: dict) -> dict:
    prompt = SCORE_PROMPT.format(
        question=compact(row["question"], 12000),
        model_answer=compact(row["model_answer"], 6000),
    )
    text = thread_llm("score").generate(prompt)
    score = OursLLMSelection._parse_score(text)
    return {"example_id": str(row["example_id"]), "score": float(score)}


def score_llm(workers: int) -> None:
    load_dotenv(REPO / ".env")
    out_path = RESULTS / "edit_llm_scores.csv"
    existing = pd.read_csv(out_path, dtype={"example_id": str}) if out_path.exists() else pd.DataFrame()
    done = set(existing.example_id.astype(str)) if len(existing) else set()
    data = pd.concat([pd.read_csv(RESULTS / "data_train.csv", dtype={"example_id": str}),
                      pd.read_csv(RESULTS / "data_test.csv", dtype={"example_id": str})], ignore_index=True)
    missing = [row for row in data.to_dict("records") if str(row["example_id"]) not in done]
    rows = existing.to_dict("records") if len(existing) else []
    status("scoring_edit_llm", completed=len(done), total=len(data), workers=workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(score_one, row): row for row in missing}
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index % 20 == 0 or index == len(missing):
                atomic_csv(pd.DataFrame(rows).drop_duplicates("example_id", keep="last"), out_path)
                status("scoring_edit_llm", completed=len(done) + index, total=len(data), workers=workers)
    status("edit_llm_scored", completed=len(data), total=len(data))


def permutation(frame: pd.DataFrame, seed: int, training: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    if training:
        order = order[: len(order) - len(order) % BATCH_SIZE]
    out = frame.iloc[order].reset_index(drop=True).copy()
    out["t"] = np.arange(len(out)) // BATCH_SIZE
    out["batch_pos"] = np.arange(len(out)) % BATCH_SIZE
    return out


def parse_indices(text: str) -> list[int]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("selected_indices"), list):
            return [int(x) for x in obj["selected_indices"]]
    except Exception:
        pass
    match = re.search(r"\{.*\}", str(text), re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return [int(x) for x in obj.get("selected_indices", [])]
        except Exception:
            pass
    return [int(x) for x in re.findall(r"-?\d+", str(text))]


def select_batch(batch: pd.DataFrame) -> list[int]:
    chunks = []
    for i, row in batch.reset_index(drop=True).iterrows():
        chunks.append(
            f"Index: {i}\nCost: 1\nTask: {row['task_type']}\n"
            f"Original prompt/evidence:\n{compact(row['question'], 4500)}\n"
            f"Model response:\n{compact(row['model_answer'], 2500)}"
        )
    prompt = SELECT_PROMPT.format(budget=BUDGET, items="\n\n---\n\n".join(chunks))
    for attempt in range(3):
        text = thread_llm("select").generate(prompt)
        selected, seen = [], set()
        for value in parse_indices(text):
            if 0 <= value < len(batch) and value not in seen and len(selected) < BUDGET:
                selected.append(value)
                seen.add(value)
        if selected:
            return selected
        print(f"[select] unparseable selection (attempt {attempt + 1}): {str(text)[:200]!r}", flush=True)
    raise RuntimeError("LLM-Select returned no valid indices after 3 attempts")


def select_llm(workers: int) -> None:
    load_dotenv(REPO / ".env")
    jobs = [("train", 7)] + [("test", seed) for seed in SEEDS]
    for split, seed in jobs:
        path = RESULTS / f"llm_select_{split}_seed{seed}.csv"
        source = pd.read_csv(RESULTS / f"data_{split}.csv", dtype={"example_id": str})
        stream = permutation(source, seed, training=(split == "train"))
        existing = pd.read_csv(path, dtype={"example_id": str}) if path.exists() else pd.DataFrame()
        completed = set(existing.t.astype(int)) if len(existing) else set()
        batches = [(int(t), b.copy()) for t, b in stream.groupby("t", sort=True) if int(t) not in completed]
        rows = existing.to_dict("records") if len(existing) else []
        status("selecting_llm", split=split, seed=seed, completed=len(completed),
               total=int(stream.t.nunique()), workers=workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(select_batch, batch): (t, batch) for t, batch in batches}
            for index, future in enumerate(as_completed(futures), 1):
                t, batch = futures[future]
                picked = set(future.result())
                for local, (_, row) in enumerate(batch.reset_index(drop=True).iterrows()):
                    rows.append({"example_id": str(row.example_id), "t": t,
                                 "batch_pos": local, "A": int(row.A),
                                 "selected": int(local in picked), "cost": 1.0})
                if index % 10 == 0 or index == len(batches):
                    atomic_csv(pd.DataFrame(rows).drop_duplicates(["t", "batch_pos"], keep="last")
                               .sort_values(["t", "batch_pos"]), path)
                    status("selecting_llm", split=split, seed=seed,
                           completed=len(completed) + index, total=int(stream.t.nunique()), workers=workers)
    status("llm_select_complete", streams=len(jobs))


def embed() -> None:
    output = RESULTS / "qwen17_features.npy"
    meta_path = RESULTS / "qwen17_features.json"
    data = pd.concat([pd.read_csv(RESULTS / "data_train.csv", dtype={"example_id": str}),
                      pd.read_csv(RESULTS / "data_test.csv", dtype={"example_id": str})], ignore_index=True)
    completed = 0
    if meta_path.exists() and output.exists():
        meta = json.loads(meta_path.read_text())
        completed = int(meta.get("completed_rows", 0))
    status("embedding_qwen17", completed=completed, total=len(data))
    encoder = Qwen3ScoreModel({
        "model_name": "/home/mll/ali/models/Qwen3-1.7B", "max_length": 2048,
        "encode_batch_size": 2, "pooling": "last", "normalize_features": True,
        "device": "cuda:0", "torch_dtype": "bfloat16", "cache_embeddings": False,
        "prompt_template": "Model response:\n\n{model_answer}\n\nOriginal RAG prompt and evidence:\n\n{question}",
    })
    dim = int(encoder.hidden_size)
    if not output.exists():
        arr = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(len(data), dim))
    else:
        arr = np.lib.format.open_memmap(output, mode="r+")
        if arr.shape != (len(data), dim):
            raise ValueError(f"Feature shape mismatch: {arr.shape} != {(len(data), dim)}")
    chunk = 16
    for start in range(completed, len(data), chunk):
        end = min(len(data), start + chunk)
        rows = [{"question": compact(r["question"], 14000),
                 "model_answer": compact(r["model_answer"], 6000)}
                for r in data.iloc[start:end].to_dict("records")]
        arr[start:end] = encoder.encode_rows(rows)
        arr.flush()
        atomic_json(meta_path, {"completed_rows": end, "total_rows": len(data),
                                "hidden_size": dim, "model": "Qwen/Qwen3-1.7B",
                                "max_length": 2048})
        status("embedding_qwen17", completed=end, total=len(data))
    status("qwen17_embedded", completed=len(data), total=len(data))


def round_metrics(labels: np.ndarray, selected: np.ndarray) -> tuple[float, float, float]:
    n = int(selected.sum())
    unselected = 1 - selected
    u = int(unselected.sum())
    type_i = 0.0 if n == 0 else float((selected * (1 - labels)).sum() / n)
    type_ii = 0.0 if u == 0 else float((unselected * labels).sum() / u)
    return type_i, type_ii, float(n)


def choose(scores: np.ndarray, alpha: float, t: int, warm: int) -> np.ndarray:
    rng = np.random.default_rng(7 + 1000003 * t)
    if t < warm:
        order = rng.permutation(len(scores))
        picked = order[:min(BUDGET, len(scores))]
    else:
        eligible = np.flatnonzero(scores >= alpha)
        rng.shuffle(eligible)
        eligible = eligible[np.argsort(-scores[eligible], kind="mergesort")]
        picked = eligible[:BUDGET]
    selected = np.zeros(len(scores), dtype=int)
    selected[picked] = 1
    return selected


def simulate_static(labels: np.ndarray, scores: np.ndarray, gamma: float, epsilon: float,
                    warm: int, details: bool = False):
    alpha = 0.0
    rows = []
    for t, start in enumerate(range(0, len(labels), BATCH_SIZE)):
        a = labels[start:start + BATCH_SIZE]
        eta = scores[start:start + BATCH_SIZE]
        selected = choose(eta, alpha, t, warm)
        ti, tii, cost = round_metrics(a, selected)
        alpha = max(0.0, alpha + gamma * (ti - epsilon))
        rows.append((ti, tii, cost, selected.copy()))
    if details:
        return rows
    values = np.array([[r[0], r[1], r[2]] for r in rows])
    return values.mean(axis=0)


def simulate_rep(labels: np.ndarray, features: np.ndarray, gamma: float, lr: float,
                 epsilon: float, warm: int, details: bool = False):
    rng = np.random.default_rng(7 + 1729)
    theta = rng.normal(0.0, 0.1, features.shape[1]).astype(np.float32)
    alpha = 0.0
    rows = []
    for t, start in enumerate(range(0, len(labels), BATCH_SIZE)):
        a = labels[start:start + BATCH_SIZE]
        x = features[start:start + BATCH_SIZE]
        eta = sigmoid(x @ theta).astype(np.float32)
        selected = choose(eta, alpha, t, warm)
        ti, tii, cost = round_metrics(a, selected)
        alpha = max(0.0, alpha + gamma * (ti - epsilon))
        n = selected.sum()
        if n:
            grad = ((selected * (eta - a))[:, None] * x).sum(axis=0) / n
            theta = (theta - lr * grad).astype(np.float32)
        rows.append((ti, tii, cost, selected.copy()))
    if details:
        return rows
    values = np.array([[r[0], r[1], r[2]] for r in rows])
    return values.mean(axis=0)


def load_arrays(split: str):
    train = pd.read_csv(RESULTS / "data_train.csv", dtype={"example_id": str})
    test = pd.read_csv(RESULTS / "data_test.csv", dtype={"example_id": str})
    all_data = pd.concat([train, test], ignore_index=True)
    scores = pd.read_csv(RESULTS / "edit_llm_scores.csv", dtype={"example_id": str})
    score_map = scores.set_index("example_id").score
    all_scores = all_data.example_id.map(score_map).to_numpy(float)
    all_features = np.load(RESULTS / "qwen17_features.npy", mmap_mode="r")
    if split == "train":
        return train, all_scores[:len(train)], all_features[:len(train)]
    return test, all_scores[len(train):], all_features[len(train):]


def stream_arrays(frame, scores, features, seed, training):
    order = np.random.default_rng(seed).permutation(len(frame))
    if training:
        order = order[:len(order) - len(order) % BATCH_SIZE]
    return frame.A.to_numpy(int)[order], scores[order], np.asarray(features[order])


def llm_select_metrics(path: Path):
    frame = pd.read_csv(path).sort_values(["t", "batch_pos"])
    vals = []
    for _, batch in frame.groupby("t", sort=True):
        vals.append(round_metrics(batch.A.to_numpy(int), batch.selected.to_numpy(int)))
    return np.asarray(vals)


def tune() -> None:
    frame, scores, features = load_arrays("train")
    streams = [stream_arrays(frame, scores, features, seed, True) for seed in SEEDS]
    select_values = llm_select_metrics(RESULTS / "llm_select_train_seed7.csv")
    select_type_i, select_type_ii = select_values[:, 0].mean(), select_values[:, 1].mean()
    cap = min(0.8, float(select_type_i + ALLOWANCE))
    epsilons = [round(float(x), 3) for x in np.arange(0.05, cap, 0.05)] + [cap]
    epsilons = sorted(set(epsilons))
    rows = []
    status("tuning", cap=cap, epsilon_count=len(epsilons))
    for epsilon in epsilons:
        best_static = None
        for gamma in GAMMAS:
            for warm in WARMS:
                metrics = np.mean([simulate_static(a, s, gamma, epsilon, warm)
                                   for a, s, _ in streams], axis=0)
                candidate = (metrics[1], metrics[2], gamma, warm, metrics[0])
                if metrics[0] <= epsilon + 1e-12 and (best_static is None or candidate < best_static):
                    best_static = candidate
        best_rep = None
        for gamma in GAMMAS:
            for lr in LRS:
                for warm in WARMS:
                    metrics = np.mean([simulate_rep(a, x, gamma, lr, epsilon, warm)
                                       for a, _, x in streams], axis=0)
                    candidate = (metrics[1], metrics[2], gamma, lr, warm, metrics[0])
                    if metrics[0] <= epsilon + 1e-12 and (best_rep is None or candidate < best_rep):
                        best_rep = candidate
        if best_static and best_rep:
            rows.append({"epsilon": epsilon, "rep_type_ii": best_rep[0],
                         "rep_cost": best_rep[1], "rep_gamma": best_rep[2],
                         "rep_lr": best_rep[3], "rep_warm": best_rep[4],
                         "rep_type_i": best_rep[5], "llm_type_ii": best_static[0],
                         "llm_cost": best_static[1], "llm_gamma": best_static[2],
                         "llm_warm": best_static[3], "llm_type_i": best_static[4]})
        atomic_csv(pd.DataFrame(rows), RESULTS / "tuning.csv")
        status("tuning", cap=cap, completed_epsilon=epsilon, epsilon_count=len(epsilons))
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("No jointly feasible EDIT-REP/EDIT-LLM training configuration")
    eligible = table[table.rep_type_ii <= min(select_type_ii, table.llm_type_ii.min()) + 0.005]
    pool = eligible if len(eligible) else table
    best = pool.sort_values(["rep_type_ii", "rep_cost", "epsilon"]).iloc[0].to_dict()
    best.update(train_llm_select_type_i=float(select_type_i),
                train_llm_select_type_ii=float(select_type_ii), type_i_cap=cap,
                selection_note=("REP best/co-best on training" if len(eligible)
                                else "No best/co-best REP point on training; minimum REP Type-II selected"))
    atomic_json(RESULTS / "selected.json", best)
    status("tuned", **best)


def cumulative_rows(method, seed, values):
    values = np.asarray(values)
    denom = np.arange(1, len(values) + 1)
    return pd.DataFrame({"method": method, "seed": seed, "round": np.arange(1, len(values) + 1),
                         "type_i": np.cumsum(values[:, 0]) / denom,
                         "type_ii": np.cumsum(values[:, 1]) / denom,
                         "cost": np.cumsum(values[:, 2]) / denom})


def evaluate() -> None:
    selected = json.loads((RESULTS / "selected.json").read_text())
    frame, scores, features = load_arrays("test")
    all_rows = []
    raw_rows = []
    for seed in SEEDS:
        a, s, x = stream_arrays(frame, scores, features, seed, False)
        rep = simulate_rep(a, x, selected["rep_gamma"], selected["rep_lr"],
                           selected["epsilon"], int(selected["rep_warm"]), True)
        llm = simulate_static(a, s, selected["llm_gamma"], selected["epsilon"],
                              int(selected["llm_warm"]), True)
        for method, detail in [("EDIT-REP", rep), ("EDIT-LLM", llm)]:
            vals = np.array([[r[0], r[1], r[2]] for r in detail])
            all_rows.append(cumulative_rows(method, seed, vals))
            raw_rows.append({"method": method, "seed": seed, "type_i": vals[:, 0].mean(),
                             "type_ii": vals[:, 1].mean(), "cost": vals[:, 2].mean()})
        select_vals = llm_select_metrics(RESULTS / f"llm_select_test_seed{seed}.csv")
        all_rows.append(cumulative_rows("LLM-Select", seed, select_vals))
        raw_rows.append({"method": "LLM-Select", "seed": seed,
                         "type_i": select_vals[:, 0].mean(),
                         "type_ii": select_vals[:, 1].mean(),
                         "cost": select_vals[:, 2].mean()})
    metrics = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame(raw_rows)
    atomic_csv(metrics, RESULTS / "running_metrics.csv")
    atomic_csv(summary, RESULTS / "test_summary_by_seed.csv")
    aggregate = summary.groupby("method").agg(
        type_i=("type_i", "mean"), type_ii=("type_ii", "mean"), cost=("cost", "mean"),
        type_i_sd=("type_i", "std"), type_ii_sd=("type_ii", "std"), cost_sd=("cost", "std"),
    ).reset_index()
    atomic_csv(aggregate, RESULTS / "test_summary.csv")
    status("evaluated", summary=aggregate.to_dict("records"))


def plot() -> None:
    metrics = pd.read_csv(RESULTS / "running_metrics.csv")
    selected = json.loads((RESULTS / "selected.json").read_text())
    sns.set_theme(context="paper", style="darkgrid")
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                         "axes.titleweight": "bold", "axes.labelweight": "bold",
                         "font.weight": "bold", "pdf.fonttype": 42})
    colors = {"EDIT-REP": "#ff7f0e", "EDIT-LLM": "#1f77b4", "LLM-Select": "#d62728"}
    order = ["EDIT-REP", "EDIT-LLM", "LLM-Select"]
    panels = [("type_ii", "Type-II error"), ("type_i", "Type-I error"),
              ("cost", "Review cost")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, (metric, title) in zip(axes, panels):
        for method in order:
            data = metrics[metrics.method == method]
            grouped = data.groupby("round")[metric]
            mean = grouped.mean()
            std = grouped.std().fillna(0)
            ci = 1.96 * std / math.sqrt(len(SEEDS))
            ax.plot(mean.index, mean, label=method, color=colors[method], linewidth=2.8)
            ax.fill_between(mean.index, mean - ci, mean + ci, color=colors[method], alpha=0.15)
        if metric == "type_i":
            ax.axhline(float(selected["epsilon"]), color="#555555", linestyle="--", linewidth=1.4)
        if metric == "cost":
            ax.axhline(BUDGET, color="#555555", linestyle="--", linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("Round")
        if metric != "cost":
            ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.03), fontsize=15)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"ragtruth_typeii_typei_cost.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    status("complete", figure=str(FIGURES / "ragtruth_typeii_typei_cost.pdf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "score-llm", "select-llm", "embed",
                                          "tune", "evaluate", "plot"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "score-llm": lambda: score_llm(args.workers),
     "select-llm": lambda: select_llm(args.workers), "embed": embed,
     "tune": tune, "evaluate": evaluate, "plot": plot}[args.stage]()


if __name__ == "__main__":
    main()
