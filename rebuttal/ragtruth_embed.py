"""Frozen Qwen3 features for RAGTruth (train then test rows, same order as run_ragtruth.load_arrays).

Usage: ragtruth_embed.py <model_dir> <max_length> <tag>
Writes rebuttal/results/ragtruth/features_<tag>.npy, resumable via features_<tag>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rebuttal"))

from models.qwen3_embedding import Qwen3ScoreModel  # noqa: E402
from run_ragtruth import RESULTS, compact  # noqa: E402

model_dir, max_length, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
out = RESULTS / f"features_{tag}.npy"
meta_path = RESULTS / f"features_{tag}.json"
data = pd.concat([pd.read_csv(RESULTS / "data_train.csv", dtype={"example_id": str}),
                  pd.read_csv(RESULTS / "data_test.csv", dtype={"example_id": str})], ignore_index=True)
done = json.loads(meta_path.read_text())["completed_rows"] if meta_path.exists() and out.exists() else 0
enc = Qwen3ScoreModel({
    "model_name": model_dir, "max_length": max_length, "encode_batch_size": 1, "pooling": "last",
    "normalize_features": True, "device": "cuda:0", "torch_dtype": "bfloat16", "cache_embeddings": False,
    "prompt_template": "Model response:\n\n{model_answer}\n\nOriginal RAG prompt and evidence:\n\n{question}",
})
dim = int(enc.hidden_size)
arr = (np.lib.format.open_memmap(out, mode="r+") if out.exists()
       else np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(len(data), dim)))
chunk = 16
for start in range(done, len(data), chunk):
    end = min(len(data), start + chunk)
    rows = [{"question": compact(r["question"], 4 * max_length * 3), "model_answer": compact(r["model_answer"], 6000)}
            for r in data.iloc[start:end].to_dict("records")]
    arr[start:end] = enc.encode_rows(rows)
    arr.flush()
    meta_path.write_text(json.dumps({"completed_rows": end, "total_rows": len(data), "model": model_dir,
                                     "max_length": max_length, "hidden_size": dim}))
    if end % 160 == 0 or end == len(data):
        print(f"[embed {tag}] {end}/{len(data)}", flush=True)
print(f"[embed {tag}] done", flush=True)
