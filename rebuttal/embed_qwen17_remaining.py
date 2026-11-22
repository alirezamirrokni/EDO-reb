"""Restore pinned official Qwen3-1.7B and cache primary GPQA/MMLU representations."""
import os
os.environ['HF_HOME']='/home/mll/alireza/or_runs/hf_restored'
os.environ['HF_XET_CACHE']='/home/mll/alireza/or_runs/hf_restored/xet'
os.environ['OMP_NUM_THREADS']='2'
import hashlib,json
from pathlib import Path
import numpy as np
from huggingface_hub import snapshot_download
from engine import Setting
from models.qwen3_embedding import Qwen3ScoreModel

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'features/qwen17_primary'
MODEL=Path('/home/mll/alireza/or_runs/models/Qwen3-1.7B')
REV='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--llama-primary',action='store_true')
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    print('DOWNLOAD pinned official Qwen3-1.7B',REV,flush=True)
    snapshot_download('Qwen/Qwen3-1.7B',revision=REV,local_dir=str(MODEL),
        allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json','LICENSE'],max_workers=2)
    enc=Qwen3ScoreModel(dict(model_name=str(MODEL),max_length=512,encode_batch_size=4,
        pooling='last',normalize_features=True,append_eos=True,device='cuda:0',torch_dtype='bfloat16',
        cache_embeddings=False,dataset_name='gpqa'))
    for model in (['llama'] if args.llama_primary else ['llama','qwen']):
        for dataset in (['MATH-500','MMLU-Pro','PopQA','GPQA'] if args.llama_primary else ['GPQA','MMLU-Pro']):
            setting=Setting(dataset,model)
            if dataset=='MATH-500':
                enc.prompt_template='Problem:\n\n{question}\n\nModel answer:\n\n{model_answer}'
            elif dataset=='PopQA':
                enc.prompt_template='Question:\n\n{question}\n\nModel answer:\n\n{model_answer}'
            else:
                enc.prompt_template='Multiple-choice question:\n\n{question}\n\nModel answer:\n\n{model_answer}'
            key=f'{model}_{dataset}';frame=setting.pool
            ids=frame.example_id.to_numpy()
            fingerprint=hashlib.sha256(frame[['example_id','question','model_answer']].to_csv(index=False).encode()).hexdigest()
            meta_path=OUT/f'{key}.json';path=OUT/f'{key}.npy'
            done=0
            if meta_path.exists():
                old=json.loads(meta_path.read_text());assert old['input_sha256']==fingerprint
                done=old['completed_rows']
            x=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype='float32',shape=(len(frame),enc.hidden_size))
            for start in range(done,len(frame),32):
                end=min(start+32,len(frame))
                x[start:end]=enc.encode_rows(frame.iloc[start:end][['question','model_answer']].to_dict('records'))
                assert np.isfinite(x[start:end]).all()
                np.testing.assert_allclose(np.linalg.norm(x[start:end],axis=1),1,atol=1e-3)
                x.flush()
                meta_path.write_text(json.dumps(dict(model='Qwen/Qwen3-1.7B',revision=REV,
                    input_sha256=fingerprint,completed_rows=end,total_rows=len(frame),pooling='last',
                    append_eos=True,normalize=True,max_length=512,prompt=enc.prompt_template),indent=2))
                print('EMBED',key,end,len(frame),flush=True)
            np.savez_compressed(OUT/f'{key}.npz',features=np.asarray(x),example_id=ids)
    print('ALL FOUR EMBEDDING CACHES COMPLETE',flush=True)

if __name__=='__main__':main()
