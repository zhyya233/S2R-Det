from __future__ import annotations
import argparse, hashlib, json, random, subprocess, time
from pathlib import Path
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet.apis import init_detector
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules
from s2r_det.training.runtime import dataset_name_to_index, failure_image_pools, file_names
from s2r_det.training.adaptation_variants import freeze_bn_running_stats

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):
            h.update(b)
    return h.hexdigest()

def write_subset(path, rows):
    Path(path).write_text("".join("{}\t{}\t{}\n".format(i,n,"" if v is None else "{:.12f}".format(v)) for i,n,v in rows))

def args_parser():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--failure-parquet",required=True)
    p.add_argument("--retention-list",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--seed",type=int,default=0)
    return p.parse_args()

def main():
    a=args_parser()
    out=Path(a.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    register_all_modules()
    cfg=Config.fromfile(a.config)
    dataset=DATASETS.build(cfg.train_dataloader.dataset)
    dataset.full_init()
    name_to_idx=dataset_name_to_index(dataset)
    idx_to_name={v:k for k,v in name_to_idx.items()}
    assert len(idx_to_name)==len(dataset)

    pools=failure_image_pools(a.failure_parquet)
    failure=set()
    for names in pools.values():
        failure.update(names)
    retention=set(file_names(a.retention_list))
    reference=failure | retention
    missing=sorted(reference-set(name_to_idx))
    assert not missing, "reference names absent from train dataset: {}".format(missing[:5])
    k=len(reference)
    assert 0 < k <= len(dataset)

    ref_rows=sorted((name_to_idx[n],n,None) for n in reference)
    rng=random.Random(a.seed)
    random_idx=sorted(rng.sample(range(len(dataset)),k))
    random_rows=[(i,idx_to_name[i],None) for i in random_idx]
    write_subset(out/"reference_union.tsv",ref_rows)
    write_subset(out/"random_subset_seed0.tsv",random_rows)

    print("TRAIN_IMAGES={}".format(len(dataset)),flush=True)
    print("REFERENCE_UNION={}".format(k),flush=True)
    print("FAILURE_UNIQUE={}".format(len(failure)),flush=True)
    print("RETENTION_UNIQUE={}".format(len(retention)),flush=True)
    print("OVERLAP={}".format(len(failure & retention)),flush=True)

    assert torch.cuda.is_available(), "CUDA required"
    model=init_detector(a.config,a.checkpoint,device="cuda:0")
    if hasattr(model.data_preprocessor,"batch_augments"):
        model.data_preprocessor.batch_augments=None
    for p in model.parameters():
        p.requires_grad_(False)
    model.train()
    freeze_bn_running_stats(model)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    scores=[]
    start=time.time()
    with torch.no_grad():
        for i in range(len(dataset)):
            si=a.seed*1000003+i
            random.seed(si)
            np.random.seed(si % (2**32))
            torch.manual_seed(si)
            torch.cuda.manual_seed_all(si)
            batch=pseudo_collate([dataset[i]])
            processed=model.data_preprocessor(batch,training=True)
            loss_dict=model._run_forward(processed,mode="loss")
            det_loss,_=model.parse_losses(loss_dict)
            loss=float(det_loss.detach().cpu())
            assert np.isfinite(loss)
            scores.append((loss,idx_to_name[i],i))
            if (i+1)%100==0 or i+1==len(dataset):
                print("SCORE_PROGRESS={}/{} elapsed_s={:.1f}".format(i+1,len(dataset),time.time()-start),flush=True)

    scores.sort(key=lambda x:(-x[0],x[1]))
    top=scores[:k]
    (out/"frozen_b00_loss_ranking.tsv").write_text("".join("{}\t{}\t{:.12f}\t{}\n".format(i,n,v,r+1) for r,(v,n,i) in enumerate(scores)))
    write_subset(out/"top_loss_subset.tsv",[(i,n,v) for v,n,i in top])

    files=["reference_union.tsv","random_subset_seed0.tsv","frozen_b00_loss_ranking.tsv","top_loss_subset.tsv"]
    manifest={
      "stage":"M00",
      "seed":a.seed,
      "source_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
      "train_image_count":len(dataset),
      "reference_unique_count":k,
      "failure_unique_count":len(failure),
      "retention_unique_count":len(retention),
      "failure_retention_overlap_count":len(failure & retention),
      "random_subset_count":len(random_rows),
      "top_loss_subset_count":len(top),
      "top_loss_definition":"Frozen B00 deterministic single-image detection loss under common train pipeline; descending loss; file_name ascending tie-break",
      "checkpoint_sha256":sha256(a.checkpoint),
      "failure_parquet_sha256":sha256(a.failure_parquet),
      "retention_list_sha256":sha256(a.retention_list),
      "files":{n:{"sha256":sha256(out/n)} for n in files}
    }
    (out/"support_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")

    print("M00_SUPPORT_FREEZE=COMPLETE",flush=True)
    print("REFERENCE_UNION={}".format(k),flush=True)
    print("RANDOM_COUNT={}".format(len(random_rows)),flush=True)
    print("TOP_LOSS_COUNT={}".format(len(top)),flush=True)

if __name__=="__main__":
    main()
