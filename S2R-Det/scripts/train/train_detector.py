import argparse
from mmengine.config import Config
from mmengine.runner import Runner

p = argparse.ArgumentParser()
p.add_argument("config")
p.add_argument("--work-dir", default=None)
p.add_argument("--resume", action="store_true")
args = p.parse_args()

cfg = Config.fromfile(args.config)

if args.work_dir:
    cfg.work_dir = args.work_dir

if args.resume:
    cfg.resume = True

Runner.from_cfg(cfg).train()
