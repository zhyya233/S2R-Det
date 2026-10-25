"""Full-image worker loader. Worker RNG follows(base_seed+worker_id) mod2**32."""

import random
from functools import partial
import numpy as np
import torch


class RawImages:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, key):
        step, source, image_id = key
        rgb, meta = self.dataset.load(image_id)
        return {
            "step": step,
            "source": source,
            "image_id": image_id,
            "rgb": rgb,
            "meta": meta,
            "gt": self.dataset.gt(image_id),
            "labels": self.dataset.labels,
        }


def initialize_worker(worker_id, seed):
    value = (seed + worker_id) % 2**32
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def keep_records(records):
    return records


class StreamBatches:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        for b in self.batches:
            yield [(b["step"], b["source"], i) for i in b["image_ids"]]


def raw_training_batches(dataset, batches, seed, workers=8):
    if workers < 0:
        raise ValueError("worker count must be nonnegative")
    return torch.utils.data.DataLoader(
        RawImages(dataset),
        batch_sampler=StreamBatches(batches),
        num_workers=workers,
        collate_fn=keep_records,
        worker_init_fn=partial(initialize_worker, seed=seed),
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=workers > 0,
    )
