# minimal_framework.py
# -*- coding: utf-8 -*-
"""
A tiny yet extensible training framework with DistributedDataParallel (DDP) support.

Run with:
torchrun --standalone --nproc_per_node=NUM_GPUS minimal_framework.py
"""
from __future__ import annotations

import os
import datetime as dt
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Protocol, Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from basedataset import BaseDataset
from basemodel import BaseModel
from basetrainer import BaseTrainer
from config import BaseFitConfig
# -----------------------------
# 1. DDP utility functions
# -----------------------------
def ddp_setup(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Create a process group for DDP.

    Args:
        rank: Local rank of the current process.
        world_size: Total number of processes.
        backend: Torch distributed backend (default ``nccl``).

    Note:
        NCCL backend requires CUDA; fallback to ``gloo`` on CPU.
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # GPU を固定


def ddp_cleanup() -> None:
    """Destroy the default process group."""
    dist.destroy_process_group()


# -----------------------------
# 5. Example user code
# -----------------------------
class ToyDataset(BaseDataset):

    def __init__(self, n_samples: int = 1024) -> None:
        super().__init__()
        self.x = torch.randn(n_samples, 10)
        self.y = (self.x.sum(dim=1) > 0).long()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"feat": self.x[index], "label": self.y[index]}


class MLP(BaseModel):
    """シンプルな多層パーセプトロン."""

    def __init__(self, in_dim: int = 10, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(
        self, batch: Dict[str, Any], epoch: int, step: int
    ) -> torch.Tensor:
        logits = self.forward(batch["feat"].cuda())
        loss = self.criterion(logits, batch["label"].cuda())
        return loss

    def validation_step(self, batch: Dict[str, Any], epoch: int) -> torch.Tensor:
        logits = self.forward(batch["feat"].cuda())
        loss = self.criterion(logits, batch["label"].cuda())
        return loss


def _run(rank: int, cfg: Any) -> None:
    """Function executed per process in DDP."""
    if cfg.world_size > 1:
        ddp_setup(rank, cfg.world_size)
    model = MLP()
    trainer = BaseTrainer(cfg, model, rank)
    ds = ToyDataset()
    trainer.fit(ds)  # toy example ⇒ no val_set
    if cfg.world_size > 1:
        ddp_cleanup()


class MyConfig(BaseFitConfig):
    lr: float = 2e-4          # Overwrite default value
    weight_decay: float = 1e-4


if __name__ == "__main__":
    config = MyConfig(epochs=3)
    if config.world_size > 1:
        mp.spawn(_run, args=(config,), nprocs=config.world_size)
    else:
        _run(rank=0, cfg=config)

