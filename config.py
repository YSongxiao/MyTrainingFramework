# config.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
from pathlib import Path
from dataclasses import dataclass, field
import torch


@dataclass
class ExperimentConfig:
    """All experiments share these basic fields."""
    log_root: Path = Path("logs")
    seed: int = 42
    world_size: int = torch.cuda.device_count() or 1
    timestamp: str = field(default_factory=lambda: dt.datetime.now().strftime("%Y%m%d_%H%M%S"))

    # helper -------------------------------------------------
    @property
    def run_dir(self) -> Path:
        """logs/20250703_165000 return exclusive log path."""
        return self.log_root / self.timestamp


@dataclass
class BaseFitConfig(ExperimentConfig):
    """Hyper-parameters for *fitting* (train/val)."""
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    num_workers: int = 4
    patience: int = 10          # early-stop
    use_amp: bool = True


@dataclass
class BaseInferConfig(ExperimentConfig):
    """Settings for an *inference* / *test* run."""
    checkpoint: Path = Path("best.pth")
    batch_size: int = 8
    num_workers: int = 2


"""
class MyConfig(BaseFitConfig):
    lr: float = 2e-4          # Overwrite default value
    weight_decay: float = 1e-4
"""
