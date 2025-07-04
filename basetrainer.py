import torch
import torch.nn as nn
import torch.optim as optim
import datetime as dt
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from typing import Protocol, Any, Dict, List, Tuple
from pathlib import Path
from basemodel import BaseModel
from basedataset import BaseDataset
from dataclasses import dataclass, asdict
from torch.optim import lr_scheduler


class _FitCfgProto(Protocol):
    # Trainer 只关心下面这些字段，别的字段会被忽略
    epochs: int
    batch_size: int
    lr: float
    num_workers: int
    patience: int
    use_amp: bool
    world_size: int
    run_dir: Path   # ← ExperimentConfig が提供


class BaseTrainer:
    def __init__(self, cfg: _FitCfgProto, model: BaseModel, rank: int = 0) -> None:
        self.cfg = cfg
        self.rank = rank
        self.device = torch.device(rank)
        self.model = model.to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.scaler = GradScaler(enabled=cfg.use_amp)

        # ---- LR Scheduler ----
        self.scheduler = self._build_scheduler()

        # DDP warper
        if cfg.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[rank]
            )

        # 监控变量
        self.best_loss: float = float("inf")
        self.no_improve: int = 0        # early-stop 计数

        # 日志目录
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = cfg.log_dir / ts
        if self.rank == 0:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    # ---------- dataloader ----------
    def _dataloader(self, ds: BaseDataset, shuffle: bool = True) -> DataLoader:
        if self.cfg.world_size > 1:
            sampler = DistributedSampler(ds, shuffle=shuffle)
            shuffle = False
        else:
            sampler = None
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    # ---------- training ----------
    def fit(self, train_set: BaseDataset, val_set: BaseDataset | None = None) -> None:
        train_loader = self._dataloader(train_set, shuffle=True)
        val_loader = self._dataloader(val_set, shuffle=False) if val_set else None

        early_stop = False
        for epoch in range(1, self.cfg.epochs + 1):
            if self.cfg.world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            # --- train one epoch ---
            self.model.train()
            for step, batch in enumerate(train_loader, 1):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with autocast(enabled=self.cfg.use_amp):
                    loss = self.model.training_step(batch, epoch, step)

                if self.cfg.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if step % 10 == 0 and self.rank == 0:
                    print(f"[{epoch}/{self.cfg.epochs}] "
                          f"step {step:04d}  loss={loss.item():.4f}")

            # --- validation & checkpoint ---
            if val_loader:
                val_loss = self._validate(val_loader, epoch)  # rank0 返回均值
                self._checkpoint_logic(val_loss)

            # --- early stop 广播 ---
            if self.cfg.world_size > 1:
                flag = [early_stop]
                # rank0 决定 early-stop，再广播
                if self.rank == 0:
                    flag[0] = self.no_improve >= self.cfg.patience
                dist.broadcast_object_list(flag, src=0)
                early_stop = bool(flag[0])
            else:
                early_stop = self.no_improve >= self.cfg.patience

            if early_stop:
                if self.rank == 0:
                    print(f"Early stopping at epoch {epoch}")
                break

        if self.rank == 0:
            self._save_checkpoint(filename="last.pth")   # 最后一次

    # ---------- validation ----------
    def _validate(self, loader: DataLoader, epoch: int) -> float:
        self.model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with autocast(enabled=self.cfg.use_amp):
                    loss = self.model.validation_step(batch, epoch)
                losses.append(float(loss))
        avg_loss = float(sum(losses) / len(losses))

        if self.rank == 0:
            print(f"Epoch {epoch}: val_loss={avg_loss:.4f}")
        return avg_loss

    # ---------- checkpoint & early-stop helper ----------
    def _checkpoint_logic(self, val_loss: float) -> None:
        """根据 val_loss 更新 best & early-stop 计数"""
        improved = val_loss < self.best_loss
        if improved:
            self.best_loss = val_loss
            self.no_improve = 0
            if self.rank == 0:
                self._save_checkpoint(filename="best.pth")
                print(f"  ↳ New best! ({val_loss:.4f})")
        else:
            self.no_improve += 1

    def _save_checkpoint(self, filename: str = "checkpoint.pth") -> None:
        """Save state_dict (+amp scaler)"""
        save_dict = {
            "model": (self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel)
                      else self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_loss": self.best_loss,
            "cfg": asdict(self.cfg),
        }
        torch.save(save_dict, self.run_dir / filename)
        print(f"Saved {filename} to {self.run_dir}")

    def _build_scheduler(self):
        name = self.cfg.scheduler_name
        kw = self.cfg.scheduler_kwargs
        if name is None:
            return None
        if name.lower() == "steplr":
            return lr_scheduler.StepLR(self.optimizer, **kw)
        if name.lower() == "cosine":
            return lr_scheduler.CosineAnnealingLR(self.optimizer, **kw)
        if name.lower() in {"plateau", "reducelronplateau"}:
            return lr_scheduler.ReduceLROnPlateau(self.optimizer, **kw)
        raise ValueError(f"Unknown scheduler {name}")
