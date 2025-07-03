import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple


class BaseModel(nn.Module):
    """A minimal model skeleton."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
        """Forward pass."""
        raise NotImplementedError

    # ---- optional hooks ----
    def training_step(
        self, batch: Dict[str, Any], epoch: int, step: int
    ) -> torch.Tensor:
        """Override for custom loss.

        Returns:
            Loss tensor to backward.
        """
        raise NotImplementedError

    def validation_step(self, batch: Dict[str, Any], epoch: int) -> torch.Tensor | None:
        """Override for custom val metrics (optional)."""
        return None
