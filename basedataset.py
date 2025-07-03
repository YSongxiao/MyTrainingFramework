from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Any, Dict, List, Tuple


# -----------------------------
# 3. Base classes
# -----------------------------
class BaseDataset(Dataset):
    """A minimal dataset skeleton."""

    def __len__(self) -> int:  # noqa: D401
        """Return dataset length."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, Any]:  # noqa: D401
        """Return one sample dict*.

        Returns:
            Dict[str, Any]: e.g. {"image": Tensor, "label": Tensor}
        """
        raise NotImplementedError