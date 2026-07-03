"""Device selection: prefer Apple-Silicon MPS, fall back to CPU."""
from __future__ import annotations

import torch


def pick_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
