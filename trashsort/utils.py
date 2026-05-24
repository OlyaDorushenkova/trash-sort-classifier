import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Reproducibility helper.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.manual_seed(seed)


def get_device() -> torch.device:
    """
    MPS (Apple Silicon) → CPU fallback.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_dirs() -> None:
    """
    Ensure project folders exist.
    """

    dirs = [
        "models",
        "models/checkpoints",
        "plots",
        "data",
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_checkpoint(model, path: str) -> None:
    """
    Save raw torch model weights (optional fallback).
    """

    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: str, device: torch.device):
    """
    Load weights into model.
    """

    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model
