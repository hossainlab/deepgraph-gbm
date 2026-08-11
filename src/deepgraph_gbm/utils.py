"""Shared utilities: seeding, logging, config IO."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml

log = logging.getLogger("deepgraph_gbm")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_label_rules(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
