from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    model_dir: Path
    candidate_k: int = 25000
    max_positive_samples: int = 200_000
    negative_ratio: int = 6
    max_seen_samples_for_pairs: int = 2_500_000
    max_activation_samples_for_priors: int = 1_000_000
    max_seen_samples_for_priors: int = 2_500_000
    random_state: int = 42


@dataclass(frozen=True)
class PredictConfig:
    model_dir: Path
    candidate_k: int = 2000

