from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# парсинг таблиц
TABLE_TO_FILENAME = {
    "people": "prod_clients.csv",
    "segments": "prizm_segments.csv",
    "transaction": "prod_financial_transaction.csv",
    "offer": "t_offer.csv",
    "merchant": "t_merchant.csv",
    "financial_account": "financial_account.csv",
    "offer_seens": "offer_seens.csv",
    "offer_activation": "offer_activation.csv",
    "offer_reward": "offer_reward.csv",
    "receipts": "receipts.csv",
}

ALLOWED_TABLES = set(TABLE_TO_FILENAME.keys())


@dataclass
class AppState:
    # Храним батчи
    batches: dict[str, dict[str, dict[int, list[dict]]]] = field(default_factory=lambda: {})
    committed_versions: set[str] = field(default_factory=set)
    data_dir: Path = field(default_factory=lambda: Path("./data_store"))
    model_dir: Path = field(default_factory=lambda: Path("./artifacts"))

    # Pipeline
    pipeline_status: str = "idle"
    pipeline_lock: threading.Lock = field(default_factory=threading.Lock)
    pipeline_error: Optional[str] = None

    # Модель
    bundle: Any = None
    model_version: str = "0"
    trained_on: str = ""

    # Последняя валидация
    last_validation_version: str = ""
    last_validation_valid: bool = True
    last_validation_checks_total: int = 0
    last_validation_checks_passed: int = 0
    last_validation_checks_failed: int = 0
    last_validation_failed_checks: list = field(default_factory=list)

    # Последний дрифт
    last_drift_version: str = ""
    last_drift_detected: bool = False
    last_drift_score: float = 0.0
    last_action_taken: str = "none"


    experiments: list = field(default_factory=list)

    @property
    def data_version(self) -> str:
        if self.committed_versions:
            return max(self.committed_versions, key=lambda v: (v,))
        return ""

    @property
    def ready(self) -> bool:
        return self.bundle is not None


_state = AppState()


def get_state() -> AppState:
    return _state
