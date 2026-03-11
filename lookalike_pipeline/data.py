from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class DataBundle:
    people: pd.DataFrame
    segments: pd.DataFrame
    transaction: pd.DataFrame
    offer: pd.DataFrame
    merchant: pd.DataFrame
    financial_account: pd.DataFrame
    offer_seens: pd.DataFrame
    offer_activation: pd.DataFrame
    receipts: pd.DataFrame


FILE_MAP = {
    "people": "prod_clients.csv",
    "segments": "prizm_segments.csv",
    "transaction": "prod_financial_transaction.csv",
    "offer": "t_offer.csv",
    "merchant": "t_merchant.csv",
    "financial_account": "financial_account.csv",
    "offer_seens": "offer_seens.csv",
    "offer_activation": "offer_activation.csv",
}


def _resolve_receipts_path(data_dir: Path) -> Path:
    candidates = [
        data_dir / "receipts.csv",
        data_dir.parent / f"{data_dir.name} 2" / "receipts.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"receipts.csv was not found for {data_dir}. "
        f"Checked: {[str(p) for p in candidates]}"
    )


def load_data(
    data_dir: Path,
    max_transaction_rows: Optional[int] = None,
    max_offer_seens_rows: Optional[int] = None,
) -> DataBundle:
    data_dir = Path(data_dir)
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in FILE_MAP.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        nrows = None
        if key == "transaction" and max_transaction_rows is not None:
            nrows = max_transaction_rows
        elif key == "offer_seens" and max_offer_seens_rows is not None:
            nrows = max_offer_seens_rows
        tables[key] = pd.read_csv(path, nrows=nrows)

    receipts_path = _resolve_receipts_path(data_dir)
    tables["receipts"] = pd.read_csv(receipts_path)
    return DataBundle(**tables)

