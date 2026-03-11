from __future__ import annotations

import argparse
import time
from pathlib import Path

from lookalike_pipeline.config import TrainConfig
from lookalike_pipeline.data import load_data
from lookalike_pipeline.features import build_feature_artifacts
from lookalike_pipeline.model import train_two_stage_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two-stage lookalike model.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to data version folder, e.g. ./v1")
    parser.add_argument("--model-dir", type=Path, default=Path("./artifacts"), help="Where to save artifacts")
    parser.add_argument("--candidate-k", type=int, default=2000, help="Stage-1 candidate count")
    parser.add_argument("--max-positive-samples", type=int, default=120_000, help="Cap positive pairs for training")
    parser.add_argument("--negative-ratio", type=int, default=3, help="Negatives per positive")
    parser.add_argument("--model-version", type=str, default="1.0", help="Model version label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        candidate_k=args.candidate_k,
        max_positive_samples=args.max_positive_samples,
        negative_ratio=args.negative_ratio,
    )

    t0 = time.perf_counter()
    data = load_data(cfg.data_dir)
    t1 = time.perf_counter()
    print(f"[1/3] data loaded in {t1 - t0:.2f}s")

    features = build_feature_artifacts(
        people=data.people,
        segments=data.segments,
        transaction=data.transaction,
        financial_account=data.financial_account,
        receipts=data.receipts,
        offer=data.offer,
        merchant=data.merchant,
    )
    t2 = time.perf_counter()
    print(f"[2/3] features built in {t2 - t1:.2f}s")

    bundle = train_two_stage_model(
        features=features,
        offer_activation=data.offer_activation,
        offer_seens=data.offer_seens,
        config=cfg,
        model_version=args.model_version,
    )
    bundle.save(cfg.model_dir)
    t3 = time.perf_counter()
    print(f"[3/3] model trained and saved in {t3 - t2:.2f}s")
    print(f"done in {t3 - t0:.2f}s; model dir: {cfg.model_dir.resolve()}")


if __name__ == "__main__":
    main()

