from __future__ import annotations

import argparse
import time
from pathlib import Path

from lookalike_pipeline.model import TrainingArtifacts, predict_top_users


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lookalike inference for one offer.")
    parser.add_argument("--model-dir", type=Path, default=Path("./artifacts"), help="Trained artifact directory")
    parser.add_argument("--merchant-id", type=int, required=True, help="merchant_id_offer")
    parser.add_argument("--offer-id", type=int, required=True, help="offer_id")
    parser.add_argument("--top-n", type=int, default=100, help="How many users to return")
    parser.add_argument("--candidate-k", type=int, default=2000, help="Stage-1 candidate pool size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = TrainingArtifacts.load(args.model_dir)
    t0 = time.perf_counter()
    pred = predict_top_users(
        bundle=bundle,
        merchant_id=args.merchant_id,
        offer_id=args.offer_id,
        top_n=args.top_n,
        candidate_k=args.candidate_k,
    )
    dt = time.perf_counter() - t0
    print(f"inference: {dt:.4f}s, rows={len(pred)}")
    print(pred.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

