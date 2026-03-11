
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from lookalike_pipeline.config import TrainConfig
from lookalike_pipeline.data import load_data
from lookalike_pipeline.features import build_feature_artifacts
from lookalike_pipeline.model import TrainingArtifacts, train_two_stage_model


TIMEOUT_SEC = 600  # 10 minutes after commit


@dataclass
class ValidationResult:
    valid: bool
    checks_total: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    failed_checks: list[dict] = field(default_factory=list)


@dataclass
class DriftResult:
    drift_detected: bool
    drift_score: float
    details: str = ""


@dataclass
class PipelineResult:
    valid: bool
    drift_detected: bool
    drift_score: float
    action_taken: str
    model_version: Optional[str] = None
    data_version: Optional[str] = None
    error: Optional[str] = None
    timings_sec: dict = field(default_factory=dict)
    validation_checks_total: int = 0
    validation_checks_passed: int = 0
    validation_checks_failed: int = 0
    validation_failed_checks: list = field(default_factory=list)


def validate_data(data_dir: Path) -> ValidationResult:
    """>5 проверок"""
    data_dir = Path(data_dir)
    failed: list[dict] = []
    total, passed = 0, 0


    from lookalike_pipeline.data import FILE_MAP
    tables_ok = True
    for key, filename in FILE_MAP.items():
        total += 1
        if (data_dir / filename).exists():
            passed += 1
        else:
            tables_ok = False
            failed.append({"table": key, "check": "file_exists", "details": f"Missing {filename}"})
    if not tables_ok:
        return ValidationResult(valid=False, checks_total=total, checks_passed=passed, checks_failed=len(failed), failed_checks=failed)

    # проверка ключей
    checks = [
        ("people", "prod_clients.csv", "user_id", "people"),
        ("transaction", "prod_financial_transaction.csv", "user_id", "transaction"),
        ("offer", "t_offer.csv", "offer_id", "offer"),
        ("segments", "prizm_segments.csv", "user_id", "segments"),
        ("merchant", "t_merchant.csv", "merchant_id_offer", "merchant"),
    ]
    for table_key, filename, col, name in checks:
        total += 1
        path = data_dir / filename
        try:
            df = pd.read_csv(path, nrows=100000)
            if df.empty:
                failed.append({"table": name, "check": f"{name}_not_empty", "details": "Table is empty"})
            elif col not in df.columns:
                failed.append({"table": name, "check": f"{col}_exists", "details": f"Column {col} missing"})
            elif df[col].isna().all():
                failed.append({"table": name, "check": f"{col}_not_null", "details": f"All values of {col} are null"})
            else:
                passed += 1
        except Exception as e:
            failed.append({"table": name, "check": "readable", "details": str(e)})

    valid = len(failed) == 0
    return ValidationResult(
        valid=valid,
        checks_total=total,
        checks_passed=passed,
        checks_failed=len(failed),
        failed_checks=failed,
    )


def _compute_drift_evidently(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
    drift_share_threshold: float = 0.5,
) -> DriftResult:
    try:
        from evidently.report import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return _compute_drift_fallback(reference_df, current_df, numerical_columns, categorical_columns)

    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(current_data=current_df, reference_data=reference_df)
        report_dict = report.as_dict()
        drift_metric = None
        for m in report_dict.get("metrics", []):
            if m.get("result", {}).get("dataset_drift") is not None:
                drift_metric = m["result"]
                break
        if drift_metric is None:
            return DriftResult(drift_detected=False, drift_score=0.0, details="No drift metric in report")
        drift = bool(drift_metric.get("dataset_drift", False))
        share = float(drift_metric.get("share_drifted_columns", 0.0))
        return DriftResult(drift_detected=drift, drift_score=round(share, 4), details="Evidently DataDriftPreset")
    except Exception as e:
        return DriftResult(drift_detected=False, drift_score=0.0, details=f"Evidently error: {e}")


def _compute_drift_fallback(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> DriftResult:
    """Simple drift: KS on first numerical column + proportion diff on first categorical."""
    import numpy as np
    score = 0.0
    for col in numerical_columns[:5]:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref = reference_df[col].dropna()
        cur = current_df[col].dropna()
        if len(ref) < 10 or len(cur) < 10:
            continue
        try:
            from scipy import stats
            _, p = stats.ks_2samp(ref, cur)
            score = max(score, 1.0 - p)
        except Exception:
            pass
    for col in categorical_columns[:3]:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        r = reference_df[col].astype(str).value_counts(normalize=True)
        c = current_df[col].astype(str).value_counts(normalize=True)
        common = r.index.union(c.index)
        r = r.reindex(common, fill_value=0).fillna(0)
        c = c.reindex(common, fill_value=0).fillna(0)
        diff = (r - c).abs().sum() / 2.0
        score = max(score, min(1.0, diff))
    drift = score >= 0.3
    return DriftResult(drift_detected=drift, drift_score=round(score, 4), details="fallback KS/categorical")


def run_pipeline(
    data_dir: Path,
    version: str,
    model_dir: Path,
    reference_data_dir: Optional[Path] = None,
    reference_features: Optional[pd.DataFrame] = None,
    current_model_version: str = "1.0",
    timeout_sec: float = TIMEOUT_SEC,
    max_transaction_rows: Optional[int] = None,
    max_offer_seens_rows: Optional[int] = None,
) -> PipelineResult:
    """
    Run full pipeline: validate → load → features → drift → [retrain if drift].
    reference_data_dir: if None, this is first run (no drift, always train).
    reference_features: optional precomputed reference for drift (e.g. user_features from previous version).
    max_transaction_rows / max_offer_seens_rows: cap size to fit in memory (e.g. 8 GB Docker).
    """
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    # 1. Validation
    t1 = time.perf_counter()
    val = validate_data(data_dir)
    timings["validation"] = time.perf_counter() - t1
    if not val.valid:
        return PipelineResult(
            valid=False,
            drift_detected=False,
            drift_score=0.0,
            action_taken="skipped",
            data_version=version,
            error="Validation failed",
            timings_sec=timings,
            validation_checks_total=val.checks_total,
            validation_checks_passed=val.checks_passed,
            validation_checks_failed=val.checks_failed,
            validation_failed_checks=val.failed_checks,
        )

    # 2. Load data
    if time.perf_counter() - t0 > timeout_sec:
        return PipelineResult(
            valid=True, drift_detected=False, drift_score=0.0, action_taken="skipped",
            data_version=version, error="Timeout before load", timings_sec=timings,
        )
    t2 = time.perf_counter()
    try:
        data = load_data(
            data_dir,
            max_transaction_rows=max_transaction_rows,
            max_offer_seens_rows=max_offer_seens_rows,
        )
    except Exception as e:
        return PipelineResult(
            valid=True, drift_detected=False, drift_score=0.0, action_taken="skipped",
            data_version=version, error=f"Load failed: {e}", timings_sec=timings,
        )
    timings["load_data"] = time.perf_counter() - t2

    # 3. Build features
    if time.perf_counter() - t0 > timeout_sec:
        return PipelineResult(
            valid=True, drift_detected=False, drift_score=0.0, action_taken="skipped",
            data_version=version, error="Timeout before features", timings_sec=timings,
        )
    t3 = time.perf_counter()
    try:
        features = build_feature_artifacts(
            people=data.people,
            segments=data.segments,
            transaction=data.transaction,
            financial_account=data.financial_account,
            receipts=data.receipts,
            offer=data.offer,
            merchant=data.merchant,
        )
    except Exception as e:
        return PipelineResult(
            valid=True, drift_detected=False, drift_score=0.0, action_taken="skipped",
            data_version=version, error=f"Features failed: {e}", timings_sec=timings,
        )
    timings["features"] = time.perf_counter() - t3

    # 4. Drift
    drift_result = DriftResult(drift_detected=False, drift_score=0.0)
    if reference_data_dir is not None or reference_features is not None:
        if time.perf_counter() - t0 > timeout_sec:
            return PipelineResult(
                valid=True, drift_detected=False, drift_score=0.0, action_taken="skipped",
                data_version=version, error="Timeout before drift", timings_sec=timings,
            )
        t4 = time.perf_counter()
        ref_df = reference_features
        if ref_df is None and reference_data_dir is not None:
            ref_data = load_data(
                reference_data_dir,
                max_transaction_rows=max_transaction_rows,
                max_offer_seens_rows=max_offer_seens_rows,
            )
            ref_feats = build_feature_artifacts(
                people=ref_data.people, segments=ref_data.segments, transaction=ref_data.transaction,
                financial_account=ref_data.financial_account, receipts=ref_data.receipts,
                offer=ref_data.offer, merchant=ref_data.merchant,
            )
            ref_df = ref_feats.user_features
        if ref_df is not None:
            cur_df = features.user_features
            num_cols = [c for c in cur_df.select_dtypes(include=["number"]).columns if c in ref_df.columns][:20]
            cat_cols = [c for c in ["segment_code", "region_size"] if c in cur_df.columns and c in ref_df.columns]
            ref_s = ref_df.sample(n=min(10000, len(ref_df)), random_state=42) if len(ref_df) > 10000 else ref_df
            cur_s = cur_df.sample(n=min(10000, len(cur_df)), random_state=42) if len(cur_df) > 10000 else cur_df
            drift_result = _compute_drift_evidently(ref_s, cur_s, num_cols, cat_cols)
        timings["drift"] = time.perf_counter() - t4
    else:
        timings["drift"] = 0.0

    # 5. Retrain only if drift (or first run: no reference => train)
    first_run = reference_data_dir is None and reference_features is None
    should_retrain = first_run or drift_result.drift_detected

    if not should_retrain:
        timings["total"] = time.perf_counter() - t0
        return PipelineResult(
            valid=True,
            drift_detected=drift_result.drift_detected,
            drift_score=drift_result.drift_score,
            action_taken="none",
            model_version=current_model_version,
            data_version=version,
            timings_sec=timings,
            validation_checks_total=val.checks_total,
            validation_checks_passed=val.checks_passed,
            validation_checks_failed=val.checks_failed,
            validation_failed_checks=val.failed_checks,
        )

    # 6. Train
    if time.perf_counter() - t0 > timeout_sec:
        return PipelineResult(
            valid=True, drift_detected=drift_result.drift_detected, drift_score=drift_result.drift_score,
            action_taken="none", model_version=current_model_version, data_version=version,
            error="Timeout before train", timings_sec=timings,
        )
    t5 = time.perf_counter()
    try:
        # При ограничении по строкам (Docker 8 GB) — меньше пар и candidate_k
        candidate_k = 12000 if max_transaction_rows is not None else 25000
        max_pos = 80_000 if max_transaction_rows is not None else 200_000
        cfg = TrainConfig(
            data_dir=data_dir,
            model_dir=model_dir,
            candidate_k=candidate_k,
            max_positive_samples=max_pos,
            negative_ratio=6,
            random_state=42,
        )
        new_version = _bump_model_version(current_model_version) if not first_run else "1.0"
        bundle = train_two_stage_model(
            features=features,
            offer_activation=data.offer_activation,
            offer_seens=data.offer_seens,
            transaction=data.transaction,
            config=cfg,
            model_version=new_version,
        )
        bundle.save(model_dir)
    except Exception as e:
        return PipelineResult(
            valid=True, drift_detected=drift_result.drift_detected, drift_score=drift_result.drift_score,
            action_taken="none", model_version=current_model_version, data_version=version,
            error=f"Train failed: {e}", timings_sec=timings,
        )
    timings["train"] = time.perf_counter() - t5
    total = time.perf_counter() - t0
    timings["total"] = total

    if total > timeout_sec:
        return PipelineResult(
            valid=True, drift_detected=drift_result.drift_detected, drift_score=drift_result.drift_score,
            action_taken="retrained", model_version=new_version, data_version=version,
            error=f"Pipeline exceeded timeout ({total:.1f}s > {timeout_sec}s)", timings_sec=timings,
        )

    return PipelineResult(
        valid=True,
        drift_detected=drift_result.drift_detected,
        drift_score=drift_result.drift_score,
        action_taken="retrained",
        model_version=new_version,
        data_version=version,
        timings_sec=timings,
        validation_checks_total=val.checks_total,
        validation_checks_passed=val.checks_passed,
        validation_checks_failed=val.checks_failed,
        validation_failed_checks=val.failed_checks,
    )


def _bump_model_version(ver: str) -> str:
    try:
        major = int(ver.split(".")[0])
        return f"{major + 1}.0"
    except Exception:
        return "2.0"
