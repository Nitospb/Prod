from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMRanker
    _HAS_LGB = True
except (ImportError, OSError):
    LGBMRanker = None
    _HAS_LGB = False

from .config import TrainConfig
from .features import FeatureArtifacts


def _l2_normalize_rows(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    norms[norms == 0.0] = 1.0
    inv = sparse.diags(1.0 / norms)
    return inv @ matrix


def _sample_df(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=random_state)


@dataclass
class TrainingArtifacts:
    model_version: str
    user_features: pd.DataFrame
    offer_features: pd.DataFrame
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_index_by_id: dict[int, int]
    brand_index_by_id: dict[str, int]
    user_ids: np.ndarray
    theme_factors: dict[str, np.ndarray]
    

    profile_vectorizer: DictVectorizer
    user_profile_matrix: sparse.csr_matrix
    global_profile_vector: np.ndarray
    
    stage1_ranker_model: LogisticRegression | None
    stage1_ranker_features: list[str]
    stage2_model: Any
    stage2_feature_columns: list[str]
    category_columns: list[str]
    numeric_columns: list[str]
    stage2_category_mappings: dict[str, dict[str, int]]
    user_offer_history: pd.DataFrame
    user_segment_by_id: dict[int, str]
    user_region_size_by_id: dict[int, str]
    user_online_pref_by_id: dict[int, float]
    user_repeat_ratio_by_id: dict[int, float]
    user_theme_share_by_id: dict[int, dict[str, float]]
    user_response_rate_by_id: dict[int, float]
    user_theme_response_rate: dict[tuple[int, str], float]
    offer_response_rate: dict[int, float]
    merchant_response_rate: dict[int, float]
    segment_theme_online_rate: dict[tuple[str, str, int], float]
    region_theme_online_rate: dict[tuple[str, str, int], float]
    global_response_rate: float
    offer_segment_rate: dict[tuple[int, str], float]
    merchant_segment_rate: dict[tuple[int, str], float]
    offer_region_rate: dict[tuple[int, str], float]
    merchant_region_rate: dict[tuple[int, str], float]
    global_segment_rate: dict[str, float]
    global_region_rate: dict[str, float]
    global_base_rate: float
    brand_purchase_count: dict[str, float]

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, model_dir / "bundle.joblib")

    @staticmethod
    def load(model_dir: Path) -> "TrainingArtifacts":
        return joblib.load(model_dir / "bundle.joblib")


def _build_interaction_matrix(
    transaction: pd.DataFrame,
    user_index_by_id: dict[int, int],
) -> tuple[sparse.csr_matrix, dict[str, int]]:
    print("Filtering transaction data...")
    tx = transaction[transaction["user_id"].isin(user_index_by_id)].copy()
    print(f"Filtered transactions: {len(tx)}")


    tx["brand_dk"] = pd.to_numeric(tx["brand_dk"], errors="coerce").fillna(-1).astype(int).astype(str).replace("-1", "unknown")
    

    unique_brands = sorted(tx["brand_dk"].unique().tolist())
    brand_index_by_id = {b: i for i, b in enumerate(unique_brands)}
    
    tx["user_idx"] = tx["user_id"].map(user_index_by_id)
    tx["brand_idx"] = tx["brand_dk"].map(brand_index_by_id)
    

    print("Grouping transactions...")
    counts = tx.groupby(["user_idx", "brand_idx"]).size().reset_index(name="count")
    print(f"Grouped transactions: {len(counts)}")
    

    rows = counts["user_idx"].values
    cols = counts["brand_idx"].values
    data = np.log1p(counts["count"].values)
    
    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_index_by_id), len(brand_index_by_id)),
        dtype=np.float32
    )

    matrix = _l2_normalize_rows(matrix)
    
    return matrix, brand_index_by_id


def _train_svd(
    matrix: sparse.csr_matrix,
    k: int = 64,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    if matrix.shape[0] < k or matrix.shape[1] < k:
        k = min(matrix.shape[0], matrix.shape[1]) - 1
        if k < 1:
            return np.zeros((matrix.shape[0], 1)), np.zeros((matrix.shape[1], 1))
            

    u, s, vt = svds(matrix, k=k, random_state=random_state)
    

    idx = np.argsort(s)[::-1]
    u = u[:, idx]
    s = s[idx]
    vt = vt[idx, :]
    

    user_factors = u @ np.diag(s)
    item_factors = vt.T
    
    return user_factors, item_factors


def _prepare_user_profile_dicts(user_features: pd.DataFrame) -> list[dict[str, Any]]:
    numeric_cols = [
        "age_bucket", "auto", "traveler", "entrepreneur", "tx_count",
        "tx_online_share", "tx_amount_mean", "tx_unique_brands", "tx_per_day",
        "accounts_total", "accounts_unique_products", "accounts_active", "accounts_closed",
        "receipts_count", "receipts_categories", "receipts_items_mean", "receipts_cost_mean",
        "receipts_per_day",
    ]
    cat_cols = ["gender_cd", "segment_code", "region_size", "vip_status", "region"]
    df = user_features.copy()
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
    for col in cat_cols:
        if col not in df.columns:
            df[col] = "unknown"

    dicts: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: dict[str, Any] = {}
        for col in cat_cols:
            rec[f"{col}={row[col]}"] = 1.0
        for col in numeric_cols:
            rec[col] = float(row[col])
        dicts.append(rec)
    return dicts


def _build_segment_priors(
    offer_activation: pd.DataFrame,
    offer_seens: pd.DataFrame,
    offer_features: pd.DataFrame,
    user_features: pd.DataFrame,
) -> tuple[dict[tuple[int, str], float], dict[tuple[int, str], float], dict[str, float], float]:
    user_seg = user_features[["user_id", "segment_code"]].copy()
    user_seg["segment_code"] = user_seg["segment_code"].fillna("unknown").astype(str)
    act = offer_activation[["user_id", "offer_id"]].drop_duplicates().merge(user_seg, on="user_id", how="left")
    seen = offer_seens[["user_id", "offer_id"]].drop_duplicates().merge(user_seg, on="user_id", how="left")
    act["segment_code"] = act["segment_code"].fillna("unknown")
    seen["segment_code"] = seen["segment_code"].fillna("unknown")

    act_cnt = act.groupby(["offer_id", "segment_code"]).size().rename("act_cnt").reset_index()
    seen_cnt = seen.groupby(["offer_id", "segment_code"]).size().rename("seen_cnt").reset_index()
    merged = seen_cnt.merge(act_cnt, on=["offer_id", "segment_code"], how="left")
    merged["act_cnt"] = merged["act_cnt"].fillna(0.0)

    global_base_rate = float((len(act) + 1.0) / (len(seen) + 2.0)) if len(seen) > 0 else 0.01
    merged["rate"] = (merged["act_cnt"] + 1.0) / (merged["seen_cnt"] + 10.0)
    merged["rate"] = 0.7 * merged["rate"] + 0.3 * global_base_rate
    offer_segment_rate = {
        (int(r.offer_id), str(r.segment_code)): float(r.rate) for r in merged.itertuples(index=False)
    }

    offer2merchant = offer_features[["offer_id", "merchant_id_offer"]].drop_duplicates()
    seen_m = seen.merge(offer2merchant, on="offer_id", how="left")
    act_m = act.merge(offer2merchant, on="offer_id", how="left")
    seen_m_cnt = seen_m.groupby(["merchant_id_offer", "segment_code"]).size().rename("seen_cnt").reset_index()
    act_m_cnt = act_m.groupby(["merchant_id_offer", "segment_code"]).size().rename("act_cnt").reset_index()
    merged_m = seen_m_cnt.merge(act_m_cnt, on=["merchant_id_offer", "segment_code"], how="left")
    merged_m["act_cnt"] = merged_m["act_cnt"].fillna(0.0)
    merged_m["rate"] = (merged_m["act_cnt"] + 1.0) / (merged_m["seen_cnt"] + 20.0)
    merged_m["rate"] = 0.7 * merged_m["rate"] + 0.3 * global_base_rate
    merchant_segment_rate = {
        (int(r.merchant_id_offer), str(r.segment_code)): float(r.rate) for r in merged_m.itertuples(index=False)
    }

    g_seen = seen.groupby("segment_code").size().rename("seen_cnt").reset_index()
    g_act = act.groupby("segment_code").size().rename("act_cnt").reset_index()
    g = g_seen.merge(g_act, on="segment_code", how="left")
    g["act_cnt"] = g["act_cnt"].fillna(0.0)
    g["rate"] = (g["act_cnt"] + 1.0) / (g["seen_cnt"] + 20.0)
    g["rate"] = 0.7 * g["rate"] + 0.3 * global_base_rate
    global_segment_rate = {str(r.segment_code): float(r.rate) for r in g.itertuples(index=False)}
    return offer_segment_rate, merchant_segment_rate, global_segment_rate, global_base_rate


def _lookup_segment_prior(
    offer_segment_rate: dict[tuple[int, str], float],
    merchant_segment_rate: dict[tuple[int, str], float],
    global_segment_rate: dict[str, float],
    global_base_rate: float,
    offer_id: int,
    merchant_id: int,
    segment_code: str,
) -> float:
    seg = str(segment_code) if segment_code is not None else "unknown"
    if (offer_id, seg) in offer_segment_rate:
        return offer_segment_rate[(offer_id, seg)]
    if (merchant_id, seg) in merchant_segment_rate:
        return merchant_segment_rate[(merchant_id, seg)]
    if seg in global_segment_rate:
        return global_segment_rate[seg]
    return global_base_rate


def _build_transaction_segment_region_priors(
    transaction: pd.DataFrame,
    offer_features: pd.DataFrame,
    offer_seens: pd.DataFrame,
    user_features: pd.DataFrame,
    max_buyers_for_priors: int = 300_000,
    random_state: int = 42,
) -> tuple[
    dict[tuple[int, str], float],
    dict[tuple[int, str], float],
    dict[str, float],
    float,
    dict[tuple[int, str], float],
    dict[tuple[int, str], float],
    dict[str, float],
]:
    """Build segment/region priors from actual purchases (transaction), not clicks. Aligns with MAP target."""
    offers = offer_features[["offer_id", "merchant_id_offer", "brand_dk", "start_date", "end_date"]].drop_duplicates()
    offers["start_date"] = pd.to_datetime(offers["start_date"], errors="coerce")
    offers["end_date"] = pd.to_datetime(offers["end_date"], errors="coerce")
    tx = transaction[["user_id", "brand_dk", "event_date"]].copy()
    tx["event_date"] = pd.to_datetime(tx["event_date"], errors="coerce")
    tx = tx[tx["brand_dk"].isin(offers["brand_dk"].dropna().unique())]
    merged = tx.merge(offers, on="brand_dk", how="inner")
    mask = (merged["event_date"] >= merged["start_date"]) & (merged["event_date"] <= merged["end_date"])
    buyers = merged.loc[mask, ["user_id", "offer_id", "merchant_id_offer"]].drop_duplicates()
    if len(buyers) > max_buyers_for_priors:
        buyers = buyers.sample(max_buyers_for_priors, random_state=random_state)
    seen = offer_seens[["user_id", "offer_id"]].drop_duplicates()
    user_seg = user_features[["user_id", "segment_code"]].copy()
    user_seg["segment_code"] = user_seg["segment_code"].fillna("unknown").astype(str)
    user_reg = user_features[["user_id", "region_size"]].copy()
    user_reg["region_size"] = user_reg["region_size"].fillna("unknown").astype(str)

    buy_seg = buyers.merge(user_seg, on="user_id", how="left")
    see_seg = seen.merge(user_seg, on="user_id", how="left")
    buy_seg["segment_code"] = buy_seg["segment_code"].fillna("unknown")
    see_seg["segment_code"] = see_seg["segment_code"].fillna("unknown")
    b_os = buy_seg.groupby(["offer_id", "segment_code"]).size().rename("buy_cnt").reset_index()
    s_os = see_seg.groupby(["offer_id", "segment_code"]).size().rename("seen_cnt").reset_index()
    m_os = s_os.merge(b_os, on=["offer_id", "segment_code"], how="left")
    m_os["buy_cnt"] = m_os["buy_cnt"].fillna(0.0)
    m_os["rate"] = (m_os["buy_cnt"] + 1.0) / (m_os["seen_cnt"] + 10.0)
    global_base = float((buyers["user_id"].nunique() + 1) / (seen["user_id"].nunique() + 2)) if len(seen) > 0 else 0.01
    m_os["rate"] = 0.8 * m_os["rate"] + 0.2 * global_base
    offer_segment_rate = {(int(r.offer_id), str(r.segment_code)): float(r.rate) for r in m_os.itertuples(index=False)}

    buy_m = buy_seg.copy()
    see_m = see_seg.merge(offer_features[["offer_id", "merchant_id_offer"]].drop_duplicates(), on="offer_id", how="left")
    b_ms = buy_m.groupby(["merchant_id_offer", "segment_code"]).size().rename("buy_cnt").reset_index()
    s_ms = see_m.groupby(["merchant_id_offer", "segment_code"]).size().rename("seen_cnt").reset_index()
    m_ms = s_ms.merge(b_ms, on=["merchant_id_offer", "segment_code"], how="left")
    m_ms["buy_cnt"] = m_ms["buy_cnt"].fillna(0.0)
    m_ms["rate"] = (m_ms["buy_cnt"] + 1.0) / (m_ms["seen_cnt"] + 20.0)
    m_ms["rate"] = 0.8 * m_ms["rate"] + 0.2 * global_base
    merchant_segment_rate = {(int(r.merchant_id_offer), str(r.segment_code)): float(r.rate) for r in m_ms.itertuples(index=False)}
    g_s = see_seg.groupby("segment_code").size().rename("seen_cnt").reset_index()
    g_b = buy_seg.groupby("segment_code").size().rename("buy_cnt").reset_index()
    g_m = g_s.merge(g_b, on="segment_code", how="left")
    g_m["buy_cnt"] = g_m["buy_cnt"].fillna(0.0)
    g_m["rate"] = (g_m["buy_cnt"] + 1.0) / (g_m["seen_cnt"] + 20.0)
    g_m["rate"] = 0.8 * g_m["rate"] + 0.2 * global_base
    global_segment_rate = {str(r.segment_code): float(r.rate) for r in g_m.itertuples(index=False)}

    buy_reg = buyers.merge(user_reg, on="user_id", how="left")
    see_reg = seen.merge(user_reg, on="user_id", how="left")
    buy_reg["region_size"] = buy_reg["region_size"].fillna("unknown")
    see_reg["region_size"] = see_reg["region_size"].fillna("unknown")
    b_or = buy_reg.groupby(["offer_id", "region_size"]).size().rename("buy_cnt").reset_index()
    s_or = see_reg.groupby(["offer_id", "region_size"]).size().rename("seen_cnt").reset_index()
    m_or = s_or.merge(b_or, on=["offer_id", "region_size"], how="left")
    m_or["buy_cnt"] = m_or["buy_cnt"].fillna(0.0)
    m_or["rate"] = (m_or["buy_cnt"] + 1.0) / (m_or["seen_cnt"] + 12.0)
    m_or["rate"] = 0.8 * m_or["rate"] + 0.2 * global_base
    offer_region_rate = {(int(r.offer_id), str(r.region_size)): float(r.rate) for r in m_or.itertuples(index=False)}
    buy_rm = buy_reg.copy()
    see_rm = see_reg.merge(offer_features[["offer_id", "merchant_id_offer"]].drop_duplicates(), on="offer_id", how="left")
    b_mr = buy_rm.groupby(["merchant_id_offer", "region_size"]).size().rename("buy_cnt").reset_index()
    s_mr = see_rm.groupby(["merchant_id_offer", "region_size"]).size().rename("seen_cnt").reset_index()
    m_mr = s_mr.merge(b_mr, on=["merchant_id_offer", "region_size"], how="left")
    m_mr["buy_cnt"] = m_mr["buy_cnt"].fillna(0.0)
    m_mr["rate"] = (m_mr["buy_cnt"] + 1.0) / (m_mr["seen_cnt"] + 20.0)
    m_mr["rate"] = 0.8 * m_mr["rate"] + 0.2 * global_base
    merchant_region_rate = {(int(r.merchant_id_offer), str(r.region_size)): float(r.rate) for r in m_mr.itertuples(index=False)}
    g_rb = buy_reg.groupby("region_size").size().rename("buy_cnt").reset_index()
    g_rs = see_reg.groupby("region_size").size().rename("seen_cnt").reset_index()
    g_rm = g_rs.merge(g_rb, on="region_size", how="left")
    g_rm["buy_cnt"] = g_rm["buy_cnt"].fillna(0.0)
    g_rm["rate"] = (g_rm["buy_cnt"] + 1.0) / (g_rm["seen_cnt"] + 20.0)
    g_rm["rate"] = 0.8 * g_rm["rate"] + 0.2 * global_base
    global_region_rate = {str(r.region_size): float(r.rate) for r in g_rm.itertuples(index=False)}
    return (
        offer_segment_rate,
        merchant_segment_rate,
        global_segment_rate,
        global_base,
        offer_region_rate,
        merchant_region_rate,
        global_region_rate,
    )


def _build_region_priors(
    offer_activation: pd.DataFrame,
    offer_seens: pd.DataFrame,
    offer_features: pd.DataFrame,
    user_features: pd.DataFrame,
    global_base_rate: float,
) -> tuple[dict[tuple[int, str], float], dict[tuple[int, str], float], dict[str, float]]:
    user_reg = user_features[["user_id", "region_size"]].copy()
    user_reg["region_size"] = user_reg["region_size"].fillna("unknown").astype(str)

    act = offer_activation[["user_id", "offer_id"]].drop_duplicates().merge(user_reg, on="user_id", how="left")
    seen = offer_seens[["user_id", "offer_id"]].drop_duplicates().merge(user_reg, on="user_id", how="left")
    act["region_size"] = act["region_size"].fillna("unknown")
    seen["region_size"] = seen["region_size"].fillna("unknown")

    a = act.groupby(["offer_id", "region_size"]).size().rename("act_cnt").reset_index()
    s = seen.groupby(["offer_id", "region_size"]).size().rename("seen_cnt").reset_index()
    m = s.merge(a, on=["offer_id", "region_size"], how="left")
    m["act_cnt"] = m["act_cnt"].fillna(0.0)
    m["rate"] = (m["act_cnt"] + 1.0) / (m["seen_cnt"] + 12.0)
    m["rate"] = 0.7 * m["rate"] + 0.3 * global_base_rate
    offer_region_rate = {(int(r.offer_id), str(r.region_size)): float(r.rate) for r in m.itertuples(index=False)}

    offer2merchant = offer_features[["offer_id", "merchant_id_offer"]].drop_duplicates()
    act_m = act.merge(offer2merchant, on="offer_id", how="left")
    seen_m = seen.merge(offer2merchant, on="offer_id", how="left")
    a2 = act_m.groupby(["merchant_id_offer", "region_size"]).size().rename("act_cnt").reset_index()
    s2 = seen_m.groupby(["merchant_id_offer", "region_size"]).size().rename("seen_cnt").reset_index()
    m2 = s2.merge(a2, on=["merchant_id_offer", "region_size"], how="left")
    m2["act_cnt"] = m2["act_cnt"].fillna(0.0)
    m2["rate"] = (m2["act_cnt"] + 1.0) / (m2["seen_cnt"] + 20.0)
    m2["rate"] = 0.7 * m2["rate"] + 0.3 * global_base_rate
    merchant_region_rate = {
        (int(r.merchant_id_offer), str(r.region_size)): float(r.rate) for r in m2.itertuples(index=False)
    }

    ga = act.groupby("region_size").size().rename("act_cnt").reset_index()
    gs = seen.groupby("region_size").size().rename("seen_cnt").reset_index()
    gm = gs.merge(ga, on="region_size", how="left")
    gm["act_cnt"] = gm["act_cnt"].fillna(0.0)
    gm["rate"] = (gm["act_cnt"] + 1.0) / (gm["seen_cnt"] + 20.0)
    gm["rate"] = 0.7 * gm["rate"] + 0.3 * global_base_rate
    global_region_rate = {str(r.region_size): float(r.rate) for r in gm.itertuples(index=False)}
    return offer_region_rate, merchant_region_rate, global_region_rate


def _lookup_region_prior(
    offer_region_rate: dict[tuple[int, str], float],
    merchant_region_rate: dict[tuple[int, str], float],
    global_region_rate: dict[str, float],
    global_base_rate: float,
    offer_id: int,
    merchant_id: int,
    region_size: str,
) -> float:
    r = str(region_size) if region_size is not None else "unknown"
    if (offer_id, r) in offer_region_rate:
        return offer_region_rate[(offer_id, r)]
    if (merchant_id, r) in merchant_region_rate:
        return merchant_region_rate[(merchant_id, r)]
    if r in global_region_rate:
        return global_region_rate[r]
    return global_base_rate


def _get_offer_top_region(offer_region_rate: dict[tuple[int, str], float], offer_id: int) -> str:
    best_r, best_v = "unknown", -1.0
    for (oid, r), v in offer_region_rate.items():
        if oid == offer_id and v > best_v:
            best_r, best_v = r, v
    return best_r


def _get_offer_top_segment(offer_segment_rate: dict[tuple[int, str], float], offer_id: int) -> str:
    best_s, best_v = "unknown", -1.0
    for (oid, s), v in offer_segment_rate.items():
        if oid == offer_id and v > best_v:
            best_s, best_v = s, v
    return best_s


def _compute_theme_match(theme: str, theme_shares: dict[str, float]) -> float:
    if not theme or theme == "other":
        return float(theme_shares.get("other", 0.0))
    return float(theme_shares.get(theme, 0.0))


def _compute_online_match(offer_is_online: float, online_pref: float) -> float:
    if offer_is_online >= 0.5:
        return float(online_pref)
    return float(1.0 - online_pref)


def _build_cohort_response_priors(
    offer_activation: pd.DataFrame,
    offer_seens: pd.DataFrame,
    offer_features: pd.DataFrame,
    user_features: pd.DataFrame,
    global_rate: float,
) -> tuple[dict[tuple[str, str, int], float], dict[tuple[str, str, int], float]]:
    offer_meta = offer_features[["offer_id", "offer_theme", "offer_is_online"]].drop_duplicates().copy()
    offer_meta["offer_theme"] = offer_meta["offer_theme"].fillna("other").astype(str)
    offer_meta["online_bin"] = (pd.to_numeric(offer_meta["offer_is_online"], errors="coerce").fillna(0.0) >= 0.5).astype(int)
    user_meta = user_features[["user_id", "segment_code", "region_size"]].copy()
    user_meta["segment_code"] = user_meta["segment_code"].fillna("unknown").astype(str)
    user_meta["region_size"] = user_meta["region_size"].fillna("unknown").astype(str)

    seen = offer_seens[["user_id", "offer_id"]].drop_duplicates().merge(offer_meta, on="offer_id", how="left").merge(
        user_meta, on="user_id", how="left"
    )
    act = offer_activation[["user_id", "offer_id"]].drop_duplicates().merge(offer_meta, on="offer_id", how="left").merge(
        user_meta, on="user_id", how="left"
    )

    s_seg = seen.groupby(["segment_code", "offer_theme", "online_bin"]).size().rename("seen").reset_index()
    a_seg = act.groupby(["segment_code", "offer_theme", "online_bin"]).size().rename("act").reset_index()
    m_seg = s_seg.merge(a_seg, on=["segment_code", "offer_theme", "online_bin"], how="left")
    m_seg["act"] = m_seg["act"].fillna(0.0)
    m_seg["rate"] = (m_seg["act"] + 1.0) / (m_seg["seen"] + 20.0)
    m_seg["rate"] = 0.7 * m_seg["rate"] + 0.3 * global_rate
    seg_rate = {
        (str(r.segment_code), str(r.offer_theme), int(r.online_bin)): float(r.rate) for r in m_seg.itertuples(index=False)
    }

    s_reg = seen.groupby(["region_size", "offer_theme", "online_bin"]).size().rename("seen").reset_index()
    a_reg = act.groupby(["region_size", "offer_theme", "online_bin"]).size().rename("act").reset_index()
    m_reg = s_reg.merge(a_reg, on=["region_size", "offer_theme", "online_bin"], how="left")
    m_reg["act"] = m_reg["act"].fillna(0.0)
    m_reg["rate"] = (m_reg["act"] + 1.0) / (m_reg["seen"] + 20.0)
    m_reg["rate"] = 0.7 * m_reg["rate"] + 0.3 * global_rate
    reg_rate = {
        (str(r.region_size), str(r.offer_theme), int(r.online_bin)): float(r.rate) for r in m_reg.itertuples(index=False)
    }
    return seg_rate, reg_rate


def _build_response_priors(
    offer_activation: pd.DataFrame,
    offer_seens: pd.DataFrame,
    offer_features: pd.DataFrame,
) -> tuple[
    dict[int, float],
    dict[tuple[int, str], float],
    dict[int, float],
    dict[int, float],
    dict[tuple[str, str, int], float],
    dict[tuple[str, str, int], float],
    float,
]:
    seen = offer_seens[["user_id", "offer_id"]].drop_duplicates()
    act = offer_activation[["user_id", "offer_id"]].drop_duplicates()
    global_rate = float((len(act) + 1.0) / (len(seen) + 2.0)) if len(seen) > 0 else 0.01

    seen_user = seen.groupby("user_id").size().rename("seen").reset_index()
    act_user = act.groupby("user_id").size().rename("act").reset_index()
    user_tbl = seen_user.merge(act_user, on="user_id", how="left")
    user_tbl["act"] = user_tbl["act"].fillna(0.0)
    user_tbl["rate"] = (user_tbl["act"] + 1.0) / (user_tbl["seen"] + 10.0)
    user_tbl["rate"] = 0.8 * user_tbl["rate"] + 0.2 * global_rate
    user_response_rate_by_id = {int(r.user_id): float(r.rate) for r in user_tbl.itertuples(index=False)}

    seen_offer = seen.groupby("offer_id").size().rename("seen").reset_index()
    act_offer = act.groupby("offer_id").size().rename("act").reset_index()
    offer_tbl = seen_offer.merge(act_offer, on="offer_id", how="left")
    offer_tbl["act"] = offer_tbl["act"].fillna(0.0)
    offer_tbl["rate"] = (offer_tbl["act"] + 1.0) / (offer_tbl["seen"] + 20.0)
    offer_tbl["rate"] = 0.8 * offer_tbl["rate"] + 0.2 * global_rate
    offer_response_rate = {int(r.offer_id): float(r.rate) for r in offer_tbl.itertuples(index=False)}

    offer2merchant = offer_features[["offer_id", "merchant_id_offer"]].drop_duplicates()
    seen_m = seen.merge(offer2merchant, on="offer_id", how="left")
    act_m = act.merge(offer2merchant, on="offer_id", how="left")
    seen_mer = seen_m.groupby("merchant_id_offer").size().rename("seen").reset_index()
    act_mer = act_m.groupby("merchant_id_offer").size().rename("act").reset_index()
    m_tbl = seen_mer.merge(act_mer, on="merchant_id_offer", how="left")
    m_tbl["act"] = m_tbl["act"].fillna(0.0)
    m_tbl["rate"] = (m_tbl["act"] + 1.0) / (m_tbl["seen"] + 30.0)
    m_tbl["rate"] = 0.8 * m_tbl["rate"] + 0.2 * global_rate
    merchant_response_rate = {int(r.merchant_id_offer): float(r.rate) for r in m_tbl.itertuples(index=False)}
    offer_meta = offer_features[["offer_id", "offer_theme", "offer_is_online"]].drop_duplicates().copy()
    offer_meta["offer_theme"] = offer_meta["offer_theme"].fillna("other").astype(str)
    offer_meta["online_bin"] = (pd.to_numeric(offer_meta["offer_is_online"], errors="coerce").fillna(0.0) >= 0.5).astype(int)

    seen_u = seen.merge(offer_meta[["offer_id", "offer_theme", "online_bin"]], on="offer_id", how="left")
    act_u = act.merge(offer_meta[["offer_id", "offer_theme", "online_bin"]], on="offer_id", how="left")
    su = seen_u.groupby(["user_id", "offer_theme"]).size().rename("seen").reset_index()
    au = act_u.groupby(["user_id", "offer_theme"]).size().rename("act").reset_index()
    ut = su.merge(au, on=["user_id", "offer_theme"], how="left")
    ut["act"] = ut["act"].fillna(0.0)
    ut["rate"] = (ut["act"] + 1.0) / (ut["seen"] + 8.0)
    ut["rate"] = 0.8 * ut["rate"] + 0.2 * global_rate
    user_theme_response_rate = {(int(r.user_id), str(r.offer_theme)): float(r.rate) for r in ut.itertuples(index=False)}

    segment_theme_online_rate: dict[tuple[str, str, int], float] = {}
    region_theme_online_rate: dict[tuple[str, str, int], float] = {}
    return (
        user_response_rate_by_id,
        user_theme_response_rate,
        offer_response_rate,
        merchant_response_rate,
        segment_theme_online_rate,
        region_theme_online_rate,
        global_rate,
    )


def _compute_stage1_feature_arrays(bundle: TrainingArtifacts, offer_row: pd.Series) -> dict[str, np.ndarray]:
    offer_id = int(offer_row["offer_id"])
    merchant_id = int(offer_row["merchant_id_offer"])
    brand_dk = str(offer_row.get("brand_dk", "unknown"))
    offer_theme = str(offer_row.get("offer_theme", "other"))
    offer_is_online = float(offer_row.get("offer_is_online", 0.0))
    online_bin = 1 if offer_is_online >= 0.5 else 0
    
    # SVD Score
    brand_idx = bundle.brand_index_by_id.get(brand_dk)
    if brand_idx is not None:
        brand_vec = bundle.item_factors[brand_idx]
    else:
        # Fallback to theme vector
        brand_vec = bundle.theme_factors.get(offer_theme, bundle.theme_factors.get("global"))
        if brand_vec is None: # Should not happen given "global"
             brand_vec = np.zeros(bundle.user_factors.shape[1])
             
    svd_scores = bundle.user_factors @ brand_vec

    segment_scores = np.array(
        [
            _lookup_segment_prior(
                offer_segment_rate=bundle.offer_segment_rate,
                merchant_segment_rate=bundle.merchant_segment_rate,
                global_segment_rate=bundle.global_segment_rate,
                global_base_rate=bundle.global_base_rate,
                offer_id=offer_id,
                merchant_id=merchant_id,
                segment_code=bundle.user_segment_by_id.get(int(uid), "unknown"),
            )
            for uid in bundle.user_ids
        ],
        dtype=float,
    )
    region_scores = np.array(
        [
            _lookup_region_prior(
                offer_region_rate=bundle.offer_region_rate,
                merchant_region_rate=bundle.merchant_region_rate,
                global_region_rate=bundle.global_region_rate,
                global_base_rate=bundle.global_base_rate,
                offer_id=offer_id,
                merchant_id=merchant_id,
                region_size=bundle.user_region_size_by_id.get(int(uid), "unknown"),
            )
            for uid in bundle.user_ids
        ],
        dtype=float,
    )
    online_scores = np.array(
        [
            _compute_online_match(
                offer_is_online=offer_is_online,
                online_pref=bundle.user_online_pref_by_id.get(int(uid), 0.0),
            )
            for uid in bundle.user_ids
        ],
        dtype=float,
    )
    theme_scores = np.array(
        [
            _compute_theme_match(
                theme=offer_theme,
                theme_shares=bundle.user_theme_share_by_id.get(int(uid), {"other": 0.0}),
            )
            for uid in bundle.user_ids
        ],
        dtype=float,
    )
    repeat_scores = np.array(
        [bundle.user_repeat_ratio_by_id.get(int(uid), 0.0) for uid in bundle.user_ids],
        dtype=float,
    )
    user_resp_scores = np.array(
        [bundle.user_response_rate_by_id.get(int(uid), bundle.global_response_rate) for uid in bundle.user_ids],
        dtype=float,
    )
    user_theme_resp_scores = np.array(
        [bundle.user_theme_response_rate.get((int(uid), offer_theme), bundle.global_response_rate) for uid in bundle.user_ids],
        dtype=float,
    )
    cohort_seg_scores = np.array(
        [
            bundle.segment_theme_online_rate.get(
                (bundle.user_segment_by_id.get(int(uid), "unknown"), offer_theme, online_bin),
                bundle.global_response_rate,
            )
            for uid in bundle.user_ids
        ],
        dtype=float,
    )
    cohort_reg_scores = np.array(
        [
            bundle.region_theme_online_rate.get(
                (bundle.user_region_size_by_id.get(int(uid), "unknown"), offer_theme, online_bin),
                bundle.global_response_rate,
            )
            for uid in bundle.user_ids
        ],
        dtype=float,
    )
    offer_resp = float(bundle.offer_response_rate.get(offer_id, bundle.global_response_rate))
    merchant_resp = float(bundle.merchant_response_rate.get(merchant_id, bundle.global_response_rate))
    brand_pop = float(bundle.brand_purchase_count.get(brand_dk, 0.0))
    
    offer_scores = np.full_like(svd_scores, offer_resp, dtype=float)
    merchant_scores = np.full_like(svd_scores, merchant_resp, dtype=float)
    brand_pop_scores = np.full_like(svd_scores, brand_pop, dtype=float)
    
    return {
        "svd_scores": svd_scores,
        "segment_scores": segment_scores,
        "region_scores": region_scores,
        "online_scores": online_scores,
        "theme_scores": theme_scores,
        "repeat_scores": repeat_scores,
        "user_resp_scores": user_resp_scores,
        "user_theme_resp_scores": user_theme_resp_scores,
        "cohort_seg_scores": cohort_seg_scores,
        "cohort_reg_scores": cohort_reg_scores,
        "offer_scores": offer_scores,
        "merchant_scores": merchant_scores,
        "brand_pop_scores": brand_pop_scores,
    }


def _stage1_feature_matrix_from_arrays(arr: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    names = [
        "svd_scores",
        "segment_scores",
        "region_scores",
        "online_scores",
        "theme_scores",
        "repeat_scores",
        "user_resp_scores",
        "user_theme_resp_scores",
        "cohort_seg_scores",
        "cohort_reg_scores",
        "offer_scores",
        "merchant_scores",
        "brand_pop_scores",
    ]
    x = np.column_stack([arr[n] for n in names]).astype(float)
    return x, names


def compute_stage1_scores(bundle: TrainingArtifacts, offer_row: pd.Series) -> np.ndarray:
    arr = _compute_stage1_feature_arrays(bundle, offer_row)
    x, names = _stage1_feature_matrix_from_arrays(arr)
    if bundle.stage1_ranker_model is not None and bundle.stage1_ranker_features == names:
        return bundle.stage1_ranker_model.predict_proba(x)[:, 1]

    offer_is_online = float(offer_row.get("offer_is_online", 0.0))
    # Adaptive weights: регион и сегмент усилены (анализ показал недоучёт)
    w_region = 0.38 * (1.0 - offer_is_online) + 0.14 * offer_is_online
    w_online = 0.03 * (1.0 - offer_is_online) + 0.12 * offer_is_online
    w_svd = 0.32
    w_segment = 0.20
    w_theme = 0.05
    w_repeat = 0.04
    w_user_resp = 0.06
    w_user_theme_resp = 0.10
    w_cohort_seg = 0.07
    w_cohort_reg = 0.08
    w_offer_resp = 0.03
    w_merchant_resp = 0.01
    w_brand_pop = 0.04
    
    return (
        w_svd * arr["svd_scores"]
        + w_segment * arr["segment_scores"]
        + w_region * arr["region_scores"]
        + w_online * arr["online_scores"]
        + w_theme * arr["theme_scores"]
        + w_repeat * arr["repeat_scores"]
        + w_user_resp * arr["user_resp_scores"]
        + w_user_theme_resp * arr["user_theme_resp_scores"]
        + w_cohort_seg * arr["cohort_seg_scores"]
        + w_cohort_reg * arr["cohort_reg_scores"]
        + w_offer_resp * arr["offer_scores"]
        + w_merchant_resp * arr["merchant_scores"]
        + w_brand_pop * arr["brand_pop_scores"]
    )


def _build_training_pairs(
    user_features: pd.DataFrame,
    offer_features: pd.DataFrame,
    user_offer_history: pd.DataFrame,
    offer_activation: pd.DataFrame,
    offer_seens: pd.DataFrame,
    transaction: pd.DataFrame,
    stage1_scores: pd.Series,
    config: TrainConfig,
) -> pd.DataFrame:
    print("Building training pairs...")
    rng = np.random.default_rng(config.random_state)

    offers_df = offer_features[["offer_id", "merchant_id_offer", "brand_dk", "start_date", "end_date"]].drop_duplicates()
    offers_df["start_date"] = pd.to_datetime(offers_df["start_date"], errors="coerce")
    offers_df["end_date"] = pd.to_datetime(offers_df["end_date"], errors="coerce")
    

    relevant_brands = offers_df["brand_dk"].dropna().unique()
    print(f"Relevant brands: {len(relevant_brands)}")
    tx_subset = transaction[transaction["brand_dk"].isin(relevant_brands)][["user_id", "brand_dk", "event_date"]].copy()
    tx_subset["event_date"] = pd.to_datetime(tx_subset["event_date"], errors="coerce")
    print(f"Transaction subset size: {len(tx_subset)}")
    

    training_offers = set(offer_activation["offer_id"].unique()) | set(offer_seens["offer_id"].unique())
    offers_df = offers_df[offers_df["offer_id"].isin(training_offers)]
    

    print("Merging transactions with offers...")
    merged = tx_subset.merge(offers_df, on="brand_dk", how="inner")
    print(f"Merged size: {len(merged)}")
    

    mask = (merged["event_date"] >= merged["start_date"]) & (merged["event_date"] <= merged["end_date"])
    positives = merged[mask][["user_id", "offer_id"]].drop_duplicates()
    positives = positives.assign(label=1)
    print(f"Positives found: {len(positives)}")
    

    
    if len(positives) > config.max_positive_samples:
        positives = positives.sample(config.max_positive_samples, random_state=config.random_state)


    print("Sampling negatives...")
    seens = offer_seens[["user_id", "offer_id"]].drop_duplicates()
    negatives = seens.merge(
        positives[["user_id", "offer_id"]],
        on=["user_id", "offer_id"],
        how="left",
        indicator=True,
    )
    negatives = negatives[negatives["_merge"] == "left_only"][["user_id", "offer_id"]]
    print(f"Potential negatives: {len(negatives)}")

    target_neg_size = len(positives) * config.negative_ratio
    if len(negatives) > target_neg_size:
        negatives = negatives.sample(target_neg_size, random_state=config.random_state)

    if len(negatives) < target_neg_size and len(user_features) > 0 and len(offer_features) > 0:
        missing = target_neg_size - len(negatives)
        user_pool = user_features["user_id"].to_numpy()
        offer_pool = positives["offer_id"].drop_duplicates().to_numpy()
        if offer_pool.size > 0:
            add = pd.DataFrame(
                {
                    "user_id": rng.choice(user_pool, size=missing, replace=True),
                    "offer_id": rng.choice(offer_pool, size=missing, replace=True),
                }
            )
            negatives = pd.concat([negatives, add], ignore_index=True).drop_duplicates()
            if len(negatives) > target_neg_size:
                negatives = negatives.sample(target_neg_size, random_state=config.random_state)

    negatives = negatives.assign(label=0)
    pairs = pd.concat([positives, negatives], ignore_index=True).drop_duplicates(["user_id", "offer_id", "label"])
    print(f"Total pairs: {len(pairs)}")
    
    print("Merging features for pairs...")
    pairs = pairs.merge(
        offer_features[
            [
                "offer_id",
                "merchant_id_offer",
                "start_date",
                "offer_duration_days",
                "offer_text_len",
                "offer_word_count",
                "offer_has_digits",
                "offer_theme",
                "offer_is_online",
                "merchant_online_share",
                "merchant_tx_cnt",
                "merchant_status",
                "brand_dk",
            ]
        ],
        on="offer_id",
        how="left",
    )
    pairs = pairs.dropna(subset=["merchant_id_offer"]).copy()
    pairs = pairs.merge(user_features, on="user_id", how="left")
    pairs = pairs.merge(
        user_offer_history[["merchant_id_offer", "user_id", "first_tx_date", "tx_count_to_merchant"]],
        on=["merchant_id_offer", "user_id"],
        how="left",
    )
    pairs["first_tx_date"] = pd.to_datetime(pairs["first_tx_date"], errors="coerce")
    pairs["start_date"] = pd.to_datetime(pairs["start_date"], errors="coerce")
    pairs["existing_before_offer"] = (pairs["first_tx_date"] < pairs["start_date"]).fillna(False).astype(int)
    pairs["tx_count_to_merchant"] = pairs["tx_count_to_merchant"].fillna(0.0)

    pairs["stage1_score"] = stage1_scores.reindex(
        list(zip(pairs["user_id"].astype(int), pairs["offer_id"].astype(int)))
    ).fillna(0.0).to_numpy()

    pairs = pairs[pairs["existing_before_offer"] == 0].copy()
    return pairs


def _fit_stage2_model(
    pairs: pd.DataFrame,
    category_columns: list[str],
    numeric_columns: list[str],
    random_state: int,
    use_ranker: bool = True,
) -> tuple[Any, dict[str, dict[str, int]], np.ndarray]:
    mappings: dict[str, dict[str, int]] = {}
    x = pairs[category_columns + numeric_columns].copy()
    for c in category_columns:
        vals = sorted(x[c].astype(str).dropna().unique().tolist())
        mappings[c] = {v: i for i, v in enumerate(vals)}
        x[c] = x[c].astype(str).map(mappings[c]).fillna(-1).astype(int)
    for c in numeric_columns:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    x_mat = x.to_numpy(dtype=float)
    y = pairs["label"].astype(int).to_numpy()

    if use_ranker and _HAS_LGB and "offer_id" in pairs.columns:
        # Learning to Rank: group by offer_id, optimize NDCG/MAP within list
        pairs_sorted = pairs.sort_values("offer_id").reset_index(drop=True)
        order = pairs_sorted.index
        x_mat = x_mat[order]
        y = y[order]
        group = pairs_sorted.groupby("offer_id", sort=True).size().tolist()
        model = LGBMRanker(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=12,
            num_leaves=128,
            min_child_samples=25,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state,
            verbosity=-1,
            objective="lambdarank",
            metric="ndcg",
        )
        model.fit(x_mat, y, group=group)
        return model, mappings, x_mat
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=0.025,
            max_iter=950,
            max_depth=18,
            min_samples_leaf=12,
            l2_regularization=0.04,
            random_state=random_state,
        )
        model.fit(x_mat, y)
        return model, mappings, x_mat


def _transform_stage2_features(
    df: pd.DataFrame,
    category_columns: list[str],
    numeric_columns: list[str],
    mappings: dict[str, dict[str, int]],
) -> np.ndarray:
    x = df[category_columns + numeric_columns].copy()
    for c in category_columns:
        mp = mappings.get(c, {})
        x[c] = x[c].astype(str).map(mp).fillna(-1).astype(int)
    for c in numeric_columns:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    return x.to_numpy(dtype=float)


def train_two_stage_model(
    features: FeatureArtifacts,
    offer_activation: pd.DataFrame,
    offer_seens: pd.DataFrame,
    transaction: pd.DataFrame,
    config: TrainConfig,
    model_version: str = "1.0",
) -> TrainingArtifacts:
    user_features = features.user_features.copy()
    offer_features = features.offer_features.copy()

    def _clean_brand(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce").fillna(-1).astype(int).astype(str).replace("-1", "unknown")

    offer_features["brand_dk"] = _clean_brand(offer_features["brand_dk"])
    

    transaction = transaction.copy()
    transaction["brand_dk"] = _clean_brand(transaction["brand_dk"])
    
    user_offer_history = features.user_offer_history.copy()
    user_features = user_features.drop_duplicates("user_id").reset_index(drop=True)
    offer_features = offer_features.drop_duplicates("offer_id").reset_index(drop=True)

    user_ids = user_features["user_id"].astype(int).to_numpy()
    user_index_by_id = {uid: idx for idx, uid in enumerate(user_ids)}
    

    print("Building interaction matrix...")
    interaction_matrix, brand_index_by_id = _build_interaction_matrix(transaction, user_index_by_id)
    print(f"Interaction matrix shape: {interaction_matrix.shape}")
    
    print("Training SVD...")
    user_factors, item_factors = _train_svd(interaction_matrix, k=320, random_state=config.random_state)
    print("SVD training complete.")
    

    offer_brands = set(offer_features["brand_dk"].fillna("unknown").astype(str).unique())
    matrix_brands = set(brand_index_by_id.keys())
    overlap = offer_brands.intersection(matrix_brands)
    print(f"Offer brands: {len(offer_brands)}")
    print(f"Matrix brands: {len(matrix_brands)}")
    print(f"Overlap: {len(overlap)}")
    

    brand_counts = transaction["brand_dk"].value_counts().to_dict()
    brand_purchase_count = {str(k): float(np.log1p(v)) for k, v in brand_counts.items()}


    theme_factors: dict[str, np.ndarray] = {}

    brand_themes = offer_features[["brand_dk", "offer_theme"]].drop_duplicates()
    brand_themes["brand_idx"] = brand_themes["brand_dk"].map(brand_index_by_id)
    brand_themes = brand_themes.dropna(subset=["brand_idx"])
    brand_themes["brand_idx"] = brand_themes["brand_idx"].astype(int)
    
    for theme, grp in brand_themes.groupby("offer_theme"):
        indices = grp["brand_idx"].values
        if len(indices) > 0:
            vecs = item_factors[indices]
            mean_vec = np.mean(vecs, axis=0)
            theme_factors[str(theme)] = mean_vec
            

    if len(item_factors) > 0:
        theme_factors["global"] = np.mean(item_factors, axis=0)
    else:
        theme_factors["global"] = np.zeros(item_factors.shape[1] if len(item_factors) > 0 else 256)

    act_all = offer_activation[["user_id", "offer_id"]].drop_duplicates()
    seen_all = offer_seens[["user_id", "offer_id"]].drop_duplicates()
    seen_priors = _sample_df(seen_all, config.max_seen_samples_for_priors, config.random_state)
    seen_pairs = _sample_df(seen_all, config.max_seen_samples_for_pairs, config.random_state)


    (
        offer_segment_rate,
        merchant_segment_rate,
        global_segment_rate,
        global_base_rate,
        offer_region_rate,
        merchant_region_rate,
        global_region_rate,
    ) = _build_transaction_segment_region_priors(
        transaction=transaction,
        offer_features=offer_features,
        offer_seens=seen_priors,
        user_features=user_features,
        max_buyers_for_priors=300_000,
        random_state=config.random_state,
    )
    act_priors = _sample_df(act_all, config.max_activation_samples_for_priors, config.random_state)
    (
        user_response_rate_by_id,
        user_theme_response_rate,
        offer_response_rate,
        merchant_response_rate,
        segment_theme_online_rate,
        region_theme_online_rate,
        global_response_rate,
    ) = _build_response_priors(
        offer_activation=act_priors,
        offer_seens=seen_priors,
        offer_features=offer_features,
    )
    segment_theme_online_rate, region_theme_online_rate = _build_cohort_response_priors(
        offer_activation=act_priors,
        offer_seens=seen_priors,
        offer_features=offer_features,
        user_features=user_features,
        global_rate=global_response_rate,
    )


    profile_vectorizer = DictVectorizer()
    user_profile_matrix = sparse.csr_matrix((0, 0))
    global_profile_vector = np.zeros(0)
    
    partial_bundle = TrainingArtifacts(
        model_version=model_version,
        user_features=user_features,
        offer_features=offer_features,
        user_factors=user_factors,
        item_factors=item_factors,
        user_index_by_id=user_index_by_id,
        brand_index_by_id=brand_index_by_id,
        user_ids=user_ids,
        profile_vectorizer=profile_vectorizer,
        user_profile_matrix=user_profile_matrix,
        global_profile_vector=global_profile_vector,
        stage1_ranker_model=None,
        stage1_ranker_features=[],
        stage2_model=None,
        stage2_feature_columns=[],
        category_columns=[],
        numeric_columns=[],
        stage2_category_mappings={},
        user_offer_history=user_offer_history,
        user_segment_by_id={
            int(r.user_id): str(r.segment_code) if pd.notna(r.segment_code) else "unknown"
            for r in user_features[["user_id", "segment_code"]].itertuples(index=False)
        },
        user_region_size_by_id={
            int(r.user_id): str(r.region_size) if pd.notna(r.region_size) else "unknown"
            for r in user_features[["user_id", "region_size"]].itertuples(index=False)
        },
        user_online_pref_by_id={
            int(r.user_id): float(r.online_pref) if pd.notna(r.online_pref) else 0.0
            for r in user_features[["user_id", "online_pref"]].itertuples(index=False)
        },
        user_repeat_ratio_by_id={
            int(r.user_id): float(r.repeat_purchase_ratio) if pd.notna(r.repeat_purchase_ratio) else 0.0
            for r in user_features[["user_id", "repeat_purchase_ratio"]].itertuples(index=False)
        },
        user_theme_share_by_id={
            int(r.user_id): {
                "food": float(getattr(r, "theme_food_share", 0.0) or 0.0),
                "beauty": float(getattr(r, "theme_beauty_share", 0.0) or 0.0),
                "pharma": float(getattr(r, "theme_pharma_share", 0.0) or 0.0),
                "kids": float(getattr(r, "theme_kids_share", 0.0) or 0.0),
                "auto": float(getattr(r, "theme_auto_share", 0.0) or 0.0),
                "electronics": float(getattr(r, "theme_electronics_share", 0.0) or 0.0),
                "home": float(getattr(r, "theme_home_share", 0.0) or 0.0),
                "fashion": float(getattr(r, "theme_fashion_share", 0.0) or 0.0),
                "travel": float(getattr(r, "theme_travel_share", 0.0) or 0.0),
                "other": float(getattr(r, "theme_other_share", 0.0) or 0.0),
            }
            for r in user_features.itertuples(index=False)
        },
        user_response_rate_by_id=user_response_rate_by_id,
        user_theme_response_rate=user_theme_response_rate,
        offer_response_rate=offer_response_rate,
        merchant_response_rate=merchant_response_rate,
        segment_theme_online_rate=segment_theme_online_rate,
        region_theme_online_rate=region_theme_online_rate,
        global_response_rate=global_response_rate,
        offer_segment_rate=offer_segment_rate,
        merchant_segment_rate=merchant_segment_rate,
        offer_region_rate=offer_region_rate,
        merchant_region_rate=merchant_region_rate,
        global_segment_rate=global_segment_rate,
        global_region_rate=global_region_rate,
        global_base_rate=global_base_rate,
        brand_purchase_count=brand_purchase_count,
        theme_factors=theme_factors,
    )

    pairs = _build_training_pairs(
        user_features=user_features,
        offer_features=offer_features,
        user_offer_history=user_offer_history,
        offer_activation=act_all,
        offer_seens=seen_pairs,
        transaction=transaction,
        stage1_scores=pd.Series(dtype=float),
        config=config,
    )
    

    print("Adding feature columns...")
    for col, default in [
        ("offer_is_online", 0.0),
        ("merchant_online_share", 0.0),
        ("merchant_tx_cnt", 0.0),
        ("offer_theme", "other"),
        ("online_pref", 0.0),
        ("repeat_purchase_ratio", 0.0),
        ("region_size", "unknown"),
        ("segment_code", "unknown"),
        ("theme_food_share", 0.0),
        ("theme_beauty_share", 0.0),
        ("theme_pharma_share", 0.0),
        ("theme_kids_share", 0.0),
        ("theme_auto_share", 0.0),
        ("theme_electronics_share", 0.0),
        ("theme_home_share", 0.0),
        ("theme_fashion_share", 0.0),
        ("theme_travel_share", 0.0),
        ("theme_other_share", 0.0),
    ]:
        if col not in pairs.columns:
            pairs[col] = default
            
    pairs["segment_prior"] = [
        _lookup_segment_prior(
            offer_segment_rate=offer_segment_rate,
            merchant_segment_rate=merchant_segment_rate,
            global_segment_rate=global_segment_rate,
            global_base_rate=global_base_rate,
            offer_id=int(oid),
            merchant_id=int(mid),
            segment_code=str(seg),
        )
        for oid, mid, seg in zip(
            pairs["offer_id"].astype(int),
            pairs["merchant_id_offer"].astype(int),
            pairs["segment_code"].fillna("unknown").astype(str),
        )
    ]
    pairs["region_prior"] = [
        _lookup_region_prior(
            offer_region_rate=offer_region_rate,
            merchant_region_rate=merchant_region_rate,
            global_region_rate=global_region_rate,
            global_base_rate=global_base_rate,
            offer_id=int(oid),
            merchant_id=int(mid),
            region_size=str(reg),
        )
        for oid, mid, reg in zip(
            pairs["offer_id"].astype(int),
            pairs["merchant_id_offer"].astype(int),
            pairs["region_size"].fillna("unknown").astype(str),
        )
    ]
    pairs["online_match"] = [_compute_online_match(float(o), float(p)) for o, p in zip(pairs["offer_is_online"], pairs["online_pref"])]
    pairs["theme_match"] = [
        _compute_theme_match(
            str(theme),
            {
                "food": float(v_food),
                "beauty": float(v_beauty),
                "pharma": float(v_pharma),
                "kids": float(v_kids),
                "auto": float(v_auto),
                "electronics": float(v_electronics),
                "home": float(v_home),
                "fashion": float(v_fashion),
                "travel": float(v_travel),
                "other": float(v_other),
            },
        )
        for theme, v_food, v_beauty, v_pharma, v_kids, v_auto, v_electronics, v_home, v_fashion, v_travel, v_other in zip(
            pairs["offer_theme"],
            pairs["theme_food_share"],
            pairs["theme_beauty_share"],
            pairs["theme_pharma_share"],
            pairs["theme_kids_share"],
            pairs["theme_auto_share"],
            pairs["theme_electronics_share"],
            pairs["theme_home_share"],
            pairs["theme_fashion_share"],
            pairs["theme_travel_share"],
            pairs["theme_other_share"],
        )
    ]
    pairs["user_response_rate"] = [
        float(user_response_rate_by_id.get(int(uid), global_response_rate))
        for uid in pairs["user_id"].astype(int)
    ]
    pairs["offer_response_rate"] = [
        float(offer_response_rate.get(int(oid), global_response_rate))
        for oid in pairs["offer_id"].astype(int)
    ]
    pairs["merchant_response_rate"] = [
        float(merchant_response_rate.get(int(mid), global_response_rate))
        for mid in pairs["merchant_id_offer"].astype(int)
    ]
    pairs["user_theme_response_rate"] = [
        float(user_theme_response_rate.get((int(uid), str(theme)), global_response_rate))
        for uid, theme in zip(pairs["user_id"].astype(int), pairs["offer_theme"].astype(str))
    ]
    pairs["segment_theme_online_rate"] = [
        float(
            segment_theme_online_rate.get(
                (str(seg), str(theme), int(float(on) >= 0.5)),
                global_response_rate,
            )
        )
        for seg, theme, on in zip(
            pairs["segment_code"].astype(str),
            pairs["offer_theme"].astype(str),
            pairs["offer_is_online"].astype(float),
        )
    ]
    pairs["region_theme_online_rate"] = [
        float(
            region_theme_online_rate.get(
                (str(reg), str(theme), int(float(on) >= 0.5)),
                global_response_rate,
            )
        )
        for reg, theme, on in zip(
            pairs["region_size"].astype(str),
            pairs["offer_theme"].astype(str),
            pairs["offer_is_online"].astype(float),
        )
    ]


    print("Computing Stage 1 scores for pairs...")
    pairs["user_idx"] = pairs["user_id"].map(user_index_by_id)
    pairs["brand_dk"] = pairs["brand_dk"].fillna("unknown").astype(str)
    pairs["brand_idx"] = pairs["brand_dk"].map(brand_index_by_id)
    
    valid_mask = pairs["user_idx"].notna() & pairs["brand_idx"].notna()
    svd_scores = np.zeros(len(pairs))

    if valid_mask.any():
        u_idx = pairs.loc[valid_mask, "user_idx"].astype(int).values
        b_idx = pairs.loc[valid_mask, "brand_idx"].astype(int).values
        u_vecs = user_factors[u_idx]
        i_vecs = item_factors[b_idx]
        svd_scores[valid_mask] = (u_vecs * i_vecs).sum(axis=1)
        

    unknown_mask = pairs["user_idx"].notna() & pairs["brand_idx"].isna()
    if unknown_mask.any():
        u_idx = pairs.loc[unknown_mask, "user_idx"].astype(int).values
        themes = pairs.loc[unknown_mask, "offer_theme"].astype(str).values
        theme_vecs = np.array([theme_factors.get(t, theme_factors.get("global")) for t in themes])
        u_vecs = user_factors[u_idx]
        svd_scores[unknown_mask] = (u_vecs * theme_vecs).sum(axis=1)
    

    w_svd = 0.32
    w_segment = 0.20
    w_theme = 0.05
    w_repeat = 0.04
    w_user_resp = 0.06
    w_user_theme_resp = 0.10
    w_cohort_seg = 0.07
    w_cohort_reg = 0.08
    w_offer_resp = 0.03
    w_merchant_resp = 0.01
    is_online = pairs["offer_is_online"].astype(float).values
    w_region = 0.38 * (1.0 - is_online) + 0.14 * is_online
    w_online = 0.03 * (1.0 - is_online) + 0.12 * is_online
    
    s1_scores = (
        w_svd * svd_scores
        + w_segment * pairs["segment_prior"].values
        + w_region * pairs["region_prior"].values
        + w_online * pairs["online_match"].values
        + w_theme * pairs["theme_match"].values
        + w_repeat * pairs["repeat_purchase_ratio"].values
        + w_user_resp * pairs["user_response_rate"].values
        + w_user_theme_resp * pairs["user_theme_response_rate"].values
        + w_cohort_seg * pairs["segment_theme_online_rate"].values
        + w_cohort_reg * pairs["region_theme_online_rate"].values
        + w_offer_resp * pairs["offer_response_rate"].values
        + w_merchant_resp * pairs["merchant_response_rate"].values
    )
    pairs["stage1_score"] = s1_scores

    pairs["svd_scores"] = svd_scores
    pairs["segment_scores"] = pairs["segment_prior"]
    pairs["region_scores"] = pairs["region_prior"]
    pairs["online_scores"] = pairs["online_match"]
    pairs["theme_scores"] = pairs["theme_match"]
    pairs["repeat_scores"] = pairs["repeat_purchase_ratio"]
    pairs["user_resp_scores"] = pairs["user_response_rate"]
    pairs["user_theme_resp_scores"] = pairs["user_theme_response_rate"]
    pairs["cohort_seg_scores"] = pairs["segment_theme_online_rate"]
    pairs["cohort_reg_scores"] = pairs["region_theme_online_rate"]
    pairs["offer_scores"] = pairs["offer_response_rate"]
    pairs["merchant_scores"] = pairs["merchant_response_rate"]
    pairs["brand_pop_scores"] = [
        float(brand_purchase_count.get(str(b), 0.0))
        for b in pairs["brand_dk"].astype(str)
    ]

    category_columns = ["gender_cd", "region", "segment_code", "region_size", "vip_status", "merchant_status"]
    numeric_columns = [
        "age_bucket",
        "auto",
        "traveler",
        "entrepreneur",
        "tx_count",
        "tx_online_share",
        "tx_amount_mean",
        "tx_unique_brands",
        "tx_per_day",
        "accounts_total",
        "accounts_unique_products",
        "accounts_active",
        "accounts_closed",
        "receipts_count",
        "receipts_categories",
        "receipts_items_mean",
        "receipts_cost_mean",
        "receipts_per_day",
        "offer_duration_days",
        "offer_text_len",
        "offer_word_count",
        "offer_has_digits",
        "offer_is_online",
        "merchant_online_share",
        "merchant_tx_cnt",
        "brand_dk",
        "tx_count_to_merchant",
        "stage1_score",
        "segment_prior",
        "region_prior",
        "online_match",
        "theme_match",
        "repeat_purchase_ratio",
        "user_response_rate",
        "offer_response_rate",
        "merchant_response_rate",
        "user_theme_response_rate",
        "segment_theme_online_rate",
        "region_theme_online_rate",
        "offer_theme_purchase_share",
        "offer_theme_is_top_for_user",
        "user_region_is_top_for_offer",
        "user_segment_is_top_for_offer",
    ]

    # Топ-регион/сегмент по офферу (из transaction priors) — бинарные фичи для Stage 2
    top_region_by_offer = {int(oid): _get_offer_top_region(offer_region_rate, int(oid)) for oid in pairs["offer_id"].unique()}
    top_segment_by_offer = {int(oid): _get_offer_top_segment(offer_segment_rate, int(oid)) for oid in pairs["offer_id"].unique()}
    pairs["user_region_is_top_for_offer"] = (
        pairs["region_size"].fillna("unknown").astype(str) == pairs["offer_id"].astype(int).map(top_region_by_offer)
    ).astype(float)
    pairs["user_segment_is_top_for_offer"] = (
        pairs["segment_code"].fillna("unknown").astype(str) == pairs["offer_id"].astype(int).map(top_segment_by_offer)
    ).astype(float)


    if "offer_theme_purchase_share" not in pairs.columns:
        pairs["offer_theme_purchase_share"] = 0.0
    for theme in ["food", "beauty", "pharma", "kids", "auto", "electronics", "home", "fashion", "travel", "other"]:
        col = f"theme_{theme}_purchase_share"
        if col in pairs.columns:
            mask = pairs["offer_theme"].astype(str) == theme
            pairs.loc[mask, "offer_theme_purchase_share"] = pairs.loc[mask, col].values


    theme_cols = [f"theme_{t}_purchase_share" for t in ["food", "beauty", "pharma", "kids", "auto", "electronics", "home", "fashion", "travel", "other"]]
    theme_names = ["food", "beauty", "pharma", "kids", "auto", "electronics", "home", "fashion", "travel", "other"]
    if all(c in pairs.columns for c in theme_cols):
        share_mat = pairs[theme_cols].fillna(0.0).to_numpy()
        top3_idx = np.argsort(-share_mat, axis=1)[:, :3]
        pairs["offer_theme_is_top_for_user"] = 0.0
        for i, theme in enumerate(theme_names):
            in_top3 = (top3_idx == i).any(axis=1)
            mask = (pairs["offer_theme"].astype(str) == theme) & in_top3
            pairs.loc[mask, "offer_theme_is_top_for_user"] = 1.0
    else:
        pairs["offer_theme_is_top_for_user"] = 0.0

    for col in category_columns:
        if col not in pairs.columns:
            pairs[col] = "unknown"
        pairs[col] = pairs[col].fillna("unknown").astype(str)
    for col in numeric_columns:
        if col not in pairs.columns:
            pairs[col] = 0.0
        pairs[col] = pd.to_numeric(pairs[col], errors="coerce").fillna(0.0)

    stage1_ranker_features = [
        "svd_scores",
        "segment_scores",
        "region_scores",
        "online_scores",
        "theme_scores",
        "repeat_scores",
        "user_resp_scores",
        "user_theme_resp_scores",
        "cohort_seg_scores",
        "cohort_reg_scores",
        "offer_scores",
        "merchant_scores",
        "brand_pop_scores",
    ]

    print("Training Stage 1 Logistic Regression...")
    X_s1 = pairs[stage1_ranker_features].fillna(0.0).to_numpy()
    y_s1 = pairs["label"].astype(int).to_numpy()
    
    s1_model = LogisticRegression(class_weight="balanced", random_state=config.random_state, max_iter=1000)
    s1_model.fit(X_s1, y_s1)
    

    pairs["stage1_score"] = s1_model.predict_proba(X_s1)[:, 1]

    stage2_feature_columns = category_columns + numeric_columns
    try:
        stage2_model, stage2_category_mappings, _ = _fit_stage2_model(
            pairs=pairs,
            category_columns=category_columns,
            numeric_columns=numeric_columns,
            random_state=config.random_state,
            use_ranker=True,
        )
    except Exception:
        stage2_model, stage2_category_mappings, _ = _fit_stage2_model(
            pairs=pairs,
            category_columns=category_columns,
            numeric_columns=numeric_columns,
            random_state=config.random_state,
            use_ranker=False,
        )
    
    partial_bundle.stage1_ranker_model = s1_model
    partial_bundle.stage1_ranker_features = stage1_ranker_features
    partial_bundle.stage2_model = stage2_model
    partial_bundle.stage2_feature_columns = stage2_feature_columns
    partial_bundle.category_columns = category_columns
    partial_bundle.numeric_columns = numeric_columns
    partial_bundle.stage2_category_mappings = stage2_category_mappings
    partial_bundle.stage1_ranker_features = stage1_ranker_features
    
    return partial_bundle


def _get_offer_row(bundle: TrainingArtifacts, offer_id: int, merchant_id: int | None = None) -> pd.Series:
    df = bundle.offer_features
    row = df[df["offer_id"] == offer_id]
    if row.empty and merchant_id is not None:
        row = df[df["merchant_id_offer"] == merchant_id].head(1)
    if row.empty:
        raise KeyError(f"Offer {offer_id} is unknown.")
    return row.iloc[0]


def _exclude_existing_customers(
    bundle: TrainingArtifacts,
    candidate_user_ids: np.ndarray,
    merchant_id: int,
    offer_start: pd.Timestamp,
) -> np.ndarray:
    hist = bundle.user_offer_history
    if hist.empty:
        return candidate_user_ids
    subset = hist[hist["merchant_id_offer"] == merchant_id][["user_id", "first_tx_date"]]
    if subset.empty:
        return candidate_user_ids
    subset = subset.copy()
    subset["first_tx_date"] = pd.to_datetime(subset["first_tx_date"], errors="coerce")
    blocked = set(subset.loc[subset["first_tx_date"] < offer_start, "user_id"].astype(int).tolist())
    if not blocked:
        return candidate_user_ids
    return np.array([uid for uid in candidate_user_ids if int(uid) not in blocked], dtype=np.int64)


def _make_stage2_features_for_offer(
    bundle: TrainingArtifacts,
    offer_row: pd.Series,
    candidate_user_ids: np.ndarray,
    stage1_scores: np.ndarray,
) -> pd.DataFrame:
    users = bundle.user_features[bundle.user_features["user_id"].isin(candidate_user_ids)].copy()
    stage2_from_user = [c for c in bundle.stage2_feature_columns if c in bundle.user_features.columns]
    theme_share_cols = [c for c in bundle.user_features.columns if c.startswith("theme_") and c.endswith("_purchase_share")]
    users = users[["user_id"] + stage2_from_user + theme_share_cols]
    users["user_id"] = users["user_id"].astype(int)

    if users.empty:
        return pd.DataFrame(columns=["user_id"] + bundle.stage2_feature_columns)

    base = pd.DataFrame({"user_id": candidate_user_ids.astype(int), "stage1_score": stage1_scores})
    feat = base.merge(users, on="user_id", how="left")

    offer_cols = [
        "merchant_id_offer",
        "offer_id",
        "start_date",
        "offer_duration_days",
        "offer_text_len",
        "offer_word_count",
        "offer_has_digits",
        "offer_theme",
        "offer_is_online",
        "merchant_online_share",
        "merchant_tx_cnt",
        "merchant_status",
        "brand_dk",
    ]
    for col in offer_cols:
        feat[col] = offer_row.get(col)

    hist = bundle.user_offer_history[
        bundle.user_offer_history["merchant_id_offer"] == int(offer_row["merchant_id_offer"])
    ][["user_id", "first_tx_date", "tx_count_to_merchant"]]
    feat = feat.merge(hist, on="user_id", how="left")
    feat["first_tx_date"] = pd.to_datetime(feat["first_tx_date"], errors="coerce")
    feat["start_date"] = pd.to_datetime(feat["start_date"], errors="coerce")
    feat["existing_before_offer"] = (feat["first_tx_date"] < feat["start_date"]).fillna(False).astype(int)
    feat["tx_count_to_merchant"] = feat["tx_count_to_merchant"].fillna(0.0)
    for col in [
        "segment_code",
        "region_size",
        "online_pref",
        "theme_food_share",
        "theme_beauty_share",
        "theme_pharma_share",
        "theme_kids_share",
        "theme_auto_share",
        "theme_electronics_share",
        "theme_home_share",
        "theme_fashion_share",
        "theme_travel_share",
        "theme_other_share",
    ]:
        if col not in feat.columns:
            feat[col] = 0.0 if col.startswith("theme_") or col == "online_pref" else "unknown"
    feat["segment_prior"] = [
        _lookup_segment_prior(
            offer_segment_rate=bundle.offer_segment_rate,
            merchant_segment_rate=bundle.merchant_segment_rate,
            global_segment_rate=bundle.global_segment_rate,
            global_base_rate=bundle.global_base_rate,
            offer_id=int(offer_row["offer_id"]),
            merchant_id=int(offer_row["merchant_id_offer"]),
            segment_code=str(seg),
        )
        for seg in feat["segment_code"].fillna("unknown").astype(str).tolist()
    ]
    feat["region_prior"] = [
        _lookup_region_prior(
            offer_region_rate=bundle.offer_region_rate,
            merchant_region_rate=bundle.merchant_region_rate,
            global_region_rate=bundle.global_region_rate,
            global_base_rate=bundle.global_base_rate,
            offer_id=int(offer_row["offer_id"]),
            merchant_id=int(offer_row["merchant_id_offer"]),
            region_size=str(reg),
        )
        for reg in feat["region_size"].fillna("unknown").astype(str).tolist()
    ]
    feat["online_match"] = [
        _compute_online_match(float(offer_row.get("offer_is_online", 0.0)), float(pref))
        for pref in feat["online_pref"].fillna(0.0).astype(float).tolist()
    ]
    feat["theme_match"] = [
        _compute_theme_match(
            theme=str(offer_row.get("offer_theme", "other")),
            theme_shares={
                "food": float(food),
                "beauty": float(beauty),
                "pharma": float(pharma),
                "kids": float(kids),
                "auto": float(auto),
                "electronics": float(electronics),
                "home": float(home),
                "fashion": float(fashion),
                "travel": float(travel),
                "other": float(other),
            },
        )
        for food, beauty, pharma, kids, auto, electronics, home, fashion, travel, other in zip(
            feat["theme_food_share"] if "theme_food_share" in feat.columns else np.zeros(len(feat)),
            feat["theme_beauty_share"] if "theme_beauty_share" in feat.columns else np.zeros(len(feat)),
            feat["theme_pharma_share"] if "theme_pharma_share" in feat.columns else np.zeros(len(feat)),
            feat["theme_kids_share"] if "theme_kids_share" in feat.columns else np.zeros(len(feat)),
            feat["theme_auto_share"] if "theme_auto_share" in feat.columns else np.zeros(len(feat)),
            feat["theme_electronics_share"] if "theme_electronics_share" in feat.columns else np.zeros(len(feat)),
            feat["theme_home_share"] if "theme_home_share" in feat.columns else np.zeros(len(feat)),
            feat["theme_fashion_share"] if "theme_fashion_share" in feat.columns else np.zeros(len(feat)),
            feat["theme_travel_share"] if "theme_travel_share" in feat.columns else np.zeros(len(feat)),
            feat["theme_other_share"] if "theme_other_share" in feat.columns else np.zeros(len(feat)),
        )
    ]
    feat["user_response_rate"] = [
        float(bundle.user_response_rate_by_id.get(int(uid), bundle.global_response_rate))
        for uid in feat["user_id"].astype(int)
    ]
    feat["offer_response_rate"] = float(
        bundle.offer_response_rate.get(int(offer_row["offer_id"]), bundle.global_response_rate)
    )
    feat["merchant_response_rate"] = float(
        bundle.merchant_response_rate.get(int(offer_row["merchant_id_offer"]), bundle.global_response_rate)
    )
    offer_theme = str(offer_row.get("offer_theme", "other"))
    online_bin = 1 if float(offer_row.get("offer_is_online", 0.0)) >= 0.5 else 0
    feat["user_theme_response_rate"] = [
        float(bundle.user_theme_response_rate.get((int(uid), offer_theme), bundle.global_response_rate))
        for uid in feat["user_id"].astype(int)
    ]
    feat["segment_theme_online_rate"] = [
        float(
            bundle.segment_theme_online_rate.get(
                (str(seg), offer_theme, online_bin),
                bundle.global_response_rate,
            )
        )
        for seg in feat["segment_code"].astype(str)
    ]
    feat["region_theme_online_rate"] = [
        float(
            bundle.region_theme_online_rate.get(
                (str(reg), offer_theme, online_bin),
                bundle.global_response_rate,
            )
        )
        for reg in feat["region_size"].astype(str)
    ]
    col_theme_share = f"theme_{offer_theme}_purchase_share"
    feat["offer_theme_purchase_share"] = feat[col_theme_share].fillna(0.0).astype(float) if col_theme_share in feat.columns else 0.0
    theme_cols_pred = [f"theme_{t}_purchase_share" for t in ["food", "beauty", "pharma", "kids", "auto", "electronics", "home", "fashion", "travel", "other"]]
    theme_names_pred = ["food", "beauty", "pharma", "kids", "auto", "electronics", "home", "fashion", "travel", "other"]
    if all(c in feat.columns for c in theme_cols_pred):
        share_mat = feat[theme_cols_pred].fillna(0.0).to_numpy()
        top3_idx = np.argsort(-share_mat, axis=1)[:, :3]
        feat["offer_theme_is_top_for_user"] = 0.0
        for i, theme in enumerate(theme_names_pred):
            if theme != offer_theme:
                continue
            in_top3 = (top3_idx == i).any(axis=1)
            feat.loc[in_top3, "offer_theme_is_top_for_user"] = 1.0
    else:
        feat["offer_theme_is_top_for_user"] = 0.0

    top_region_offer = _get_offer_top_region(bundle.offer_region_rate, int(offer_row["offer_id"]))
    top_segment_offer = _get_offer_top_segment(bundle.offer_segment_rate, int(offer_row["offer_id"]))
    feat["user_region_is_top_for_offer"] = (feat["region_size"].fillna("unknown").astype(str) == top_region_offer).astype(float)
    feat["user_segment_is_top_for_offer"] = (feat["segment_code"].fillna("unknown").astype(str) == top_segment_offer).astype(float)

    for col in bundle.category_columns:
        if col not in feat.columns:
            feat[col] = "unknown"
        feat[col] = feat[col].fillna("unknown").astype(str)
    for col in bundle.numeric_columns:
        if col not in feat.columns:
            feat[col] = 0.0
        feat[col] = pd.to_numeric(feat[col], errors="coerce").fillna(0.0)

    return feat[["user_id"] + bundle.stage2_feature_columns]


def predict_top_users(
    bundle: TrainingArtifacts,
    merchant_id: int,
    offer_id: int,
    top_n: int = 100,
    candidate_k: int = 2000,
) -> pd.DataFrame:
    if top_n < 1 or top_n > 1000:
        raise ValueError("top_n must be in range [1, 1000]")
    offer_row = _get_offer_row(bundle, offer_id=offer_id, merchant_id=merchant_id)
    scores = compute_stage1_scores(bundle=bundle, offer_row=offer_row)

    if candidate_k > len(scores):
        candidate_k = len(scores)
    idx = np.argpartition(scores, -candidate_k)[-candidate_k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    candidate_user_ids = bundle.user_ids[idx]
    candidate_scores = scores[idx]

    offer_start = pd.to_datetime(offer_row["start_date"], errors="coerce")
    if pd.isna(offer_start):
        offer_start = pd.Timestamp("1970-01-01")
    candidate_user_ids = _exclude_existing_customers(
        bundle=bundle,
        candidate_user_ids=candidate_user_ids,
        merchant_id=int(offer_row["merchant_id_offer"]),
        offer_start=offer_start,
    )
    if candidate_user_ids.size == 0:
        return pd.DataFrame(columns=["user_id", "score"])

    score_by_user = {int(uid): float(sc) for uid, sc in zip(bundle.user_ids[idx], candidate_scores)}
    stage1_for_candidates = np.array([score_by_user[int(uid)] for uid in candidate_user_ids], dtype=float)
    stage2_input = _make_stage2_features_for_offer(bundle, offer_row, candidate_user_ids, stage1_for_candidates)
    if stage2_input.empty:
        return pd.DataFrame(columns=["user_id", "score"])

    x = _transform_stage2_features(
        df=stage2_input,
        category_columns=bundle.category_columns,
        numeric_columns=bundle.numeric_columns,
        mappings=bundle.stage2_category_mappings,
    )
    if hasattr(bundle.stage2_model, "predict_proba"):
        probs = bundle.stage2_model.predict_proba(x)[:, 1]
    else:
        raw = bundle.stage2_model.predict(x)
        rmin, rmax = float(np.min(raw)), float(np.max(raw))
        probs = (raw - rmin) / (rmax - rmin + 1e-9) if rmax > rmin else np.zeros_like(raw)
    s1 = stage2_input["stage1_score"].to_numpy(dtype=float)
    s1_min = float(np.min(s1)) if len(s1) else 0.0
    s1_max = float(np.max(s1)) if len(s1) else 1.0
    if s1_max > s1_min:
        s1_norm = (s1 - s1_min) / (s1_max - s1_min)
    else:
        s1_norm = np.zeros_like(s1)
    final_score = 0.95 * probs + 0.05 * s1_norm
    out = pd.DataFrame({"user_id": stage2_input["user_id"].astype(int), "score": final_score.astype(float)})
    out = out.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    return out
