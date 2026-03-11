from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


AMOUNT_BUCKET_MAP = {
    "0-100": 50.0,
    "100-500": 300.0,
    "500-1k": 750.0,
    "1k+": 1500.0,
    "10k+": 10000.0,
}

# определяем тему
THEME_KEYWORDS = {
    "food": ["еда", "кафе", "ресторан", "пицц", "бургер", "суши", "кофе", "продукт", "напит", "food"],
    "beauty": ["beauty", "крас", "уход", "маник", "парик", "космет", "spa", "спа"],
    "pharma": ["аптек", "pharm", "лекар", "здоров", "мед"],
    "kids": ["дет", "baby", "игруш", "школ", "подгуз", "ребен"],
    "auto": ["авто", "шины", "запчаст", "car", "бенз", "азс"],
    "electronics": ["смартф", "телефон", "ноут", "электрон", "техник", "gadget"],
    "home": ["дом", "ремонт", "мебел", "декор", "хоз"],
    "fashion": ["одежд", "обув", "fashion", "style", "аксессуар"],
    "travel": ["travel", "путеш", "авиа", "отел", "тур", "билет"],
}


def _parse_segment(segment: pd.Series) -> pd.Series:
    return segment.fillna("unknown").astype(str).str.extract(r"^([a-z]_\d+)")[0].fillna("unknown")


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.fillna(0.0)


def _classify_theme(text: str) -> str:
    s = str(text).lower()
    for theme, kws in THEME_KEYWORDS.items():
        if any(kw in s for kw in kws):
            return theme
    return "other"


def _is_online_offer(text: str) -> float:
    s = str(text).lower()
    # Проверка на онлайн flag
    online_kws = ["онлайн", "online", "интернет", "сайт", "app", "приложен", "доставк", "web", "www"]
    return 1.0 if any(kw in s for kw in online_kws) else 0.0


@dataclass
class FeatureArtifacts:
    user_features: pd.DataFrame
    offer_features: pd.DataFrame
    user_offer_history: pd.DataFrame


def build_user_features(
    people: pd.DataFrame,
    segments: pd.DataFrame,
    transaction: pd.DataFrame,
    financial_account: pd.DataFrame,
    receipts: pd.DataFrame,
) -> pd.DataFrame:
    people_f = people.copy()
    people_f["age_bucket"] = pd.to_numeric(people_f["age_bucket"], errors="coerce").fillna(0)
    people_f["gender_cd"] = people_f["gender_cd"].fillna("U")
    people_f["region"] = people_f["region"].fillna("UNKNOWN")

    seg = segments.copy()
    seg["segment_code"] = _parse_segment(seg["segment"])
    seg["region_size"] = seg["region_size"].fillna("unknown")
    for col in ["auto", "traveler", "entrepreneur"]:
        seg[col] = pd.to_numeric(seg[col], errors="coerce").fillna(0.0)
    seg["vip_status"] = seg["vip_status"].fillna("unknown")
    seg = seg[
        [
            "user_id",
            "segment_code",
            "region_size",
            "auto",
            "traveler",
            "entrepreneur",
            "vip_status",
        ]
    ]

    tx = transaction.copy()
    tx["event_date"] = _to_datetime(tx["event_date"])
    tx["amount_num"] = tx["amount_bucket"].map(AMOUNT_BUCKET_MAP).fillna(0.0)
    tx["is_online"] = (tx["online_transaction_flg"] == "Y").astype(float)
    tx_agg = (
        tx.groupby("user_id", as_index=False)
        .agg(
            tx_count=("transaction_id", "count"),
            tx_online_share=("is_online", "mean"),
            tx_amount_mean=("amount_num", "mean"),
            tx_unique_brands=("brand_dk", "nunique"),
            tx_days_active=("event_date", lambda x: x.nunique()),
        )
        .fillna(0.0)
    )
    tx_agg["tx_per_day"] = _safe_div(tx_agg["tx_count"], tx_agg["tx_days_active"])

    fa = financial_account.copy()
    fa["account_status_cd"] = fa["account_status_cd"].fillna("UNK")
    fa_agg = (
        fa.groupby("user_id", as_index=False)
        .agg(
            accounts_total=("product_cd", "count"),
            accounts_unique_products=("product_cd", "nunique"),
            accounts_active=("account_status_cd", lambda s: (s == "ACT").sum()),
            accounts_closed=("account_status_cd", lambda s: (s == "CLO").sum()),
        )
        .fillna(0.0)
    )

    rc = receipts.copy()
    rc["date_operated"] = _to_datetime(rc["date_operated"])
    rc["items_count_num"] = pd.to_numeric(rc["items_count"], errors="coerce").fillna(0.0)
    rc["items_cost_num"] = rc["items_cost"].map(AMOUNT_BUCKET_MAP).fillna(0.0)
    rc_agg = (
        rc.groupby("user_id", as_index=False)
        .agg(
            receipts_count=("date_operated", "count"),
            receipts_days=("date_operated", "nunique"),
            receipts_categories=("category_name", "nunique"),
            receipts_items_mean=("items_count_num", "mean"),
            receipts_cost_mean=("items_cost_num", "mean"),
        )
        .fillna(0.0)
    )
    rc_agg["receipts_per_day"] = _safe_div(rc_agg["receipts_count"], rc_agg["receipts_days"])
    rc["theme"] = rc["category_name"].fillna("other").astype(str).map(_classify_theme)
    theme_pivot = (
        rc.groupby(["user_id", "theme"]).size().rename("cnt").reset_index()
        .pivot(index="user_id", columns="theme", values="cnt")
        .fillna(0.0)
    )
    theme_pivot.columns = [f"theme_{c}_count" for c in theme_pivot.columns]
    theme_pivot = theme_pivot.reset_index()
    theme_cols = [c for c in theme_pivot.columns if c.startswith("theme_") and c.endswith("_count")]
    if theme_cols:
        total = theme_pivot[theme_cols].sum(axis=1).replace(0, np.nan)
        for c in theme_cols:
            share_col = c.replace("_count", "_share")
            theme_pivot[share_col] = (theme_pivot[c] / total).fillna(0.0)
    else:
        theme_pivot["theme_other_share"] = 0.0

    tx_brand_cnt = tx.groupby(["user_id", "brand_dk"]).size().rename("cnt").reset_index()
    repeat_stats = tx_brand_cnt.groupby("user_id", as_index=False).agg(
        merchants_total=("brand_dk", "nunique"),
        merchants_repeat=("cnt", lambda s: int((s >= 2).sum())),
    )
    repeat_stats["repeat_purchase_ratio"] = _safe_div(repeat_stats["merchants_repeat"], repeat_stats["merchants_total"])

    user = people_f.merge(seg, on="user_id", how="left")
    user = user.merge(tx_agg, on="user_id", how="left")
    user = user.merge(fa_agg, on="user_id", how="left")
    user = user.merge(rc_agg, on="user_id", how="left")
    user = user.merge(theme_pivot, on="user_id", how="left")
    user = user.merge(repeat_stats[["user_id", "repeat_purchase_ratio"]], on="user_id", how="left")
    user["online_pref"] = user.get("tx_online_share", 0.0)

    for col in user.columns:
        if col == "user_id":
            continue
        if user[col].dtype.kind in {"f", "i", "u"}:
            user[col] = user[col].fillna(0.0)
        else:
            user[col] = user[col].fillna("unknown")
    return user


def build_offer_features(offer: pd.DataFrame, merchant: pd.DataFrame, transaction: pd.DataFrame) -> pd.DataFrame:
    offers = offer.copy()
    offers["start_date"] = _to_datetime(offers["start_date"])
    offers["end_date"] = _to_datetime(offers["end_date"])
    offers["offer_duration_days"] = (offers["end_date"] - offers["start_date"]).dt.days.fillna(0).clip(lower=0)
    offers["offer_text"] = offers["offer_text"].fillna("")
    offers["offer_text_len"] = offers["offer_text"].str.len().astype(float)
    offers["offer_word_count"] = offers["offer_text"].str.split().str.len().fillna(0).astype(float)
    offers["offer_has_digits"] = offers["offer_text"].str.contains(r"\d", regex=True).astype(float)
    offers["offer_theme"] = offers["offer_text"].map(_classify_theme)
    offers["offer_is_online_text"] = offers["offer_text"].map(_is_online_offer)

    m = merchant.copy()
    m["merchant_status"] = m["merchant_status"].fillna("UNK")
    m["brand_dk"] = pd.to_numeric(m["brand_dk"], errors="coerce")
    m = m[["merchant_id_offer", "merchant_status", "brand_dk"]]

    tx = transaction.copy()
    tx["brand_dk"] = pd.to_numeric(tx["brand_dk"], errors="coerce")
    tx["is_online"] = (tx["online_transaction_flg"] == "Y").astype(float)
    merchant_online = (
        tx.merge(m[["merchant_id_offer", "brand_dk"]], on="brand_dk", how="inner")
        .groupby("merchant_id_offer", as_index=False)
        .agg(merchant_online_share=("is_online", "mean"), merchant_tx_cnt=("is_online", "count"))
    )

    out = offers.merge(m, on="merchant_id_offer", how="left")
    out = out.merge(merchant_online, on="merchant_id_offer", how="left")
    out["merchant_status"] = out["merchant_status"].fillna("UNK")
    out["brand_dk"] = out["brand_dk"].fillna(-1)
    out["merchant_online_share"] = out["merchant_online_share"].fillna(0.0)
    out["merchant_tx_cnt"] = out["merchant_tx_cnt"].fillna(0.0)
    out["offer_is_online"] = np.where(
        out["merchant_tx_cnt"] >= 20,
        out["merchant_online_share"],
        0.7 * out["offer_is_online_text"] + 0.3 * out["merchant_online_share"],
    )
    return out


def build_user_merchant_history(transaction: pd.DataFrame, merchant: pd.DataFrame) -> pd.DataFrame:
    tx = transaction.copy()
    tx["brand_dk"] = pd.to_numeric(tx["brand_dk"], errors="coerce")
    tx["event_date"] = _to_datetime(tx["event_date"])
    tx = tx[["user_id", "brand_dk", "event_date"]]

    m = merchant.copy()
    m["brand_dk"] = pd.to_numeric(m["brand_dk"], errors="coerce")
    m = m[["merchant_id_offer", "brand_dk"]].dropna()

    merged = tx.merge(m, on="brand_dk", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["merchant_id_offer", "user_id", "first_tx_date", "tx_count_to_merchant"])

    hist = (
        merged.groupby(["merchant_id_offer", "user_id"], as_index=False)
        .agg(
            first_tx_date=("event_date", "min"),
            tx_count_to_merchant=("event_date", "count"),
        )
        .sort_values(["merchant_id_offer", "user_id"])
    )
    return hist


def _add_theme_purchase_shares(
    user_features: pd.DataFrame,
    transaction: pd.DataFrame,
    offer_features: pd.DataFrame,
) -> pd.DataFrame:
    of = offer_features[["brand_dk", "offer_theme"]].drop_duplicates()
    of["brand_dk"] = pd.to_numeric(of["brand_dk"], errors="coerce").fillna(-1).astype(int).astype(str).replace("-1", "unknown")
    of["offer_theme"] = of["offer_theme"].fillna("other").astype(str)
    brand_theme = of.groupby("brand_dk")["offer_theme"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) else "other"
    ).to_dict()
    tx = transaction[["user_id", "brand_dk"]].copy()
    tx["brand_dk"] = pd.to_numeric(tx["brand_dk"], errors="coerce").fillna(-1).astype(int).astype(str).replace("-1", "unknown")
    tx["theme"] = tx["brand_dk"].map(brand_theme).fillna("other")
    theme_cnt = tx.groupby(["user_id", "theme"]).size().rename("cnt").reset_index()
    pivot = theme_cnt.pivot(index="user_id", columns="theme", values="cnt").fillna(0.0)
    total = pivot.sum(axis=1).replace(0, np.nan)
    for c in pivot.columns:
        pivot[c] = (pivot[c] / total).fillna(0.0)
    pivot.columns = [f"theme_{c}_purchase_share" for c in pivot.columns]
    pivot = pivot.reset_index()
    user_features = user_features.merge(pivot, on="user_id", how="left")
    for c in pivot.columns:
        if c != "user_id" and c in user_features.columns:
            user_features[c] = user_features[c].fillna(0.0)
    return user_features


def build_feature_artifacts(
    people: pd.DataFrame,
    segments: pd.DataFrame,
    transaction: pd.DataFrame,
    financial_account: pd.DataFrame,
    receipts: pd.DataFrame,
    offer: pd.DataFrame,
    merchant: pd.DataFrame,
) -> FeatureArtifacts:
    user_features = build_user_features(people, segments, transaction, financial_account, receipts)
    offer_features = build_offer_features(offer, merchant, transaction)
    user_features = _add_theme_purchase_shares(user_features, transaction, offer_features)
    user_offer_history = build_user_merchant_history(transaction, merchant)
    return FeatureArtifacts(
        user_features=user_features,
        offer_features=offer_features,
        user_offer_history=user_offer_history,
    )


def extract_main_segment(segment_code: str) -> str:
    match = re.match(r"^([a-z]_\d+)$", str(segment_code))
    return match.group(1) if match else "unknown"

