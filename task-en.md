[[_TOC_]]

# Look-a-Like Service

## Business Context

You work at a bank. The bank processes anonymized customer transaction data from merchant partners.

Partners launch **offers** (cashback promotions) to acquire new customers. Impressions cost money, so the key objective is to show a **specific offer** to users most likely to **respond to it**.

### Problem

Each partner already has a customer base of users who have transacted with them. Every offer has its own characteristics: type, minimum purchase amount, text, and validity period. For the partner, it is important to:

- **Not show** the offer to existing customers (they already know the brand)
- **Show** the offer to new customers who are highly likely to **respond to this exact offer**

You need to build a **look-a-like audience for a specific offer**: bank customers who:
- Are behaviorally similar to users who responded to similar offers
- Have not previously transacted with the given partner

### Drift Scenario: New Offer

The model is trained on historical data: which customers respond to which offers. But the world changes:

1. A partner launches a **new offer** - potentially for a different audience, with a different spend range, in a different period
2. **New transactions** appear - customer behavior shifts
3. **New customers** arrive whom the model has never seen

Your task is to **automatically detect** data changes, **decide independently** whether retraining is needed, and if so - **retrain** the model.

---

## What You Need to Build

1. Build an **ML model** for look-a-like audience selection for a specific offer
2. Wrap it as **Docker + REST API**
3. Implement **data ingestion** via a streaming batch API
4. Implement **data validation** with **Great Expectations**
5. Set up **data/model versioning** with **DVC**
6. Implement **data drift monitoring** with **Evidently**
7. Implement **experiment tracking** with **MLflow**
8. Implement an **automatic pipeline**: when new data arrives, the service validates data, checks drift, and decides whether to retrain

Data is received via REST API. The service must store incoming data in **S3 (MinIO)** and version it with DVC. The pipeline must be **reproducible**: `dvc repro` should work.

How you design the solution is up to you, including infrastructure: Dockerfile, docker-compose, S3 (MinIO) for data/models/checkpoints/artifacts.

---

## Input Data

### Development Data

Two versioned data archives are provided for development:

- `v1.zip`
- `v2.zip`

Each archive contains its version folder (`v1/` or `v2/`) with CSV tables:

- `prod_clients.csv`
- `prod_financial_transaction.csv`
- `financial_account.csv`
- `offer_activation.csv`
- `offer_reward.csv`
- `offer_seens.csv`
- `prizm_segments.csv`
- `t_merchant.csv`
- `t_offer.csv`

Use v1 and v2 to:
1. Understand data structure and table relationships
2. Build features and train a model on v1
3. Compare v1 and v2 distributions to identify drift
4. Configure Evidently and choose drift_score thresholds
5. Verify that your pipeline retrains correctly when drift is present

> **Important:** local `v1/v2` is for development only. During evaluation, the checker sends **different data versions** of the same format. Do not hardcode to specific values or distributions.

Additionally, `receipts_v1.zip` and `receipts_v2.zip` are provided in the repository root (same as `v1.zip`/`v2.zip`) with the `receipts` table (`user_id`, `date_operated`, `category_name`, `items_count`, `items_cost`) for feature engineering; this source **is part of** the `/data/batch` API contract as a separate `receipts` table.

### Data Delivery

During evaluation, data is sent to your service **via REST API** in JSON format. The checker sends records in **batches** (`POST /data/batch`) - for one or more tables at a time. After all batches for a version are uploaded, the checker sends **commit** (`POST /data/commit`) to trigger processing.

For `POST /data/batch`, use logical table names:

- `people`
- `segments`
- `transaction`
- `offer`
- `merchant`
- `financial_account`
- `offer_seens`
- `offer_activation`
- `offer_reward`
- `receipts`

### Table Relationships

Main links in the current dataset:

- `people` (`prod_clients.csv`) is linked with `segments`, `financial_account`, and `transaction` by `user_id`.
- `offer` (`t_offer.csv`) is linked with `merchant` (`t_merchant.csv`) by `merchant_id_offer`.
- `transaction` contains `brand_dk`, and `merchant` contains both `brand_dk` and `merchant_id_offer`.
- To connect transactions with offers, use: `transaction.brand_dk -> merchant.brand_dk -> merchant.merchant_id_offer -> offer.merchant_id_offer`.
- Tables `offer_seens`, `offer_activation`, `offer_reward` link users and offers by (`user_id`, `offer_id`).
- `receipts` is linked with `people` by `user_id` and provides an additional behavioral signal layer for features.

---

### Logical Tables and Files

| Logical table (API) | File in `template/data` | Key fields |
|---------|----------|----------|
| `people` | `prod_clients.csv` | `user_id`, `age_bucket`, `gender_cd`, `region`, `last_activity_day` |
| `segments` | `prizm_segments.csv` | `user_id`, `segment`, `region_size`, `auto`, `traveler`, `entrepreneur`, `vip_status` |
| `transaction` | `prod_financial_transaction.csv` | `transaction_id`, `user_id`, `merchant_id_tx`, `event_date`, `amount_bucket`, `online_transaction_flg`, `brand_dk` |
| `offer` | `t_offer.csv` | `offer_id`, `merchant_id_offer`, `start_date`, `end_date`, `offer_text` |
| `merchant` | `t_merchant.csv` | `merchant_id_offer`, `merchant_status`, `brand_dk` |
| `financial_account` | `financial_account.csv` | `user_id`, `product_cd`, `open_month`, `close_month`, `account_status_cd` |
| `offer_seens` | `offer_seens.csv` | `user_id`, `offer_id`, `start_date`, `end_date` |
| `offer_activation` | `offer_activation.csv` | `user_id`, `offer_id`, `activation_date` |
| `offer_reward` | `offer_reward.csv` | `user_id`, `offer_id`, `event_date`, `reward_amt` |
| `receipts` | `receipts.csv` | `user_id`, `date_operated`, `category_name`, `items_count`, `items_cost` |

### Segment Dictionary

Below is a mapping of `segment` codes and their labels.

#### Moscow (`m_*`)

| Code | Name |
|-----|----------|
| `m_01` | Stable well-being |
| `m_02` | Active lifestyle |
| `m_03` | Young and promising |
| `m_04` | Family-oriented residents of Moscow |
| `m_05` | People approaching retirement age |
| `m_06` | Professionals who moved to the capital |
| `m_07` | People valuing work-life balance |
| `m_08` | Specialists in practical professions |
| `m_09` | Older generation of the capital |
| `m_10` | Specialists with foundational qualifications |
| `m_11` | Youth of Moscow |

#### Large Cities (`u_*`)

| Code | Name |
|-----|----------|
| `u_01` | High-income residents of large cities |
| `u_02` | Young affluent residents of large cities |
| `u_03` | Wealthy residents of million-plus cities |
| `u_04` | Financially stable urban residents |
| `u_05` | Searching for career and life trajectory |
| `u_06` | Family-oriented residents of large cities |
| `u_07` | Residents adapting to big-city life |
| `u_08` | Mature residents focused on personal comfort |
| `u_09` | Youth of large cities |
| `u_10` | Residents of million-plus cities with limited income |
| `u_11` | Adult residents of million-plus cities with a stable lifestyle |
| `u_12` | At the beginning of an independent path |

#### Cities (`t_*`)

| Code | Name |
|-----|----------|
| `t_01` | Financially stable city residents |
| `t_02` | Young promising city residents |
| `t_03` | Affluent city residents |
| `t_04` | Family-oriented city residents |
| `t_05` | Specialists in practical and production sectors |
| `t_06` | Residents oriented toward a stable lifestyle |
| `t_07` | Residents adapting to the city pace |
| `t_08` | Older generation of the city |
| `t_09` | City youth |
| `t_10` | City residents with limited income |
| `t_11` | Adult city residents with a stable lifestyle |
| `t_12` | Starting an independent path in the city |

#### Rural Areas (`r_*`)

| Code | Name |
|-----|----------|
| `r_01` | Financially stable rural residents |
| `r_02` | Young promising rural residents |
| `r_03` | Economically successful rural residents |
| `r_04` | Workers in practical and production sectors |
| `r_05` | Older rural generation |
| `r_06` | Adult rural residents with a stable lifestyle |
| `r_07` | Family-oriented rural residents |
| `r_08` | Rural youth |
| `r_09` | Focused on professional development |
| `r_10` | At the beginning of an independent rural path |

---

## How Evaluation Works

The checker sequentially sends **multiple data versions** through `POST /data/batch` + `POST /data/commit`. Your service must **decide on its own** whether drift exists and whether retraining is needed.

Some versions contain **real drift**, some do **not**. One version may contain **corrupted data** (missing batches, missing tables, invalid records). The order and number of versions are unknown in advance.

```
Checker                                       Your service
──────                                        ────────────

    ═══ PHASE 1: initial training ═══

1. POST /data/batch (people, v1)              Accepts, buffers
   POST /data/batch (transaction, v1, 1/3)
   POST /data/batch (offer, v1)
   POST /data/batch (merchant, v1)
   POST /data/batch (transaction, v1, 2/3)    Batches arrive
   POST /data/batch (segments, v1)            in mixed order
   POST /data/batch (offer_seens, v1)
   POST /data/batch (offer_activation, v1)
   POST /data/batch (offer_reward, v1)
   POST /data/batch (receipts, v1)
   POST /data/batch (transaction, v1, 3/3)
   POST /data/batch (financial_account, v1)

2. POST /data/commit {"version":"v1"}    →   All v1 data received.
                                              Validation (GE) → features → training
3. Polls GET /status                        ←   ready: true, model_version: "1.0"

4. POST /lookalike (set of offers)          ←   Predictions from baseline model
   → Measures MAP@100 (baseline)

    ═══ PHASE 2+: update series ═══

   For EACH version vN:

5. POST /data/batch (tables of vN)          →   Accepts batches
6. POST /data/commit {"version":"vN"}     →   Processes:
                                              validation (GE) → data/features update
                                              → Evidently → decision
                                              - invalid data? → skip
                                              - drift? → retrain model
                                              - no drift? → data updated,
                                                model unchanged
7. Polls GET /status                        ←   pipeline_status: "idle"

8. GET /monitoring/data-quality             ←   valid: true/false
9. GET /monitoring/drift                    ←   drift_detected: true/false
10. GET /model/info                         ←   did model_version change?

11. POST /lookalike (SAME offers)           ←   MAP@100 (vN)
    → Compares with previous                    Δ = gain / stability
```

Key point: the checker does **not tell** your service to retrain. It only sends data and commit. After each commit, the service **always** updates the data catalog (new offers, transactions, features). The **model retraining** decision is separate - only when statistical drift is detected. If the service retrains when there is no drift, or does not retrain when drift exists, points are reduced.

> **Important:** the appearance of new offers or merchants in data is **NOT** drift. Drift means a **statistical shift in feature distributions** (transaction amount shift, purchase frequency changes, segment distribution changes, etc.). Some versions may contain new entities but stable distributions - in this case retraining is **not** required.

> **How MAP@100 is calculated:** the checker has **real future transaction data**. For each offer, ground truth is **new customers** who actually made a transaction at the merchant **during the offer period**, but **had not transacted with that merchant before** (before `start_date`). Ground truth is aligned with the filtering rule: both your service and checker exclude existing customers. For this dataset, transaction-to-offer-merchant linkage is built through `brand_dk` and `merchant_id_offer`. Offer attributes (period, minimum purchase amount, text) **affect** who appears in ground truth.

> **Timeout:** after each `POST /data/commit`, the checker waits at most **10 minutes** until `pipeline_status: "idle"`. If the pipeline does not finish in time, all points for that version are zeroed and checker proceeds to the next version.

> **Fail-soft (production):** an error in a single batch, data version, or pipeline stage must not bring down the whole service. `/lookalike` must keep serving from the latest successfully deployed model. If a data version is invalid, training is skipped and the service remains available.

---

## What You Submit

A repository with your solution. It must include:

- `Dockerfile` - Docker image for your service
- `docker-compose.yml` - orchestration for your service and required infrastructure (MinIO, MLflow, etc.)
- `dvc.yaml` - DVC pipeline
- `params.yaml` - pipeline parameters
- `reports/` - generated reports (output of `dvc repro`)

File/folder structure and module decomposition are up to you.

> **Important:** We provide only the data and `openapi.yml`. All other code, Dockerfile, docker-compose, and infrastructure are **yours**.

### Run

Checker runs from the repository root:

```bash
docker compose up -d
```

Then it expects `GET http://localhost:80/ready` → 200 within **180 seconds**. All your services (MinIO, MLflow, etc.) must be up and ready in this time. Services communicate via **local docker network**.

If `/ready` does not return 200 - **0 points for the whole solution**.

---

## API Contract

> Full specification: [`openapi.yml`](openapi.yml)

The service **must** listen on port **80**.

### `GET /ready`

```json
{"status": "ok"}
```

### `POST /data/batch`

Uploads a batch of records for one table. Can be called **multiple times** - batches may arrive in arbitrary order and interleaved across tables. Large tables are split into multiple batches.

**Request:**
```json
{
  "version": "v2",
  "table": "transaction",
  "batch_id": 1,
  "total_batches": 3,
  "records": [
    {"transaction_id": 1, "user_id": 42, "merchant_id_tx": 75, "event_date": "2025-06-01", "amount_bucket": "10k+", "online_transaction_flg": "N", "brand_dk": 18601},
    {"transaction_id": 2, "user_id": 15, "merchant_id_tx": 110, "event_date": "2025-06-02", "amount_bucket": "1k+", "online_transaction_flg": "Y", "brand_dk": 29256}
  ]
}
```

**Response (200 OK):**
```json
{"status": "accepted", "table": "transaction", "batch_id": 1}
```

> **Idempotency:** repeated call with the same `version` + `table` + `batch_id` **must not** duplicate records.
>
> Required request fields: `version`, `table`, `batch_id`, `total_batches`, `records`.
> Allowed `table` values: `people`, `segments`, `transaction`, `offer`, `merchant`, `financial_account`, `offer_seens`, `offer_activation`, `offer_reward`, `receipts`.
> Unknown table name or missing required field must return `400`.
> Empty `records` is a valid no-op (without crashing the service).

### `POST /data/commit`

Signal that all data for the specified version has been uploaded. The service starts **asynchronous** processing (in background).

**Request:**
```json
{"version": "v2"}
```

**Response (200 OK):**
```json
{"status": "accepted", "tables_received": ["people", "segments", "transaction", "offer", "merchant", "financial_account", "offer_seens", "offer_activation", "offer_reward", "receipts"]}
```

> **Idempotency:** repeated commit with the same `version` **must not** re-run the pipeline.
>
> The first successful `commit` freezes (closes) that data version.
> `commit` before any batches must be handled safely (no `500`): the version is treated as invalid, and training is skipped (`action_taken: "skipped"`).

### `GET /status`

Current service and pipeline state.

```json
{
  "ready": true,
  "model_version": "2.0",
  "data_version": "v2",
  "pipeline_status": "idle"
}
```

- `ready` - service is ready to serve `/lookalike` (even if pipeline runs in background, previous model keeps serving requests)
- `data_version` - latest successfully processed data-catalog version (it may be newer than the current model's `trained_on`)
- `pipeline_status` - `"idle"` | `"running"` | `"failed"` (after successful processing, status returns to `"idle"`)

### `POST /lookalike`

**Request:**
```json
{
  "merchant_id": 75,
  "offer_id": 42,
  "top_n": 100
}
```

**Response (200 OK):**
```json
{
  "merchant_id": 75,
  "offer_id": 42,
  "audience": [
    {"user_id": 28, "score": 0.95},
    {"user_id": 43, "score": 0.91}
  ],
  "audience_size": 100,
  "model_version": "1.0",
  "reasons": [
    {"feature": "segment=u_02", "impact": 0.34},
    {"feature": "tx_count_30d", "impact": 0.21},
    {"feature": "online_share_90d", "impact": -0.08}
  ]
}
```

- `audience` - sorted by descending `score`
- `score` - relevance score (0..1)
- `reasons` - top factors/features that influenced ranking for this response; each item includes `feature` and numeric `impact` (can be negative)
- `reasons` must pass a sanity check: when one key user feature is changed in a controlled way (`segment`, online spend share, etc.), top reason and/or `score` should change predictably, and `impact` sign should remain logical
- **Do not include** current merchant customers (users with transactions before the offer period)
- Different offers of the same merchant may produce **different** audiences
- `top_n` - integer in range `1..1000`; out-of-range values must return `400`

**Codes:** `200`, `400`, `404`, `503`

### `POST /lookalike/batch`

**Request:**
```json
{
  "requests": [
    {"merchant_id": 75, "offer_id": 42, "top_n": 50},
    {"merchant_id": 110, "offer_id": 99, "top_n": 100}
  ]
}
```

**Response (200 OK):**
```json
{
  "results": [
    {
      "merchant_id": 75,
      "offer_id": 42,
      "audience": [...],
      "audience_size": 50,
      "model_version": "1.0",
      "reasons": [...]
    }
  ]
}
```

Each element in `results` must follow the same contract as `/lookalike`, including `reasons`.

### `GET /model/info`

Model metadata, including **lineage** - what data it was trained on.

```json
{
  "model_name": "lookalike-cf",
  "model_version": "2.0",
  "trained_on": "v2",
  "features_count": 47,
  "train_metrics": {
    "precision_at_100": 0.42
  }
}
```

Response fields are extensible, but `model_name`, `model_version`, and `trained_on` are **required**.

### `GET /monitoring/drift`

```json
{
  "drift_detected": true,
  "drift_score": 0.73,
  "action_taken": "retrained"
}
```

- `drift_detected` - drift presence indicator
- `drift_score` - numeric drift score
- `action_taken` - action taken by the service (`"retrained"` | `"none"` | `"skipped"`)

Extra fields in this response are implementation-defined, but `drift_detected`, `drift_score`, and `action_taken` are **required**.

### `GET /monitoring/data-quality`

Validation result for the latest data version (Great Expectations).

```json
{
  "version": "v2",
  "valid": true,
  "checks_total": 12,
  "checks_passed": 11,
  "checks_failed": 1,
  "failed_checks": [
    {
      "table": "transaction",
      "check": "user_id_not_null",
      "details": "3 rows with null user_id"
    }
  ]
}
```

- `valid` (bool) - data passed validation (all critical checks passed)
- `checks_total` / `checks_passed` / `checks_failed` - validation stats
- `failed_checks` - list of failed checks

Validation must be **substantive** - at least **5 checks** (`checks_total ≥ 5`) covering different aspects of data quality. One or two formal checks ("table is not empty") are **not enough**.

Examples of meaningful checks: no `NaN` in key fields (`user_id`, `merchant_id_tx`, `offer_id`), allowed categorical values, date correctness, referential integrity between tables, minimum record counts, ID uniqueness.

> **Attention:** one checker version contains **intentionally prepared data errors**. Formal validation **may not catch** them. If your service trains on corrupted data, model quality will degrade and points for related checks will be lost.

> **Important:** the checker verifies that your service actually diagnosed the issue: `failed_checks` must contain relevant reasons, not just a generic failure flag. Returning only `valid: false` without meaningful diagnostics is not sufficient.

> If data validation fails (`valid: false`), the service must not train the model on that data. In this case `action_taken` in `/monitoring/drift` should be `"skipped"`.

### `GET /experiments`

Experiment history from MLflow. Each training/retraining run must be logged as an experiment.

```json
{
  "experiments": [
    {
      "run_id": "a1b2c3",
      "data_version": "v1",
      "model_version": "1.0",
      "metrics": {
        "precision_at_100": 0.31,
        "map_at_100": 0.29
      },
      "params": {
        "model_type": "als",
        "n_factors": 64
      },
      "timestamp": "2026-01-15T10:30:00"
    },
    {
      "run_id": "d4e5f6",
      "data_version": "v2",
      "model_version": "2.0",
      "metrics": {
        "precision_at_100": 0.44,
        "map_at_100": 0.41
      },
      "params": {
        "model_type": "als",
        "n_factors": 64
      },
      "timestamp": "2026-01-15T11:15:00"
    }
  ]
}
```

Required fields for each experiment: `run_id`, `data_version`, `model_version`, `metrics`, `timestamp`. `params` is optional. The list must contain **all** training runs in the current session.

---

## Constraints

| Parameter | Limit |
|----------|-------|
| Startup time | < 180 seconds until `/ready` (otherwise 0 for everything) |
| Processing time after `/data/commit` | < 10 minutes until `pipeline_status: "idle"` |
| Memory | < 8 GB |
| CPU | 4 cores |
| GPU | No |
| `/lookalike` response time | < 2 seconds (p95) |
| Docker image size | < 4 GB |
| Network | **No internet access** during run and evaluation |
| Total service runtime within a single evaluation session | ≤ 60 minutes (after that, evaluation may be stopped) |

---

## Scoring - 100 Points

The score consists of **100 binary checks** (0 or 1 point each), grouped into **5 blocks**.

> **Global startup rule:** if `GET /ready` does not return 200 within **180 seconds** after startup - **0 points for the whole solution**.

> **Global quality rule:** if MAP@100 on the baseline model (after first training) is **worse than random ranking** - **0 points for the whole solution**. No sense validating MLOps if ML does not work.

> **Anti-fraud:** any attempt to game the evaluation system leads to **zeroing the entire result**. This includes, but is not limited to:
> - Hardcoded responses or pre-baked model that does not use input data
> - Same audience for all offers
> - Same `score` for all users (no ranking)
> - `model_version` changes without actual retraining (metrics and predictions unchanged)
> - Fake MLflow experiments (duplicate `run_id`, invalid timestamps)
> - Using external code without understanding how it works

> **Interview:** after automated evaluation there is an **interview**. You must explain your solution: architecture, model choice, drift detection logic, pipeline design. The interview may **change results both ways**:
> - If a participant **cannot explain** their code - points are **deducted** (up to full zeroing)
> - If functionality is formal-only or non-working - points for related blocks are **deducted**
> - If checker missed points due to a technical error, but participant **demonstrates and explains** a working solution - points may be **restored**

---

### Block 1: Pipeline - 20 checks

| # | Check |
|---|----------|
| 1.1 | `GET /ready` → 200, response matches contract |
| 1.2 | `POST /data/batch` → 200, response matches contract |
| 1.3 | Batches for different tables are accepted in arbitrary order |
| 1.4 | Idempotency: repeated batch does not duplicate records |
| 1.5 | `POST /data/commit` → 200, `tables_received` is correct |
| 1.6 | Idempotency: repeated commit does not restart pipeline |
| 1.7 | Pipeline completes within timeout |
| 1.8 | `GET /status` → 200, required fields present |
| 1.9 | `pipeline_status` correctly reflects state (`running` → `idle`) |
| 1.10 | `ready: true` after pipeline completion |
| 1.11 | `tables_received` matches actually submitted tables |
| 1.12 | Batch with unknown table name → 400, not 500 |
| 1.13 | Commit before batches → handled correctly, not 500 |
| 1.14 | `GET /model/info` → 200, required fields |
| 1.15 | `GET /monitoring/drift` → 200, required fields |
| 1.16 | `GET /monitoring/data-quality` → 200, required fields |
| 1.17 | `GET /experiments` → 200, ≥1 experiment with required fields |
| 1.18 | `/data/batch` with empty `records` array → handled correctly, not 500 |
| 1.19 | `/data/batch` missing required field → 400, not 500 |
| 1.20 | DVC configuration is present and valid: `dvc.yaml` + `params.yaml`, with correctly defined pipeline stages |

---

### Block 2: Model Quality - 15 checks

| # | Check |
|---|----------|
| 2.1 | `POST /lookalike` → 200 for known offer |
| 2.2 | Response matches contract (required fields, types, format, including `reasons`) |
| 2.3 | Filtering: existing merchant customers are excluded from audience |
| 2.4 | `audience_size` ≤ `top_n` |
| 2.5 | Counterfactual `reasons` sanity-check: under 2-3 controlled single-feature perturbations, `score` and/or top reason changes, the touched feature becomes top reason with a logical `impact` sign; for materially different offers, reasons should also differ materially |
| 2.6 | MAP@100 > 15% |
| 2.7 | MAP@100 > 20% |
| 2.8 | MAP@100 > 25% |
| 2.9 | MAP@100 > 30% |
| 2.10 | MAP@100 > 35% |
| 2.11 | MAP@100 > 40% |
| 2.12 | MAP@100 > 45% |
| 2.13 | MAP@100 > 50% |
| 2.14 | MAP@100 > 55% |
| 2.15 | MAP@100 > 60% |

Each next MAP level is harder than the previous one. Checker uses a hidden offer set. Reminder: if MAP@100 is below random ranking - **0 points for the whole solution**.

---

### Block 3: Reliability - 15 checks

| # | Check |
|---|----------|
| 3.1 | `POST /lookalike/batch` → 200, number of results is correct |
| 3.2 | Batch inference is consistent with single requests |
| 3.3 | Non-existing `merchant_id` or `offer_id` → 404 |
| 3.4 | Invalid `/lookalike` request → 400 |
| 3.5 | Boundary values for `top_n` are handled correctly |
| 3.6 | Determinism: repeated request → identical response and order |
| 3.7 | 10-50 concurrent `/lookalike` and `POST /data/batch` requests → all succeed without errors |
| 3.8 | Under concurrent batch writes and inference, there are no state races: data is not corrupted and responses stay consistent |
| 3.9 | `/lookalike` response time < 2s (p95) |
| 3.10 | `/lookalike` → 200 while pipeline is running (without stopping the service) |
| 3.11 | Response remains valid during pipeline run - old model keeps serving |
| 3.12 | `pipeline_status: "running"` is correctly shown during processing |
| 3.13 | New offer before new version is processed → 404 |
| 3.14 | New offer after processing (even without model retraining) → valid response |
| 3.15 | If a pipeline run fails, `/lookalike` still returns `200` using the previous model (fail-soft, no `500`) |

Service **must not** return 500 under any circumstances.

---

### Block 4: Drift Detection - 20 checks

| # | Check |
|---|----------|
| 4.1 | `drift_detected: true` on a drifted version |
| 4.2 | `action_taken: "retrained"` |
| 4.3 | `model_version` changes after retraining |
| 4.4 | `data_version` in `/status` is updated |
| 4.5 | MAP@100 did not drop after retraining |
| 4.6 | MAP@100 improved after retraining |
| 4.7 | New experiment appears in `/experiments` |
| 4.8 | `drift_detected: false` on non-drift version |
| 4.9 | `action_taken: "none"` |
| 4.10 | `model_version` does not change without drift |
| 4.11 | MAP@100 remains stable without drift |
| 4.12 | Experiment count does not increase without drift |
| 4.13 | `valid: true` on correct data, `checks_total` ≥ 5 |
| 4.14 | `trained_on` matches training data |
| 4.15 | `model_version` in `/model/info` matches `/status` |
| 4.16 | `drift_score` is adequate when drift is present |
| 4.17 | `drift_score` is adequate when drift is absent |
| 4.18 | Metrics in `/experiments` match `train_metrics` in `/model/info` |
| 4.19 | Series handling: drift → no drift |
| 4.20 | Series handling: no drift → drift |

One correct cycle is enough: detect drift → retrain → quality improves. Among test versions there will be data with **new offers but no statistical drift** - do not retrain on those.

---

### Block 5: Production Readiness - 30 checks

| # | Check |
|---|----------|
| 5.1 | `valid: false` on invalid data |
| 5.2 | `action_taken: "skipped"` when `valid: false` |
| 5.3 | `model_version` does not change after invalid data |
| 5.4 | MAP@100 remains stable after invalid data |
| 5.5 | Service does not crash after invalid data (`/ready` → 200) |
| 5.6 | Experiment count does not increase after skip |
| 5.7 | Full series: alternating drift and normal versions |
| 5.8 | Full series: multiple no-drift versions in a row |
| 5.9 | Full series: invalid data between normal versions |
| 5.10 | `action_taken` is correct across full series |
| 5.11 | `drift_score` is adequate across full series |
| 5.12 | `data_version` is up to date after each update |
| 5.13 | `trained_on` is correct after series |
| 5.14 | `model_version` is correct after series |
| 5.15 | `/monitoring/drift` updates after each version |
| 5.16 | `/monitoring/data-quality` updates after each version |
| 5.17 | Experiment count = number of actual retraining events |
| 5.18 | `/experiments`: chronology is correct (timestamps increase) |
| 5.19 | `/experiments`: metrics actually differ between experiments |
| 5.20 | No MAP regression throughout the series |
| 5.21 | MAP@100 after series ≥ baseline |
| 5.22 | Audience changes after retraining on drifted data |
| 5.23 | Audience for existing offers remains stable on no-drift version |
| 5.24 | Scores change after retraining |
| 5.25 | `data_version`, `trained_on`, DVC lineage (`dvc.lock`), and `/experiments` are mutually consistent |
| 5.26 | New offer after version processing → valid response |
| 5.27 | Service remains stable after 3+ update cycles |
| 5.28 | `/data/batch` is accepted after series completion (ready for new data) |
| 5.29 | `dvc repro` runs successfully in the evaluation environment without manual steps |
| 5.30 | `/status` → `ready: true` at the end of all checks |

Your service must behave like a production system: no crashes, no state loss, correct processing of any incoming data.

---

## FAQ

**Which programming language should I use?** <br/>
Python is recommended.

**Can I use any ML model?** <br/>
Yes, but it must be **local**. There is no internet access during run and evaluation.

**Can I use GPU?** <br/>
No. CPU only.

**Which version of the solution will be tested?** <br/>
The last commit in the `main` branch at the time of the deadline will be sent for verification.
