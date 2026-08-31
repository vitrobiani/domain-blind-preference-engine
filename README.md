# Preference Engine

A domain-blind, PySpark-based recommendation engine. Given any dataset that fits `(user, item, interaction)` records plus a small adapter and a YAML config, it returns a ranked list of items for any user with a per-signal score breakdown.

The whole system is built around one idea: the core knows nothing about the domain. All domain knowledge lives in adapter files. Every signal, aggregator, and combiner operates on the canonical schema `(user_id, item_id, value, ts)`. Write one adapter for a new dataset and every signal that makes sense for it works for free.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [End-to-End Lifecycle](#end-to-end-lifecycle)
- [Signals](#signals)
  - [popularity](#popularity)
  - [trends](#trends)
  - [segments](#segments)
  - [content](#content)
  - [als_cf](#als_cf)
  - [rules](#rules)
- [Adapters](#adapters)
  - [synthetic](#synthetic)
  - [movies (MovieLens 100K)](#movies-movielens-100k)
  - [elections (INES 2025)](#elections-ines-2025)
- [Heuristics Configuration](#heuristics-configuration)
- [Feature Store](#feature-store)
- [Streaming and HTTP Serving](#streaming-and-http-serving)
- [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)
- [Diagrams to Add](#diagrams-to-add)

---

## Overview

Two ideas hold the system together:

1. **Domain-blind core.** Every signal, feature-store builder, and combiner works on the canonical schema `(user_id, item_id, value, ts)` defined in [schema.py](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/schema.py). None of them know what a "movie" or "party" is.
2. **Adapters.** Domain knowledge is quarantined to one file per dataset. An adapter subclasses [`DomainAdapter`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/adapter/base.py) and returns the canonical schema plus two feature tables. Add a new adapter, get every signal for free.

The three built-in adapters (`synthetic`, `movies`, `elections`) are radically different domains, but the same signals score all of them without touching any core code.

---

## Quick Start

End-to-end walkthrough against MovieLens. Every step is a single command; the second half is meant to be run live in front of an audience.

### 0. One-time setup

```bash
# venv (on NixOS, add --system-site-packages so numpy/BLAS resolve)
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

The `-c` flag on every `preference-engine` command below picks a heuristics YAML, which in turn picks the adapter (`domain:` field) and sets signal weights. See [Heuristics Configuration](#heuristics-configuration).

### 1. Start Kafka

```bash
docker compose up -d
docker ps --filter name=preference-engine-kafka   # confirm broker is up
```

### 2. Bootstrap an empty movie feature store

Writes only the user + movie catalog to `.data/ml-100k/feature_store/`. No interactions.

```bash
preference-engine -c adapters/movies/heuristics.yaml bootstrap
```

### 3. Start the server

```bash
preference-engine -c adapters/movies/heuristics.yaml serve
```

On startup the server fits every active signal against the empty interaction table. Signals that need interactions (ALS) will be skipped and logged; popularity / trends / content / segments fit against the catalog only. FastAPI listens on `:8000`.

### 4. See what "no data" looks like

In a second terminal:

```bash
curl http://localhost:8000/recommend/196
```

Every candidate comes back with the same score. There are no interactions yet, so each scorer collapses to a constant fallback and min-max normalization pins them to 0.5. This is the "cold" baseline.

### 5. Watch Kafka (optional, third terminal)

```bash
./kafka_consumer_viewer.sh
```

Tails the `preference-engine.interactions` topic so you can watch each event flow through Kafka as you POST.

### 6. Convert MovieLens ratings to a POST body

```bash
python3 udata_to_json.py data/ml-100k/u.data interactions.json
```

Produces an ~8.7 MB JSON file matching the [POST /interactions](#endpoints) schema: `{"interactions": [{"user_id":..., "item_id":..., "value":..., "ts":...}, ...]}`.

### 7. Feed the whole dataset in

```bash
curl -X POST http://localhost:8000/interactions \
     -H 'Content-Type: application/json' \
     --data @interactions.json
```

Publishes all 100,000 ratings to Kafka (visible in the terminal from step 5), appends them to the feature store, and triggers a full retrain. The request blocks until retraining finishes; expect several minutes on a laptop.

### 8. See what "with data" looks like

```bash
curl http://localhost:8000/recommend/196
```

Scores now vary. The engine has real interaction history to rank against.

---

For alternatives to the REST path (batch feature-store build from raw CSVs, CLI-only fit/recommend, ingesting through the Kafka producer instead of HTTP), see [CLI Reference](#cli-reference).

---

## Architecture

Three layers, each domain-blind past the adapter:

<details open>
<summary><b>Layer 1: Adapters (domain-specific I/O)</b></summary>

Files: [`adapters/<name>/adapter.py`](https://github.com/vitrobiani/domain-blind-preference-engine/tree/main/adapters).

Every adapter subclasses [`DomainAdapter`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/adapter/base.py) and returns:

| Method | Returns |
|---|---|
| `interactions(spark)` | DataFrame with `(user_id, item_id, value, ts)` |
| `user_features(spark)` | DataFrame with `user_id` plus any user columns |
| `item_features(spark)` | DataFrame with `item_id` plus any item columns |
| `feature_specs()` | Descriptors for each feature column (numeric / categorical / ordinal / text, applies to user or item) |

The adapter is also where `dataset_name` (used to derive `.data/<dataset_name>/` storage paths) and `interaction_type` (`EXPLICIT` for ratings, `IMPLICIT` for counts/binary) are declared.

Discovery is by convention. Adapters are looked up by directory name via [`registry.py`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/adapter/registry.py) which scans `adapters/` for subdirectories containing `adapter.py`.

</details>

<details open>
<summary><b>Layer 2: Feature Store (domain-blind preprocessing)</b></summary>

File: [`streaming/feature_store.py`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/streaming/feature_store.py).

Parquet-backed cache of five tables:

| Table | Contents | Written by |
|---|---|---|
| `interactions` | Raw canonical interactions | Adapter, streaming append |
| `user_features` | User columns from adapter | Adapter |
| `item_features` | Item columns from adapter | Adapter |
| `popularity` | `(item_id, count)` | Aggregation |
| `recency` | `(item_id, recency_score)` with exponential time decay | Aggregation |

Layout: `.data/<dataset_name>/feature_store/<table>/`. The dataset_name comes from the adapter, so building against movies and elections never clobber each other.

The `FeatureStore` dataclass caches read DataFrames in memory. `clear_cache()` is called after streaming appends so signals re-read fresh data before retraining.

</details>

<details open>
<summary><b>Layer 3: Signals + Combiner + Engine</b></summary>

Files: [`signals/`](https://github.com/vitrobiani/domain-blind-preference-engine/tree/main/src/preference_engine/signals), [`combiner/`](https://github.com/vitrobiani/domain-blind-preference-engine/tree/main/src/preference_engine/combiner), [`serving/query.py`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/serving/query.py).

Each signal is a `Scorer` subclass that implements `fit(ctx, params)` and `raw_score(user_id, candidates)`. The base [`Scorer.score()`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/signals/base.py#L73) wraps raw scores with min-max normalization to `[0, 1]`.

The [`RecommendationEngine`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/serving/query.py#L26) loads scorers from disk, trains anything missing, calls each active scorer against candidate items, and hands the per-signal DataFrames to the [`combine()`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/combiner/combiner.py#L25) function which does the weighted sum and returns top-k with a per-signal breakdown map.

Cold cases (user missing from ALS training, no user features for segments) return an empty DataFrame from `raw_score()`. The combiner treats missing signals as zero and continues.

</details>

---

## End-to-End Lifecycle

There are two supported flows.

### Batch (build once, recommend forever)

```
1. build-features --batch   adapter -> parquet feature store
2. fit                      trains + persists every active signal to .data/<dataset>/models/
3. recommend --user <id>    loads models, scores every candidate, returns top-k
```

Everything is deterministic given the same feature store. Rerunning `fit` without `--refit` loads existing models from disk instead of retraining.

### Streaming (bootstrap empty, feed live)

```
1. bootstrap                writes empty interactions table + full user/item catalog
2. serve                    boots FastAPI on :8000 with in-memory engine + Kafka producer
3. POST /interactions       for each batch:
                              a. best-effort publish to Kafka (audit trail)
                              b. append rows to interactions parquet
                              c. refresh_aggregates() recomputes popularity + recency
                              d. engine.fit(force=True) retrains every active signal
                              e. atomic-swap the new model set on disk
4. GET /recommend/<user_id>  scores against the freshly retrained models
```

Concurrent POSTs are serialized by a threading.Lock so retrain doesn't race with itself. The response reports which signals were retrained and which were skipped (per-signal failures are logged and don't kill the whole retrain).

---

## Signals

All signals live under [`src/preference_engine/signals/`](https://github.com/vitrobiani/domain-blind-preference-engine/tree/main/src/preference_engine/signals) and share the contract in [`base.py`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/signals/base.py):

- `fit(ctx, params)` prepares the model against the [`FeatureStore`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/streaming/feature_store.py#L53)
- `raw_score(user_id, candidates)` returns `DataFrame[item_id, raw_score]`
- Base `.score(...)` normalizes to `[0, 1]` per signal via min-max

Cold cases return empty DataFrames and the combiner ignores them.

Registered signals live in [`signals/registry.py`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/signals/registry.py). The full catalog:

| Signal | Personalized | Needs history | Best when |
|---|---|---|---|
| popularity | no | no | always (fallback baseline) |
| trends | no | no | interactions span time |
| segments | yes | no (uses demographics) | rich user features |
| content | yes | yes | rich item features |
| als_cf | yes | yes | dense multi-item histories |
| rules | yes | yes (basket >= 2) | multi-item transactions |

<details>
<summary><h3 id="popularity">popularity</h3></summary>

File: [signals/popularity.py](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/signals/popularity.py)

**What it does.** Ranks items by their global interaction count.

**Fit.** Reads the precomputed `popularity` table from the feature store. That table is a simple `groupBy(item_id).count()` computed by [`_compute_popularity`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/streaming/feature_store.py#L107).

**Score.** Left-joins candidates with the popularity table, defaults missing items to zero.

**Params.** None.

**When it works.** Always. Every dataset has interaction counts.

**When it fails.** Never. It's the fallback baseline.

</details>

<details>
<summary><h3 id="trends">trends</h3></summary>

File: [signals/trends.py](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/signals/trends.py)

**What it does.** Popularity, but weighted by recency. Old interactions decay exponentially.

**Fit.** Reads the precomputed `recency` table. Recency scores are computed by [`_compute_recency`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/streaming/feature_store.py#L124) as:

```
recency_score(item) = sum(exp(-lambda * age_in_days))
lambda = ln(2) / halflife_days
```

An interaction that's `halflife_days` old counts for exactly half of a fresh one.

**Score.** Left-join candidates with recency, default missing to zero.

**Params.** `halflife_days` (default 30).

**When it works.** Datasets where interactions arrive over meaningful time.

**When it fails.** All timestamps in the same window (like the elections survey), because every item's decay weight is the same and `recency` degenerates to a scaled `popularity`.

</details>

<details>
<summary><h3 id="segments">segments</h3></summary>

File: [signals/segments.py](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/signals/segments.py)

**What it does.** Clusters users by their demographic features, then for each cluster measures which items the cluster prefers. Scores candidates by the user's cluster's preferences.

**Fit.**
1. Reads `user_features`, StringIndexes categoricals, VectorAssembles into feature vectors.
2. Runs Spark MLlib KMeans with `k` clusters.
3. Computes per-cluster item preferences: for each `(cluster, item_id)`, the score is `avg(value) * count(*)`. This dual factor is important: pure `avg(value)` alone collapses to 1.0 for binary interactions (like elections v104), losing all "how many people picked this" signal. Multiplying by count preserves both the rating quality axis (for explicit ratings) and the support axis (for binary interactions).

**Score.** Looks up user's cluster, returns cluster preferences joined onto candidates. Cold user (no features) returns empty.

**Params.** `k` (number of clusters). Rule of thumb: aim for 30 to 100 users per cluster. For 1,588 elections respondents, `k: 15` gives ~106 per cluster.

**When it works.** Rich user demographics, especially valuable when interaction history is sparse or single-shot.

**When it fails.** No user_features. Also: VectorAssembler is set to `handleInvalid="skip"`, so users with null values in any demographic column are silently dropped from clustering. Prefer bucketed columns with no nulls over raw numeric columns that might be sparse.

</details>

<details>
<summary><h3 id="content">content</h3></summary>

File: [signals/content.py](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/signals/content.py)

**What it does.** Cosine similarity between a user's taste profile and each item's feature vector.

**Fit.**
1. Builds a normalized feature vector for each item from `item_features` (StringIndexer for categoricals, VectorAssembler + Normalizer).
2. Builds a user profile per user as the weighted average of vectors of items they've interacted with, weighted by `value`.

**Score.** Cosine similarity between the user's profile and each candidate's vector. Cold user (no history) returns empty.

**Params.** `feature_weights` map (currently unused inside fit, reserved for future weighting).

**When it works.** Rich item metadata (genres, categories, numerical attributes).

**When it fails.** No item metadata beyond IDs. Also degenerates for single-choice datasets: if the user's history has exactly one item, the user profile equals that item's vector and cosine similarity just recommends the same item back.

</details>

<details>
<summary><h3 id="als_cf">als_cf</h3></summary>

File: [signals/als_cf.py](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/signals/als_cf.py)

**What it does.** Collaborative filtering via Spark MLlib's Alternating Least Squares matrix factorization. Learns latent user and item factors and predicts ratings as their dot product.

**Fit.**
1. Uses [`IDMapper`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/ids.py) to convert string IDs to the integer IDs ALS requires.
2. Filters nulls (explicit mode) or coalesces to 1.0 (implicit mode) in the `value` column.
3. Trains ALS with `rank`, `regParam`, `maxIter`, `implicitPrefs`, `coldStartStrategy="drop"`, `seed=42`.

**Score.** Converts user_id to int, generates user-item pairs, calls `model.transform()`, unwraps predictions. Cold user (not in training data) returns empty.

**Params.** `rank` (latent factor dimension), `reg` (L2 regularization), `max_iter`, plus `implicit` (constructor arg, not in params).

**When it works.** Dense user-item matrices with multi-item histories per user.

**When it fails.** Sparse or single-choice datasets. If every user has exactly one interaction, factorization has no co-occurrence signal to learn from.

**Persistence.** The trained `ALSModel` is saved via Spark ML's native `.save()`, and the `IDMapper` labels are saved as parquet. Load reconstructs via `ALSModel.load()` and `StringIndexerModel.from_labels()`.

</details>

<details>
<summary><h3 id="rules">rules</h3></summary>

File: [signals/rules.py](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/signals/rules.py)

**What it does.** Association rule mining with FP-Growth. Treats each user's history as a shopping basket. Discovers rules like `{Toy Story, Star Wars} -> {Empire}` with confidence and support.

**Fit.** Groups interactions by user into `collect_set(item_id)` baskets, fits `pyspark.ml.fpm.FPGrowth`, keeps the resulting association rules DataFrame.

**Score.** Finds rules whose antecedent is a subset of the user's basket, unions their consequents, aggregates by max confidence per item.

**Params.** `min_support` (minimum fraction of baskets containing the itemset), `min_confidence` (minimum conditional probability of consequent given antecedent).

**When it works.** Datasets with real multi-item transactions.

**When it fails.** Baskets of size 1. No co-occurrence to discover, no rules generated.

</details>

---

## Adapters

Each adapter directory lives at `adapters/<name>/`. Discovered by name from the `domain:` field in the heuristics YAML. See [`registry.py`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/adapter/registry.py).

<details>
<summary><h3 id="synthetic">synthetic</h3></summary>

File: [adapters/synthetic/adapter.py](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/adapters/synthetic/adapter.py)

Fake data generator for tests and pipeline smoke tests. No external data files.

**Config:**
- `n_users` (default 200)
- `n_items` (default 100)
- `n_interactions` (default 2000)
- `seed` (default 42)

**Interactions:** random `(user_i, item_j)` pairs with 1-5 star ratings and timestamps spread over the last year (base date fixed at 2025-06-01 for reproducibility).

**User features:** `age_group` (young / adult / senior), `activity_level` (0-100 float).

**Item features:** `category` (A / B / C / D), `quality_score` (0-10 float).

**dataset_name:** `synthetic`. All output goes to `.data/synthetic/`.

Used by the persistence test suite ([tests/test_persistence.py](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/tests/test_persistence.py)) to round-trip every scorer without needing MovieLens installed.

</details>

<details>
<summary><h3 id="movies-movielens-100k">movies (MovieLens 100K)</h3></summary>

File: [adapters/movies/adapter.py](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/adapters/movies/adapter.py) - Config: [adapters/movies/heuristics.yaml](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/adapters/movies/heuristics.yaml)

The classic. 100,000 ratings from 943 users on 1,682 movies. Expects the dataset at `data/ml-100k/`.

**Interactions:** loaded from `u.data` (tab-separated `user_id, item_id, rating, timestamp`). Rating is 1-5 stars. `interaction_type = EXPLICIT`.

**User features:** loaded from `u.user` (pipe-separated `user_id, age, gender, occupation, zip`).
- `age` (numeric)
- `gender` (categorical)
- `occupation` (categorical)

**Item features:** loaded from `u.item` (pipe-separated, latin-1 encoded). Genre flags are 19 binary columns.
- `year` extracted from title via regex `\((\d{4})\)`
- `primary_genre` (categorical, first matching genre)
- 19 binary genre flags (`genre_Action`, `genre_Comedy`, etc.)

**dataset_name:** `ml-100k`. All output goes to `.data/ml-100k/`.

**Default signal weights** (see [heuristics.yaml](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/adapters/movies/heuristics.yaml)):

| Signal | Weight |
|---|---|
| popularity | 0.1 |
| als_cf | 0.5 |
| content | 0.2 |
| segments | 0.1 |
| trends | 0.1 |
| rules | 0.0 (disabled) |

ALS is the workhorse because the user-item matrix is dense (avg ~106 ratings per user).

</details>

<details open>
<summary><h3 id="elections-ines-2025">elections (INES 2025)</h3></summary>

File: [adapters/elections/adapter.py](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/adapters/elections/adapter.py) - Config: [adapters/elections/heuristics.yaml](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/adapters/elections/heuristics.yaml)

Israeli National Election Study, off-cycle 2025 survey. 1,588 respondents (1,308 Jews + 119 Arabs plus others). Data at `data/Elections/2025_STATA.csv`. Original source: [Tel Aviv University INES](https://socsci4.tau.ac.il/mu2/ines/data/our-data/).

**The design challenge.** Every respondent has exactly one interaction: their answer to question 104 ("if elections were held today, which party would you vote for?"). This shape completely changes the signal mix:

| Signal | Works? | Why |
|---|---|---|
| popularity | yes | Trivially counts party votes |
| segments | yes (main workhorse) | Rich demographics, sparse interactions |
| trends | no | All timestamps in the same survey window |
| als_cf | no | One interaction per user, no factorization signal |
| content | no | Parties have no metadata; single-item user profile is degenerate |
| rules | no | Baskets of size 1 can't produce co-occurrence rules |

**Interactions:** `(resp_id, v104_party)` with `value=1.0` flat, `ts=date_of_survey`.

**User features** (from [`user_features`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/adapters/elections/adapter.py#L107)):
- `agegroup` (categorical, 6 buckets; used instead of raw `age` which is 58.8% null)
- `gender`
- `educ`
- `religiosity`
- `sector`
- `district` (from `District_CBS`, geographic bucket to preserve respondent anonymity per Israeli privacy law)

All demographic codes are cast to string so segments' StringIndexer treats them as categorical. The code orderings aren't documented as ordinal in the appendix, so treating them as numeric would be wrong.

**Party codes.** The v104 answer is an integer code. The adapter maps codes 1-13 to readable slugs via [`PARTY_NAMES`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/adapters/elections/adapter.py#L41):

| Code | Party |
|---|---|
| 1 | Likud (Netanyahu) |
| 2 | YeshAtid (Lapid) |
| 3 | NationalUnity (Gantz) |
| 4 | ReligiousZionism (Smotrich) |
| 5 | Shas (Deri) |
| 6 | UnitedTorahJudaism (Gafni) |
| 7 | YisraelBeiteinu (Lieberman) |
| 8 | HaDemocratim / Labor-Meretz (Golan) |
| 9 | OtzmaYehudit (Ben Gvir) |
| 10 | NationalRight (Saar) |
| 11 | HadashTaal (Odeh) |
| 12 | Raam (Abbas) |
| 13 | Balad (Abu Shahadeh) |

Sentinel codes are filtered out in `interactions()` and never enter the recommendation universe:
- 30 = other (specify)
- 94 = don't intend to vote
- 96 = blank ballot
- 97 = undecided
- 98 = don't know
- 99 = refuse

Without this filter the recommender surfaces "undecided" as its top result (it's the plurality answer in the sample), which is useless.

**`item_features`** returns the fixed 13-party catalog so downstream joins have every party available even if a party drew zero votes in the sample.

**Default signal weights:**

| Signal | Weight | Reason |
|---|---|---|
| popularity | 0.3 | Baseline party ranking |
| segments | 0.7 | Rich demographics = the main personalization signal |
| als_cf, content, rules, trends | 0.0 | Not applicable to this dataset shape |

`filters.exclude_seen` is set to `false` so the user's actually-voted-for party stays in the recommendations (so you can eyeball "did the model predict what they voted for?" as a sanity check).

**dataset_name:** `Elections`. All output goes to `.data/Elections/`.

</details>

---

## Heuristics Configuration

File: [`combiner/heuristics.py`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/combiner/heuristics.py) (loader), [`config/heuristics.example.yaml`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/config/heuristics.example.yaml) (reference).

The YAML is validated with pydantic. All fields:

```yaml
domain: movies                   # adapter directory name under adapters/
top_k: 20                        # number of items in the response
normalization: minmax            # minmax | zscore (per-signal normalization inside Scorer.score)

signals:
  popularity:
    weight: 0.1
  als_cf:
    weight: 0.5
    params:
      rank: 32
      reg: 0.1
      max_iter: 10
  content:
    weight: 0.2
    feature_weights: {}          # feature_name -> weight; empty means equal
  segments:
    weight: 0.1
    params:
      k: 8
  rules:
    weight: 0.0                  # weight: 0 disables the signal entirely
    params:
      min_support: 0.01
      min_confidence: 0.2
  trends:
    weight: 0.1
    params:
      halflife_days: 30

filters:
  exclude_seen: true             # skip items the user already interacted with
  exclude_items: []              # list of item IDs to never recommend
  require_features: {}           # (reserved) item must have feature=value
```

Signals with weight 0 are skipped during `fit()` and `recommend()`, saving computation. Change a weight to enable or disable a signal without touching code.

The `normalization` field is currently only honored for per-signal normalization in `Scorer.score()`. Combiner-level normalization always sums weighted contributions.

---

## Feature Store

File: [`streaming/feature_store.py`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/streaming/feature_store.py)

Persistent parquet cache. Directory layout:

```
.data/
  <dataset_name>/
    feature_store/
      interactions/       (user_id, item_id, value, ts)
      user_features/      (user_id, ...)
      item_features/      (item_id, ...)
      popularity/         (item_id, count)
      recency/            (item_id, recency_score)
      .checkpoints/       (streaming mode only)
    models/
      popularity/manifest.json
      als_cf/als_model/, id_mapper/, manifest.json
      segments/kmeans_model/, user_clusters/, cluster_preferences/, manifest.json
      content/item_vectors/, user_profiles/, manifest.json
      rules/rules/, user_items/, manifest.json
      trends/manifest.json
```

The `<dataset_name>` prefix comes from the adapter, so builds against different datasets never overwrite each other.

**API surface:**
- `FeatureStore(path, spark).interactions() / .popularity() / .recency() / .user_features() / .item_features()` returns cached DataFrames.
- `build_features_batch(adapter, spark, path)` runs the adapter and writes all five tables.
- `bootstrap_features(adapter, spark, path)` writes only the catalog (empty interactions/popularity/recency); used for the streaming cold-start.
- `append_interactions(spark, path, events)` appends new rows to `interactions/`.
- `refresh_aggregates(spark, path)` recomputes and overwrites `popularity/` and `recency/`.
- `build_features_streaming(spark, bootstrap_servers, path)` starts a Spark Structured Streaming job that consumes Kafka and appends to `interactions/`.

**Model persistence** ([`signals/persistence.py`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/signals/persistence.py)):
- `save_scorers(scorers, root)` writes each scorer to `root/.<name>.tmp/` then atomic-renames to `root/<name>/`. Concurrent readers never see a half-written directory.
- `load_scorers(root, ctx)` iterates root, reads each `manifest.json` to determine the scorer class, delegates to `cls.load()`.

---

## Streaming and HTTP Serving

File: [`serving/api.py`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/serving/api.py)

FastAPI app with three endpoints. Lifespan manages Spark, engine, and Kafka producer.

### Endpoints

| Method | Path | Body / Params | Response |
|---|---|---|---|
| `GET` | `/health` | none | `{status, signals, kafka, bootstrap_servers}` |
| `POST` | `/interactions` | `InteractionBatchIn` | `IngestResponse` (202 Accepted) |
| `GET` | `/recommend/{user_id}` | none | `RecommendResponse` (top-k) |

### Request/response schemas

`InteractionBatchIn` (min 1, max 100000 events per batch):

```json
{
  "interactions": [
    {"user_id": "u1", "item_id": "i5", "value": 4.5, "ts": "2026-08-25T14:32:00"},
    {"user_id": "u1", "item_id": "i9", "value": null}
  ]
}
```

`value` and `ts` are optional (ts defaults to server-side `now()`).

`IngestResponse`:

```json
{
  "status": "accepted",
  "accepted": 2,
  "published_to_kafka": true,
  "retrained_signals": ["popularity", "segments"],
  "skipped_signals": ["als_cf"]
}
```

### The POST /interactions flow (per batch)

1. Best-effort Kafka publish. Each event goes to `preference-engine.interactions` topic. Kafka is optional; if the broker is down at boot, the endpoint sets `published_to_kafka: false` and continues.
2. `append_interactions()` appends new rows to `interactions/`.
3. `feature_store.clear_cache()` so signals re-read from disk.
4. `refresh_aggregates()` recomputes popularity + recency.
5. `feature_store.clear_cache()` again.
6. `engine.fit(force=True)` retrains every active signal from scratch. Per-signal failures are logged and reported in `skipped_signals` but don't kill the retrain.
7. `save_scorers()` does the atomic-swap so readers never see partial state.

Concurrent POSTs are serialized by `app.state.retrain_lock` so the retrain loop can assume it owns the model directory.

### Kafka setup

File: [`docker-compose.yml`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/docker-compose.yml)

Single-node Kafka in KRaft mode (no Zookeeper), Confluent 7.6.0 image. Listens on `localhost:9092`. Persistent volume for the log.

```bash
docker compose up -d           # start
docker compose down            # stop, preserve data
docker compose down -v         # stop, wipe data
```

Watch the interactions topic:

```bash
./kafka_consumer_viewer.sh
```

This shells into the container and runs `kafka-console-consumer` with `print.timestamp=true`.

The producer library is [`kafka-python`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/ingest/producer.py). Spark uses the `spark-sql-kafka-0-10_2.12` connector for the streaming consumer.

---

## CLI Reference

File: [`cli.py`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/src/preference_engine/cli.py)

Entry point registered in [`pyproject.toml`](https://github.com/vitrobiani/domain-blind-preference-engine/blob/main/pyproject.toml) as `preference-engine`.

Global flag:
- `-c` / `--heuristics PATH` (default `config/heuristics.example.yaml`)

Subcommands:

| Subcommand | Purpose | Key flags |
|---|---|---|
| `ingest` | Replay adapter interactions to Kafka | `--bootstrap-servers` |
| `build-features` | Build feature store from adapter (batch) or Kafka (streaming) | `--batch`, `--output`, `--bootstrap-servers` |
| `bootstrap` | Empty feature store from adapter catalog only (for streaming cold start) | `--output` |
| `fit` | Load or train every active signal, persist to disk | `--feature-store`, `--models`, `--refit` |
| `recommend` | Print top-k recommendations for a user | `--user`, `--feature-store`, `--models`, `--refit` |
| `serve` | Run the FastAPI server | `--host`, `--port`, `--feature-store`, `--models`, `--bootstrap-servers` |

Path defaults are derived from the adapter's `dataset_name`. Explicit `--feature-store`, `--models`, `--output` override.

Examples:

```bash
# MovieLens end-to-end
preference-engine -c adapters/movies/heuristics.yaml build-features --batch
preference-engine -c adapters/movies/heuristics.yaml fit
preference-engine -c adapters/movies/heuristics.yaml recommend --user 42

# Elections end-to-end
preference-engine -c adapters/elections/heuristics.yaml build-features --batch
preference-engine -c adapters/elections/heuristics.yaml fit
preference-engine -c adapters/elections/heuristics.yaml recommend --user 295709

# Streaming with HTTP
preference-engine -c adapters/movies/heuristics.yaml bootstrap
preference-engine -c adapters/movies/heuristics.yaml serve --port 8000
curl -X POST localhost:8000/interactions -H 'Content-Type: application/json' \
  -d '{"interactions":[{"user_id":"u1","item_id":"i5","value":4.5}]}'
curl localhost:8000/recommend/u1

# Force retraining of everything
preference-engine -c adapters/movies/heuristics.yaml fit --refit
```

---

## Project Structure

```
final_project/
  README.md                                this file
  pyproject.toml                           package definition, deps, CLI entry point
  docker-compose.yml                       single-node Kafka (KRaft, no Zookeeper)
  kafka_consumer_viewer.sh                 tail the interactions topic

  src/preference_engine/                   the domain-blind core
    __init__.py
    cli.py                                 argparse subcommands
    schema.py                              canonical column names + FeatureSpec dataclass
    spark_session.py                       SparkSession factory (with optional Kafka connector)
    ids.py                                 IDMapper (string <-> int for ALS)

    adapter/
      base.py                              DomainAdapter ABC
      registry.py                          discover + instantiate by name

    combiner/
      combiner.py                          weighted-sum aggregator + filters
      heuristics.py                        pydantic YAML loader

    signals/
      base.py                              Scorer ABC + normalization
      registry.py                          scorer name -> class map
      persistence.py                       atomic save/load of the full scorer set
      popularity.py
      trends.py
      segments.py
      content.py
      als_cf.py
      rules.py

    ingest/
      topics.py                            Kafka topic name + InteractionEvent
      producer.py                          kafka-python producer

    serving/
      api.py                               FastAPI app + lifespan + POST /interactions
      query.py                             RecommendationEngine

    streaming/
      feature_store.py                     parquet feature store + Kafka structured streaming

  adapters/                                one directory per domain
    synthetic/
      adapter.py
    movies/
      adapter.py
      heuristics.yaml
    elections/
      adapter.py
      heuristics.yaml

  config/
    heuristics.example.yaml                reference heuristics config

  data/                                    external datasets (gitignored)
    ml-100k/                               MovieLens 100K
    Elections/                             INES 2025

  .data/                                   generated feature stores + models (gitignored)
    <dataset_name>/
      feature_store/...
      models/...

  tests/
    test_persistence.py                    scorer save/load round-trip
```
