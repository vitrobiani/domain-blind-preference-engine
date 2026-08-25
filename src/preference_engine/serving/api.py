"""
HTTP serving layer.

FastAPI app exposing:
- POST /interactions  - publishes a new interaction event to Kafka
- GET  /recommend/{user_id}  - returns top-k recommendations from the in-memory engine
- GET  /health  - sanity check

The engine and feature store live for the lifetime of the process; Kafka is
optional (endpoints depending on it return 503 if the producer is unavailable).
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from preference_engine.combiner.combiner import FINAL_SCORE, BREAKDOWN
from preference_engine.combiner.heuristics import load_heuristics
from preference_engine.ingest.producer import _create_producer
from preference_engine.ingest.topics import INTERACTIONS_TOPIC, InteractionEvent
from preference_engine.serving.query import RecommendationEngine, DEFAULT_MODELS_PATH
from preference_engine.spark_session import get_spark_session
from preference_engine.streaming.feature_store import (
    FeatureStore,
    DEFAULT_FEATURE_STORE_PATH,
    append_interactions,
    refresh_aggregates,
)


class InteractionIn(BaseModel):
    """A single interaction inside the POST /interactions batch."""

    user_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    value: float | None = None
    ts: datetime | None = None  # defaults to now() if omitted


class InteractionBatchIn(BaseModel):
    """Request body for POST /interactions - always a batch."""

    interactions: list[InteractionIn] = Field(min_length=1, max_length=100000)


class IngestResponse(BaseModel):
    status: str
    accepted: int
    published_to_kafka: bool
    retrained_signals: list[str]
    skipped_signals: list[str]


class RecommendationOut(BaseModel):
    """One row of the recommendation response."""

    item_id: str
    score: float
    breakdown: dict[str, float]


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendationOut]


def create_app(
    heuristics_path: Path,
    feature_store_path: Path = DEFAULT_FEATURE_STORE_PATH,
    models_path: Path = DEFAULT_MODELS_PATH,
    bootstrap_servers: str = "localhost:9092",
) -> FastAPI:
    """
    Build the FastAPI app with lifespan-managed Spark / engine / Kafka.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(f"Loading heuristics from {heuristics_path}...")
        heuristics = load_heuristics(heuristics_path)

        print("Starting SparkSession...")
        spark = get_spark_session(enable_kafka=False)
        spark.sparkContext.setLogLevel("WARN")

        print(f"Loading feature store from {feature_store_path}...")
        feature_store = FeatureStore(path=feature_store_path, spark=spark)

        print(f"Fitting engine (models_path={models_path})...")
        engine = RecommendationEngine(heuristics, feature_store, models_path=models_path)
        engine.fit()

        print(f"Connecting to Kafka at {bootstrap_servers}...")
        producer = None
        try:
            producer = _create_producer(bootstrap_servers)
            print("  Kafka producer ready.")
        except Exception as exc:  # noqa: BLE001
            # A broker-less startup is still useful for /recommend
            print(f"  Kafka unavailable ({exc}); /interactions will return 503.")

        app.state.spark = spark
        app.state.engine = engine
        app.state.heuristics = heuristics
        app.state.producer = producer
        app.state.bootstrap_servers = bootstrap_servers
        app.state.feature_store_path = feature_store_path
        # Serialize retrains so concurrent POSTs don't race on model dirs
        app.state.retrain_lock = threading.Lock()

        try:
            yield
        finally:
            print("Shutting down...")
            if producer is not None:
                producer.close()
            spark.stop()

    app = FastAPI(title="Preference Engine", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        engine: RecommendationEngine = app.state.engine
        return {
            "status": "ok",
            "signals": sorted(engine._scorers.keys()),
            "kafka": app.state.producer is not None,
            "bootstrap_servers": app.state.bootstrap_servers,
        }

    @app.post("/interactions", response_model=IngestResponse, status_code=202)
    def post_interactions(batch: InteractionBatchIn) -> IngestResponse:
        """
        Ingest a batch of interactions.

        For each POST we:
          1. Publish every event to Kafka (best-effort, non-blocking).
          2. Append the batch to the feature store's interactions table.
          3. Recompute popularity + recency aggregates.
          4. Retrain every active signal against the fresh feature store.
          5. Atomically swap in the new model set on disk.

        The handler blocks until step 4 completes; the response reflects the
        new model set. Concurrent POSTs are serialized by a retrain lock.
        """
        events = [
            InteractionEvent(
                user_id=i.user_id,
                item_id=i.item_id,
                value=i.value,
                ts=i.ts or datetime.now(),
            )
            for i in batch.interactions
        ]

        engine: RecommendationEngine = app.state.engine
        spark = app.state.spark
        producer = app.state.producer
        fs_path: Path = app.state.feature_store_path

        with app.state.retrain_lock:
            # 1. Best-effort Kafka publish (an audit trail; not on the critical path)
            published = False
            if producer is not None:
                for event in events:
                    producer.send(INTERACTIONS_TOPIC, value=event.to_json())
                producer.flush()
                published = True

            # 2. Append to the interactions parquet table
            append_interactions(spark, fs_path, events)

            # 3. Cache is now stale, force a re-read before aggregating
            engine.feature_store.clear_cache()
            refresh_aggregates(spark, fs_path)

            # 4. Recompute everything else - clear cache again so signals see
            #    the fresh popularity/recency tables, then force a full retrain.
            engine.feature_store.clear_cache()
            engine.fit(force=True)

        return IngestResponse(
            status="accepted",
            accepted=len(events),
            published_to_kafka=published,
            retrained_signals=sorted(engine._scorers.keys()),
            skipped_signals=sorted(
                set(engine.heuristics.get_active_signals()) - engine._scorers.keys()
            ),
        )

    @app.get("/recommend/{user_id}", response_model=RecommendResponse)
    def get_recommendations(user_id: str) -> RecommendResponse:
        engine: RecommendationEngine = app.state.engine

        rows = engine.recommend(user_id).collect()
        recs = [
            RecommendationOut(
                item_id=row["item_id"],
                score=float(row[FINAL_SCORE]),
                breakdown={k: float(v) for k, v in (row[BREAKDOWN] or {}).items()},
            )
            for row in rows
        ]
        return RecommendResponse(user_id=user_id, recommendations=recs)

    return app
