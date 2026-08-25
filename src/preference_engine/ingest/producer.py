"""
Kafka producer for replaying dataset interactions.

Publishes interaction events to Kafka for consumption by the feature store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kafka import KafkaProducer

from preference_engine.ingest.topics import INTERACTIONS_TOPIC, InteractionEvent
from preference_engine.schema import USER_ID, ITEM_ID, VALUE, TS

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def _create_producer(bootstrap_servers: str) -> KafkaProducer:
    """Create a KafkaProducer instance."""
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: v,  # Already bytes from InteractionEvent.to_json()
        acks="all",
        retries=3,
    )


def replay_interactions(
    interactions: "DataFrame",
    bootstrap_servers: str = "localhost:9092",
    batch_size: int = 1000,
) -> int:
    """
    Replay interactions from a DataFrame to Kafka.

    Publishes each interaction as a JSON event to the interactions topic.
    Collects data in batches for efficiency.

    Args:
        interactions: DataFrame with canonical interaction columns.
        bootstrap_servers: Kafka bootstrap servers address.
        batch_size: Number of rows to collect per batch.

    Returns:
        Number of events published.
    """
    producer = _create_producer(bootstrap_servers)
    count = 0

    try:
        # Process in batches to avoid memory issues with large datasets
        # Collect all rows (for small datasets) or use toLocalIterator for large ones
        rows = interactions.select(USER_ID, ITEM_ID, VALUE, TS).collect()

        for row in rows:
            event = InteractionEvent(
                user_id=row[USER_ID],
                item_id=row[ITEM_ID],
                value=row[VALUE],
                ts=row[TS],
            )
            producer.send(INTERACTIONS_TOPIC, value=event.to_json())
            count += 1

            # Flush periodically to avoid memory buildup
            if count % batch_size == 0:
                producer.flush()

        # Final flush
        producer.flush()

    finally:
        producer.close()

    return count


def publish_event(
    event: InteractionEvent,
    bootstrap_servers: str = "localhost:9092",
) -> None:
    """
    Publish a single interaction event to Kafka.

    Args:
        event: Interaction event to publish.
        bootstrap_servers: Kafka bootstrap servers address.
    """
    producer = _create_producer(bootstrap_servers)
    try:
        producer.send(INTERACTIONS_TOPIC, value=event.to_json())
        producer.flush()
    finally:
        producer.close()
