"""
Kafka topic definitions and (de)serialization utilities.

Defines topic names and JSON encoding/decoding for events.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Topic names
INTERACTIONS_TOPIC = "preference-engine.interactions"


@dataclass
class InteractionEvent:
    """
    An interaction event to be published to Kafka.

    Attributes:
        user_id: User identifier.
        item_id: Item identifier.
        value: Interaction value (rating, count, etc.).
        ts: Timestamp of the interaction.
    """

    user_id: str
    item_id: str
    value: float | None
    ts: datetime

    def to_json(self) -> bytes:
        """Serialize to JSON bytes for Kafka."""
        return json.dumps(
            {
                "user_id": self.user_id,
                "item_id": self.item_id,
                "value": self.value,
                "ts": self.ts.isoformat(),
            }
        ).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> "InteractionEvent":
        """Deserialize from JSON bytes."""
        obj = json.loads(data.decode("utf-8"))
        return cls(
            user_id=obj["user_id"],
            item_id=obj["item_id"],
            value=obj.get("value"),
            ts=datetime.fromisoformat(obj["ts"]),
        )


def parse_kafka_value(value: Any) -> dict[str, Any]:
    """
    Parse a Kafka message value to a dictionary.

    Args:
        value: Raw Kafka value (bytes or string).

    Returns:
        Parsed dictionary.
    """
    if isinstance(value, bytes):
        return json.loads(value.decode("utf-8"))
    elif isinstance(value, str):
        return json.loads(value)
    else:
        raise ValueError(f"Unexpected value type: {type(value)}")
