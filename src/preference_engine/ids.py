"""
ID mapping utilities for string-to-integer conversion.

Spark's ALS requires integer user/item IDs, but canonical IDs are strings.
This module provides bidirectional mapping built once and reused across signals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.ml.feature import StringIndexer, StringIndexerModel, IndexToString
from pyspark.sql import DataFrame

from preference_engine.schema import USER_ID, ITEM_ID, USER_ID_INT, ITEM_ID_INT

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


class IDMapper:
    """
    Bidirectional string-to-integer ID mapper for users and items.

    Built once from an interactions DataFrame and reused by all signals
    that require integer IDs (e.g., ALS).

    Attributes:
        user_indexer: Fitted StringIndexer for user IDs.
        item_indexer: Fitted StringIndexer for item IDs.
        user_converter: IndexToString for reverse user mapping.
        item_converter: IndexToString for reverse item mapping.
    """

    def __init__(self) -> None:
        self._user_indexer: StringIndexer | None = None
        self._item_indexer: StringIndexer | None = None
        self._user_model: object | None = None
        self._item_model: object | None = None

    def fit(self, interactions: DataFrame) -> IDMapper:
        """
        Fit the ID mappers on an interactions DataFrame.

        Args:
            interactions: DataFrame with USER_ID and ITEM_ID string columns.

        Returns:
            Self for method chaining.
        """
        self._user_indexer = StringIndexer(
            inputCol=USER_ID,
            outputCol=USER_ID_INT,
            handleInvalid="keep",
        )
        self._item_indexer = StringIndexer(
            inputCol=ITEM_ID,
            outputCol=ITEM_ID_INT,
            handleInvalid="keep",
        )

        self._user_model = self._user_indexer.fit(interactions)
        self._item_model = self._item_indexer.fit(interactions)

        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Add integer ID columns to a DataFrame.

        Args:
            df: DataFrame with USER_ID and/or ITEM_ID columns.

        Returns:
            DataFrame with added USER_ID_INT and/or ITEM_ID_INT columns.
        """
        if self._user_model is None or self._item_model is None:
            raise RuntimeError("IDMapper must be fit before transform")

        columns = df.columns
        result = df

        if USER_ID in columns:
            result = self._user_model.transform(result)
        if ITEM_ID in columns:
            result = self._item_model.transform(result)

        return result

    def user_id_to_int(self, user_id: str) -> int | None:
        """
        Convert a single user ID string to integer.

        Args:
            user_id: String user ID.

        Returns:
            Integer ID, or None if not found.
        """
        if self._user_model is None:
            raise RuntimeError("IDMapper must be fit before use")

        labels = self._user_model.labels
        try:
            return labels.index(user_id)
        except ValueError:
            return None

    def item_id_to_int(self, item_id: str) -> int | None:
        """
        Convert a single item ID string to integer.

        Args:
            item_id: String item ID.

        Returns:
            Integer ID, or None if not found.
        """
        if self._item_model is None:
            raise RuntimeError("IDMapper must be fit before use")

        labels = self._item_model.labels
        try:
            return labels.index(item_id)
        except ValueError:
            return None

    def int_to_user_id(self, idx: int) -> str | None:
        """
        Convert an integer index back to user ID string.

        Args:
            idx: Integer index.

        Returns:
            String user ID, or None if out of range.
        """
        if self._user_model is None:
            raise RuntimeError("IDMapper must be fit before use")

        labels = self._user_model.labels
        if 0 <= idx < len(labels):
            return labels[idx]
        return None

    def int_to_item_id(self, idx: int) -> str | None:
        """
        Convert an integer index back to item ID string.

        Args:
            idx: Integer index.

        Returns:
            String item ID, or None if out of range.
        """
        if self._item_model is None:
            raise RuntimeError("IDMapper must be fit before use")

        labels = self._item_model.labels
        if 0 <= idx < len(labels):
            return labels[idx]
        return None

    def inverse_transform_users(self, df: DataFrame) -> DataFrame:
        """
        Convert USER_ID_INT back to USER_ID strings.

        Args:
            df: DataFrame with USER_ID_INT column.

        Returns:
            DataFrame with USER_ID column added.
        """
        if self._user_model is None:
            raise RuntimeError("IDMapper must be fit before use")

        converter = IndexToString(
            inputCol=USER_ID_INT,
            outputCol=USER_ID,
            labels=self._user_model.labels,
        )
        return converter.transform(df)

    HANDLE_INVALID = "keep"

    def save(self, path: Path) -> None:
        """
        Persist the fitted mapper to `path` (a directory).

        Writes user and item labels as parquet, plus a small manifest.
        Requires an active SparkSession.
        """
        if self._user_model is None or self._item_model is None:
            raise RuntimeError("IDMapper must be fit before save")

        from pyspark.sql import SparkSession

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("No active SparkSession - cannot save IDMapper")

        self._write_labels(spark, list(self._user_model.labels), path / "user_labels")
        self._write_labels(spark, list(self._item_model.labels), path / "item_labels")

        manifest = {"handle_invalid": self.HANDLE_INVALID}
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @staticmethod
    def _write_labels(spark: "SparkSession", labels: list[str], target: Path) -> None:
        rows = [(i, label) for i, label in enumerate(labels)]
        df = spark.createDataFrame(rows, "idx int, label string")
        df.write.mode("overwrite").parquet(str(target))

    @classmethod
    def load(cls, path: Path, spark: "SparkSession") -> "IDMapper":
        """Rebuild a fitted mapper from disk."""
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        handle_invalid = manifest.get("handle_invalid", cls.HANDLE_INVALID)

        user_labels = cls._read_labels(spark, path / "user_labels")
        item_labels = cls._read_labels(spark, path / "item_labels")

        instance = cls()
        instance._user_model = StringIndexerModel.from_labels(
            user_labels,
            inputCol=USER_ID,
            outputCol=USER_ID_INT,
            handleInvalid=handle_invalid,
        )
        instance._item_model = StringIndexerModel.from_labels(
            item_labels,
            inputCol=ITEM_ID,
            outputCol=ITEM_ID_INT,
            handleInvalid=handle_invalid,
        )
        return instance

    @staticmethod
    def _read_labels(spark: "SparkSession", source: Path) -> list[str]:
        rows = (
            spark.read.parquet(str(source))
            .orderBy("idx")
            .select("label")
            .collect()
        )
        return [r["label"] for r in rows]

    def inverse_transform_items(self, df: DataFrame) -> DataFrame:
        """
        Convert ITEM_ID_INT back to ITEM_ID strings.

        Args:
            df: DataFrame with ITEM_ID_INT column.

        Returns:
            DataFrame with ITEM_ID column added.
        """
        if self._item_model is None:
            raise RuntimeError("IDMapper must be fit before use")

        converter = IndexToString(
            inputCol=ITEM_ID_INT,
            outputCol=ITEM_ID,
            labels=self._item_model.labels,
        )
        return converter.transform(df)
