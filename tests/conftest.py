"""
Pytest configuration and shared fixtures.

Provides session-scoped SparkSession and temporary feature store paths.
"""

import tempfile
from pathlib import Path

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """
    Session-scoped SparkSession for tests.

    Uses local mode with minimal resources for fast test execution.
    Kafka connector is disabled for unit tests.
    """
    session = (
        SparkSession.builder
        .appName("preference-engine-tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "localhost")
        .getOrCreate()
    )

    # Set log level to reduce noise
    session.sparkContext.setLogLevel("WARN")

    yield session

    session.stop()


@pytest.fixture
def temp_feature_store(tmp_path: Path) -> Path:
    """
    Temporary feature store path for tests.

    Cleaned up automatically after each test.
    """
    store_path = tmp_path / "feature_store"
    store_path.mkdir(parents=True, exist_ok=True)
    return store_path
