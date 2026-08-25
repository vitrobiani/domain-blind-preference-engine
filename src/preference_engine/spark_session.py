"""
SparkSession factory for the preference engine.

Provides a configured SparkSession with Kafka connector and appropriate
settings for local development.
"""

from pyspark.sql import SparkSession


def get_spark_session(
    app_name: str = "preference-engine",
    enable_kafka: bool = True,
) -> SparkSession:
    """
    Create or get a SparkSession configured for the preference engine.

    Args:
        app_name: Name for the Spark application.
        enable_kafka: Whether to include the Kafka SQL connector package.

    Returns:
        Configured SparkSession instance.
    """
    builder = SparkSession.builder.appName(app_name)

    if enable_kafka:
        # Add Kafka connector for Structured Streaming
        # Match Scala version (2.12) to PySpark's bundled Scala
        builder = builder.config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1",
        )

    # Local mode settings for development
    builder = builder.config("spark.sql.shuffle.partitions", "4")

    return builder.getOrCreate()
