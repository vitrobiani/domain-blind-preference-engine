import argparse
import sys
from pathlib import Path


DATA_ROOT = Path(".data")


def _dataset_paths(heuristics, args: argparse.Namespace) -> tuple[Path, Path]:
    """
    Compute the (feature_store, models) paths for the current dataset.

    Layout: `.data/<dataset_name>/{feature_store, models}` - one root per
    dataset so builds against different domains never clobber each other.
    `--feature-store` / `--models` / `--output` flags still override.
    """
    from preference_engine.adapter.registry import get_adapter

    adapter = get_adapter(heuristics.domain)
    root = DATA_ROOT / adapter.dataset_name

    fs_override = getattr(args, "feature_store", None) or getattr(args, "output", None)
    models_override = getattr(args, "models", None)

    fs = Path(fs_override) if fs_override else root / "feature_store"
    models = Path(models_override) if models_override else root / "models"
    return fs, models


def cmd_ingest(args: argparse.Namespace) -> int:
    """Run the Kafka producer to ingest data."""
    from preference_engine.combiner.heuristics import load_heuristics
    from preference_engine.adapter.registry import get_adapter
    from preference_engine.spark_session import get_spark_session
    from preference_engine.ingest.producer import replay_interactions

    print(f"Loading heuristics from {args.heuristics}...")
    heuristics = load_heuristics(args.heuristics)

    print(f"Loading adapter '{heuristics.domain}'...")
    adapter = get_adapter(heuristics.domain)

    print("Starting SparkSession...")
    spark = get_spark_session(enable_kafka=False)
    spark.sparkContext.setLogLevel("WARN")

    try:
        print("Loading interactions from adapter...")
        interactions = adapter.interactions(spark)

        print(f"Publishing to Kafka at {args.bootstrap_servers}...")
        count = replay_interactions(interactions, args.bootstrap_servers)

        print(f"Successfully published {count} interaction events.")
        return 0

    finally:
        spark.stop()


def cmd_build_features(args: argparse.Namespace) -> int:
    """Build the feature store."""
    from preference_engine.combiner.heuristics import load_heuristics
    from preference_engine.adapter.registry import get_adapter
    from preference_engine.spark_session import get_spark_session
    from preference_engine.streaming.feature_store import (
        build_features_batch,
        build_features_streaming,
    )

    print(f"Loading heuristics from {args.heuristics}...")
    heuristics = load_heuristics(args.heuristics)

    output_path, _ = _dataset_paths(heuristics, args)

    if args.batch:
        print(f"Loading adapter '{heuristics.domain}'...")
        adapter = get_adapter(heuristics.domain)

        print("Starting SparkSession...")
        spark = get_spark_session(enable_kafka=False)
        spark.sparkContext.setLogLevel("WARN")

        try:
            print(f"Building features (batch mode) to {output_path}...")
            build_features_batch(adapter, spark, output_path)
            print("Feature store built successfully.")
            return 0
        finally:
            spark.stop()
    else:
        print("Starting SparkSession with Kafka support...")
        spark = get_spark_session(enable_kafka=True)
        spark.sparkContext.setLogLevel("WARN")

        try:
            print(f"Building features (streaming mode) to {output_path}...")
            build_features_streaming(spark, args.bootstrap_servers, output_path)
            print("Streaming job started.")
            return 0
        finally:
            # Note: streaming job runs indefinitely, so this won't be reached
            # unless interrupted
            spark.stop()


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """Bootstrap an empty feature store from the adapter catalog only."""
    from preference_engine.combiner.heuristics import load_heuristics
    from preference_engine.adapter.registry import get_adapter
    from preference_engine.spark_session import get_spark_session
    from preference_engine.streaming.feature_store import bootstrap_features

    print(f"Loading heuristics from {args.heuristics}...")
    heuristics = load_heuristics(args.heuristics)

    print(f"Loading adapter '{heuristics.domain}'...")
    adapter = get_adapter(heuristics.domain)

    output_path, _ = _dataset_paths(heuristics, args)

    print("Starting SparkSession...")
    spark = get_spark_session(enable_kafka=False)
    spark.sparkContext.setLogLevel("WARN")

    try:
        print(f"Bootstrapping empty feature store at {output_path}...")
        bootstrap_features(adapter, spark, output_path)
        print("Bootstrap complete. Start `serve` and POST /interactions to feed data.")
        return 0
    finally:
        spark.stop()


def cmd_fit(args: argparse.Namespace) -> int:
    """Fit all active signals."""
    from preference_engine.combiner.heuristics import load_heuristics
    from preference_engine.spark_session import get_spark_session
    from preference_engine.streaming.feature_store import FeatureStore
    from preference_engine.serving.query import RecommendationEngine

    print(f"Loading heuristics from {args.heuristics}...")
    heuristics = load_heuristics(args.heuristics)

    feature_store_path, models_path = _dataset_paths(heuristics, args)

    print("Starting SparkSession...")
    spark = get_spark_session(enable_kafka=False)
    spark.sparkContext.setLogLevel("WARN")

    try:
        print(f"Loading feature store from {feature_store_path}...")
        feature_store = FeatureStore(path=feature_store_path, spark=spark)

        print(f"Fitting signals (models_path={models_path}, refit={args.refit})...")
        engine = RecommendationEngine(heuristics, feature_store, models_path=models_path)
        engine.fit(force=args.refit)

        print(f"Ready. {len(engine._scorers)} signals: {sorted(engine._scorers.keys())}")
        return 0

    finally:
        spark.stop()


def cmd_recommend(args: argparse.Namespace) -> int:
    """Generate recommendations for a user."""
    from preference_engine.combiner.heuristics import load_heuristics
    from preference_engine.spark_session import get_spark_session
    from preference_engine.streaming.feature_store import FeatureStore
    from preference_engine.serving.query import RecommendationEngine
    from preference_engine.combiner.combiner import FINAL_SCORE, BREAKDOWN

    print(f"Loading heuristics from {args.heuristics}...")
    heuristics = load_heuristics(args.heuristics)

    feature_store_path, models_path = _dataset_paths(heuristics, args)

    print("Starting SparkSession...")
    spark = get_spark_session(enable_kafka=False)
    spark.sparkContext.setLogLevel("WARN")

    try:
        print(f"Loading feature store from {feature_store_path}...")
        feature_store = FeatureStore(path=feature_store_path, spark=spark)

        print(f"Fitting signals (models_path={models_path}, refit={args.refit})...")
        engine = RecommendationEngine(heuristics, feature_store, models_path=models_path)
        engine.fit(force=args.refit)

        print(f"\nGenerating recommendations for user '{args.user}'...")
        recommendations = engine.recommend(args.user)

        # Display results
        print(f"\nTop {heuristics.top_k} recommendations:")
        print("-" * 70)

        results = recommendations.collect()
        if not results:
            print("No recommendations found.")
            return 0

        for i, row in enumerate(results, 1):
            item_id = row["item_id"]
            score = row[FINAL_SCORE]
            breakdown = row[BREAKDOWN]

            # Format breakdown
            breakdown_str = ", ".join(
                f"{k}: {v:.3f}" for k, v in sorted(breakdown.items())
            )

            print(f"{i:2d}. {item_id:8s}  score={score:.4f}  [{breakdown_str}]")

        print("-" * 70)
        return 0

    finally:
        spark.stop()


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the FastAPI serving process."""
    import uvicorn
    from preference_engine.combiner.heuristics import load_heuristics
    from preference_engine.serving.api import create_app

    heuristics = load_heuristics(args.heuristics)
    feature_store_path, models_path = _dataset_paths(heuristics, args)

    app = create_app(
        heuristics_path=args.heuristics,
        feature_store_path=feature_store_path,
        models_path=models_path,
        bootstrap_servers=args.bootstrap_servers,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="preference-engine",
        description="Domain-blind preference engine for ranked recommendations",
    )
    parser.add_argument(
        "--heuristics",
        "-c",
        type=Path,
        default=Path("config/heuristics.example.yaml"),
        help="Path to heuristics YAML configuration",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Replay dataset interactions to Kafka",
    )
    ingest_parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)",
    )
    ingest_parser.set_defaults(func=cmd_ingest)

    # build-features command
    build_parser = subparsers.add_parser(
        "build-features",
        help="Build the feature store from data",
    )
    build_parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch mode instead of streaming (no Kafka required)",
    )
    build_parser.add_argument(
        "--output",
        "-o",
        help="Output path for feature store (default: .data/<dataset>/feature_store)",
    )
    build_parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers for streaming mode",
    )
    build_parser.set_defaults(func=cmd_build_features)

    # bootstrap command
    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="Create an empty feature store (catalog only, no interactions)",
    )
    bootstrap_parser.add_argument(
        "--output",
        "-o",
        help="Output path for feature store (default: .data/<dataset>/feature_store)",
    )
    bootstrap_parser.set_defaults(func=cmd_bootstrap)

    # fit command
    fit_parser = subparsers.add_parser(
        "fit",
        help="Fit all active signals against the feature store",
    )
    fit_parser.add_argument(
        "--feature-store",
        "-f",
        help="Path to feature store (default: .data/<dataset>/feature_store)",
    )
    fit_parser.add_argument(
        "--models",
        "-m",
        help="Path to models directory (default: .data/<dataset>/models)",
    )
    fit_parser.add_argument(
        "--refit",
        action="store_true",
        help="Retrain every active signal even if a saved model exists",
    )
    fit_parser.set_defaults(func=cmd_fit)

    # recommend command
    rec_parser = subparsers.add_parser(
        "recommend",
        help="Generate recommendations for a user",
    )
    rec_parser.add_argument(
        "--user",
        "-u",
        required=True,
        help="User ID to generate recommendations for",
    )
    rec_parser.add_argument(
        "--feature-store",
        "-f",
        help="Path to feature store (default: .data/<dataset>/feature_store)",
    )
    rec_parser.add_argument(
        "--models",
        "-m",
        help="Path to models directory (default: .data/<dataset>/models)",
    )
    rec_parser.add_argument(
        "--refit",
        action="store_true",
        help="Retrain every active signal even if a saved model exists",
    )
    rec_parser.set_defaults(func=cmd_recommend)

    # serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Run the HTTP serving process (FastAPI + Kafka producer)",
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port")
    serve_parser.add_argument(
        "--feature-store",
        "-f",
        help="Path to feature store (default: .data/<dataset>/feature_store)",
    )
    serve_parser.add_argument(
        "--models",
        "-m",
        help="Path to models directory (default: .data/<dataset>/models)",
    )
    serve_parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)",
    )
    serve_parser.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
