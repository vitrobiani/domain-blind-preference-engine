"""
MovieLens adapter (Phase 6).

Loads MovieLens 100K dataset and transforms to canonical schema.
Validates that the preference engine is truly domain-blind.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    DoubleType,
    TimestampType,
)

from preference_engine.adapter.base import DomainAdapter
from preference_engine.schema import (
    InteractionType,
    FeatureSpec,
    FeatureKind,
    Applies,
    USER_ID,
    ITEM_ID,
    VALUE,
    TS,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


# Path to the MovieLens 100K dataset
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "ml-100k"

# Genre names in order (matching the binary flags in u.item)
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Childrens",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "FilmNoir", "Horror", "Musical", "Mystery", "Romance",
    "SciFi", "Thriller", "War", "Western"
]


class MoviesAdapter(DomainAdapter):
    """
    Adapter for MovieLens 100K dataset.

    Transforms MovieLens ratings/movies/users to canonical schema.
    Phase 6 implementation - validates the domain-blind core.

    Dataset structure:
    - u.data: ratings (user_id, item_id, rating, timestamp)
    - u.user: user demographics (user_id, age, gender, occupation, zip)
    - u.item: movie metadata (movie_id, title, release_date, ..., genre_flags)
    """

    name = "movies"
    dataset_name = "ml-100k"
    interaction_type = InteractionType.EXPLICIT

    def interactions(self, spark: "SparkSession") -> "DataFrame":
        """
        Load MovieLens ratings as interactions.

        u.data format: user_id \t item_id \t rating \t timestamp
        Rating is 1-5 stars (explicit feedback).
        """
        ratings_path = str(DATA_DIR / "u.data")

        # Define schema for ratings
        schema = StructType([
            StructField("_user_id", IntegerType(), False),
            StructField("_item_id", IntegerType(), False),
            StructField("_rating", IntegerType(), False),
            StructField("_timestamp", IntegerType(), False),
        ])

        # Read tab-separated file
        df = spark.read.csv(
            ratings_path,
            schema=schema,
            sep="\t",
            header=False,
        )

        # Transform to canonical schema
        return df.select(
            F.col("_user_id").cast(StringType()).alias(USER_ID),
            F.col("_item_id").cast(StringType()).alias(ITEM_ID),
            F.col("_rating").cast(DoubleType()).alias(VALUE),
            F.from_unixtime(F.col("_timestamp")).cast(TimestampType()).alias(TS),
        )

    def user_features(self, spark: "SparkSession") -> "DataFrame":
        """
        Load user demographics as features.

        u.user format: user_id | age | gender | occupation | zip_code
        """
        users_path = str(DATA_DIR / "u.user")

        # Define schema for users
        schema = StructType([
            StructField("_user_id", IntegerType(), False),
            StructField("age", IntegerType(), True),
            StructField("gender", StringType(), True),
            StructField("occupation", StringType(), True),
            StructField("zip_code", StringType(), True),
        ])

        # Read pipe-separated file
        df = spark.read.csv(
            users_path,
            schema=schema,
            sep="|",
            header=False,
        )

        # Transform to canonical schema
        # Convert age to age_group for better segmentation
        return df.select(
            F.col("_user_id").cast(StringType()).alias(USER_ID),
            F.col("age").cast(DoubleType()).alias("age"),
            F.col("gender"),
            F.col("occupation"),
        )

    def item_features(self, spark: "SparkSession") -> "DataFrame":
        """
        Load movie metadata as features.

        u.item format: movie_id | title | release_date | video_release | imdb_url | genre_flags...
        Genre flags are 19 binary columns (one per genre).
        """
        items_path = str(DATA_DIR / "u.item")

        # Build schema: movie_id, title, release_date, video_release, imdb_url, then 19 genre flags
        fields = [
            StructField("_item_id", IntegerType(), False),
            StructField("title", StringType(), True),
            StructField("release_date", StringType(), True),
            StructField("video_release", StringType(), True),
            StructField("imdb_url", StringType(), True),
        ]
        for genre in GENRES:
            fields.append(StructField(f"genre_{genre}", IntegerType(), True))

        schema = StructType(fields)

        # Read pipe-separated file with latin-1 encoding (some titles have special chars)
        df = spark.read.csv(
            items_path,
            schema=schema,
            sep="|",
            header=False,
            encoding="ISO-8859-1",
        )

        # Extract year from title (format: "Title (YYYY)")
        # Also create a primary genre column (first matching genre)
        genre_cols = [f"genre_{g}" for g in GENRES]

        # Build CASE expression to find primary genre
        primary_genre_expr = F.when(F.col(genre_cols[0]) == 1, F.lit(GENRES[0]))
        for i, genre in enumerate(GENRES[1:], 1):
            primary_genre_expr = primary_genre_expr.when(
                F.col(genre_cols[i]) == 1, F.lit(genre)
            )
        primary_genre_expr = primary_genre_expr.otherwise(F.lit("unknown"))

        # Extract year from title using regex
        year_pattern = r"\((\d{4})\)"

        result = df.select(
            F.col("_item_id").cast(StringType()).alias(ITEM_ID),
            F.col("title"),
            F.regexp_extract(F.col("title"), year_pattern, 1).alias("year_str"),
            primary_genre_expr.alias("primary_genre"),
            # Also keep all genre flags for content-based filtering
            *[F.col(c).cast(DoubleType()).alias(c) for c in genre_cols]
        )

        # Convert year to numeric (null if not found)
        result = result.withColumn(
            "year",
            F.when(F.col("year_str") != "", F.col("year_str").cast(DoubleType()))
            .otherwise(F.lit(None))
        ).drop("year_str")

        return result

    def feature_specs(self) -> list[FeatureSpec]:
        """
        Describe MovieLens features for vectorization.

        User features: age (numeric), gender (categorical), occupation (categorical)
        Item features: year (numeric), primary_genre (categorical), genre_* (numeric binary flags)
        """
        specs = [
            # User features
            FeatureSpec(name="age", kind=FeatureKind.NUMERIC, applies=Applies.USER),
            FeatureSpec(name="gender", kind=FeatureKind.CATEGORICAL, applies=Applies.USER),
            FeatureSpec(name="occupation", kind=FeatureKind.CATEGORICAL, applies=Applies.USER),
            # Item features
            FeatureSpec(name="year", kind=FeatureKind.NUMERIC, applies=Applies.ITEM),
            FeatureSpec(name="primary_genre", kind=FeatureKind.CATEGORICAL, applies=Applies.ITEM),
        ]

        # Add binary genre flags as numeric features
        for genre in GENRES:
            specs.append(
                FeatureSpec(
                    name=f"genre_{genre}",
                    kind=FeatureKind.NUMERIC,
                    applies=Applies.ITEM,
                )
            )

        return specs
