"""
Israeli National Elections Study 2025 adapter.

Loads the STATA CSV export of the 2025 INES off-cycle survey and models
each respondent's answer to v104 ("if elections were held today, which
party would you vote for?") as a single interaction. All other columns
worth using are stable demographics that flow into user_features so the
segments signal can cluster voters by profile.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import functions as F
from pyspark.sql.types import StringType

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


DATA_DIR = Path(__file__).parent.parent.parent / "data" / "Elections"
CSV_PATH = DATA_DIR / "2025_STATA.csv"


# v104 party codes → readable slugs (from the March 2025 questionnaire).
# Codes 30/94/96/97/98/99 are non-party sentinels (other/won't-vote/blank/
# undecided/don't-know/refuse) those are excluded upstream in
# interactions() so they never enter the recommendation universe.
PARTY_NAMES: dict[int, str] = {
    1: "Likud",
    2: "YeshAtid",
    3: "NationalUnity",
    4: "ReligiousZionism",
    5: "Shas",
    6: "UnitedTorahJudaism",
    7: "YisraelBeiteinu",
    8: "HaDemocratim",
    9: "OtzmaYehudit",
    10: "NationalRight",
    11: "HadashTaal",
    12: "Raam",
    13: "Balad",
}


class ElectionsAdapter(DomainAdapter):
    """
    Adapter for the INES 2025 off-cycle survey.

    Every respondent has exactly one interaction (their v104 answer), so
    signals that need multi-item baskets — als_cf, content, rules — cannot
    learn anything. Popularity works trivially; the segments signal, driven
    by demographics, carries the real predictive load here.
    """

    name = "elections"
    dataset_name = "Elections"
    interaction_type = InteractionType.EXPLICIT

    def _raw(self, spark: "SparkSession") -> "DataFrame":
        # ~120 columns; inferSchema saves us from listing them all.
        return spark.read.csv(
            str(CSV_PATH),
            header=True,
            inferSchema=True,
            nullValue="",
        )

    def interactions(self, spark: "SparkSession") -> "DataFrame":
        """
        One interaction per respondent: (resp_id, party_from_v104).

        Only real party codes (1-13, per the March 2025 questionnaire) are
        kept. Sentinels 30/94/96/97/98/99 (other/won't-vote/blank/undecided/
        don't-know/refuse) are dropped — they aren't parties and letting
        them through means the recommender surfaces "undecided" as its top
        pick, which is useless.
        """
        df = self._raw(spark)
        party_map = F.create_map(
            *[x for code, name in PARTY_NAMES.items()
              for x in (F.lit(code), F.lit(name))]
        )
        return (
            df
            .filter(F.col("v104").isin(list(PARTY_NAMES.keys())))
            .select(
                F.col("resp_id").cast(StringType()).alias(USER_ID),
                party_map[F.col("v104")].alias(ITEM_ID),
                F.lit(1.0).alias(VALUE),
                F.to_timestamp(F.col("date"), "dd/MM/yyyy").alias(TS),
            )
        )

    def user_features(self, spark: "SparkSession") -> "DataFrame":
        """
        Demographics for segment clustering.

        Uses `agegroup` (0% null) instead of raw `age` (~59% null) because
        segments' VectorAssembler uses handleInvalid="skip", which would
        silently drop the majority of respondents from clustering.

        Coded columns are cast to string so segments' StringIndexer treats
        them as categorical — we don't know the code orderings without the
        Appendix, so treating them as ordinal/numeric would be wrong.
        """
        df = self._raw(spark)
        return df.select(
            F.col("resp_id").cast(StringType()).alias(USER_ID),
            F.col("agegroup").cast(StringType()).alias("agegroup"),
            F.col("gender").cast(StringType()).alias("gender"),
            F.col("educ").cast(StringType()).alias("educ"),
            F.col("religiosity").cast(StringType()).alias("religiosity"),
            F.col("sector").cast(StringType()).alias("sector"),
            F.col("District_CBS").cast(StringType()).alias("district"),
        )

    def item_features(self, spark: "SparkSession") -> "DataFrame":
        """
        Parties have no metadata in this dataset — return the fixed party
        catalog (from PARTY_NAMES) so downstream joins have every party
        available even if it drew zero votes.
        """
        return spark.createDataFrame(
            [(name,) for name in PARTY_NAMES.values()],
            schema=f"{ITEM_ID} STRING",
        )

    def feature_specs(self) -> list[FeatureSpec]:
        return [
            FeatureSpec(name="agegroup", kind=FeatureKind.CATEGORICAL, applies=Applies.USER),
            FeatureSpec(name="gender", kind=FeatureKind.CATEGORICAL, applies=Applies.USER),
            FeatureSpec(name="educ", kind=FeatureKind.CATEGORICAL, applies=Applies.USER),
            FeatureSpec(name="religiosity", kind=FeatureKind.CATEGORICAL, applies=Applies.USER),
            FeatureSpec(name="sector", kind=FeatureKind.CATEGORICAL, applies=Applies.USER),
            FeatureSpec(name="district", kind=FeatureKind.CATEGORICAL, applies=Applies.USER),
        ]
