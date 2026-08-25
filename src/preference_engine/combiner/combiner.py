"""
Score combiner for weighted signal aggregation.

Takes normalized per-signal scores and combines them into a final ranking
with per-signal breakdown.
"""

from typing import TYPE_CHECKING
from functools import reduce

from pyspark.sql import functions as F

from preference_engine.schema import ITEM_ID
from preference_engine.combiner.heuristics import FiltersConfig

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


# Column names for combined results
FINAL_SCORE = "final_score"
BREAKDOWN = "breakdown"


def combine(
    scores: dict[str, "DataFrame"],
    weights: dict[str, float],
    filters: FiltersConfig,
    top_k: int,
    seen_items: "DataFrame | None" = None,
) -> "DataFrame":
    """
    Combine per-signal scores into a final ranking.

    Computes: score(item) = sum_k(w_k * signal_k(item))
    where missing signals contribute 0.

    Args:
        scores: Mapping from signal name to DataFrame[item_id, score].
        weights: Signal weights from heuristics (0 = signal skipped).
        filters: Filter configuration for exclusions.
        top_k: Number of items to return.
        seen_items: Optional DataFrame of seen items for filtering.

    Returns:
        DataFrame[item_id, final_score, breakdown] where breakdown is a
        map of signal_name -> contribution for each item, sorted by
        final_score descending, limited to top_k.
    """
    if not scores:
        raise ValueError("No scores provided to combiner")

    # Get the spark session from one of the DataFrames
    first_df = next(iter(scores.values()))
    spark = first_df.sparkSession

    # Start with the first signal's items as the base
    signal_names = list(scores.keys())

    # Rename score columns to signal-specific names and compute weighted contributions
    renamed_dfs = []
    for name, df in scores.items():
        weight = weights.get(name, 0.0)
        if weight > 0 and not df.isEmpty():
            # Rename score column and add weighted contribution
            renamed = (
                df
                .withColumnRenamed("score", f"score_{name}")
                .withColumn(f"contrib_{name}", F.col(f"score_{name}") * F.lit(weight))
            )
            renamed_dfs.append(renamed)

    if not renamed_dfs:
        # No active signals - return empty DataFrame
        schema = f"{ITEM_ID} string, {FINAL_SCORE} double, {BREAKDOWN} map<string, double>"
        return spark.createDataFrame([], schema)

    # Join all signal scores on item_id
    combined = renamed_dfs[0]
    for df in renamed_dfs[1:]:
        combined = combined.join(df, on=ITEM_ID, how="outer")

    # Fill nulls with 0 for missing scores
    for name in signal_names:
        if weights.get(name, 0.0) > 0:
            combined = combined.fillna({f"score_{name}": 0.0, f"contrib_{name}": 0.0})

    # Compute final score as sum of contributions
    contrib_cols = [f"contrib_{name}" for name in signal_names if weights.get(name, 0.0) > 0]
    combined = combined.withColumn(
        FINAL_SCORE,
        reduce(lambda a, b: a + b, [F.col(c) for c in contrib_cols])
    )

    # Build breakdown map
    breakdown_entries = []
    for name in signal_names:
        if weights.get(name, 0.0) > 0:
            breakdown_entries.append(F.lit(name))
            breakdown_entries.append(F.col(f"contrib_{name}"))

    if breakdown_entries:
        combined = combined.withColumn(
            BREAKDOWN,
            F.create_map(*breakdown_entries)
        )
    else:
        combined = combined.withColumn(BREAKDOWN, F.lit(None))

    # Apply filters
    if filters.exclude_seen and seen_items is not None:
        combined = apply_exclude_seen(combined, seen_items)

    if filters.exclude_items:
        combined = apply_exclude_items(combined, filters.exclude_items)

    # Select final columns, sort, and limit
    result = (
        combined
        .select(ITEM_ID, FINAL_SCORE, BREAKDOWN)
        .orderBy(F.col(FINAL_SCORE).desc())
        .limit(top_k)
    )

    return result


def apply_exclude_seen(df: "DataFrame", seen_items: "DataFrame") -> "DataFrame":
    """
    Exclude items the user has already seen.

    Args:
        df: DataFrame with items.
        seen_items: DataFrame with item_id column of seen items.

    Returns:
        Filtered DataFrame.
    """
    return df.join(
        seen_items.select(ITEM_ID),
        on=ITEM_ID,
        how="left_anti"
    )


def apply_exclude_items(df: "DataFrame", exclude_list: list[str]) -> "DataFrame":
    """
    Exclude specific items by ID.

    Args:
        df: DataFrame with items.
        exclude_list: List of item IDs to exclude.

    Returns:
        Filtered DataFrame.
    """
    if not exclude_list:
        return df
    return df.filter(~F.col(ITEM_ID).isin(exclude_list))


def apply_filters(
    df: "DataFrame",
    user_id: str,
    filters: FiltersConfig,
    seen_items: "DataFrame | None" = None,
) -> "DataFrame":
    """
    Apply exclusion filters to candidate items.

    Args:
        df: DataFrame with items to filter.
        user_id: Current user (for exclude_seen).
        filters: Filter configuration.
        seen_items: Optional DataFrame of items the user has seen.

    Returns:
        Filtered DataFrame.
    """
    result = df

    if filters.exclude_seen and seen_items is not None:
        result = apply_exclude_seen(result, seen_items)

    if filters.exclude_items:
        result = apply_exclude_items(result, filters.exclude_items)

    return result
