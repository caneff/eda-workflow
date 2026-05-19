import enum
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd


AnalysisFunction = Callable[..., dict[str, Any]]


class AnalysisStepName(enum.StrEnum):
    """Registered analysis step names."""

    PROFILE_DATASET = "profile_dataset"
    ANALYZE_MISSINGNESS = "analyze_missingness"
    COMPUTE_AGGREGATES = "compute_aggregates"
    ANALYZE_RELATIONSHIPS = "analyze_relationships"


@dataclass(frozen=True)
class AnalysisStep:
    """Named deterministic analysis step used to build the workflow graph."""

    name: AnalysisStepName
    analyze: AnalysisFunction


def _round_or_none(value: Any) -> float | None:
    """Round finite numeric values and normalize missing values to None."""
    if pd.isna(value):
        return None

    number = float(value)
    if not math.isfinite(number):
        return None

    return round(number, 2)


def profile_dataset(
    df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate dataset profile with basic statistics."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "str"],
    ).columns.tolist()

    return {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "numeric_summary": (
            df[numeric_cols].describe().to_dict() if numeric_cols else {}
        ),
        "categorical_summary": {
            col: df[col].value_counts().head(10).to_dict() for col in categorical_cols
        },
    }


def analyze_missingness(
    df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Analyze missing values in the dataset."""
    missing_count = df.isnull().sum().to_dict()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    high_missing = {col: pct for col, pct in missing_pct.items() if pct > 20}

    return {
        "total_rows": len(df),
        "missing_count": missing_count,
        "missing_percentage": missing_pct,
        "high_missing_columns": high_missing,
        "complete_rows": int(df.dropna().shape[0]),
        "complete_rows_pct": (
            round(df.dropna().shape[0] / len(df) * 100, 2) if len(df) > 0 else 0
        ),
    }


def _get_eligible_groups(
    df: pd.DataFrame,
    categorical_column: str,
    max_groups: int,
    min_percentage: float,
) -> tuple[dict[Any, dict[str, float | int | None]], list[Any]]:
    """Return sufficiently common category groups and their row coverage."""
    if df[categorical_column].nunique(dropna=True) <= 1:
        return {}, []

    group_counts = cast(
        pd.Series,
        df[categorical_column].value_counts(dropna=True),
    )
    group_summary = pd.DataFrame({
        "count": group_counts,
        "percentage": group_counts / len(df) * 100,
    })
    eligible_group_summary = cast(
        pd.DataFrame,
        group_summary.loc[group_summary["percentage"] >= min_percentage].nlargest(
            max_groups,
            "count",
        ),
    )
    if len(eligible_group_summary) <= 1:
        return {}, []

    eligible_groups = {}
    for group in eligible_group_summary.index:
        eligible_groups[group] = {
            "count": int(cast(Any, eligible_group_summary.at[group, "count"])),
            "percentage": _round_or_none(
                eligible_group_summary.at[group, "percentage"],
            ),
        }

    return eligible_groups, list(eligible_group_summary.index)


def _summarize_categorical_column(
    df: pd.DataFrame,
    categorical_column: str,
    numeric_columns: list[str],
    max_groups: int,
    min_percentage: float,
) -> dict[str, Any]:
    """Summarize numeric columns for one categorical column."""
    eligible_groups, eligible_group_names = _get_eligible_groups(
        df,
        categorical_column,
        max_groups,
        min_percentage,
    )
    if not eligible_groups:
        return {}

    numeric_summaries = {}
    eligible_df = df.loc[df[categorical_column].isin(eligible_group_names)]
    if not numeric_columns:
        return {}

    overall_values = df[numeric_columns]
    q1 = overall_values.quantile(0.25)
    q3 = overall_values.quantile(0.75)
    overall_stats = (
        pd.DataFrame({
            "median": overall_values.median(),
            "mean": overall_values.mean(),
            "std": overall_values.std(),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
        })
        .dropna(subset=["median"])
        .rename_axis("numeric_column")
        .reset_index()
    )
    if overall_stats.empty:
        return {}

    # Compute every numeric summary in one groupby call. Pandas returns a wide
    # table with a two-level column index: numeric column, then statistic.
    group_stats = cast(
        pd.DataFrame,
        eligible_df.groupby(categorical_column, observed=True)[numeric_columns].agg(
            ["median", "mean", "std"]
        ),
    )
    # Reshape to one row per category/numeric-column pair so it can be merged
    # with the overall numeric summary and filtered by the IQR rule below.
    group_stats_long = cast(
        pd.DataFrame,
        group_stats.stack(level=0, future_stack=True),
    )
    group_stats_long.index.names = [categorical_column, "numeric_column"]
    group_stats_long = cast(pd.DataFrame, group_stats_long.reset_index())
    group_stats_long = group_stats_long.dropna(subset=["median"])

    notable_stats = group_stats_long.merge(
        overall_stats,
        on="numeric_column",
        suffixes=("_group", "_overall"),
    )
    notable_stats = notable_stats.loc[
        (notable_stats["median_group"] < notable_stats["q1"])
        | (notable_stats["median_group"] > notable_stats["q3"])
    ]

    for numeric_column, numeric_stats in notable_stats.groupby(
        "numeric_column",
        sort=False,
    ):
        overall_row = numeric_stats.iloc[0]
        notable_groups = {}

        for _, stats in numeric_stats.iterrows():
            group = stats[categorical_column]
            notable_groups[group] = {
                "count": eligible_groups[group]["count"],
                "percentage": eligible_groups[group]["percentage"],
                "median": _round_or_none(stats["median_group"]),
                "mean": _round_or_none(stats["mean_group"]),
                "std": _round_or_none(stats["std_group"]),
            }

        numeric_summaries[numeric_column] = {
            "overall": {
                "median": _round_or_none(overall_row["median_overall"]),
                "mean": _round_or_none(overall_row["mean_overall"]),
                "std": _round_or_none(overall_row["std_overall"]),
                "q1": _round_or_none(overall_row["q1"]),
                "q3": _round_or_none(overall_row["q3"]),
                "iqr": _round_or_none(overall_row["iqr"]),
            },
            "groups": notable_groups,
        }

    if not numeric_summaries:
        return {}

    return {
        "eligible_groups": eligible_groups,
        "numeric_summaries": numeric_summaries,
    }


def compute_aggregates(
    df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Compute notable categorical group differences for numeric columns.

    This step depends on prior workflow results from ``profile_dataset`` and
    ``analyze_missingness``. The profile step identifies numeric and categorical
    columns, while missingness excludes columns with high missing values.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset being analyzed.
    **kwargs
        Workflow context. ``results`` should contain prior analysis results.
        Optional ``max_groups`` limits the number of category groups considered,
        and ``min_percentage`` is the minimum percentage of total rows a group
        must cover to be analyzed.

    Returns
    -------
    dict[str, Any]
        Nested aggregate summaries by categorical column. Each category includes
        eligible group coverage and numeric summaries for groups whose medians
        fall outside the overall interquartile range.
    """
    results = kwargs.get("results")
    if not isinstance(results, Mapping) or len(df) == 0:
        return {}

    profile = results.get(AnalysisStepName.PROFILE_DATASET.value)
    missingness = results.get(AnalysisStepName.ANALYZE_MISSINGNESS.value)
    if not isinstance(profile, Mapping) or not isinstance(missingness, Mapping):
        return {}

    max_groups = int(kwargs.get("max_groups", 10))
    min_percentage = float(kwargs.get("min_percentage", 5.0))
    if max_groups <= 0:
        return {}

    high_missing_columns = set(missingness.get("high_missing_columns", {}))

    # Read upstream profile context and exclude columns already flagged as sparse.
    numeric_columns = [
        col
        for col in profile.get("numeric_columns", [])
        if col in df.columns and col not in high_missing_columns
    ]
    categorical_columns = [
        col
        for col in profile.get("categorical_columns", [])
        if col in df.columns and col not in high_missing_columns
    ]

    aggregate_results: dict[str, Any] = {}

    for categorical_column in categorical_columns:
        categorical_summary = _summarize_categorical_column(
            df,
            categorical_column,
            numeric_columns,
            max_groups,
            min_percentage,
        )
        if categorical_summary:
            aggregate_results[categorical_column] = categorical_summary

    return aggregate_results


def analyze_relationships(
    df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Analyze relationships between variables."""
    return {}


# Register all above analysis steps in this list. extract_observations will be called after every step
# and the next step called in order.
ANALYSIS_STEPS = [
    AnalysisStep(AnalysisStepName.PROFILE_DATASET, profile_dataset),
    AnalysisStep(AnalysisStepName.ANALYZE_MISSINGNESS, analyze_missingness),
    AnalysisStep(AnalysisStepName.COMPUTE_AGGREGATES, compute_aggregates),
]
