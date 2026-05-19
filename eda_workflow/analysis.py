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
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

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
    total_rows = len(df)

    for categorical_column in categorical_columns:
        if df[categorical_column].nunique(dropna=True) <= 1:
            continue

        # Choose sufficiently common category groups before comparing numerics.
        group_counts = cast(
            pd.Series,
            df[categorical_column].value_counts(dropna=True),
        )
        group_summary = pd.DataFrame({
            "count": group_counts,
            "percentage": group_counts / total_rows * 100,
        })
        eligible_group_summary = cast(
            pd.DataFrame,
            group_summary.loc[group_summary["percentage"] >= min_percentage].nlargest(
                max_groups,
                "count",
            ),
        )
        if len(eligible_group_summary) <= 1:
            continue

        eligible_group_names = list(eligible_group_summary.index)

        eligible_groups = {
            group: {
                "count": int(row["count"]),
                "percentage": _round_or_none(row["percentage"]),
            }
            for group, row in eligible_group_summary.iterrows()
        }

        numeric_summaries = {}
        eligible_df = df.loc[df[categorical_column].isin(eligible_group_names)]

        for numeric_column in numeric_columns:
            overall_values = df[numeric_column].dropna()
            if overall_values.empty:
                continue

            # Compute the overall reference band used for outlier group filtering.
            q1 = overall_values.quantile(0.25)
            q3 = overall_values.quantile(0.75)
            overall = {
                "median": _round_or_none(overall_values.median()),
                "mean": _round_or_none(overall_values.mean()),
                "std": _round_or_none(overall_values.std()),
                "q1": _round_or_none(q1),
                "q3": _round_or_none(q3),
                "iqr": _round_or_none(q3 - q1),
            }

            group_stats = (
                eligible_df.groupby(categorical_column, observed=True)[numeric_column]
                .agg(["median", "mean", "std"])
                .dropna(subset=["median"])
            )

            notable_groups = {}
            for group, stats in group_stats.iterrows():
                group_median = stats["median"]
                if q1 <= group_median <= q3:
                    continue

                # Keep only groups whose median sits outside the overall IQR.
                notable_groups[group] = {
                    "count": eligible_groups[group]["count"],
                    "percentage": eligible_groups[group]["percentage"],
                    "median": _round_or_none(group_median),
                    "mean": _round_or_none(stats["mean"]),
                    "std": _round_or_none(stats["std"]),
                }

            if notable_groups:
                numeric_summaries[numeric_column] = {
                    "overall": overall,
                    "groups": notable_groups,
                }

        if numeric_summaries:
            aggregate_results[categorical_column] = {
                "eligible_groups": eligible_groups,
                "numeric_summaries": numeric_summaries,
            }

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
