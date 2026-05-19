from collections.abc import Mapping
from typing import Any

import pandas

import eda_workflow.analysis as analysis


def assert_dict_contains(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    for key, expected_value in expected.items():
        assert key in actual
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            assert isinstance(actual_value, Mapping)
            assert_dict_contains(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def test_profile_dataset_summarizes_dataframe(sample_eda_dataframe: pandas.DataFrame):
    profile = analysis.profile_dataset(sample_eda_dataframe)

    assert_dict_contains(
        profile,
        {
            "shape": {"rows": 3, "columns": 3},
            "numeric_columns": ["sales", "units"],
            "categorical_columns": ["region"],
            "categorical_summary": {
                "region": {"North": 2, "South": 1},
            },
        },
    )


def test_analyze_missingness_summarizes_missing_values(
    sample_eda_dataframe: pandas.DataFrame,
):
    missingness = analysis.analyze_missingness(sample_eda_dataframe)

    assert_dict_contains(
        missingness,
        {
            "missing_count": {"sales": 1, "units": 1},
            "missing_percentage": {"sales": 33.33, "units": 33.33},
            "high_missing_columns": {"sales": 33.33, "units": 33.33},
            "complete_rows": 1,
        },
    )


def test_compute_aggregates_summarizes_notable_group_differences():
    dataframe = pandas.DataFrame({
        "segment": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D"],
        "thresholded_segment": [
            "large",
            "large",
            "large",
            "large",
            "large",
            "small",
            "small",
            "small",
            "small",
            "tiny",
        ],
        "single_group": ["all"] * 10,
        "sparse_category": [None, None, None, None, None, None, "x", "x", "y", "y"],
        "revenue": [10, 12, 14, 100, 110, 120, 50, 55, 60, 200],
        "unstable_score": [1, None, None, None, None, None, None, None, None, None],
    })
    results = {
        "profile_dataset": {
            "numeric_columns": ["revenue", "unstable_score"],
            "categorical_columns": [
                "segment",
                "thresholded_segment",
                "single_group",
                "sparse_category",
            ],
        },
        "analyze_missingness": {
            "high_missing_columns": {"unstable_score": 90.0, "sparse_category": 60.0},
        },
    }

    aggregates = analysis.compute_aggregates(
        dataframe,
        results=results,
    )

    assert aggregates["segment"] == {
        "eligible_groups": {
            "A": {"count": 3, "percentage": 30.0},
            "B": {"count": 3, "percentage": 30.0},
            "C": {"count": 3, "percentage": 30.0},
            "D": {"count": 1, "percentage": 10.0},
        },
        "numeric_summaries": {
            "revenue": {
                "overall": {
                    "median": 57.5,
                    "mean": 73.1,
                    "std": 60.21,
                    "q1": 23.0,
                    "q3": 107.5,
                    "iqr": 84.5,
                },
                "groups": {
                    "A": {
                        "count": 3,
                        "percentage": 30.0,
                        "median": 12.0,
                        "mean": 12.0,
                        "std": 2.0,
                    },
                    "B": {
                        "count": 3,
                        "percentage": 30.0,
                        "median": 110.0,
                        "mean": 110.0,
                        "std": 10.0,
                    },
                    "D": {
                        "count": 1,
                        "percentage": 10.0,
                        "median": 200.0,
                        "mean": 200.0,
                        "std": None,
                    },
                },
            },
        },
    }

    top_two_aggregates = analysis.compute_aggregates(
        dataframe,
        results=results,
        max_groups=2,
    )

    assert list(top_two_aggregates["segment"]["eligible_groups"]) == ["A", "B"]

    thresholded_aggregates = analysis.compute_aggregates(
        dataframe,
        results=results,
        min_percentage=20.0,
    )

    assert "tiny" not in thresholded_aggregates["thresholded_segment"][
        "eligible_groups"
    ]


def test_compute_aggregates_without_prior_results_returns_empty_dict(
    sample_eda_dataframe: pandas.DataFrame,
):
    assert analysis.compute_aggregates(sample_eda_dataframe) == {}
