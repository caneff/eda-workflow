from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas
import pytest

import eda_workflow.eda_workflow


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


@pytest.fixture
def sample_eda_dataframe() -> pandas.DataFrame:
    return pandas.DataFrame({
        "region": ["North", "South", "North"],
        "sales": [10, None, 30],
        "units": [1, 2, None],
    })


@pytest.fixture
def sample_eda_csv_path(
    tmp_path: Path,
    sample_eda_dataframe: pandas.DataFrame,
) -> Path:
    csv_path = tmp_path / "sample_eda.csv"
    sample_eda_dataframe.to_csv(csv_path, index=False)
    return csv_path


def test_eda_state_from_graph_with_existing_state_returns_same_instance():
    state = eda_workflow.eda_workflow.EDAState(dataframe={"sales": {0: 10}})

    result = eda_workflow.eda_workflow.EDAState.from_graph(state)

    assert result is state


def test_eda_state_from_graph_with_mapping_applies_defaults():
    state = eda_workflow.eda_workflow.EDAState.from_graph({
        "dataframe": {"sales": {0: 10}},
    })

    expected_defaults = {
        "results": {},
        "observations": {},
        "current_step": "",
        "summary": "",
        "recommendations": [],
    }
    for attribute, expected_value in expected_defaults.items():
        assert getattr(state, attribute) == expected_value


def test_workflow_getters_before_invoke_return_none():
    workflow = eda_workflow.eda_workflow.EDAWorkflow()

    assert workflow.get_summary() is None
    assert workflow.get_recommendations() is None
    assert workflow.get_results() is None
    assert workflow.get_observations() is None


def test_invoke_workflow_without_model_profiles_dataset_and_missingness(
    sample_eda_csv_path: Path,
):
    workflow = eda_workflow.eda_workflow.EDAWorkflow(model=None)

    result = workflow.invoke_workflow(str(sample_eda_csv_path))

    assert result is None
    results = workflow.get_results()
    assert results is not None
    assert {"profile_dataset", "analyze_missingness"}.issubset(results)

    profile = results["profile_dataset"]
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

    missingness = results["analyze_missingness"]
    assert_dict_contains(
        missingness,
        {
            "missing_count": {"sales": 1, "units": 1},
            "missing_percentage": {"sales": 33.33, "units": 33.33},
            "high_missing_columns": {"sales": 33.33, "units": 33.33},
            "complete_rows": 1,
        },
    )


def test_invoke_workflow_without_model_sets_no_llm_summary(
    sample_eda_csv_path: Path,
):
    workflow = eda_workflow.eda_workflow.EDAWorkflow(model=None)

    workflow.invoke_workflow(str(sample_eda_csv_path))

    assert workflow.get_summary() == "No LLM provided for synthesis"
    assert workflow.get_recommendations() == []
    assert workflow.get_observations() == {}


@pytest.mark.parametrize(
    ("file_prefix", "template_inputs", "expected_values"),
    [
        (
            "extract_observations",
            {
                "step_name": "Analyze Missingness",
                "results": r"{'missing_count': {'sales': 1}}",
            },
            ["Analyze Missingness", "'sales': 1"],
        ),
        (
            "synthesize_findings",
            {
                "observations": "Analyze Missingness:\n  - Sales has missing values.",
            },
            ["Analyze Missingness", "Sales has missing values."],
        ),
    ],
)
def test_load_prompt_pair_renders_runtime_inputs(
    file_prefix: str,
    template_inputs: dict[str, str],
    expected_values: list[str],
):
    prompt = eda_workflow.eda_workflow.load_prompt_pair(file_prefix)

    messages = prompt.format_messages(**template_inputs)

    rendered_text = "\n".join(cast(str, message.content) for message in messages)
    for expected_value in expected_values:
        assert expected_value in rendered_text
    for template_key in template_inputs:
        assert f"{{{template_key}}}" not in rendered_text
    assert "TODO" not in rendered_text
