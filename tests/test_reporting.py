from pathlib import Path

import eda_workflow.reporting as reporting


STEPS = (("profile_dataset", "Dataset Profile"),)
RESULTS = {
    "profile_dataset": {
        "total_rows": 3,
        "shape": {"rows": 3, "columns": 2},
        "columns": ["sales", "region"],
    },
}
OBSERVATIONS = {
    "profile_dataset": [
        "The sample has enough rows for a smoke test.",
    ],
}


def test_format_text_report_summarizes_nested_results_by_default():
    report = reporting.format_text_report(
        results=RESULTS,
        observations=OBSERVATIONS,
        summary="The dataset is small but usable.",
        recommendations=["Run the full workflow on the showcase data."],
        steps=STEPS,
    )

    assert "DATASET PROFILE" in report
    assert "total_rows: 3" in report
    assert "shape: dict with 2 items" in report
    assert "columns: list with 2 items" in report
    assert '"rows": 3' not in report
    assert "The sample has enough rows for a smoke test." in report
    assert "Run the full workflow on the showcase data." in report


def test_format_text_report_expands_nested_results_when_detailed():
    report = reporting.format_text_report(
        results=RESULTS,
        observations=OBSERVATIONS,
        summary="The dataset is small but usable.",
        recommendations=[],
        steps=STEPS,
        detailed=True,
    )

    assert "shape:" in report
    assert '"rows": 3' in report
    assert '"sales"' in report
    assert "shape: dict with 2 items" not in report


def test_format_markdown_report_collapses_nested_results():
    report = reporting.format_markdown_report(
        csv_path=Path("data/sample.csv"),
        model_name="gpt-test",
        graph_path="graph.png",
        results=RESULTS,
        observations=OBSERVATIONS,
        summary="The dataset is small but usable.",
        recommendations=["Run the full workflow on the showcase data."],
        steps=STEPS,
    )

    assert "# EDA Workflow Report" in report
    assert "## Run Metadata" in report
    assert "| CSV | [`data/sample.csv`](data/sample.csv) |" in report
    assert "| Model | `gpt-test` |" in report
    assert "| Graph | [`graph.png`](graph.png) |" in report
    assert "| `total_rows` | `3` |" in report
    assert (
        "<summary><strong><code>shape</code></strong> "
        "(dict, 2 items)</summary>"
    ) in report
    assert (
        "<summary><strong><code>columns</code></strong> "
        "(list, 2 items)</summary>"
    ) in report
    assert "```json" in report
    assert '"rows": 3' in report
    assert "- The sample has enough rows for a smoke test." in report
    assert "### Summary\nThe dataset is small but usable." in report
    assert "- Run the full workflow on the showcase data." in report


def test_format_markdown_report_handles_missing_optional_content():
    report = reporting.format_markdown_report(
        results={},
        observations={},
        summary=None,
        recommendations=[],
        steps=STEPS,
    )

    assert "_No results._" in report
    assert "_No observations._" in report
    assert "_No summary._" in report
    assert "_No recommendations._" in report


def test_format_html_report_uses_template_with_embedded_styles():
    report = reporting.format_html_report(
        csv_path=Path("data/sample.csv"),
        model_name="gpt-test",
        graph_path="graph.png",
        results={
            "profile_dataset": {
                "total_rows": 3,
                "shape": {"rows": 3, "columns": 2},
                "source": "<script>alert('x')</script>",
            },
        },
        observations=OBSERVATIONS,
        summary="The dataset is small but usable.",
        recommendations=["Run the full workflow on the showcase data."],
        steps=STEPS,
    )

    assert "<!doctype html>" in report
    assert "<style>" in report
    assert '<a href="data/sample.csv"><code>data/sample.csv</code></a>' in report
    assert '<a href="graph.png"><code>graph.png</code></a>' in report
    assert "<td><code>total_rows</code></td>" in report
    assert "<td><code>3</code></td>" in report
    assert "<summary><code>shape</code> (dict, 2 items)</summary>" in report
    assert "&#34;rows&#34;: 3" in report
    assert "&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;" in report
    assert "<li>The sample has enough rows for a smoke test.</li>" in report
    assert "<p>The dataset is small but usable.</p>" in report
