"""Format EDA workflow output for text, markdown, and HTML reports."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jinja2


AnalysisStepDisplay = tuple[str, str]
TEMPLATES_DIR = Path(__file__).with_name("templates")


def format_text_report(
    *,
    results: Mapping[str, Any],
    observations: Mapping[str, Sequence[str]],
    summary: str | None,
    recommendations: Sequence[str],
    steps: Sequence[AnalysisStepDisplay],
    detailed: bool = False,
) -> str:
    """Format workflow output as terminal-friendly text.

    Parameters
    ----------
    results
        Analysis step results keyed by step name.
    observations
        LLM observations keyed by step name.
    summary
        Final LLM summary text.
    recommendations
        Final LLM recommendations.
    steps
        Ordered pairs of ``(step_key, display_title)``.
    detailed
        When true, expand nested dict and list values as JSON. When false,
        show compact summaries for nested values.

    Returns
    -------
    str
        A formatted text report.
    """
    sections: list[str] = []

    for step_key, step_title in steps:
        sections.append(_format_text_section(step_title))
        sections.append(_format_text_results(step_key, results, detailed=detailed))
        sections.append(_format_text_observations(step_key, observations))

    sections.append(_format_text_final_synthesis(summary, recommendations))

    return "\n\n".join(section for section in sections if section)


def format_markdown_report(
    *,
    results: Mapping[str, Any],
    observations: Mapping[str, Sequence[str]],
    summary: str | None,
    recommendations: Sequence[str],
    steps: Sequence[AnalysisStepDisplay],
    csv_path: Path | str | None = None,
    model_name: str | None = None,
    graph_path: Path | str | None = None,
) -> str:
    """Format workflow output as a portable markdown report."""
    lines = ["# EDA Workflow Report", ""]
    metadata = _format_markdown_metadata(
        csv_path=csv_path,
        model_name=model_name,
        graph_path=graph_path,
    )
    if metadata:
        lines.extend([metadata, ""])

    for step_key, step_title in steps:
        lines.extend(
            [
                f"## {step_title}",
                "",
                "### Results",
                _format_markdown_results(step_key, results),
                "",
                "### Observations",
                _format_markdown_observations(step_key, observations),
                "",
            ]
        )

    lines.extend(
        [
            "## Final Synthesis",
            "",
            "### Summary",
            summary if summary else "_No summary._",
            "",
            "### Recommendations",
            _format_markdown_recommendations(recommendations),
            "",
        ]
    )

    return "\n".join(lines)


def format_html_report(
    *,
    results: Mapping[str, Any],
    observations: Mapping[str, Sequence[str]],
    summary: str | None,
    recommendations: Sequence[str],
    steps: Sequence[AnalysisStepDisplay],
    csv_path: Path | str | None = None,
    model_name: str | None = None,
    graph_path: Path | str | None = None,
) -> str:
    """Format workflow output as a self-contained HTML report."""
    template = _html_environment().get_template("report.html.j2")
    return template.render(
        metadata=_html_metadata(
            csv_path=csv_path,
            model_name=model_name,
            graph_path=graph_path,
        ),
        steps=_html_steps(
            results=results,
            observations=observations,
            steps=steps,
        ),
        summary=summary,
        recommendations=list(recommendations),
    )


def _format_json(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)


def _html_environment() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _format_text_section(title: str) -> str:
    return "\n".join([
        "=" * 60,
        title.upper(),
        "=" * 60,
    ])


def _format_text_results(
    step_key: str,
    results: Mapping[str, Any],
    *,
    detailed: bool,
) -> str:
    step_results = results.get(step_key)
    if not isinstance(step_results, Mapping) or not step_results:
        return "(No results)"

    return "\n".join(
        _format_text_result_item(key, value, detailed=detailed)
        for key, value in step_results.items()
    )


def _format_text_result_item(key: str, value: Any, *, detailed: bool) -> str:
    if isinstance(value, (dict, list)):
        if detailed:
            return f"{key}:\n{_format_json(value)}"

        return f"{key}: {type(value).__name__} with {len(value)} items"

    return f"{key}: {value}"


def _format_text_observations(
    step_key: str,
    observations: Mapping[str, Sequence[str]],
) -> str:
    lines = ["Observations:"]
    step_observations = observations.get(step_key, [])
    if not step_observations:
        lines.append("  (No observations)")
    else:
        lines.extend(f"  - {observation}" for observation in step_observations)

    return "\n".join(lines)


def _format_text_final_synthesis(
    summary: str | None,
    recommendations: Sequence[str],
) -> str:
    lines = [
        _format_text_section("Final Synthesis"),
        "",
        "Summary:",
        summary if summary else "(Not implemented yet)",
        "",
        "Recommendations:",
    ]
    if recommendations:
        lines.extend(f"  - {recommendation}" for recommendation in recommendations)
    else:
        lines.append("  (Not implemented yet)")

    return "\n".join(lines)


def _format_markdown_metadata(
    *,
    csv_path: Path | str | None,
    model_name: str | None,
    graph_path: Path | str | None,
) -> str:
    rows = []
    if csv_path is not None:
        rows.append(("CSV", _format_markdown_path_link(csv_path)))
    if model_name is not None:
        rows.append(("Model", f"`{model_name}`"))
    if graph_path is not None:
        rows.append(("Graph", _format_markdown_path_link(graph_path)))

    if not rows:
        return ""

    lines = [
        "## Run Metadata",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    return "\n".join(lines)


def _format_markdown_path_link(path: Path | str) -> str:
    path_text = str(path)
    return f"[`{path_text}`]({path_text})"


def _format_markdown_results(
    step_key: str,
    results: Mapping[str, Any],
) -> str:
    step_results = results.get(step_key)
    if not isinstance(step_results, Mapping) or not step_results:
        return "_No results._"

    scalar_table = _format_markdown_scalar_results(step_results)
    nested_sections = [
        _format_markdown_result_item(key, value)
        for key, value in step_results.items()
        if isinstance(value, (dict, list))
    ]
    parts = [scalar_table] if scalar_table else []
    parts.extend(nested_sections)

    return "\n\n".join(parts)


def _format_markdown_scalar_results(step_results: Mapping[str, Any]) -> str:
    scalar_rows = [
        (key, value)
        for key, value in step_results.items()
        if not isinstance(value, (dict, list))
    ]
    if not scalar_rows:
        return ""

    lines = [
        "| Field | Value |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| `{key}` | `{_escape_markdown_table_cell(value)}` |"
        for key, value in scalar_rows
    )
    return "\n".join(lines)


def _format_markdown_result_item(key: str, value: Any) -> str:
    if isinstance(value, (dict, list)):
        result_type = type(value).__name__
        return "\n".join(
            [
                "<details>",
                (
                    f"<summary><strong><code>{key}</code></strong> "
                    f"({result_type}, {len(value)} items)</summary>"
                ),
                "",
                "```json",
                _format_json(value),
                "```",
                "",
                "</details>",
            ]
        )

    return f"- `{key}`: {value}"


def _escape_markdown_table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _format_markdown_observations(
    step_key: str,
    observations: Mapping[str, Sequence[str]],
) -> str:
    step_observations = observations.get(step_key, [])
    if not step_observations:
        return "_No observations._"

    return "\n".join(f"- {observation}" for observation in step_observations)


def _format_markdown_recommendations(recommendations: Sequence[str]) -> str:
    if not recommendations:
        return "_No recommendations._"

    return "\n".join(f"- {recommendation}" for recommendation in recommendations)


def _html_metadata(
    *,
    csv_path: Path | str | None,
    model_name: str | None,
    graph_path: Path | str | None,
) -> list[dict[str, str | None]]:
    metadata = []
    if csv_path is not None:
        metadata.append({
            "label": "CSV",
            "value": str(csv_path),
            "href": str(csv_path),
        })
    if model_name is not None:
        metadata.append({
            "label": "Model",
            "value": model_name,
            "href": None,
        })
    if graph_path is not None:
        metadata.append({
            "label": "Graph",
            "value": str(graph_path),
            "href": str(graph_path),
        })

    return metadata


def _html_steps(
    *,
    results: Mapping[str, Any],
    observations: Mapping[str, Sequence[str]],
    steps: Sequence[AnalysisStepDisplay],
) -> list[dict[str, Any]]:
    return [
        _html_step(
            step_key=step_key,
            title=title,
            results=results,
            observations=observations,
        )
        for step_key, title in steps
    ]


def _html_step(
    *,
    step_key: str,
    title: str,
    results: Mapping[str, Any],
    observations: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    step_results = results.get(step_key)
    scalar_results = []
    nested_results = []

    if isinstance(step_results, Mapping):
        scalar_results = [
            {"key": key, "value": value}
            for key, value in step_results.items()
            if not isinstance(value, (dict, list))
        ]
        nested_results = [
            {
                "key": key,
                "type": type(value).__name__,
                "count": len(value),
                "json": _format_json(value),
            }
            for key, value in step_results.items()
            if isinstance(value, (dict, list))
        ]

    return {
        "title": title,
        "scalar_results": scalar_results,
        "nested_results": nested_results,
        "observations": list(observations.get(step_key, [])),
        "has_results": bool(scalar_results or nested_results),
    }
