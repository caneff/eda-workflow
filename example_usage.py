"""
Example usage of the EDA Workflow with OpenAI.

Requires OPENAI_API_KEY in a .env file or environment variable.
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import eda_workflow.reporting as reporting
from eda_workflow.workflow import EDAWorkflow


DEFAULT_CSV_PATH = Path("data/eda_showcase.csv")
DEFAULT_MARKDOWN_PATH = Path("eda_report.md")
DEFAULT_HTML_PATH = Path("eda_report.html")
GRAPH_PATH = "graph.png"
MODEL_NAME = "gpt-4o-mini"
ANALYSIS_STEPS = (
    ("profile_dataset", "Dataset Profile"),
    ("analyze_missingness", "Missingness Analysis"),
    ("compute_aggregates", "Aggregates Analysis"),
    ("analyze_relationships", "Relationships Analysis"),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the example script."""
    parser = argparse.ArgumentParser(
        description="Run the EDA workflow on a CSV file.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to the CSV dataset. Defaults to {DEFAULT_CSV_PATH}.",
    )
    parser.add_argument(
        "--markdown-path",
        type=Path,
        default=DEFAULT_MARKDOWN_PATH,
        help=f"Path for the markdown report. Defaults to {DEFAULT_MARKDOWN_PATH}.",
    )
    parser.add_argument(
        "--html-path",
        type=Path,
        default=DEFAULT_HTML_PATH,
        help=f"Path for the HTML report. Defaults to {DEFAULT_HTML_PATH}.",
    )
    parser.add_argument(
        "--text-detail",
        action="store_true",
        help="Expand nested dict and list output in the terminal report.",
    )
    return parser.parse_args()


def write_report(path: Path, report: str) -> None:
    """Write a report, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    """Run the example EDA workflow."""
    args = parse_args()
    load_dotenv()

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
    )
    workflow = EDAWorkflow(model=llm)

    workflow.draw_graph(output_file_path=GRAPH_PATH)
    print(f"Graph diagram saved to {GRAPH_PATH}\n")

    print(f"Running EDA analysis on {args.csv_path}...\n")
    workflow.invoke_workflow(str(args.csv_path))

    summary = workflow.get_summary()
    recommendations = workflow.get_recommendations() or []
    observations = workflow.get_observations() or {}
    results = workflow.get_results() or {}

    print(
        reporting.format_text_report(
            results=results,
            observations=observations,
            summary=summary,
            recommendations=recommendations,
            steps=ANALYSIS_STEPS,
            detailed=args.text_detail,
        )
    )

    markdown_report = reporting.format_markdown_report(
        csv_path=args.csv_path,
        model_name=MODEL_NAME,
        graph_path=GRAPH_PATH,
        summary=summary,
        recommendations=recommendations,
        observations=observations,
        results=results,
        steps=ANALYSIS_STEPS,
    )
    write_report(args.markdown_path, markdown_report)
    print(f"\nMarkdown report saved to {args.markdown_path}")

    html_report = reporting.format_html_report(
        csv_path=args.csv_path,
        model_name=MODEL_NAME,
        graph_path=GRAPH_PATH,
        summary=summary,
        recommendations=recommendations,
        observations=observations,
        results=results,
        steps=ANALYSIS_STEPS,
    )
    write_report(args.html_path, html_report)
    print(f"HTML report saved to {args.html_path}")


if __name__ == "__main__":
    main()
