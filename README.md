# EDA Workflow

Fork maintained at [caneff/eda-workflow](https://github.com/caneff/eda-workflow). Upstream: [Future-Proof-DS/eda-workflow](https://github.com/Future-Proof-DS/eda-workflow).

An AI-assisted exploratory data analysis workflow for consistent first-pass dataset review. It combines deterministic pandas analysis with LangGraph orchestration, LLM-generated observations after each step, final synthesis with recommendations, and reusable text, Markdown, and HTML reporting.

## How It Works

The workflow follows a fixed analyze, observe, synthesize loop:

1. **Analyze**: Run deterministic pandas analysis steps on the dataset.
2. **Observe**: Ask the LLM for concise observations after each analysis step.
3. **Synthesize**: Summarize the observations into final findings and recommendations.
4. **Report**: Format results for terminal output, portable Markdown, or styled HTML.

Active analysis steps:

- `profile_dataset`: basic shape, columns, dtypes, numeric summaries, and categorical summaries.
- `analyze_missingness`: missing counts, percentages, high-missing columns, and complete row counts.
- `compute_aggregates`: notable categorical group differences for numeric columns.
- `analyze_relationships`: strongest positive and negative Spearman relationships between numeric columns.

The deterministic steps can run without an LLM. Observation extraction and final synthesis require a chat model.

## Setup

### Prerequisites

- **Python 3.14 or higher** ([uv](https://docs.astral.sh/uv/) will install one if needed)
- **uv** for dependency management
- **OpenAI API key** for the LLM-powered example and synthesis

### Installation

1. Install uv if needed:

   **Windows PowerShell**
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   **macOS/Linux**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

3. Create a `.env` file and add your OpenAI API key:

   **Windows**
   ```powershell
   copy .env.example .env
   ```

   **macOS/Linux**
   ```bash
   cp .env.example .env
   ```

   ```env
   OPENAI_API_KEY=sk-your-key-here
   ```

To pin the project Python version:

```bash
uv python pin 3.14
uv sync
```

## Usage

### Python Workflow API

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from eda_workflow.workflow import EDAWorkflow

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
workflow = EDAWorkflow(model=llm)

workflow.invoke_workflow("data/eda_showcase.csv")

summary = workflow.get_summary()
recommendations = workflow.get_recommendations()
observations = workflow.get_observations()
results = workflow.get_results()
```

To run only deterministic analysis and skip LLM observations/synthesis:

```python
from eda_workflow.workflow import EDAWorkflow

workflow = EDAWorkflow(model=None)
workflow.invoke_workflow("data/eda_showcase.csv")
results = workflow.get_results()
```

### Reporting API

`eda_workflow.reporting` formats workflow output without rerunning analysis.

```python
import eda_workflow.reporting as reporting

steps = (
    ("profile_dataset", "Dataset Profile"),
    ("analyze_missingness", "Missingness Analysis"),
    ("compute_aggregates", "Aggregates Analysis"),
    ("analyze_relationships", "Relationships Analysis"),
)

text_report = reporting.format_text_report(
    results=results or {},
    observations=observations or {},
    summary=summary,
    recommendations=recommendations or [],
    steps=steps,
)

detailed_text_report = reporting.format_text_report(
    results=results or {},
    observations=observations or {},
    summary=summary,
    recommendations=recommendations or [],
    steps=steps,
    detailed=True,
)

markdown_report = reporting.format_markdown_report(
    results=results or {},
    observations=observations or {},
    summary=summary,
    recommendations=recommendations or [],
    steps=steps,
    csv_path="data/eda_showcase.csv",
    model_name="gpt-4o-mini",
    graph_path="graph.png",
)

html_report = reporting.format_html_report(
    results=results or {},
    observations=observations or {},
    summary=summary,
    recommendations=recommendations or [],
    steps=steps,
    csv_path="data/eda_showcase.csv",
    model_name="gpt-4o-mini",
    graph_path="graph.png",
)
```

Text output defaults to compact summaries for nested results. Use `detailed=True` to expand nested dictionaries and lists. Markdown keeps full results in collapsible sections. HTML uses the Jinja2 template in `eda_workflow/templates/report.html.j2`.

### Running The Example

```bash
uv run python example_usage.py
```

By default, the example analyzes `data/eda_showcase.csv` and writes:

- `graph.png`: LangGraph workflow diagram
- `eda_report.md`: portable Markdown report
- `eda_report.html`: styled, self-contained HTML report

Useful flags:

```bash
uv run python example_usage.py --csv-path data/cafe_sales.csv
uv run python example_usage.py --markdown-path reports/analysis.md
uv run python example_usage.py --html-path reports/analysis.html
uv run python example_usage.py --text-detail
```

The full example calls the configured OpenAI model. Make sure `OPENAI_API_KEY` is set before running it.

## Datasets

- `data/cafe_sales.csv`: small sample dataset.
- `data/eda_showcase.csv`: synthetic showcase dataset designed to exercise missingness, aggregate differences, and numeric relationships.

Regenerate the showcase dataset with:

```bash
uv run python scripts/generate_showcase_dataset.py
```

The generated reports and graph image are ignored by git. The two sample CSVs are intentionally committed.

## Project Structure

```text
eda-workflow/
├── data/
│   ├── cafe_sales.csv
│   └── eda_showcase.csv
├── eda_workflow/
│   ├── analysis.py                 # Deterministic pandas analysis steps
│   ├── reporting.py                # Text, Markdown, and HTML report formatters
│   ├── workflow.py                 # LangGraph workflow orchestration
│   ├── prompts/                    # LLM prompt templates
│   └── templates/
│       └── report.html.j2          # Styled HTML report template
├── scripts/
│   └── generate_showcase_dataset.py
├── tests/
│   ├── test_analysis.py
│   ├── test_reporting.py
│   └── test_workflow.py
├── example_usage.py
├── pyproject.toml
├── uv.lock
└── README.md
```

`uv.lock` is committed so users get identical, tested dependency versions.
