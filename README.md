# EDA Workflow

Fork maintained at [caneff/eda-workflow](https://github.com/caneff/eda-workflow). Upstream: [Future-Proof-DS/eda-workflow](https://github.com/Future-Proof-DS/eda-workflow).

An AI-powered exploratory data analysis workflow that performs consistent, first-pass analysis of datasets using LangChain and LangGraph. The workflow runs a fixed set of analysis tools, uses an LLM to extract observations after each step, and synthesizes findings into a summary with actionable recommendations.

## How It Works

The workflow follows a sequential process:
1. **Analyze**: Runs a fixed set of predefined analysis tools on the dataset
2. **Observe**: After each tool, the LLM extracts concise observations from the results
3. **Synthesize**: Once all tools have run, the LLM summarizes findings and provides actionable recommendations

This approach combines deterministic pandas-based analysis with LLM-powered interpretation.

## Setup

### Prerequisites

- **Python 3.14 or higher** ([uv](https://docs.astral.sh/uv/) will install one if needed)
- **uv** (dependency manager)
- **OpenAI API Key**

### Installation Steps

1. **Install uv** (if not already installed):

   **Windows (PowerShell)**:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   **macOS/Linux**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   After installation, restart your terminal.

2. **Install dependencies**:
   ```bash
   uv sync
   ```

   This uses `uv.lock` for reproducible installs.

3. **Set up your OpenAI API key**:
   
   **Windows**:
   ```powershell
   copy .env.example .env
   ```
   
   **macOS/Linux**:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

### Multiple Python Versions?

To pin the project Python (see `.python-version`):

```bash
uv python pin 3.14
uv sync
```

## Usage

### Python API

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from eda_workflow.workflow import EDAWorkflow

load_dotenv()

# Initialize the workflow with an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
workflow = EDAWorkflow(model=llm)

# Run analysis on a dataset
workflow.invoke_workflow("data/cafe_sales.csv")

# Retrieve results
summary = workflow.get_summary()  # str
recommendations = workflow.get_recommendations()  # list[str]
observations = workflow.get_observations()  # dict[str, list[str]]
results = workflow.get_results()  # dict
```

### Running the Example

```bash
uv run python example_usage.py
```

This runs a full analysis on the sample dataset and prints the results for each step.

## Project Structure

```
eda-agent/
├── data/
│   └── cafe_sales.csv             # Sample dataset
├── eda_workflow/
│   ├── __init__.py
│   ├── eda_workflow.py             # Main workflow class and graph
│   └── prompts/                   # LLM prompt templates
│       ├── extract_observations_system.md
│       ├── extract_observations_human.md
│       ├── synthesize_findings_system.md
│       └── synthesize_findings_human.md
├── .env.example                   # Environment variable template
├── example_usage.py               # Example script
├── pyproject.toml                 # Dependencies configuration
├── uv.lock                        # Locked dependency versions
└── README.md
```

**Important**: The `uv.lock` file is committed to ensure all users get identical, tested dependency versions.
