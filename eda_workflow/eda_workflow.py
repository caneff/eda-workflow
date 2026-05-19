import enum
import logging
import math
import os
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd
import pydantic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

logger = logging.getLogger(__name__)
WORKFLOW_NAME = "eda_workflow"
LOG_PATH = os.path.join(os.getcwd(), "logs/")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


class ObservationOutput(pydantic.BaseModel):
    """Structured observations returned by the observation LLM call."""

    observations: list[str] = pydantic.Field(
        description="1-2 concise, actionable observations"
    )


class SynthesisOutput(pydantic.BaseModel):
    """Structured response returned by the synthesis LLM call."""

    summary: str = pydantic.Field(
        description="A concise 2-3 sentence summary of key findings"
    )
    recommendations: list[str] = pydantic.Field(
        description="3-5 actionable recommendations"
    )


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


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    with open(prompt_path, "r") as f:
        return f.read()


def load_prompt_pair(file_prefix: str) -> ChatPromptTemplate:
    """Load system and human prompt templates for a shared file prefix."""
    return ChatPromptTemplate.from_messages([
        ("system", load_prompt(f"{file_prefix}_system.md")),
        ("human", load_prompt(f"{file_prefix}_human.md")),
    ])


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


def extract_observations(
    model: BaseChatModel | None,
    step_name: str,
    step_results: dict[str, Any],
) -> list[str]:
    """Extract concise observations for one analysis step using an LLM."""
    if model is None:
        return []

    observation_prompt = load_prompt_pair("extract_observations")
    chain = observation_prompt | model.with_structured_output(ObservationOutput)
    response = cast(
        ObservationOutput,
        chain.invoke({
            "step_name": step_name.replace("_", " ").title(),
            "results": str(step_results),
        }),
    )

    return response.observations


@dataclass
class EDAState:
    """LangGraph workflow state."""

    dataframe_dict: dict
    results: dict = field(default_factory=dict)
    observations: dict[str, list[str]] = field(default_factory=dict)
    current_step: str = ""
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)

    @classmethod
    def from_graph(cls, state: EDAState | Mapping[str, Any]) -> EDAState:
        """Coerce LangGraph invoke output; omitted keys use dataclass defaults."""
        if isinstance(state, EDAState):
            return state

        return cls(**dict(state))


class EDAWorkflow:
    """
    Exploratory Data Analysis workflow that performs consistent, first-pass analysis of datasets.

    Uses a fixed set of predefined analysis tools to produce structured, tabular outputs.
    Operates sequentially and deterministically through baseline EDA steps.

    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.

    Attributes
    ----------
    response : EDAState or None
        Stores the full response after invoke_workflow() is called.
    """

    model: BaseChatModel | None
    log: bool
    log_path: str | None
    checkpointer: Checkpointer | None
    response: EDAState | None
    _compiled_graph: CompiledStateGraph

    def __init__(
        self,
        model: BaseChatModel | None = None,
        log: bool = False,
        log_path: str | None = None,
        checkpointer: Checkpointer | None = None,
    ) -> None:
        self.model = model
        self.log = log
        self.log_path = log_path
        self.checkpointer = checkpointer
        self.response = None
        self._compiled_graph = make_eda_baseline_workflow(
            model=model,
            log=log,
            log_path=log_path,
            checkpointer=checkpointer,
        )

    def invoke_workflow(self, filepath: str, **kwargs):
        """
        Run EDA analysis on the provided dataset.

        Parameters
        ----------
        filepath : str
            Path to the dataset file.
        **kwargs
            Additional arguments passed to the underlying graph invoke method.

        Returns
        -------
        None
            Results are stored in self.response and accessed via getter methods.
        """
        df = pd.read_csv(filepath)

        response = self._compiled_graph.invoke(
            EDAState(dataframe_dict=df.to_dict()),
            **kwargs,
        )

        self.response = EDAState.from_graph(response)
        return None

    def draw_graph(self, output_file_path: str) -> None:
        """Save a Mermaid PNG diagram of the compiled workflow graph."""
        self._compiled_graph.get_graph().draw_mermaid_png(
            output_file_path=output_file_path
        )

    def get_summary(self) -> str | None:
        """Retrieves the analysis summary."""
        if self.response:
            return self.response.summary

    def get_recommendations(self) -> list[str] | None:
        """Retrieves the recommendations."""
        if self.response:
            return self.response.recommendations

    def get_results(self) -> dict | None:
        """Retrieves the full analysis results."""
        if self.response:
            return self.response.results

    def get_observations(self) -> dict[str, list[str]] | None:
        """Retrieves all observations from analysis steps."""
        if self.response:
            return self.response.observations


def make_eda_baseline_workflow(
    model: BaseChatModel | None = None,
    log: bool = False,
    log_path: str | None = None,
    checkpointer: Checkpointer | None = None,
) -> CompiledStateGraph:
    """
    Factory function that creates a compiled LangGraph workflow for baseline EDA.

    Performs automated first-pass analysis with fixed analysis steps.

    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.

    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow ready to process EDA requests.
    """
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    def make_analysis_node(step: AnalysisStep):
        """Create a graph node for one registered analysis step."""

        def analysis_node(state: EDAState):
            step_name = step.name.value
            logger.info("Running analysis step: %s", step_name)
            df = pd.DataFrame.from_dict(state.dataframe_dict)
            prior_results = dict(state.results)
            step_results = step.analyze(df, results=prior_results)
            results = {**prior_results, step_name: step_results}

            return {
                "current_step": step_name,
                "results": results,
            }

        return analysis_node

    def extract_observations_node(state: EDAState):
        """Extract observations from the latest analysis results using LLM."""
        logger.info("Extracting observations")

        current_step = state.current_step
        results = state.results
        observations = dict(state.observations)

        if not current_step or current_step not in results:
            return {"observations": observations}

        step_results = cast(dict[str, Any], results[current_step])
        step_observations = extract_observations(model, current_step, step_results)

        if step_observations:
            observations[current_step] = step_observations

        return {
            "observations": observations,
        }

    def synthesize_findings_node(state: EDAState):
        """Synthesize accumulated findings into summary and recommendations."""
        logger.info("Synthesizing findings")

        observations = state.observations

        if model is None:
            return {
                "summary": "No LLM provided for synthesis",
                "recommendations": [],
            }

        all_observations = []
        for step_name, step_obs in observations.items():
            all_observations.append(f"\n{step_name.replace('_', ' ').title()}:")
            for obs in step_obs:
                all_observations.append(f"  - {obs}")

        observations_text = "\n".join(all_observations)

        synthesis_prompt = load_prompt_pair("synthesize_findings")

        chain = synthesis_prompt | model.with_structured_output(SynthesisOutput)
        response = cast(
            SynthesisOutput,
            chain.invoke({"observations": observations_text}),
        )

        return {
            "summary": response.summary,
            "recommendations": response.recommendations,
        }

    step_names = [step.name.value for step in ANALYSIS_STEPS]
    next_step_by_name = {
        step_name: step_names[index + 1]
        for index, step_name in enumerate(step_names[:-1])
    }

    def route_after_observations(state: EDAState) -> str:
        return next_step_by_name.get(state.current_step, "synthesize_findings")

    route_targets: dict[Hashable, str] = {
        step_name: step_name for step_name in step_names[1:]
    }
    route_targets["synthesize_findings"] = "synthesize_findings"

    workflow = StateGraph(EDAState)

    for step in ANALYSIS_STEPS:
        workflow.add_node(step.name.value, make_analysis_node(step))
    workflow.add_node("extract_observations", extract_observations_node)
    workflow.add_node("synthesize_findings", synthesize_findings_node)

    workflow.set_entry_point(ANALYSIS_STEPS[0].name.value)

    for step in ANALYSIS_STEPS:
        workflow.add_edge(step.name.value, "extract_observations")
    workflow.add_conditional_edges(
        "extract_observations",
        route_after_observations,
        route_targets,
    )
    workflow.add_edge("synthesize_findings", END)

    app = workflow.compile(checkpointer=checkpointer, name=WORKFLOW_NAME)

    return app
