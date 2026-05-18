import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

logger = logging.getLogger(__name__)
WORKFLOW_NAME = "eda_workflow"
LOG_PATH = os.path.join(os.getcwd(), "logs/")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


class SynthesisResponse(TypedDict):
    """Structured response returned by the synthesis LLM call."""

    summary: str
    recommendations: list[str]


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    with open(prompt_path, "r") as f:
        return f.read()


@dataclass
class EDAState:
    """LangGraph workflow state."""

    dataframe: dict
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

        return cls(**cast(dict[str, Any], dict(state)))


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
            EDAState(dataframe=df.to_dict()),
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

    def profile_dataset_node(state: EDAState):
        """Generate dataset profile with basic statistics."""
        logger.info("Profiling dataset")
        df = pd.DataFrame.from_dict(state.dataframe)
        results = dict(state.results)

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "numeric_summary": (
                df[numeric_cols].describe().to_dict() if numeric_cols else {}
            ),
            "categorical_summary": {
                col: df[col].value_counts().head(10).to_dict()
                for col in categorical_cols
            },
        }

        results["profile_dataset"] = profile

        return {
            "current_step": "profile_dataset",
            "results": results,
        }

    def analyze_missingness_node(state: EDAState):
        """Analyze missing values in the dataset."""
        logger.info("Analyzing missingness")
        df = pd.DataFrame.from_dict(state.dataframe)
        results = dict(state.results)

        missing_count = df.isnull().sum().to_dict()
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2).to_dict()

        high_missing = {col: pct for col, pct in missing_pct.items() if pct > 20}

        missingness = {
            "total_rows": len(df),
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "high_missing_columns": high_missing,
            "complete_rows": int(df.dropna().shape[0]),
            "complete_rows_pct": (
                round(df.dropna().shape[0] / len(df) * 100, 2) if len(df) > 0 else 0
            ),
        }

        results["analyze_missingness"] = missingness

        return {
            "current_step": "analyze_missingness",
            "results": results,
        }

    def compute_aggregates_node(state: EDAState):
        """Compute group-by aggregates on key columns.

        TODO: Implement this analysis tool.

        See profile_dataset_node and analyze_missingness_node for reference.
        Store your results in results["compute_aggregates"] and return
        {"current_step": "compute_aggregates", "results": results}.
        """
        logger.info("Computing aggregates")

    def analyze_relationships_node(state: EDAState):
        """Analyze relationships between variables.

        TODO: Implement this analysis tool.

        See profile_dataset_node and analyze_missingness_node for reference.
        Store your results in results["analyze_relationships"] and return
        {"current_step": "analyze_relationships", "results": results}.
        """
        logger.info("Analyzing relationships")

    def extract_observations_node(state: EDAState):
        """Extract observations from the latest analysis results using LLM."""
        logger.info("Extracting observations")

        current_step = state.current_step
        results = state.results
        observations = dict(state.observations)

        if model is None or not current_step or current_step not in results:
            return {"observations": observations}

        step_results = results.get(current_step, {})

        observation_schema = {
            "title": "ObservationOutput",
            "description": "Observations extracted from an analysis step.",
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "1-2 concise, actionable observations",
                },
            },
            "required": ["observations"],
        }

        observation_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("extract_observations_system.txt")),
            ("human", load_prompt("extract_observations_human.txt")),
        ])

        chain = observation_prompt | model.with_structured_output(observation_schema)
        response = cast(
            dict[str, list[str]],
            chain.invoke({
                "step_name": current_step.replace("_", " ").title(),
                "results": str(step_results),
            }),
        )

        observations[current_step] = response["observations"]

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

        synthesis_schema = {
            "title": "SynthesisOutput",
            "description": "Synthesized findings from EDA observations.",
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "A concise 2-3 sentence summary of key findings"
                    ),
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-5 actionable recommendations",
                },
            },
            "required": ["summary", "recommendations"],
        }

        all_observations = []
        for step_name, step_obs in observations.items():
            all_observations.append(f"\n{step_name.replace('_', ' ').title()}:")
            for obs in step_obs:
                all_observations.append(f"  - {obs}")

        observations_text = "\n".join(all_observations)

        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("synthesize_findings_system.txt")),
            ("human", load_prompt("synthesize_findings_human.txt")),
        ])

        chain = synthesis_prompt | model.with_structured_output(synthesis_schema)
        response = cast(
            SynthesisResponse,
            chain.invoke({"observations": observations_text}),
        )

        return {
            "summary": response["summary"],
            "recommendations": response["recommendations"],
        }

    workflow = StateGraph(EDAState)

    workflow.add_node("profile_dataset", profile_dataset_node)
    workflow.add_node("extract_observations_1", extract_observations_node)
    workflow.add_node("analyze_missingness", analyze_missingness_node)
    workflow.add_node("extract_observations_2", extract_observations_node)
    workflow.add_node("compute_aggregates", compute_aggregates_node)
    workflow.add_node("extract_observations_3", extract_observations_node)
    workflow.add_node("analyze_relationships", analyze_relationships_node)
    workflow.add_node("extract_observations_4", extract_observations_node)
    workflow.add_node("synthesize_findings", synthesize_findings_node)

    workflow.set_entry_point("profile_dataset")

    workflow.add_edge("profile_dataset", "extract_observations_1")
    workflow.add_edge("extract_observations_1", "analyze_missingness")
    workflow.add_edge("analyze_missingness", "extract_observations_2")
    workflow.add_edge("extract_observations_2", "compute_aggregates")
    workflow.add_edge("compute_aggregates", "extract_observations_3")
    workflow.add_edge("extract_observations_3", "analyze_relationships")
    workflow.add_edge("analyze_relationships", "extract_observations_4")
    workflow.add_edge("extract_observations_4", "synthesize_findings")
    workflow.add_edge("synthesize_findings", END)

    app = workflow.compile(checkpointer=checkpointer, name=WORKFLOW_NAME)

    return app
