# ruff: noqa: E501
"""Report generation helper."""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report_stub(metrics: MetricsReport) -> str:
    """Return a minimal report stub."""
    return f"""# Day 08 Lab Report

## Metrics summary

- Total scenarios: {metrics.total_scenarios}
- Success rate: {metrics.success_rate:.2%}
- Average nodes visited: {metrics.avg_nodes_visited:.2f}
- Total retries: {metrics.total_retries}
- Total interrupts: {metrics.total_interrupts}

## Architecture & Implementation Details

The architecture utilizes LangGraph to create an agentic workflow with robust error handling and conditional routing.

- **State Schema:** Uses a lean TypedDict with append-only lists for `messages`, `events`, `tool_results`, and `errors`. Overwritable scalars (`attempt`, `route`, `risk_level`) are used for simple tracking. This provides a balance of determinism and auditability.
- **Failure Modes:** Explicit handling for transient tool failures through a bounded retry loop (evaluate -> retry -> tool). If failures persist past `max_attempts`, they are sent to a `dead_letter` queue.
- **Improvement Plan:** Moving forward, the heuristic routing policy should be replaced by a structured LLM call with few-shot examples for accuracy, and `evaluate_node` should use LLM-as-a-judge for rigorous tool output validation.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report_stub(metrics), encoding="utf-8")
