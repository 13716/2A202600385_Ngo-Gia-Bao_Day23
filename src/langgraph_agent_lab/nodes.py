"""Node skeletons for the LangGraph workflow.

Each function should be small, testable, and return a partial state update. Avoid mutating the
input state in place.
"""

from __future__ import annotations

from .state import AgentState, ApprovalDecision, Route, make_event


def intake_node(state: AgentState) -> dict:
    """Normalize raw query into state fields."""
    query = state.get("query", "").strip()
    # Basic normalization and metadata extraction
    normalized_query = " ".join(query.split())
    has_pii = any(char.isdigit() for char in normalized_query if len(normalized_query) > 10)
    
    return {
        "query": normalized_query,
        "messages": [f"intake:{normalized_query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized", pii_flag=has_pii)],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route."""
    query = state.get("query", "").lower()
    route = Route.SIMPLE
    risk_level = "low"
    
    # Improved routing policy based on keywords
    if any(keyword in query for keyword in ["refund", "delete", "send", "cancel", "remove"]):
        route = Route.RISKY
        risk_level = "high"
    elif any(keyword in query for keyword in ["status", "order", "lookup", "find", "search"]):
        route = Route.TOOL
    elif len(query.split()) < 4 or any(keyword in query for keyword in ["it", "that", "this"]):
        route = Route.MISSING_INFO
    elif any(keyword in query for keyword in ["timeout", "fail", "error", "bug", "crash"]):
        route = Route.ERROR
        
    return {
        "route": route.value,
        "risk_level": risk_level,
        "events": [make_event("classify", "completed", f"route={route.value}")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")
    question = f"Could you provide more context or clarify what you mean by '{query}'?"
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "missing information requested")],
    }


def tool_node(state: AgentState) -> dict:
    """Call a mock tool.

    Simulates transient failures for error-route scenarios to demonstrate retry loops.
    """
    attempt = int(state.get("attempt", 0))
    scenario = state.get("scenario_id", "unknown")
    if state.get("route") == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient failure attempt={attempt} scenario={scenario}"
    else:
        # Structured tool results
        result = f"SUCCESS: tool_execution_id={scenario}_{attempt} data='Mock data retrieved'"
    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", f"tool executed attempt={attempt}")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for approval."""
    action = f"Action: execute risky operation for query '{state.get('query', '')}';"
    return {
        "proposed_action": f"{action} Justification: required by policy; approval required",
        "events": [make_event("risky_action", "pending_approval", "approval required")],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval step with optional LangGraph interrupt().

    Set LANGGRAPH_INTERRUPT=true to use real interrupt() for HITL demos.
    Default uses mock decision so tests and CI run offline.
    """
    import os

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt({
            "proposed_action": state.get("proposed_action"),
            "risk_level": state.get("risk_level"),
        })
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        decision = ApprovalDecision(approved=True, comment="mock approval for lab")
    return {
        "approval": decision.model_dump(),
        "events": [make_event("approval", "completed", f"approved={decision.approved}")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt or fallback decision."""
    attempt = int(state.get("attempt", 0)) + 1
    # Bounded retry and exponential backoff metadata conceptually
    backoff_ms = (2 ** attempt) * 100
    errors = [f"transient failure attempt={attempt}"]
    return {
        "attempt": attempt,
        "errors": errors,
        "events": [
            make_event(
                "retry",
                "completed",
                "retry attempt recorded",
                attempt=attempt,
                backoff_ms=backoff_ms,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response."""
    approval = state.get("approval") or {}
    if state.get("tool_results"):
        answer = f"I found the following based on our tools: {state['tool_results'][-1]}"
        if approval.get("approved"):
            answer += " This action was approved."
    elif pending := state.get("pending_question"):
        answer = pending
    else:
        answer = f"Based on your query '{state.get('query', '')}', here is the safe mock answer."
    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "answer generated")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the 'done?' check that enables retry loops."""
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""
    # More structured validation check
    if "ERROR:" in latest or "failure" in latest.lower():
        return {
            "evaluation_result": "needs_retry",
            "events": [
                make_event("evaluate", "completed", "tool result indicates failure, retry needed")
            ],
        }
    return {
        "evaluation_result": "success",
        "events": [make_event("evaluate", "completed", "tool result satisfactory")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Log unresolvable failures for manual review.

    Third layer of error strategy: retry -> fallback -> dead letter.
    """
    attempt = state.get("attempt", 0)
    ans = (
        f"Request could not be completed after {attempt} "
        "retry attempts. Logged for manual review."
    )
    msg = f"max retries exceeded, attempt={attempt}, ticket_created=True"
    return {
        "final_answer": ans,
        "events": [make_event("dead_letter", "completed", msg)],
    }


def finalize_node(state: AgentState) -> dict:
    """Finalize the run and emit a final audit event."""
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
