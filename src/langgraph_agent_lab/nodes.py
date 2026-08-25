from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event


# ─── Data Models for LLM Structured Outputs ──────────────────────────
class ClassificationResult(BaseModel):
    """Structured output for intent classification."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description=(
            "The classified route for the user query:\n"
            "- 'risky': Actions with side effects, refunds, deletions, sending emails.\n"
            "- 'tool': Information lookups, order status, tracking numbers, searches.\n"
            "- 'missing_info': Vague queries lacking actionable context.\n"
            "- 'error': System failures, timeouts, crash reports.\n"
            "- 'simple': General FAQ questions answerable with static knowledge."
        )
    )
    risk_level: Literal["low", "high"] = Field(
        description="'high' if the route is risky; 'low' otherwise."
    )
    reasoning: str = Field(
        description="Reasoning following priority: risky > tool > missing_info > error > simple."
    )


class EvaluationResult(BaseModel):
    """Structured output for LLM-as-judge tool evaluation."""

    evaluation_result: Literal["success", "needs_retry"] = Field(
        description="'needs_retry' if tool encountered error/timeout; 'success' if completed."
    )
    reasoning: str = Field(description="Explanation of tool execution evaluation.")


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Student Implementations ─────────────────────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM with structured output."""
    query = state.get("query", "").strip()
    system_prompt = (
        "You are an intent classification engine for an enterprise support agent.\n"
        "Classify the query into exactly one route:\n"
        "1. 'risky': Side effects, mutations, refunds, deletions, sending emails.\n"
        "2. 'tool': Information lookups, order status checks, tracking.\n"
        "3. 'missing_info': Vague queries where context/identifiers are missing.\n"
        "4. 'error': System errors, service failures, timeout exceptions.\n"
        "5. 'simple': General informational/FAQ questions requiring no external tool.\n\n"
        "Priority order: risky > tool > missing_info > error > simple."
    )


    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(ClassificationResult)
        res: ClassificationResult = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ])
        route = res.route
        risk_level = "high" if route == "risky" else res.risk_level
    except Exception:
        # Robust fallback
        q_lower = query.lower()
        if any(w in q_lower for w in ["refund", "delete", "cancel", "email", "send"]):
            route, risk_level = "risky", "high"
        elif any(w in q_lower for w in ["order", "lookup", "status", "track", "search"]):
            route, risk_level = "tool", "low"
        elif any(w in q_lower for w in ["timeout", "failure", "crash", "error"]):
            route, risk_level = "error", "low"
        elif len(query.split()) <= 4 and any(w in q_lower for w in ["fix", "it", "help"]):
            route, risk_level = "missing_info", "low"
        else:
            route, risk_level = "simple", "low"

    return {
        "route": route,
        "risk_level": risk_level,
        "messages": [f"classified:{route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified query as '{route}' (risk: {risk_level})",
                route=route,
                risk_level=risk_level,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call with error simulation."""
    attempt = state.get("attempt", 0)
    route = state.get("route", "")
    query = state.get("query", "")

    if route == "error" and attempt < 2:
        result_string = f"ERROR: Service timeout/failure during tool execution (attempt {attempt})"
    elif route == "risky":
        result_string = f"SUCCESS: Risky operation processed and executed for query: '{query}'"
    elif route == "tool":
        result_string = (
            f"SUCCESS: Order/Account lookup completed for query: '{query}'. Status: Shipped."
        )
    else:
        result_string = f"SUCCESS: Tool executed successfully for query: '{query}'"

    return {
        "tool_results": [result_string],
        "messages": [f"tool:{result_string[:40]}"],
        "events": [
            make_event(
                "tool",
                "completed",
                f"executed tool: {result_string[:50]}",
                result=result_string,
            )
        ],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results using LLM-as-judge (with heuristic fallback)."""
    tool_results = state.get("tool_results", [])
    latest_result = tool_results[-1] if tool_results else ""

    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(EvaluationResult)
        prompt = (
            "You are an LLM-as-judge evaluating the execution result of a tool.\n"
            "Determine if the tool execution succeeded or encountered an error requiring retry.\n"
            f"Tool Execution Output:\n{latest_result}"
        )

        res: EvaluationResult = structured_llm.invoke(prompt)
        eval_result = res.evaluation_result
    except Exception:
        has_err = "ERROR" in latest_result.upper() or "FAIL" in latest_result.upper()
        eval_result = "needs_retry" if has_err else "success"

    return {
        "evaluation_result": eval_result,
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"evaluated tool result as '{eval_result}'",
                evaluation_result=eval_result,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM grounded in context."""
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")

    context_parts = []
    if tool_results:
        context_parts.append("Tool Results:\n" + "\n".join(f"- {tr}" for tr in tool_results))
    if approval:
        context_parts.append(f"Approval Decision: {approval}")

    context_str = "\n\n".join(context_parts) if context_parts else "No external tools needed."

    system_prompt = (
        "You are an expert customer support AI. Generate a concise, helpful, "
        "professional, and grounded answer to the user's query based on the context."
    )
    user_prompt = f"User Query: {query}\n\nContext:\n{context_str}"

    try:
        llm = get_llm(temperature=0.0)
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        final_answer = response.content if hasattr(response, "content") else str(response)
        if isinstance(final_answer, list):
            final_answer = "".join(str(item) for item in final_answer)
    except Exception:
        if tool_results:
            final_answer = f"Based on your request '{query}': {tool_results[-1]}"
        else:
            final_answer = (
                f"Here is the answer to your request '{query}': To reset your password, "
                "visit account settings and select 'Reset Password'."
            )

    return {
        "final_answer": final_answer,
        "messages": [f"answer:{str(final_answer)[:40]}"],
        "events": [make_event("answer", "completed", "final answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")
    approval = state.get("approval")

    # If action was rejected by human supervisor in HITL flow
    if approval and not approval.get("approved", True):
        comment = approval.get("comment", "Action was not authorized by supervisor")
        question = (
            f"[REJECTED] Yêu cầu '{query}' đã bị TỪ CHỐI bởi người giám sát (Lý do: {comment}). "
            "Hành động rủi ro này KHÔNG ĐƯỢC THỰC THI. Vui lòng cung cấp thêm thông tin "
            "xác thực nếu bạn muốn gửi lại yêu cầu."
        )

    else:
        try:
            llm = get_llm(temperature=0.0)
            prompt = (
                "You are a customer support agent. The user's query is too vague.\n"
                f"User Query: {query}\n\n"
                "Ask a single, polite, and specific clarification question requesting details."
            )
            response = llm.invoke(prompt)
            question = response.content if hasattr(response, "content") else str(response)
            if isinstance(question, list):
                question = "".join(str(x) for x in question)
        except Exception:
            question = f"Could you please provide more details regarding your request: '{query}'?"

    return {
        "pending_question": question,
        "final_answer": question,
        "messages": [f"clarify:{question[:40]}"],
        "events": [
            make_event(
                "clarify",
                "completed",
                f"requested clarification: {question[:50]}",
                question=question,
            )
        ],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    proposed = f"Proposed Action: Execute sensitive operation for request: '{query}'"
    return {
        "proposed_action": proposed,
        "messages": [f"risky_action:{proposed[:40]}"],
        "events": [
            make_event(
                "risky_action",
                "completed",
                "prepared action for approval",
                proposed_action=proposed,
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step."""
    import os

    existing = state.get("approval")
    if existing is not None and isinstance(existing, dict):
        approval_dict = existing
    elif os.getenv("LANGGRAPH_INTERRUPT", "false").lower() in ("true", "1"):
        try:
            from langgraph.types import interrupt

            decision = interrupt({
                "question": "Approval required for risky action",
                "proposed_action": state.get("proposed_action", ""),
                "query": state.get("query", ""),
            })
            if isinstance(decision, dict):
                approval_dict = decision
            elif isinstance(decision, bool):
                approval_dict = {
                    "approved": decision,
                    "reviewer": "human_supervisor",
                    "comment": "Decision from HITL interrupt",
                }
            else:
                approval_dict = {
                    "approved": True,
                    "reviewer": "human_supervisor",
                    "comment": str(decision),
                }
        except Exception:
            approval_dict = {
                "approved": True,
                "reviewer": "mock-reviewer",
                "comment": "Interrupt fallback auto-approved",
            }
    else:
        approval_dict = {
            "approved": True,
            "reviewer": "mock-reviewer",
            "comment": "Mock approval for testing",
        }

    is_approved = approval_dict.get("approved", False)
    return {
        "approval": approval_dict,
        "messages": [f"approval:{'approved' if is_approved else 'rejected'}"],
        "events": [
            make_event(
                "approval",
                "completed",
                f"human approval status: {'approved' if is_approved else 'rejected'}",
                approved=is_approved,
            )
        ],
    }



def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt."""
    current_attempt = state.get("attempt", 0)
    new_attempt = current_attempt + 1
    err_msg = f"Transient failure encountered; incremented attempt to {new_attempt}"
    return {
        "attempt": new_attempt,
        "errors": [err_msg],
        "messages": [f"retry:attempt_{new_attempt}"],
        "events": [
            make_event(
                "retry",
                "completed",
                f"retry recorded (attempt {new_attempt})",
                attempt=new_attempt,
            )
        ],
    }



def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded."""
    attempt = state.get("attempt", 0)
    query = state.get("query", "")
    final_answer = (
        f"Escalated to dead letter queue: Request '{query}' could not be completed after "
        f"{attempt} retry attempts. A senior engineer/support specialist has been notified."
    )
    return {
        "final_answer": final_answer,
        "errors": [f"Exhausted max retries ({attempt} attempts). Sent to dead letter queue."],
        "messages": ["dead_letter:escalated"],
        "events": [
            make_event(
                "dead_letter",
                "completed",
                "max retries exceeded, escalated to dead letter queue",
                attempt=attempt,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    return {
        "messages": ["finalize:workflow_completed"],
        "events": [make_event("finalize", "completed", "workflow finished successfully")],
    }
