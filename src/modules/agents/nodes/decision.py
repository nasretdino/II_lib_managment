from src.core import get_logger
from src.modules.agents.state import AgentState


logger = get_logger(module="agents", node="decide")


async def decision_node(state: AgentState) -> AgentState:
    iteration = state.get("iteration", 0) + 1
    max_iterations = state.get("max_iterations", 3)

    is_approved = state.get("is_approved", False)
    needs_more_context = state.get("needs_more_context", False)

    if is_approved or iteration >= max_iterations:
        route_after_decision = "end"
    elif needs_more_context:
        route_after_decision = "retry_retrieve"
    else:
        route_after_decision = "retry_analyze"

    logger.info(
        "Decide node routed: route={}, approved={}, needs_more_context={}, iteration={}/{}",
        route_after_decision,
        is_approved,
        needs_more_context,
        iteration,
        max_iterations,
    )

    critique = state.get("critique", "")
    correction = critique if route_after_decision != "end" else ""

    events = list(state.get("events", []))
    events.append(
        {
            "type": "decision_made",
            "data": f"Route={route_after_decision}",
            "iteration": iteration,
        }
    )

    return {
        **state,
        "iteration": iteration,
        "correction": correction,
        "route_after_decision": route_after_decision,
        "events": events,
    }


def should_continue(state: AgentState) -> str:
    route = state.get("route_after_decision", "end")
    logger.debug("Conditional edge decision: {}", route)
    return route
