from src.core import get_logger
from src.modules.agents.state import AgentState


logger = get_logger(module="agents", node="emit")


async def emitter_node(state: AgentState) -> AgentState:
    final_answer = state.get("draft_answer") or "No answer produced."
    logger.info(
        "Emit node finalized response: answer_len={}, iteration={}",
        len(final_answer),
        state.get("iteration", 0),
    )

    events = list(state.get("events", []))
    events.append(
        {
            "type": "emit_completed",
            "data": "Final answer is ready",
            "iteration": state.get("iteration", 0),
        }
    )

    return {
        **state,
        "final_answer": final_answer,
        "events": events,
    }
