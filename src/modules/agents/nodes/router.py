from src.core import get_logger
from src.modules.agents.state import AgentState, QueryType


logger = get_logger(module="agents", node="ask")


def _classify_query_type(question: str) -> QueryType:
    lowered = question.lower()

    if any(token in lowered for token in ("podcast", "podkast", "audio script", "script episode")):
        return "podcast"

    analytics_hints = (
        "compare",
        "difference",
        "across",
        "trend",
        "summary of all",
        "срав",
        "разниц",
        "по всем",
        "аналит",
    )
    if any(token in lowered for token in analytics_hints):
        return "analytics"

    return "search"


async def router_node(state: AgentState) -> AgentState:
    question = state["question"]
    query_type = _classify_query_type(question)
    logger.info(
        "Ask node routed query: query_type={}, question_len={}, iteration={}",
        query_type,
        len(question),
        state.get("iteration", 0),
    )

    events = list(state.get("events", []))
    events.append(
        {
            "type": "ask_routed",
            "data": f"Query routed as '{query_type}'",
            "iteration": state.get("iteration", 0),
        }
    )

    return {
        **state,
        "query_type": query_type,
        "events": events,
    }
