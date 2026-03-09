from langgraph.graph import END, StateGraph

from src.core import get_logger
from src.modules.agents.nodes import (
    AnalystNode,
    CriticNode,
    build_researcher_node,
    decision_node,
    emitter_node,
    router_node,
    should_continue,
)
from src.modules.agents.state import AgentState
from src.modules.agents.tools import AgentTools
from src.modules.rag.services import RagService


logger = get_logger(module="agents", component="workflow")


def build_workflow(rag: RagService):
    logger.info("Building ARCADE workflow graph")
    tools = AgentTools(rag)
    researcher_node = build_researcher_node(tools)

    analyst_node = AnalystNode()
    critic_node = CriticNode()

    graph = StateGraph(AgentState)

    graph.add_node("ask", router_node)
    graph.add_node("retrieve", researcher_node)
    graph.add_node("analyze", analyst_node)
    graph.add_node("critique", critic_node)
    graph.add_node("decide", decision_node)
    graph.add_node("emit", emitter_node)

    graph.set_entry_point("ask")
    graph.add_edge("ask", "retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "critique")
    graph.add_edge("critique", "decide")
    

    graph.add_conditional_edges(
        "decide",
        should_continue,
        {
            "retry_retrieve": "retrieve",
            "retry_analyze": "analyze",
            "end": "emit",
        },
    )
    graph.add_edge("emit", END)

    compiled = graph.compile()
    logger.info("ARCADE workflow graph compiled successfully")
    return compiled
