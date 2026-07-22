from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from nodes import agent_node
from state import AgentState
from tools import TOOLS


def build_graph():
    graph_builder = StateGraph(AgentState)

    # The single node
    graph_builder.add_node("agent", agent_node)
    # Executes whatever tool the agent calls
    graph_builder.add_node("tools", ToolNode(TOOLS))

    graph_builder.add_edge(START, "agent")
    # If the agent asked for a tool, go to "tools"; otherwise finish.
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    # After running tools, give control back to the agent
    graph_builder.add_edge("tools", "agent")

    return graph_builder.compile()
