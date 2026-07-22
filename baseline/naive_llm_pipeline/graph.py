from langgraph.graph import END, START, StateGraph
from nodes import (
    aggregate_outputs,
    load_raw_data,
    process_chunks_with_openai,
    split_into_large_chunks,
)
from state import BaselineState

def build_graph():
    graph_builder = StateGraph(BaselineState)

    graph_builder.add_node("load_raw_data", load_raw_data)
    graph_builder.add_node("split_into_large_chunks", split_into_large_chunks)
    graph_builder.add_node("process_chunks_with_openai", process_chunks_with_openai)
    graph_builder.add_node("aggregate_outputs", aggregate_outputs)

    graph_builder.add_edge(START, "load_raw_data")
    graph_builder.add_edge("load_raw_data", "split_into_large_chunks")
    graph_builder.add_edge("split_into_large_chunks", "process_chunks_with_openai")
    graph_builder.add_edge("process_chunks_with_openai", "aggregate_outputs")
    graph_builder.add_edge("aggregate_outputs", END)

    return graph_builder.compile()
