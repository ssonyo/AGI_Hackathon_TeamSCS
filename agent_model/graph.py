import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

# import nodes(modules)
from nodes.parse_document import parse_document_node
from nodes.is_valid import is_valid_node
from nodes.embed_docs import embed_docs_node
from nodes.calculate_yield import calculate_yield_node
from nodes.evaluate_risk import evaluate_risk_node
from nodes.generate_report import generate_report_node

# define states
class RealEstateState(TypedDict, total=False):
    file_path: str
    query: str
    document_text: str
    is_valid: bool
    vectordb_path: str
    yield_result: str
    risk_result: str
    final_report: str

graph_builder = StateGraph(state_schema=RealEstateState)

# register nodes
graph_builder.add_node("parse_document", parse_document_node)
graph_builder.add_node("check_validity", is_valid_node)
graph_builder.add_node("embed_docs", embed_docs_node)
graph_builder.add_node("calculate_yield", calculate_yield_node)
graph_builder.add_node("evaluate_risk", evaluate_risk_node)
graph_builder.add_node("generate_report", generate_report_node)

# define agentif flow
graph_builder.set_entry_point("parse_document")
graph_builder.add_edge("parse_document", "check_validity")

# 유효성 검사 후 분기 (invalid 면 건너뛰도록)
graph_builder.add_conditional_edges(
    "check_validity",
    lambda state: "generate_report" if not state["is_valid"] else "embed_docs"
)

graph_builder.add_edge("embed_docs", "calculate_yield")
graph_builder.add_edge("calculate_yield", "evaluate_risk")
graph_builder.add_edge("evaluate_risk", "generate_report")
graph_builder.add_edge("generate_report", END)

app = graph_builder.compile()
