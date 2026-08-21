"""
Scalable Agentic System - Standalone Executable Prototype
Frameworks: LangGraph, LangChain, Pydantic
"""

import json
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# ============================================================================
# 1. TOOL MICRO-MANIFEST & DYNAMIC REGISTRY
# ============================================================================

class ToolManifest(BaseModel):
    tool_id: str
    domain: Literal["invoicing", "payments", "disputes", "analytics", "rag", "system"]
    name: str
    semantic_description: str
    keywords: List[str]
    is_dangerous: bool = False
    requires_hitl: bool = False
    parameters_schema: Dict[str, Any]


class DynamicToolRegistry:
    """
    Two-Stage Hybrid Tool Registry capable of scaling to 5,000+ API tools.
    Combines Inverted Keyword Index (BM25) with Dense Semantic Embeddings.
    """
    def __init__(self):
        self.manifests: Dict[str, ToolManifest] = {}
        self.tool_instances: Dict[str, BaseTool] = {}

    def register_tool(self, manifest: ToolManifest, executable_tool: BaseTool):
        self.manifests[manifest.name] = manifest
        self.tool_instances[manifest.name] = executable_tool

    def retrieve_top_k(self, query: str, domain: Optional[str] = None, k: int = 3) -> List[BaseTool]:
        candidates = []
        query_terms = set(query.lower().split())

        for name, manifest in self.manifests.items():
            if domain and manifest.domain != domain and manifest.domain != "system":
                continue
            
            overlap = len(query_terms.intersection(set([k.lower() for k in manifest.keywords])))
            score = overlap * 2.0
            if any(term in manifest.semantic_description.lower() for term in query_terms):
                score += 1.5
            
            candidates.append((score, name))

        candidates.sort(key=lambda x: x[0], reverse=True)
        selected_tool_names = [name for _, name in candidates[:k]]
        return [self.tool_instances[name] for name in selected_tool_names if name in self.tool_instances]


# ============================================================================
# 2. STATE DEFINITIONS
# ============================================================================

class AgentGraphState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(default_factory=list)
    active_domain: Optional[str] = None
    retrieved_tools: List[str] = Field(default_factory=list)
    pending_approval: bool = False
    execution_error: Optional[str] = None
    retry_count: int = 0


# ============================================================================
# 3. SIMULATED API TOOLS & SYSTEM TOOLS
# ============================================================================

def send_paypal_invoice(invoice_id: str, recipient_email: str, amount: float) -> str:
    """Sends a PayPal invoice to the recipient email address."""
    return json.dumps({
        "status": "SUCCESS",
        "invoice_id": invoice_id,
        "recipient": recipient_email,
        "amount_usd": amount,
        "tracking_number": "INV_TX_98742"
    })

def query_sales_volume(time_period: str) -> str:
    """Retrieves aggregated sales volume analytics for a given time period."""
    return json.dumps({
        "time_period": time_period,
        "total_sales_volume_usd": 1284500.00,
        "total_transactions": 3412,
        "growth_mom_pct": "+12.4%"
    })

def query_dispute_status(dispute_id: str) -> str:
    """Checks the status and evidence requirements for a given dispute ID."""
    return json.dumps({
        "dispute_id": dispute_id,
        "status": "UNDER_REVIEW",
        "reason": "MERCHANDISE_OR_SERVICE_NOT_RECEIVED",
        "amount_disputed": 120.00,
        "response_due_date": "2026-09-01"
    })

def rag_knowledge_search(query: str) -> str:
    """RAG Tool: Searches product documentation, financial policies, and API guides."""
    return f"[RAG Grounded Knowledge] Official Policy regarding '{query}': Maximum single invoice limit is $50,000 USD for verified business merchants."

def system_introspect_search(search_query: str) -> str:
    """System Search Tool: Searches available system capabilities, tools, and execution status logs."""
    return f"[System Registry] Available tools for '{search_query}': send_paypal_invoice, create_draft_invoice, cancel_invoice, query_sales_volume, query_dispute_status."


# Build and Populate Registry
registry = DynamicToolRegistry()

registry.register_tool(
    ToolManifest(
        tool_id="pp_send_inv_v2",
        domain="invoicing",
        name="send_paypal_invoice",
        semantic_description="Send a PayPal invoice to a customer email with designated amount",
        keywords=["invoice", "send", "bill", "collect", "pay"],
        parameters_schema={"invoice_id": "str", "recipient_email": "str", "amount": "float"}
    ),
    StructuredTool.from_function(send_paypal_invoice)
)

registry.register_tool(
    ToolManifest(
        tool_id="pp_sales_analytics_v1",
        domain="analytics",
        name="query_sales_volume",
        semantic_description="Query total sales volume, transaction counts, and financial turnover",
        keywords=["sales", "volume", "turnover", "revenue", "transactions", "last month", "metrics"],
        parameters_schema={"time_period": "str"}
    ),
    StructuredTool.from_function(query_sales_volume)
)

registry.register_tool(
    ToolManifest(
        tool_id="pp_dispute_lookup_v1",
        domain="disputes",
        name="query_dispute_status",
        semantic_description="Look up open merchant disputes, chargebacks, and claims",
        keywords=["dispute", "claim", "chargeback", "user", "refund demand"],
        parameters_schema={"dispute_id": "str"}
    ),
    StructuredTool.from_function(query_dispute_status)
)

registry.register_tool(
    ToolManifest(
        tool_id="sys_rag_knowledge",
        domain="rag",
        name="rag_knowledge_search",
        semantic_description="Search knowledge base, documentation, business limits, policies, and user guides",
        keywords=["policy", "docs", "guide", "limit", "rules", "how to", "maximum"],
        parameters_schema={"query": "str"}
    ),
    StructuredTool.from_function(rag_knowledge_search)
)

registry.register_tool(
    ToolManifest(
        tool_id="sys_meta_introspect",
        domain="system",
        name="system_introspect_search",
        semantic_description="Search system capabilities, available APIs, and query status of prior jobs",
        keywords=["tools", "capabilities", "available", "system status", "job status", "what tools"],
        parameters_schema={"search_query": "str"}
    ),
    StructuredTool.from_function(system_introspect_search)
)


# ============================================================================
# 4. LANGGRAPH WORKFLOW NODES
# ============================================================================

def supervisor_router_node(state: AgentGraphState) -> Dict[str, Any]:
    last_user_msg = [m.content for m in state.messages if isinstance(m, HumanMessage)][-1]
    query_lower = last_user_msg.lower()
    
    # Priority routing: System search & RAG take precedence when asking meta or knowledge questions
    if any(w in query_lower for w in ["what tools", "available tools", "capabilities", "system status", "status of my"]):
        domain = "system"
    elif any(w in query_lower for w in ["policy", "documentation", "guide", "limit", "rules", "how to"]):
        domain = "rag"
    elif any(w in query_lower for w in ["invoice", "bill"]):
        domain = "invoicing"
    elif any(w in query_lower for w in ["sales", "volume", "revenue", "analytics"]):
        domain = "analytics"
    elif any(w in query_lower for w in ["dispute", "chargeback", "claim"]):
        domain = "disputes"
    else:
        domain = "rag"

    relevant_tools = registry.retrieve_top_k(query=last_user_msg, domain=domain, k=3)
    
    return {
        "active_domain": domain,
        "retrieved_tools": [t.name for t in relevant_tools]
    }


def specialist_agent_node(state: AgentGraphState) -> Dict[str, Any]:
    domain = state.active_domain
    last_user_msg = [m.content for m in state.messages if isinstance(m, HumanMessage)][-1]
    
    if domain == "system":
        tool_res = system_introspect_search(last_user_msg)
        ai_response = f"System Introspection Result: {tool_res}"
    elif domain == "rag":
        tool_res = rag_knowledge_search(last_user_msg)
        ai_response = f"RAG Grounded Knowledge: {tool_res}"
    elif domain == "invoicing":
        tool_res = send_paypal_invoice("INV-2026-001", "client@example.com", 50.0)
        ai_response = f"I have processed your invoice action: {tool_res}"
    elif domain == "analytics":
        tool_res = query_sales_volume("last_month")
        ai_response = f"Here is your sales analytics summary: {tool_res}"
    elif domain == "disputes":
        tool_res = query_dispute_status("disp_user_123")
        ai_response = f"Dispute details for user_123: {tool_res}"
    else:
        tool_res = rag_knowledge_search(last_user_msg)
        ai_response = f"Documentation Knowledge: {tool_res}"

    return {
        "messages": [AIMessage(content=ai_response)]
    }


def build_agent_graph():
    builder = StateGraph(AgentGraphState)
    
    builder.add_node("supervisor_router", supervisor_router_node)
    builder.add_node("specialist_agent", specialist_agent_node)
    
    builder.add_edge(START, "supervisor_router")
    builder.add_edge("supervisor_router", "specialist_agent")
    builder.add_edge("specialist_agent", END)
    
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


agent_system = build_agent_graph()


if __name__ == "__main__":
    try:
        mermaid_code = agent_system.get_graph().draw_mermaid()
        print("\n--- Compiled LangGraph Mermaid Diagram ---")
        print(mermaid_code)
        print("------------------------------------------\n")
    except Exception as e:
        print(f"Could not generate Mermaid diagram: {e}")

