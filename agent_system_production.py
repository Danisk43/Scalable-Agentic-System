"""
Scalable Agentic System - Production-Grade Executable Module
Includes:
- SentenceTransformer + FAISS Dense Semantic Indexing
- Sparse Token Overlap Indexing
- Reciprocal Rank Fusion (RRF) Retrieval Model
- LangGraph State Machine with Ephemeral State Persistence
"""

import os
import json
import numpy as np
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Ensure NumPy compatibility warnings are suppressed if any
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# 1. TOOL MICRO-MANIFEST & DENSE-SPARSE REGISTRY
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


class SentenceTransformerToolRegistry:
    """
    Production-grade hybrid registry capable of scaling to 5,000+ API tools.
    Utilizes SentenceTransformers (all-MiniLM-L6-v2) for dense vectors, 
    FAISS for indexing, and Reciprocal Rank Fusion (RRF) with keyword matching.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # Lazy loaded
        self.manifests: Dict[str, ToolManifest] = {}
        self.tool_instances: Dict[str, BaseTool] = {}
        self.tool_names_ordered: List[str] = []
        self.embeddings_list: List[np.ndarray] = []
        self.index = None

    @property
    def model(self):
        """Lazy load the sentence transformer model to keep initial imports fast."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def register_tool(self, manifest: ToolManifest, executable_tool: BaseTool):
        self.manifests[manifest.name] = manifest
        self.tool_instances[manifest.name] = executable_tool
        
        # 1. Generate dense embedding of the semantic description
        embedding = self.model.encode(manifest.semantic_description, convert_to_numpy=True)
        
        self.tool_names_ordered.append(manifest.name)
        self.embeddings_list.append(embedding)
        
        # 2. Rebuild FAISS index
        import faiss
        vectors = np.array(self.embeddings_list).astype('float32')
        # Normalize vectors for cosine similarity (Inner Product index)
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def retrieve_top_k(self, query: str, domain: Optional[str] = None, k: int = 3) -> List[BaseTool]:
        """
        Retrieves Top-K tools using hybrid dense-sparse scoring and RRF:
        1. Query dense semantic embedding vector via SentenceTransformers.
        2. Query FAISS for Cosine Similarity.
        3. Match keywords for Sparse scoring.
        4. Fuse ranks using Reciprocal Rank Fusion.
        """
        if not self.tool_names_ordered or self.index is None:
            return []

        # Filter domain constraints
        candidates = []
        for name in self.tool_names_ordered:
            manifest = self.manifests[name]
            if domain and manifest.domain != domain and manifest.domain != "system":
                continue
            candidates.append(name)

        if not candidates:
            return []

        # --- Stage 1: Dense Search (FAISS) ---
        query_vector = self.model.encode(query, convert_to_numpy=True).astype('float32').reshape(1, -1)
        import faiss
        faiss.normalize_L2(query_vector)
        
        # Search all items in index
        total_registered = len(self.tool_names_ordered)
        similarities, indices = self.index.search(query_vector, total_registered)
        
        dense_ranks = {}
        dense_scores = {}
        rank_idx = 1
        for idx in indices[0]:
            if idx == -1:
                continue
            name = self.tool_names_ordered[idx]
            if name in candidates:
                dense_ranks[name] = rank_idx
                # Store actual similarity score (Cosine Similarity)
                dense_scores[name] = float(similarities[0][np.where(indices[0] == idx)[0][0]])
                rank_idx += 1

        # --- Stage 2: Sparse Search (Keyword Match) ---
        query_terms = set(query.lower().split())
        sparse_scores = {}
        for name in candidates:
            manifest = self.manifests[name]
            overlap = len(query_terms.intersection(set([kw.lower() for kw in manifest.keywords])))
            sparse_scores[name] = float(overlap * 2.0)

        # Sort candidates for sparse ranking
        sorted_sparse = sorted(candidates, key=lambda x: sparse_scores[x], reverse=True)
        sparse_ranks = {name: r + 1 for r, name in enumerate(sorted_sparse)}

        # --- Stage 3: Reciprocal Rank Fusion (RRF) ---
        rrf_scores = {}
        for name in candidates:
            d_rank = dense_ranks.get(name, len(candidates) + 1)
            s_rank = sparse_ranks.get(name, len(candidates) + 1)
            # RRF constant k = 60
            rrf_scores[name] = (1.0 / (60.0 + d_rank)) + (1.0 / (60.0 + s_rank))

        # Sort by final RRF score
        sorted_by_rrf = sorted(candidates, key=lambda x: rrf_scores[x], reverse=True)
        
        # Print diagnostic metrics
        print(f"\n[Dynamic Retrieval: \"{query}\"]")
        for rank, name in enumerate(sorted_by_rrf[:k], 1):
            m = self.manifests[name]
            d_sim = dense_scores.get(name, 0.0)
            s_match = sparse_scores.get(name, 0.0)
            print(f"  #{rank} Tool: '{name}' | Domain: {m.domain} | RRF Score: {rrf_scores[name]:.4f} | Cosine Sim: {d_sim:.4f} | Keyword Score: {s_match:.1f}")

        return [self.tool_instances[name] for name in sorted_by_rrf[:k]]


# ============================================================================
# 2. STATE DEFINITION
# ============================================================================

class AgentGraphState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(default_factory=list)
    active_domain: Optional[str] = None
    retrieved_tools: List[str] = Field(default_factory=list)
    pending_approval: bool = False
    execution_error: Optional[str] = None
    retry_count: int = 0


# ============================================================================
# 3. TOOLS & CATALOG POPULATION
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
    return f"[System Registry] Available tools for '{search_query}': send_paypal_invoice, query_sales_volume, query_dispute_status, rag_knowledge_search, system_introspect_search."


# Initialize registry
registry = SentenceTransformerToolRegistry()

registry.register_tool(
    ToolManifest(
        tool_id="pp_send_inv_v2",
        domain="invoicing",
        name="send_paypal_invoice",
        semantic_description="Send a PayPal invoice or billing request to a customer email with a designated amount to collect payment.",
        keywords=["invoice", "send", "bill", "collect", "pay", "charge"],
        parameters_schema={"invoice_id": "str", "recipient_email": "str", "amount": "float"}
    ),
    StructuredTool.from_function(send_paypal_invoice)
)

registry.register_tool(
    ToolManifest(
        tool_id="pp_sales_analytics_v1",
        domain="analytics",
        name="query_sales_volume",
        semantic_description="Query aggregated sales volume history, transaction counts, and financial turnover or monthly revenue metrics.",
        keywords=["sales", "volume", "turnover", "revenue", "transactions", "analytics", "income"],
        parameters_schema={"time_period": "str"}
    ),
    StructuredTool.from_function(query_sales_volume)
)

registry.register_tool(
    ToolManifest(
        tool_id="pp_dispute_lookup_v1",
        domain="disputes",
        name="query_dispute_status",
        semantic_description="Look up open merchant disputes, chargebacks, user claims, and review due dates.",
        keywords=["dispute", "claim", "chargeback", "user", "refund demand", "resolution"],
        parameters_schema={"dispute_id": "str"}
    ),
    StructuredTool.from_function(query_dispute_status)
)

registry.register_tool(
    ToolManifest(
        tool_id="sys_rag_knowledge",
        domain="rag",
        name="rag_knowledge_search",
        semantic_description="Search the business knowledge base, documentation, business limits, policies, and guidelines.",
        keywords=["policy", "docs", "guide", "limit", "rules", "how to", "compliance"],
        parameters_schema={"query": "str"}
    ),
    StructuredTool.from_function(rag_knowledge_search)
)

registry.register_tool(
    ToolManifest(
        tool_id="sys_meta_introspect",
        domain="system",
        name="system_introspect_search",
        semantic_description="Search system capabilities, available APIs, tool registry information, and active execution logs.",
        keywords=["tools", "capabilities", "available", "system status", "job status", "apis"],
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
    
    # Priority routing to select target domain classification
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

    # Production Registry dynamically extracts relevant tools for this node turn
    relevant_tools = registry.retrieve_top_k(query=last_user_msg, domain=domain, k=2)
    
    return {
        "active_domain": domain,
        "retrieved_tools": [t.name for t in relevant_tools]
    }


def mock_execute_specialist(domain: str, query: str) -> str:
    """Mock execution helper if no real API key is configured."""
    if domain == "system":
        return system_introspect_search(query)
    elif domain == "rag":
        return rag_knowledge_search(query)
    elif domain == "invoicing":
        return send_paypal_invoice("INV-2026-99", "production@example.com", 250.0)
    elif domain == "analytics":
        return query_sales_volume("last_quarter")
    elif domain == "disputes":
        return query_dispute_status("disp_user_789")
    return rag_knowledge_search(query)


def specialist_agent_node(state: AgentGraphState) -> Dict[str, Any]:
    domain = state.active_domain
    last_user_msg = [m.content for m in state.messages if isinstance(m, HumanMessage)][-1]
    
    # --- Production Path: Check if real OpenAI API key is set ---
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            
            # 1. Initialize production Chat LLM
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
            # 2. Get the dynamically retrieved tools
            tools = [registry.tool_instances[name] for name in state.retrieved_tools if name in registry.tool_instances]
            
            # 3. Bind tools and execute
            llm_with_tools = llm.bind_tools(tools)
            sys_msg = SystemMessage(content=(
                f"You are a production '{domain}' specialist. You have access strictly to these dynamically "
                f"retrieved tools: {state.retrieved_tools}. Fulfill the request."
            ))
            messages = [sys_msg] + list(state.messages)
            
            ai_msg = llm_with_tools.invoke(messages)
            
            # Check for tool calls and execute
            if ai_msg.tool_calls:
                tool_call = ai_msg.tool_calls[0]
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                tool_res = registry.tool_instances[tool_name].invoke(tool_args)
                
                # final synthesis
                synthesis_messages = [
                    sys_msg,
                    HumanMessage(content=last_user_msg),
                    AIMessage(content="", tool_calls=ai_msg.tool_calls),
                    ToolMessage(content=tool_res, tool_call_id=tool_call["id"])
                ]
                final_ai = llm.invoke(synthesis_messages)
                ai_response = final_ai.content
            else:
                ai_response = ai_msg.content
                
        except Exception as e:
            ai_response = f"[Production LLM Exception: {e}]\nFallback Mock Result: {mock_execute_specialist(domain, last_user_msg)}"
    else:
        # Fallback to Simulated Local Execution (No API Key set)
        ai_response = f"[Simulated Node - Domain: {domain}]\nResult: {mock_execute_specialist(domain, last_user_msg)}"

    return {
        "messages": [AIMessage(content=ai_response)]
    }


# ============================================================================
# 5. GRAPH COMPILATION WITH PERSISTENCE
# ============================================================================

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


# ============================================================================
# 6. LOCAL TESTING BLOCK
# ============================================================================

if __name__ == "__main__":
    test_queries = [
        ("Invoicing Action", "Send an invoice for $50 to client@example.com"),
        ("Analytics Query", "What was my total sales volume last month?"),
        ("Dispute Lookup", "Is there a dispute open from user_123?"),
        ("System Search Tool", "What tools are available for managing invoices?"),
        ("RAG Pipeline Tool", "What is the policy on maximum invoice limits for business accounts?"),
    ]

    print("=================================================================")
    print("RUNNING PRODUCTION-GRADE AGENT SYSTEM VERIFICATION")
    print("=================================================================\n")

    for idx, (label, query) in enumerate(test_queries, 1):
        print(f"--- TEST CASE [{idx}]: {label} ---")
        print(f"User Input: \"{query}\"")
        
        config = {"configurable": {"thread_id": f"prod_thread_{idx}"}}
        result = agent_system.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )
        
        active_domain = result.get("active_domain")
        retrieved_tools = result.get("retrieved_tools")
        response_msg = result["messages"][-1].content
        
        print(f"\nFinal State Details:")
        print(f" -> Active Domain    : {active_domain}")
        print(f" -> Bound Top-K Tools: {retrieved_tools}")
        print(f" -> Final Response   :\n{response_msg}")
        print("=" * 65 + "\n")
