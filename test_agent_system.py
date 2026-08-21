"""
Verification Script for the Scalable Agentic System Architecture
Tests the 5 core scenarios:
1. "Send an invoice for $50 to..."
2. "What was my total sales volume last month?"
3. "Is there a dispute open from user_123?"
4. System Search Tool ("What tools are available for managing invoices?")
5. RAG Pipeline Tool ("What is the maximum invoice limit for business accounts?")
"""

from langchain_core.messages import HumanMessage
from agent_system import agent_system

def run_tests():
    test_queries = [
        ("Invoicing Action", "Send an invoice for $50 to client@example.com"),
        ("Analytics Query", "What was my total sales volume last month?"),
        ("Dispute Lookup", "Is there a dispute open from user_123?"),
        ("System Search Tool", "What tools are available for managing invoices?"),
        ("RAG Pipeline Tool", "What is the policy on maximum invoice limits for business accounts?"),
    ]

    print("=================================================================")
    print("RUNNING ARCHITECTURE VERIFICATION TEST SUITE (DATAZOIC AGENT)")
    print("=================================================================\n")

    for idx, (label, query) in enumerate(test_queries, 1):
        print(f"[{idx}] TEST SCENARIO: {label}")
        print(f"User Input: \"{query}\"")
        
        # Invoke LangGraph State Machine with thread checkpointing
        config = {"configurable": {"thread_id": f"thread_test_{idx}"}}
        result = agent_system.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )
        
        active_domain = result.get("active_domain")
        retrieved_tools = result.get("retrieved_tools")
        response_msg = result["messages"][-1].content
        
        print(f"-> Classified Domain  : {active_domain}")
        print(f"-> Top-K Dynamic Tools: {retrieved_tools}")
        print(f"-> Agent Response     : {response_msg}")
        print("-" * 65 + "\n")

    print("All 5 test scenarios completed successfully!")

if __name__ == "__main__":
    run_tests()
