"""Sentinel Mesh — autonomous, agentic security operations for the mid-market.

Layer map (PRD Section 5.2, Figure 2)::

    1. Sources        sentinel.ingest.replay        alert/log replay service
    2. Ingestion      sentinel.ingest.bus           event bus
                      sentinel.ingest.normalizer    heterogeneous -> canonical Alert
    3. Intelligence   sentinel.ml.*                 anomaly ensemble, GNN, bandit, RAG
    4. Orchestration  sentinel.agents.*             LangGraph + 5 agents      [Part 3]
    5. Action         sentinel.connectors.*         EDR/firewall/git/slack    [Part 4]
    6. Oversight      sentinel.audit.*              hash-chained audit log
                      web/                          Analyst Copilot dashboard [Part 5]
"""

__version__ = "0.1.0"
