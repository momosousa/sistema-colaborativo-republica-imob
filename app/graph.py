"""Montagem do StateGraph + checkpointer (padrão ReAct para RAG).

Single graph, single thread, dois "atores humanos" multiplexados pelo Streamlit
via `current_user`. O `MemorySaver` persiste o State entre turnos da mesma thread.

Topologia:

- START → classificador_node (LLM extrai intent)
- classificador_node → { agente_responde | proposta_node | analisa_consenso | finalizacao_node }
- analisa_consenso → agente_responde   (sempre — o agente comunica o resultado)
- agente_responde → tool_node          (se o LLM emitiu tool_calls)
- tool_node       → agente_responde    (loop ReAct: agente formula resposta com o resultado da tool)
- agente_responde → END                (se não houve tool_calls)
- proposta_node   → END
- finalizacao_node → END

Para trocar por persistência durável (sobrevive a reinício do Streamlit),
basta substituir `MemorySaver()` por `SqliteSaver.from_conn_string(...)`.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .nodes import (
    agente_responde,
    analisa_consenso,
    classificador_node,
    finalizacao_node,
    proposta_node,
)
from .rag import consultar_documentos_imobiliaria
from .router import route_after_agent, route_by_intent
from .state import GraphState


def construir_grafo():
    """Constrói o StateGraph compilado seguindo o padrão ReAct."""
    builder = StateGraph(GraphState)

    builder.add_node("classificador_node", classificador_node)
    builder.add_node("agente_responde", agente_responde)
    builder.add_node("tool_node", ToolNode([consultar_documentos_imobiliaria]))
    builder.add_node("proposta_node", proposta_node)
    builder.add_node("analisa_consenso", analisa_consenso)
    builder.add_node("finalizacao_node", finalizacao_node)

    builder.add_edge(START, "classificador_node")
    builder.add_edge("proposta_node", END)
    builder.add_edge("finalizacao_node", END)
    builder.add_edge("analisa_consenso", "agente_responde")
    builder.add_edge("tool_node", "agente_responde")

    builder.add_conditional_edges(
        "classificador_node",
        route_by_intent,
        {
            "agente_responde": "agente_responde",
            "proposta_node": "proposta_node",
            "analisa_consenso": "analisa_consenso",
            "finalizacao_node": "finalizacao_node",
        },
    )

    builder.add_conditional_edges(
        "agente_responde",
        route_after_agent,
        {"tool_node": "tool_node", END: END},
    )

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def get_graph():
    """Retorna o grafo compilado (singleton compartilhado entre reruns do Streamlit)."""
    return construir_grafo()
