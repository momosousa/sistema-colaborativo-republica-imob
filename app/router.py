"""Funções de roteamento condicional do grafo.

São apenas duas, conforme a arquitetura ReAct:

- `route_by_intent`: após o `classificador_node`, decide qual ramo executar.
  As intenções `pergunta_rag` e `chat_geral` vão direto para o `agente_responde`
  (o LLM decide se invoca a tool ou não); `aprovacao`/`rejeicao` passam pelo
  `analisa_consenso` antes de voltarem ao agente para gerar a resposta.

- `route_after_agent`: padrão ReAct — se o LLM emitiu `tool_calls`, vamos
  para o `tool_node`; senão, o turno termina.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.graph import END

from .state import GraphState


def route_by_intent(state: GraphState) -> str:
    """Mapeia `state['intent']` para o nó correspondente."""
    intent = state.get("intent")
    if intent in ("pergunta_rag", "chat_geral"):
        return "agente_responde"
    if intent == "proposta":
        return "proposta_node"
    if intent in ("aprovacao", "rejeicao"):
        return "analisa_consenso"
    if intent == "finalizar":
        return "finalizacao_node"
    return "agente_responde"


def route_after_agent(state: GraphState) -> str:
    """Padrão ReAct: se a última msg do LLM tem tool_calls → executa a tool.

    Caso contrário, encerra o turno — o painel do Streamlit já refletiu
    as mudanças no `dossie_final`/`item_pendente`.
    """
    messages = state.get("messages", [])
    if not messages:
        return END
    ultima = messages[-1]
    if isinstance(ultima, AIMessage) and getattr(ultima, "tool_calls", None):
        return "tool_node"
    return END
