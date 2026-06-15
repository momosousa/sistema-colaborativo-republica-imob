"""Definição do State do LangGraph.

Toda a coordenação entre os dois moradores (Morador 1 e Morador 2) vive
neste TypedDict. O Streamlit injeta `current_user` a cada `invoke` e o
grafo lê/escreve os demais campos para implementar o fluxo colaborativo
de construção do Dossiê de Dúvidas.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


Intent = Literal[
    "pergunta_rag",
    "chat_geral",
    "proposta",
    "aprovacao",
    "rejeicao",
    "finalizar",
]


CategoriaItem = Literal[
    "aluguel",
    "iptu",
    "seguro_incendio",
    "manutencao",
    "caucao",
    "fiador",
    "multa_rescisoria",
    "outros",
]


Usuario = Literal["Morador 1", "Morador 2"]


class ItemDossie(TypedDict, total=False):
    """Um item aprovado (ou pendente) do Dossiê de Dúvidas.

    `aprovador` é opcional porque um item pendente ainda não tem aprovador.
    `fonte_pdf` também é opcional para propostas não baseadas em RAG.
    """

    descricao: str
    categoria: CategoriaItem
    fonte_pdf: Optional[str]
    proponente: str
    aprovador: Optional[str]


class GraphState(TypedDict, total=False):
    """Estado completo do grafo de coordenação dos moradores."""

    messages: Annotated[list[BaseMessage], add_messages]

    current_user: Usuario

    intent: Optional[Intent]

    rag_context: Optional[str]
    rag_sources: Optional[list[str]]

    item_pendente: Optional[ItemDossie]
    proponente: Optional[str]

    dossie_final: list[ItemDossie]

    erro_coordenacao: Optional[str]


def estado_inicial() -> GraphState:
    """Retorna um State zerado, útil para o primeiro `invoke` da thread."""
    return {
        "messages": [],
        "current_user": "Morador 1",
        "intent": None,
        "rag_context": None,
        "rag_sources": None,
        "item_pendente": None,
        "proponente": None,
        "dossie_final": [],
        "erro_coordenacao": None,
    }
