"""Utilitários para exibir o diagrama do grafo (padrão ReAct).

Mantém o diagrama ASCII canônico e um Mermaid estático equivalentes ao
grafo compilado em `app/graph.py`. `mermaid_dinamico()` extrai o Mermaid
direto do grafo já compilado — útil para a sidebar do Streamlit confirmar
que código e arquitetura estão sincronizados.
"""

from __future__ import annotations


ASCII_DIAGRAM = r"""
                                ┌───────────────┐
                                │     START     │
                                └───────┬───────┘
                                        │
                                        ▼
                          ┌─────────────────────────────┐
                          │     classificador_node      │
                          │  (LLM → intent ∈ 6 valores) │
                          └──────────────┬──────────────┘
                                         │
        ┌──────────────────┬─────────────┼─────────────────┬─────────────┐
        │                  │             │                 │             │
        ▼                  ▼             ▼                 ▼             ▼
 pergunta_rag /        proposta    aprovacao /         finalizar     (default)
   chat_geral             │         rejeicao              │             │
        │                 ▼             │                 ▼             │
        │         ┌──────────────┐      │         ┌─────────────────┐   │
        │         │ proposta_    │      │         │ finalizacao_    │   │
        │         │ node         │      │         │ node            │   │
        │         │ (extrai item │      │         │ (gera dossiê    │   │
        │         │  → pendente) │      │         │  formal)        │   │
        │         └──────┬───────┘      │         └────────┬────────┘   │
        │                │              ▼                  │            │
        │                │      ┌──────────────────┐       │            │
        │                │      │ analisa_consenso │       │            │
        │                │      │ (anti-autovoto;  │       │            │
        │                │      │  muta dossiê)    │       │            │
        │                │      └────────┬─────────┘       │            │
        │                │               │                 │            │
        ▼                │               ▼                 │            ▼
 ┌───────────────────────┴──────────────────────────────────────────────────┐
 │                            agente_responde                               │
 │  (LLM + bind_tools([consultar_documentos_imobiliaria]))                  │
 │  Decide via tool_calls se invoca a ferramenta ou responde direto.        │
 └────────────┬──────────────────────────────────────────────────┬──────────┘
              │                                                  │
       sem tool_calls                                     com tool_calls
              │                                                  │
              ▼                                                  ▼
      ┌──────────────┐                                ┌────────────────────┐
      │     END      │                                │     tool_node      │
      └──────────────┘                                │ consultar_         │
              ▲                                       │ documentos_        │
              │                                       │ imobiliaria        │
              │                                       └─────────┬──────────┘
              │                                                 │
              └─────────────── loop ReAct ──────────────────────┘
                       (ToolMessage volta ao agente_responde)
"""


MERMAID_DIAGRAM = """
flowchart TD
    Start([START]) --> Classif[classificador_node<br/>LLM extrai intent]

    Classif -->|"pergunta_rag | chat_geral"| Agente[agente_responde<br/>LLM + bind_tools]
    Classif -->|proposta| Proposta[proposta_node<br/>seta item_pendente + proponente]
    Classif -->|"aprovacao | rejeicao"| Consenso[analisa_consenso<br/>regra anti-autovoto]
    Classif -->|finalizar| Final[finalizacao_node<br/>dossiê formal]

    Consenso --> Agente

    Agente -->|tool_calls| Tool[tool_node<br/>consultar_documentos_imobiliaria]
    Tool --> Agente
    Agente -->|sem tool_calls| Fim([END])

    Proposta --> Fim
    Final --> Fim
"""


def mermaid_dinamico() -> str:
    """Extrai o Mermaid do grafo já compilado (validação ao vivo do código)."""
    from .graph import get_graph

    return get_graph().get_graph().draw_mermaid()


if __name__ == "__main__":
    print(ASCII_DIAGRAM)
    print("\n--- Mermaid (estático) ---")
    print(MERMAID_DIAGRAM)
    print("\n--- Mermaid (dinâmico, extraído do grafo compilado) ---")
    try:
        print(mermaid_dinamico())
    except Exception as exc:
        print(f"(não foi possível extrair dinamicamente: {exc})")
