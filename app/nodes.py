"""Implementação dos 6 nós do grafo (padrão ReAct para RAG).

Nós:
- classificador_node      → classifica a intenção do current_user em 6 categorias
- agente_responde         → LLM com bind_tools([consultar_documentos_imobiliaria])
- proposta_node           → extrai item proposto e marca como pendente
- analisa_consenso        → processa aprovação/rejeição (regra anti-autovoto)
- finalizacao_node        → gera o documento formal do dossiê

(O nó `tool_node` é injetado direto em graph.py via ToolNode prebuilt.)

Todos os nós seguem `def node(state: GraphState) -> dict`, retornando
apenas as chaves do State que precisam ser atualizadas (LangGraph faz o
merge usando os reducers — em particular `add_messages` para o histórico).
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from .rag import consultar_documentos_imobiliaria
from .state import CategoriaItem, GraphState, Intent, ItemDossie


CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "gemini-2.5-flash-lite")
EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "gemini-2.5-flash")


def _chat_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)


def _classifier_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=CLASSIFIER_MODEL, temperature=0)


def _extractor_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=EXTRACTOR_MODEL, temperature=0)


def _ultima_humana(messages: list[BaseMessage]) -> Optional[HumanMessage]:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    return None


def _renderizar_dossie(dossie: list[ItemDossie]) -> str:
    if not dossie:
        return "(Dossiê vazio)"
    linhas = []
    for i, item in enumerate(dossie, start=1):
        linhas.append(
            f"{i}. [{item.get('categoria', 'outros')}] {item.get('descricao', '')}"
            f" — proposto por {item.get('proponente')}, aprovado por {item.get('aprovador')}"
        )
    return "\n".join(linhas)


class IntentDecision(BaseModel):
    """Saída estruturada do classificador de intenção."""

    intent: Intent = Field(
        description=(
            "Classifique a última mensagem do usuário em UMA destas categorias:\n"
            "- pergunta_rag: pergunta sobre o contrato/IPTU/seguro que requer consulta aos PDFs.\n"
            "- chat_geral: conversa casual, esclarecimento sem precisar dos PDFs.\n"
            "- proposta: o usuário sugere adicionar algo ao Dossiê de Dúvidas.\n"
            "- aprovacao: o usuário concorda com um item_pendente (ex: 'aprovo', 'concordo', 'pode adicionar').\n"
            "- rejeicao: o usuário recusa o item_pendente (ex: 'rejeito', 'não concordo', 'descarta').\n"
            "- finalizar: o usuário quer gerar o documento final do dossiê para a imobiliária."
        )
    )


def classificador_node(state: GraphState) -> dict:
    """Nó de entrada: classifica a intenção do `current_user` em uma das 6 categorias."""
    ultima = _ultima_humana(state.get("messages", []))
    if ultima is None:
        return {"intent": "chat_geral"}

    item_pendente = state.get("item_pendente")
    current_user = state.get("current_user", "Morador 1")

    contexto_pendente = ""
    if item_pendente:
        contexto_pendente = (
            f"\n\nATENÇÃO: existe um item PENDENTE proposto por "
            f"{state.get('proponente')}: '{item_pendente.get('descricao')}' "
            f"(categoria: {item_pendente.get('categoria')}).\n"
            f"Como o usuário atual é {current_user}, priorize classificar como "
            "'aprovacao' ou 'rejeicao' se a mensagem indicar concordância ou recusa."
        )

    sistema = SystemMessage(
        content=(
            "Você é o roteador de intenções de um sistema colaborativo onde dois "
            "moradores discutem contratos de aluguel para montar um dossiê de dúvidas. "
            "Classifique a mensagem em UMA das 6 intenções disponíveis."
            + contexto_pendente
        )
    )

    llm = _classifier_llm().with_structured_output(IntentDecision)
    decisao: IntentDecision = llm.invoke([sistema, ultima])
    return {"intent": decisao.intent}


def agente_responde(state: GraphState) -> dict:
    """Nó do agente: LLM com tool-calling (padrão ReAct).

    O LLM decide autonomamente se invoca `consultar_documentos_imobiliaria`
    (quando a pergunta exige dados dos PDFs) ou se responde diretamente.
    Se invocar, o `tool_node` (em graph.py) executa a tool e volta para
    cá — então o LLM formula a resposta final ao usuário.
    """
    messages = list(state.get("messages", []))
    current_user = state.get("current_user", "Morador 1")
    erro = state.get("erro_coordenacao")
    item_pendente = state.get("item_pendente")
    proponente = state.get("proponente")
    dossie = state.get("dossie_final", [])

    blocos_sistema = [
        "Você é o Agente da República — assistente IA que ajuda DOIS moradores "
        "(Morador 1 e Morador 2) a entenderem contratos de aluguel, IPTU e "
        "seguro incêndio, e a montarem juntos um Dossiê de Dúvidas para a "
        "imobiliária.",
        f"O usuário falando agora é: {current_user}.",
        f"Estado atual do Dossiê:\n{_renderizar_dossie(dossie)}",
        "FERRAMENTA DISPONÍVEL: consultar_documentos_imobiliaria(query). "
        "Use-a SEMPRE que a pergunta exigir valores, prazos, cláusulas, taxas "
        "ou qualquer fato presente nos PDFs do contrato. Para conversa casual "
        "ou para confirmar uma ação que já aconteceu (ex: aprovação registrada), "
        "responda direto sem invocar a ferramenta.",
    ]

    if item_pendente:
        blocos_sistema.append(
            f"Há um item PENDENTE de aprovação, proposto por {proponente}: "
            f"'{item_pendente.get('descricao')}' "
            f"(categoria: {item_pendente.get('categoria')}). "
            "Quem precisa aprovar é o outro morador."
        )

    if erro:
        blocos_sistema.append(
            f"IMPORTANTE — Erro de coordenação detectado no passo anterior: {erro}\n"
            "Explique gentilmente a situação ao usuário antes de continuar."
        )

    sistema = SystemMessage(content="\n\n".join(blocos_sistema))
    llm_com_tools = _chat_llm().bind_tools([consultar_documentos_imobiliaria])
    resposta: AIMessage = llm_com_tools.invoke([sistema] + messages)

    return {
        "messages": [resposta],
        "erro_coordenacao": None,
    }


class PropostaExtraida(BaseModel):
    """Estrutura usada para extrair a proposta de item do Dossiê."""

    descricao: str = Field(
        description="Descrição clara e objetiva do item/dúvida a ser enviada à imobiliária."
    )
    categoria: CategoriaItem = Field(
        description="Categoria do item, entre: aluguel, iptu, seguro_incendio, "
        "manutencao, caucao, fiador, multa_rescisoria, outros."
    )


def proposta_node(state: GraphState) -> dict:
    """Nó de proposta: extrai o item, marca como pendente e pede aprovação."""
    ultima = _ultima_humana(state.get("messages", []))
    current_user = state.get("current_user", "Morador 1")
    outro = "Morador 2" if current_user == "Morador 1" else "Morador 1"

    if ultima is None:
        return {}

    sistema = SystemMessage(
        content=(
            "Extraia da fala do morador a proposta de item para o Dossiê de Dúvidas. "
            "Seja conciso, em linguagem formal apropriada para uma imobiliária. "
            "Escolha a categoria mais apropriada."
        )
    )
    llm = _extractor_llm().with_structured_output(PropostaExtraida)
    extraido: PropostaExtraida = llm.invoke([sistema, ultima])

    novo_pendente: ItemDossie = {
        "descricao": extraido.descricao,
        "categoria": extraido.categoria,
        "fonte_pdf": None,
        "proponente": current_user,
        "aprovador": None,
    }

    mensagem = AIMessage(
        content=(
            f"{current_user} sugeriu adicionar ao Dossiê:\n\n"
            f"> **[{extraido.categoria}]** {extraido.descricao}\n\n"
            f"{outro}, você aprova esse item? Responda **'Aprovo'** para incluir "
            f"ou **'Rejeito'** para descartar."
        )
    )

    return {
        "item_pendente": novo_pendente,
        "proponente": current_user,
        "messages": [mensagem],
    }


def analisa_consenso(state: GraphState) -> dict:
    """Nó de coordenação: processa aprovação OU rejeição do item pendente.

    Aplica a regra anti-autovoto: se `proponente == current_user`, registra
    `erro_coordenacao` e devolve sem mutar o dossiê. Caso contrário, muta
    o State (`dossie_final` cresce ou `item_pendente` é descartado) e
    devolve controle ao `agente_responde`, que comunica o resultado ao
    usuário em linguagem natural.

    Este nó **não escreve em `messages`** de propósito — quem fala é o
    agente_responde no próximo passo.
    """
    intent = state.get("intent")
    current_user = state.get("current_user", "Morador 1")
    proponente = state.get("proponente")
    item = state.get("item_pendente")
    eh_aprovacao = intent == "aprovacao"
    acao = "aprovar" if eh_aprovacao else "rejeitar"

    if item is None or proponente is None:
        return {"erro_coordenacao": f"Não há item pendente para {acao}."}

    if proponente == current_user:
        return {
            "erro_coordenacao": (
                f"{current_user}, você não pode {acao} sua própria proposta. "
                "Aguarde o outro morador."
            )
        }

    if eh_aprovacao:
        novo_item: ItemDossie = {**item, "aprovador": current_user}
        return {
            "dossie_final": list(state.get("dossie_final", [])) + [novo_item],
            "item_pendente": None,
            "proponente": None,
        }

    return {"item_pendente": None, "proponente": None}


def finalizacao_node(state: GraphState) -> dict:
    """Nó de finalização: gera o documento formal do Dossiê para a imobiliária."""
    dossie = state.get("dossie_final", [])

    if not dossie:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "O Dossiê ainda está vazio. Discutam os pontos do contrato "
                        "e proponham itens antes de finalizar."
                    )
                )
            ]
        }

    por_categoria: dict[str, list[ItemDossie]] = {}
    for item in dossie:
        cat = item.get("categoria", "outros")
        por_categoria.setdefault(cat, []).append(item)

    linhas_brutas = ["=== ITENS DO DOSSIÊ (agrupados por categoria) ==="]
    for cat, itens in por_categoria.items():
        linhas_brutas.append(f"\n## {cat.upper()}")
        for item in itens:
            fonte = item.get("fonte_pdf") or "—"
            linhas_brutas.append(
                f"- {item.get('descricao')} "
                f"(proposto por {item.get('proponente')}, "
                f"aprovado por {item.get('aprovador')}, fonte: {fonte})"
            )
    dossie_bruto = "\n".join(linhas_brutas)

    sistema = SystemMessage(
        content=(
            "Você é o redator final do Dossiê de Dúvidas que dois moradores "
            "vão enviar para a imobiliária. Reorganize os itens abaixo em um "
            "documento formal, agrupado por categoria, com tom cortês e objetivo. "
            "Inclua uma introdução curta e numere as perguntas. Não invente itens; "
            "use apenas o que está listado."
        )
    )
    humano = HumanMessage(content=dossie_bruto)
    resposta: AIMessage = _chat_llm().invoke([sistema, humano])

    return {"messages": [resposta]}
