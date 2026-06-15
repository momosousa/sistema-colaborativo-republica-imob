"""UI Streamlit do Sistema Colaborativo da República (entrypoint).

Rode com: `streamlit run app.py`

Layout:
- Sidebar: uploader (até 5 PDFs) + radio "Quem está falando?" + botões
  "Gerar Dossiê Final" / "Nova sessão" + expander com o diagrama.
- Tela principal: 2 colunas (st.columns([2, 1]))
    Esquerda  → Chat (histórico + input)
    Direita   → Dossiê de Dúvidas em tempo real (lido do State do grafo)

Integração com o LangGraph: cada submit faz `graph.invoke({...},
config={thread_id})`. O snapshot do State (`graph.get_state(...)`) alimenta
o painel do dossiê e o item pendente.

Sobre `app.py` vs pacote `app/`: o Python lida bem com a coexistência —
`streamlit run app.py` faz esse arquivo ser `__main__`, e `from app.X import Y`
resolve para o pacote (entradas distintas em `sys.modules`).
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.diagram import MERMAID_DIAGRAM
from app.graph import get_graph
from app.rag import MAX_PDFS, set_pdf_sources


load_dotenv()

st.set_page_config(
    page_title="República — Dossiê Colaborativo",
    page_icon=":house:",
    layout="wide",
)

TEMP_UPLOAD_BASE = Path(tempfile.gettempdir()) / "republica_uploads"


def _inicializar_sessao() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"republica-{uuid.uuid4()}"
        st.session_state.upload_dir = TEMP_UPLOAD_BASE / st.session_state.thread_id
        st.session_state.upload_dir.mkdir(parents=True, exist_ok=True)
        st.session_state.uploaded_names = []
    if "current_user" not in st.session_state:
        st.session_state.current_user = "Morador 1"


def _config() -> dict[str, Any]:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def _snapshot() -> dict[str, Any]:
    state = get_graph().get_state(_config())
    return state.values if state else {}


def _processar_uploads(uploaded_files) -> None:
    """Salva os PDFs uploaded no tempdir da sessão e alimenta o retriever.

    Idempotente: se a lista de nomes não mudou, não reindexa.
    """
    if not uploaded_files:
        return

    nomes_atuais = sorted(f.name for f in uploaded_files)
    if nomes_atuais == st.session_state.uploaded_names:
        return

    if len(uploaded_files) > MAX_PDFS:
        st.sidebar.error(f"Envie no máximo {MAX_PDFS} PDFs.")
        return

    for antigo in st.session_state.upload_dir.glob("*.pdf"):
        antigo.unlink()

    paths: list[Path] = []
    for f in uploaded_files:
        destino = st.session_state.upload_dir / f.name
        destino.write_bytes(f.getbuffer())
        paths.append(destino)

    with st.sidebar:
        with st.spinner(f"Indexando {len(paths)} PDF(s)…"):
            set_pdf_sources(paths)
    st.session_state.uploaded_names = nomes_atuais
    st.sidebar.success(f"{len(paths)} PDF(s) indexado(s).")


def _enviar_mensagem(texto: str) -> None:
    get_graph().invoke(
        {
            "messages": [
                HumanMessage(
                    content=texto,
                    additional_kwargs={"name": st.session_state.current_user},
                )
            ],
            "current_user": st.session_state.current_user,
        },
        config=_config(),
    )
    st.rerun()


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Configuração")

        uploaded = st.file_uploader(
            "Documentos da imobiliária",
            type="pdf",
            accept_multiple_files=True,
            help=f"Até {MAX_PDFS} arquivos PDF (contrato, IPTU, seguro etc.)",
        )
        _processar_uploads(uploaded)

        st.divider()
        st.session_state.current_user = st.radio(
            "Quem está falando?",
            options=["Morador 1", "Morador 2"],
            index=0 if st.session_state.current_user == "Morador 1" else 1,
            horizontal=True,
        )

        st.divider()
        if st.button("Gerar Dossiê Final", type="primary", use_container_width=True):
            _enviar_mensagem(
                "Por favor, gere o dossiê final para enviarmos à imobiliária."
            )
        if st.button("Nova sessão (reset)", use_container_width=True):
            for chave in list(st.session_state.keys()):
                del st.session_state[chave]
            set_pdf_sources(None)
            st.rerun()

        st.caption(f"thread_id: `{st.session_state.thread_id}`")
        with st.expander("Arquitetura do grafo"):
            st.markdown(f"```mermaid\n{MERMAID_DIAGRAM}\n```")


def _avatar(autor: str) -> str:
    return {"Morador 1": "👨", "Morador 2": "👩"}.get(autor, "🤖")


def _render_chat(messages: list) -> None:
    for msg in messages:
        if isinstance(msg, HumanMessage):
            autor = msg.additional_kwargs.get("name") or "Usuário"
            with st.chat_message("user", avatar=_avatar(autor)):
                st.markdown(f"**{autor}:** {msg.content}")
        elif isinstance(msg, AIMessage):
            if not msg.content and getattr(msg, "tool_calls", None):
                with st.chat_message("assistant", avatar="🔍"):
                    nomes = ", ".join(tc["name"] for tc in msg.tool_calls)
                    st.caption(f"Consultando documentos via `{nomes}`…")
            elif msg.content:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg.content)
        elif isinstance(msg, ToolMessage):
            continue


def _render_dossie(snapshot: dict[str, Any]) -> None:
    st.subheader("Dossiê de Dúvidas")
    st.caption("Atualizado em tempo real conforme os moradores votam.")

    pendente = snapshot.get("item_pendente")
    if pendente:
        st.warning(
            f"**Item pendente** — _proposto por {snapshot.get('proponente')}_\n\n"
            f"**[{pendente.get('categoria')}]** {pendente.get('descricao')}\n\n"
            "Aguardando aprovação do outro morador."
        )

    dossie = snapshot.get("dossie_final", []) or []
    if not dossie:
        st.info("O dossiê ainda está vazio. Proponham itens no chat.")
        return

    for i, item in enumerate(dossie, start=1):
        with st.container(border=True):
            st.markdown(f"**{i}. [{item.get('categoria')}]** {item.get('descricao')}")
            rodape = (
                f"Proposto por {item.get('proponente')} • "
                f"Aprovado por {item.get('aprovador')}"
            )
            if item.get("fonte_pdf"):
                rodape += f" • Fonte: {item.get('fonte_pdf')}"
            st.caption(rodape)


def _validar_google_key() -> bool:
    if not os.getenv("GOOGLE_API_KEY"):
        st.error(
            "Defina a variável `GOOGLE_API_KEY` no `.env`. "
            "Pegue uma chave gratuita em https://aistudio.google.com/app/apikey."
        )
        return False
    return True


def main() -> None:
    st.title("Sistema Colaborativo da República")
    st.caption(
        "Dois moradores analisando contratos (aluguel, IPTU, seguro incêndio) "
        "para montar juntos um Dossiê de Dúvidas para a imobiliária."
    )
    if not _validar_google_key():
        st.stop()

    _inicializar_sessao()
    _render_sidebar()
    snapshot = _snapshot()

    col_chat, col_dossie = st.columns([2, 1], gap="large")

    with col_chat:
        st.subheader("Conversa")
        _render_chat(snapshot.get("messages", []) or [])
        prompt = st.chat_input(f"Digite como {st.session_state.current_user}…")
        if prompt:
            _enviar_mensagem(prompt)

    with col_dossie:
        _render_dossie(snapshot)


if __name__ == "__main__":
    main()
