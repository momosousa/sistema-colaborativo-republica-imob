"""Camada de RAG: ingestão de PDFs e retriever in-memory sobre os contratos.

Backend: FAISS in-memory + pypdf.PdfReader. Sem persistência em disco —
o vectorstore vive enquanto o processo Streamlit estiver de pé.

API pública:
    - construir_vectorstore(pdf_sources=None, max_files=5)
        Constrói o FAISS. Se pdf_sources for None, auto-descobre em docs/.
    - consultar_contexto(query) -> (texto_formatado, fontes)
        Helper interno usado pelo rag_node do LangGraph.
    - consultar_documentos_imobiliaria  (@tool do LangChain)
        Wrapper sobre consultar_contexto, pronto para uso por agentes.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

MAX_PDFS = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

_PDF_SOURCES_OVERRIDE: Optional[list[Path]] = None


def _extrair_texto_pdf(path: Path) -> list[Document]:
    """Extrai o texto de um PDF página a página usando pypdf.

    Páginas em branco (sem texto extraível) são descartadas para evitar
    poluir o vectorstore com chunks vazios.
    """
    reader = PdfReader(str(path))
    docs: list[Document] = []
    for i, page in enumerate(reader.pages):
        texto = (page.extract_text() or "").strip()
        if not texto:
            continue
        docs.append(
            Document(
                page_content=texto,
                metadata={"source": path.name, "page": i},
            )
        )
    return docs


def _carregar_pdfs(pdf_sources: list[Path]) -> list[Document]:
    documentos: list[Document] = []
    for pdf in pdf_sources:
        documentos.extend(_extrair_texto_pdf(pdf))
    return documentos


def _dividir(documentos: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documentos)


def _embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def construir_vectorstore(
    pdf_sources: Optional[list[Path]] = None,
    max_files: int = MAX_PDFS,
) -> Optional[FAISS]:
    """Constrói o vectorstore FAISS in-memory a partir dos PDFs.

    Args:
        pdf_sources: Lista de caminhos de PDFs. Se None, auto-descobre os
            primeiros `max_files` PDFs (em ordem alfabética) em `docs/`.
        max_files: Limite máximo de PDFs aceitos (default: 5).

    Returns:
        FAISS pronto para uso, ou None se não houver PDFs / texto extraível.

    Raises:
        ValueError: se `pdf_sources` for explicitamente passado com mais
            de `max_files` arquivos.
    """
    if pdf_sources is None:
        pdf_sources = sorted(DOCS_DIR.glob("*.pdf"))[:max_files]
    else:
        if len(pdf_sources) > max_files:
            raise ValueError(
                f"Máximo de {max_files} PDFs permitidos "
                f"(recebido {len(pdf_sources)})."
            )

    if not pdf_sources:
        return None

    documentos = _carregar_pdfs(pdf_sources)
    if not documentos:
        return None

    chunks = _dividir(documentos)
    return FAISS.from_documents(chunks, _embeddings())


@lru_cache(maxsize=1)
def get_retriever() -> Optional[VectorStoreRetriever]:
    """Retorna um retriever singleton sobre o FAISS in-memory.

    Se `set_pdf_sources(...)` foi chamado, usa essa lista. Caso contrário,
    auto-descobre os PDFs em `docs/`. Retorna None se não houver fontes.
    Use `reset_retriever_cache()` para forçar reconstrução.
    """
    vs = construir_vectorstore(pdf_sources=_PDF_SOURCES_OVERRIDE)
    if vs is None:
        return None
    return vs.as_retriever(search_kwargs={"k": TOP_K})


def consultar_contexto(query: str) -> tuple[str, list[str]]:
    """Executa a busca semântica e devolve (contexto_concatenado, fontes únicas).

    O contexto vem formatado com marcadores `[fonte: arquivo.pdf, pag. N]`
    para que o LLM consumidor saiba citar.
    """
    retriever = get_retriever()
    if retriever is None:
        return ("(Nenhum PDF indexado em docs/.)", [])

    docs: list[Document] = retriever.invoke(query)
    if not docs:
        return ("", [])

    blocos: list[str] = []
    fontes: list[str] = []
    fontes_vistas: set[str] = set()

    for doc in docs:
        fonte_nome = doc.metadata.get("source", "desconhecido")
        pagina = doc.metadata.get("page")
        marcador = f"[fonte: {fonte_nome}"
        if pagina is not None:
            marcador += f", pag. {int(pagina) + 1}"
        marcador += "]"
        blocos.append(f"{marcador}\n{doc.page_content.strip()}")

        if fonte_nome not in fontes_vistas:
            fontes.append(fonte_nome)
            fontes_vistas.add(fonte_nome)

    contexto = "\n\n---\n\n".join(blocos)
    return contexto, fontes


@tool("consultar_documentos_imobiliaria")
def consultar_documentos_imobiliaria(query: str) -> str:
    """Consulta os documentos da imobiliária (contrato de aluguel, IPTU,
    seguro incêndio, condomínio etc.) e retorna trechos relevantes em
    linguagem natural, com marcadores de fonte no formato
    `[fonte: arquivo.pdf, pag. N]`.

    Use esta ferramenta sempre que precisar de informação factual sobre
    valores, prazos, cláusulas, taxas, multas ou regras presentes no
    contrato. A `query` deve ser uma pergunta em linguagem natural sobre
    o conteúdo dos PDFs (ex: "Qual o valor do aluguel?", "Como funciona
    o reajuste anual?", "Existe multa por rescisão antecipada?").
    """
    contexto, _fontes = consultar_contexto(query)
    return contexto


def reset_retriever_cache() -> None:
    """Limpa o cache do retriever (útil após adicionar novos PDFs em docs/)."""
    get_retriever.cache_clear()


def set_pdf_sources(paths: Optional[list[Path]]) -> None:
    """Define os PDFs do singleton (sobrescreve a auto-descoberta em docs/).

    Passe `None` para voltar ao auto-discovery em `docs/`. Limpa o cache do
    retriever para que a próxima consulta reindexe com as novas fontes.

    Usado pela UI Streamlit quando o usuário faz upload de PDFs.
    """
    global _PDF_SOURCES_OVERRIDE
    _PDF_SOURCES_OVERRIDE = paths
    reset_retriever_cache()
