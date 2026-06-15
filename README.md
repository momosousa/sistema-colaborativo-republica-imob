# Sistema de Análise de Contratos para Repúblicas

Dois estudantes que vão dividir uma república analisam juntos os contratos
da imobiliária (aluguel, IPTU, seguro incêndio) com a ajuda de um agente IA
e, ao final, montam de forma colaborativa um **Dossiê de Dúvidas** para
enviar à imobiliária.

**Stack**: LangGraph (orquestração) + RAG com FAISS in-memory e Google Gemini
(consulta aos PDFs) + Streamlit (UI).

---

## Visão geral

- **Cenário**: dois moradores (Morador 1 e Morador 2) compartilham um chat e
  decidem juntos o que perguntar à imobiliária.
- **Agente IA**: lê os PDFs do contrato via RAG (FAISS + Google Gemini) e
  responde dúvidas factuais com citação de fontes.
- **3C na prática**: **C**omunicação no chat compartilhado; **C**olaboração
  via RAG conjunto; **C**oordenação por regra anti-autovoto — nenhum item
  entra no dossiê sem o aval do outro morador.
- **Como rodar**: `pip install -r requirements.txt` → `streamlit run app.py`.

## Sumário

- [Visão geral em 30 segundos](#visão-geral-em-30-segundos)
- [Modelo 3C aplicado](#modelo-3c-aplicado)
- [Diagrama do grafo](#diagrama-do-grafo)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Execução](#execução)
- [Fluxo de uso (4 passos)](#fluxo-de-uso-4-passos)
- [Variáveis de ambiente](#variáveis-de-ambiente)
- [Publicação no GitHub](#publicação-no-github)
- [Notas técnicas](#notas-técnicas)

---

## Modelo 3C aplicado

O sistema foi desenhado em cima dos três pilares clássicos de groupware
(Comunicação, Colaboração, Coordenação). Cada pilar tem uma implementação
concreta e identificável no código.

### Comunicação — chat compartilhado entre os dois moradores

- O grafo roda numa **thread única** no LangGraph com `MemorySaver`
  ([`app/graph.py`](app/graph.py)). Ambos os moradores conversam no mesmo
  histórico — o `messages` do `GraphState` ([`app/state.py`](app/state.py))
  usa o reducer `add_messages` e acumula tudo.
- A UI ([`app.py`](app.py)) tem um `st.radio("Quem está falando?")` na
  sidebar que define `st.session_state.current_user`. A cada submit do
  chat, esse valor é injetado de duas formas:
  1. no campo `current_user` do State (que os nós leem para decidir);
  2. em `additional_kwargs={"name": current_user}` da `HumanMessage`
     (que a UI lê para renderizar o avatar correto no histórico).
- Resultado: Morador 1 vê o que Morador 2 falou, e vice-versa, em tempo
  real, na mesma tela.

### Colaboração — agente IA com RAG ajudando ambos

- O nó `agente_responde` ([`app/nodes.py`](app/nodes.py)) usa o padrão
  **ReAct**: `llm.bind_tools([consultar_documentos_imobiliaria])`.
- Quando qualquer um dos moradores pergunta algo factual sobre o
  contrato (*"Qual o valor do aluguel?"*, *"Existe multa por rescisão?"*),
  o LLM emite `tool_calls` e o `tool_node` executa a `@tool` definida em
  [`app/rag.py`](app/rag.py), que faz a busca semântica sobre os PDFs
  via FAISS.
- A resposta volta como `ToolMessage`, o `agente_responde` recebe o
  contexto e formula a resposta natural com citação de fontes
  (`[fonte: contrato.pdf, pag. 3]`). Os dois moradores leem a mesma
  resposta e podem reagir a ela em conjunto.

### Coordenação — aprovação do dossiê com regra anti-autovoto

Esta é a peça central do sistema. Construir o dossiê **exige consenso**:
nenhum item entra na lista oficial sem que **o outro** morador concorde.

Fluxo concreto:

1. **Proposta** — Morador 1 digita: *"Vamos pedir o conserto do
   vazamento antes de assinar"*. O `classificador_node` rota como
   `intent="proposta"`. O `proposta_node` extrai o item (descrição +
   categoria), grava no State:
   ```python
   item_pendente = {"descricao": "Conserto do vazamento", "categoria": "manutencao", ...}
   proponente    = "Morador 1"
   ```
   e responde no chat: *"Morador 1 sugeriu adicionar X. Morador 2, você
   aprova?"*.

2. **Aprovação** — Morador 2 muda o radio da sidebar para "Morador 2"
   e digita *"Aprovo"*. O classificador detecta `intent="aprovacao"`
   e roteia para `analisa_consenso`, que aplica a **regra
   anti-autovoto**:
   ```python
   if proponente == current_user:
       erro_coordenacao = "Você não pode aprovar sua própria proposta."
   ```
   Como `"Morador 1" != "Morador 2"`, a regra passa: o item é
   carimbado com `aprovador="Morador 2"`, anexado a `dossie_final`,
   e o `item_pendente` é limpo. Em seguida o `agente_responde`
   confirma em linguagem natural e a coluna direita da UI atualiza
   instantaneamente.

3. **Tentativa de autoaprovação (o caso bloqueado)** — se Morador 1
   tentasse aprovar a própria proposta sem trocar o radio, o
   `analisa_consenso` setaria `erro_coordenacao` e o `agente_responde`
   recusaria educadamente: *"Você não pode aprovar sua própria proposta.
   Aguarde a resposta do outro morador."* O `dossie_final` permanece
   intacto.

A mesma lógica vale para rejeições: `analisa_consenso` é um nó único
que processa aprovação **ou** rejeição olhando o `intent`, e aplica a
regra de consenso em ambos os caminhos.

---

## Diagrama do grafo

```text
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
```

Leitura rápida:

- O **único ponto de decisão "grosso"** é o `classificador_node`. Ele
  manda cada turno para um dos 4 ramos.
- O ramo de aprovação/rejeição passa por `analisa_consenso` (que muta
  o State silenciosamente) e em seguida pelo `agente_responde` (que
  comunica o resultado em linguagem natural).
- O loop `agente_responde ↔ tool_node` é o coração do RAG: o LLM
  decide quando consultar os PDFs; o `tool_node` executa; a resposta
  volta para o agente sintetizar.

---

## Estrutura do projeto

```
app.py                # UI Streamlit (entrypoint: `streamlit run app.py`)
app/
├── state.py          # GraphState (TypedDict) + ItemDossie
├── rag.py            # pypdf + FAISS in-memory + @tool consultar_documentos_imobiliaria
├── nodes.py          # 5 nós (classificador, agente_responde, proposta, analisa_consenso, finalizacao)
├── router.py         # route_by_intent + route_after_agent
├── graph.py          # StateGraph + MemorySaver + ToolNode
└── diagram.py        # Diagramas ASCII e Mermaid
docs/                 # PDFs de fallback (alternativa ao upload pela UI — até 5)
.env.example
requirements.txt
```

---

## Instalação

Requer Python 3.10+.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

**Linux / macOS (Bash):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edite o `.env` recém-criado e preencha sua chave do Google AI Studio
(gere uma gratuita em [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)):

```
GOOGLE_API_KEY=AIza...
```

## Execução

```powershell
streamlit run app.py
```

O Streamlit abre em `http://localhost:8501`.

---

## Fluxo de uso (4 passos)

1. **Upload dos PDFs** — na sidebar, envie até 5 arquivos (contrato,
   IPTU, seguro incêndio, condomínio etc.). Eles são indexados em
   memória no FAISS. Alternativa: coloque os PDFs em `docs/` antes de
   subir o Streamlit.
2. **Selecione o morador** — radio "Quem está falando?" → escolha
   Morador 1.
3. **Conversem e proponham** — peçam ao agente esclarecimentos
   factuais sobre o contrato; quando quiserem registrar algo no
   dossiê, escrevam *"Vamos colocar X no dossiê"*. Troquem o radio
   para o outro morador para aprovar ou rejeitar.
4. **Finalize** — quando o dossiê estiver completo, clique em
   **Gerar Dossiê Final** na sidebar. O `finalizacao_node` produz
   um documento formal agrupado por categoria, pronto para copiar
   e enviar à imobiliária.

---

## Variáveis de ambiente

| Variável             | Obrigatória | Default                       |
|----------------------|-------------|-------------------------------|
| `GOOGLE_API_KEY`     | sim         | —                             |
| `CHAT_MODEL`         | não         | `gemini-2.5-flash`            |
| `CLASSIFIER_MODEL`   | não         | `gemini-2.5-flash-lite`       |
| `EXTRACTOR_MODEL`    | não         | `gemini-2.5-flash`            |
| `EMBEDDING_MODEL`    | não         | `models/gemini-embedding-001` |

Os três modelos de chat podem ser configurados independentemente —
útil para usar um modelo mais barato no classificador e um mais forte
no agente principal.

---

## Publicação no GitHub

### 1. Pré-checagem de segurança

Antes do primeiro push, confirme:

- [ ] `.env` (com sua chave real) está no `.gitignore` — já está protegido
- [ ] `.env.example` contém **apenas placeholder** (`cole-sua-chave-...`), nunca uma chave real
- [ ] PDFs reais em `docs/` estão ignorados — `.gitignore` já bloqueia `docs/*.pdf`

### 2. Criar o repositório no GitHub

1. Acesse [github.com/new](https://github.com/new).
2. Nome sugerido: `republica-dossier-langgraph`.
3. Marque como **Public** (a atividade exige repositório público).
4. **NÃO** marque "Add a README", "Add .gitignore" nem "Add license" — o
   projeto já tem.
5. Clique em **Create repository** e copie a URL HTTPS que aparece
   (ex.: `https://github.com/<seu-user>/republica-dossier-langgraph.git`).

### 3. Inicializar git e fazer o primeiro push

Na raiz do projeto, execute (PowerShell ou Bash):

```bash
git init
git add .
git status
```

Confira na saída do `git status` que **NÃO aparecem**: `.env`, `.venv/`,
nem PDFs reais. Se aparecer algum, ajuste o `.gitignore` antes de
prosseguir.

```bash
git commit -m "Sistema colaborativo de análise de contratos (LangGraph + RAG + Streamlit)"
git branch -M main
git remote add origin https://github.com/<seu-user>/republica-dossier-langgraph.git
git push -u origin main
```

### 4. Verificar no navegador

Abra a URL do repositório e confirme:

- O `README.md` renderiza corretamente na home (com o diagrama ASCII e o
  sumário).
- Não há `.env` nem chave da API exposta em nenhum arquivo.
- A pasta `.venv/` não foi enviada.

---

## Notas técnicas

- **Persistência in-memory**: o `MemorySaver` ([`app/graph.py`](app/graph.py))
  guarda o State em RAM, e o FAISS ([`app/rag.py`](app/rag.py)) também.
  Tudo se perde ao reiniciar o processo. Para durabilidade, troque por
  `SqliteSaver.from_conn_string(...)` no `construir_grafo()`.
- **Tool única como interface ao RAG**: no padrão ReAct adotado, o
  agente só acessa os PDFs via a `@tool consultar_documentos_imobiliaria`.
  Isso facilita auditar quando o agente consultou os documentos
  (basta filtrar `tool_calls` no histórico).
- **Anti-autovoto sem `interrupt()`**: a coordenação é resolvida com
  estado puro (`item_pendente`, `proponente`, `current_user`) — não
  precisamos pausar o grafo. Cada turno é um `invoke` completo que
  termina em `END`.
