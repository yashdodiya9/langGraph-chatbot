# LangGraph HITL Chatbot

A Streamlit chatbot backed by a [LangGraph](https://github.com/langchain-ai/langgraph) agent
with **human-in-the-loop** controls: the agent pauses mid-turn to ask for missing tool
arguments, and pauses again before running sensitive tools so a human can approve, edit,
or reject the call. It also supports per-thread PDF retrieval (chat with a document),
persistent multi-thread history, auto-generated chat titles, and pinning/deleting chats.

## Features

- **Human-in-the-loop tool calls**
  - *Clarification*: if the model's tool call is missing a required argument, the graph
    interrupts and asks the user to fill it in before continuing.
  - *Approval*: tools that mutate data (see `SENSITIVE_TOOLS` in `app/config.py`) pause for
    an explicit approve / edit-and-approve / reject decision. **Every** tool call in a model
    message is reviewed (a model can request several at once); rejected calls are answered with
    a "cancelled" tool result so the chat history stays valid for the LLM API.
- **Chat with a PDF**: upload a PDF per-thread; it's chunked, embedded, and indexed into a
  FAISS store the agent can search via a dedicated tool.
- **Multi-thread persistence**: every conversation is checkpointed in SQLite via
  `AsyncSqliteSaver`, so closing and reopening the app resumes exactly where you left off.
- **Auto-named chats**: the first message of a new conversation gets a short, LLM-generated
  title (like ChatGPT/Claude), generated in the background so it never blocks the reply.
- **Pin & delete chats**: pinned chats always sort to the top of the sidebar; deleting a
  chat wipes its checkpoints, metadata, and any indexed PDF.
- **Streaming responses** with a live "🔧 Using `<tool>`…" status indicator while tools run.
- **Markdown rendering**: tables, code blocks and formatting render correctly in both live streaming and history replay.

## Project layout

```
app/
  config.py                  # env vars & tunable constants - the only place that reads os.getenv()
  backend/
    async_runtime.py         # shared background event loop (Streamlit is sync, LangGraph is async)
    llm.py                   # base model + tool-bound model
    state.py                 # ChatState TypedDict
    human_review.py          # the HITL node (clarification + approval)
    graph.py                 # builds/compiles the StateGraph
    tools/                   # search, stock price, PDF RAG, MCP-provided tools
    persistence/
      checkpointer.py        # LangGraph's SQLite checkpointer
      thread_meta.py         # title / pinned / timestamps side-table + CRUD helpers
  frontend/
    streamlit_app.py         # session setup, streaming, turn orchestration
    components/
      sidebar.py             # new-chat button, PDF uploader, thread list (pin/delete)
      hitl_panel.py           # renders whatever the graph is currently paused on
main.py                       # entry point - run this with `streamlit run main.py`
tests/                        # pytest unit tests (fast, no network/model downloads)
scripts/init_db.py            # one-off DB bootstrap helper
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate   # or your tool of choice
pip install -r requirements.txt

cp .env.example .env
# then edit .env: at minimum set GROQ_API_KEY

streamlit run main.py
```

> Run `streamlit run main.py` from the repo root - `main.py` exists specifically so the
> `app.*` absolute imports resolve correctly no matter how you invoke the command.

### Running with Docker

```bash
cp .env.example .env   # fill in real values first
docker compose up --build
```

### Running tests

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

The suite runs fast, offline and **without any API keys**: `tests/conftest.py` points the app at a
throwaway sqlite file, and the tests use a scripted fake LLM (`tests/fakes.py`) and a fake embedding
model. It includes end-to-end graph tests (plain chat, clarification, approve / edit / reject, several
tool calls in one message, auto-titles, pin, delete) and headless Streamlit UI tests that drive
`main.py` like a user would.

## Known caveats

- **`AsyncSqliteSaver` isn't meant for production concurrency.** LangGraph's own docs
  recommend it only for demos/small projects due to SQLite's write-locking. For a real
  multi-user deployment, swap `app/backend/persistence/checkpointer.py` to
  `AsyncPostgresSaver` - it's the same interface, so nothing else in the codebase changes.
- **MCP integration is disabled by default** (`ENABLE_MCP_TOOLS=false`) because connecting
  to unreachable/misconfigured MCP servers at startup is a common source of slow boots and
  flaky tests. The `from langchain.mcp import MCPAdapter` import in
  `app/backend/tools/mcp_tools.py` is carried over from the original project as-is; depending
  on your installed langchain/langgraph version you may need the separate
  `langchain-mcp-adapters` package (`MultiServerMCPClient`) instead. Update that module to
  match whatever MCP client library you're actually using before setting
  `ENABLE_MCP_TOOLS=true`.
- **Web search needs the `ddgs` package** (not the older `duckduckgo-search`) with current
  `langchain-community` versions - it's already in `requirements.txt`.
- **`langchain-community` is being sunset** upstream; you'll see a DeprecationWarning for the PDF
  loader import. It works today; plan a migration to the standalone integration packages.
- **Empty chats leave a tiny row.** Every fresh browser session registers a new thread id; threads
  that never receive a message aren't shown in the sidebar but keep a small `thread_meta` row.
  A periodic cleanup of rows without checkpoints would tidy this up.
- **No auth.** All threads are currently global (visible to anyone using the app). Add a
  login layer and scope threads by `user_id` before deploying this somewhere public.

## Further improvements (not yet implemented)

- Swap SQLite → Postgres checkpointer for real concurrency.
- Add authentication and per-user thread scoping.
- LangSmith tracing for observability into tool calls and interrupts.
- Message editing / regeneration and a "stop generating" button.
- Rate limiting around the per-chat title-generation LLM call.
