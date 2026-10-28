# AI Twin - Contextual Document Understanding and Personal Knowledge Retrieval

**Subtasks 1–3: ingestion, RAG chat, and LangGraph orchestration.** The backend covers PDF/image
ingestion, pgvector embeddings, owner-scoped documents, persistent conversations, and a
**LangGraph**-orchestrated chat workflow (query planning, multi-source retrieval, evidence
assessment, bounded retries, citation validation). Streamlit **AI Twin Chat** uses the same
LiteLLM / OpenAI configuration as embeddings.

**Not built yet:** the Playground UI (private-doc chat), cross-user sharing of `global` documents,
and real authentication. Memory, summaries, and checkpoint recovery are documented in §20–21;
evaluation and performance tooling in §23.

> **Security status:** the app uses a single *development identity* (see
> [Authentication](#development-identity-and-replacing-it)). It is **not** secure multi-user
> software. Do not expose it to untrusted networks.

---

## 1. Architecture

```
Streamlit (frontend/) --HTTP only--> FastAPI (backend/) --> PostgreSQL + pgvector (Supabase)
                                         |-> Supabase Storage  (original PDFs and images)
                                         '-> OpenAI / LiteLLM (embeddings + chat)
```

**Retrieval policy (AI Twin):** only the current owner's documents with
`visibility=global` and `status=ready` are searched (`twin_retrievable_filter()` in
`document_repository.py`). Private documents never appear in Twin chat. `global` means
eligible for this owner's Twin, not world-readable. There is no sharing table.
```

| Layer | Location | Responsibility |
|---|---|---|
| Routes | `backend/app/api/routes/` | HTTP contract and Pydantic validation only |
| Dependencies | `backend/app/dependencies/` | Database sessions, owner resolution, service wiring |
| Services | `backend/app/services/` | Business logic, independent of FastAPI routes |
| Repository | `backend/app/repositories/` | All SQL. Owner-scoped. Never commits |
| Models / migrations | `backend/app/db/`, `backend/migrations/` | Schema. Alembic is the source of truth |
| Frontend | `frontend/` | UI only. `services/api_client.py` is the only code that talks to the backend |

These are the backend services in `backend/app/services/`:

- `storage_service.py` is the Supabase Storage REST client (via httpx). It streams uploads and downloads, and its deletes are idempotent.
- `pdf_extractor.py` uses PyMuPDF for validation and page-by-page text extraction.
- `image_extractor.py` validates PNG/JPEG with Pillow and transcribes them with Tesseract OCR.
- `chunking_service.py` does semantic chunking (sentence embeddings + similarity breakpoints, token caps and overlap) and records the page range of each chunk.
- `embedding_service.py` calls `AsyncOpenAI` with batching, bounded retries with backoff, and order and dimension validation.
- `ingestion_service.py` orchestrates the steps from claiming a document through to marking it ready or failed.
- `job_runner.py` provides the `IngestionJobRunner` protocol and its `BackgroundTasks` implementation.
- `document_service.py` handles the upload, retry, and delete workflows across the database and storage.

**Secrets live only in the backend.** Streamlit only knows `STREAMLIT_API_URL`.

## 2. Prerequisites

- Python 3.12+ (`uv` recommended, plain `pip` works too)
- A Supabase project (PostgreSQL with pgvector, plus Storage)
- An OpenAI API key
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for PNG/JPEG and PDF page/figure OCR
  (`sudo apt install tesseract-ocr` on Debian/Ubuntu). Unit tests mock OCR and do not need it.
- Optional: Docker, or `pgserver`, for a local Postgres+pgvector used by the database tests

## 3. Environment configuration

There is one configuration mechanism: environment variables, which the backend also loads from
`backend/.env` (via pydantic-settings).

```bash
cp backend/.env.example backend/.env       # fill in real values; .env is gitignored
cp frontend/.env.example frontend/.env     # only STREAMLIT_API_URL
```

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | - | `postgresql://` or `postgres://` URLs are **rewritten to `postgresql+asyncpg://`** and `sslmode=` becomes `ssl=` (asyncpg's name for it). See `to_async_database_url` in `backend/app/core/config.py`. |
| `DATABASE_USE_PGBOUNCER` | `false` | Set to `true` for the Supabase pooler in transaction mode (port 6543). This disables prepared-statement caches. |
| `OPENAI_API_KEY` | - | Backend only (OpenAI direct, or LiteLLM if `LITELLM_API_KEY` is unset) |
| `LITELLM_API_BASE` | (empty) | If set (e.g. `http://localhost:4000`), embeddings call the LiteLLM **OpenAI-compatible** proxy at `{base}/v1` |
| `LITELLM_API_KEY` | (empty) | LiteLLM master / proxy key; falls back to `OPENAI_API_KEY` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | With LiteLLM, use the model name your proxy expects (e.g. `openai/text-embedding-3-small`) |
| `OPENAI_EMBEDDING_DIMENSIONS` | `1536` | Must match the `vector(1536)` column. The app refuses to start on a mismatch. |
| `OPENAI_TIMEOUT_SECONDS` / `OPENAI_MAX_RETRIES` | `30` / `4` | Bounded retries with exponential backoff |
| `OPENAI_CHAT_MODEL` | `gpt-6.1-sol` | Chat / RAG answers; with LiteLLM use the model name your proxy exposes |
| `RETRIEVAL_TOP_K` / `RETRIEVAL_MAX_CANDIDATES` | `8` / `30` | pgvector cosine search; `max_candidates >= top_k` |
| `MAX_CONTEXT_TOKENS` | `10000` | Evidence packed into the prompt |
| `MAX_CHAT_HISTORY_MESSAGES` | `10` | Prior turns sent to the LLM (separate from retrieval query) |
| `CHAT_TEMPERATURE` | `0.2` | Generation temperature |
| `LANGGRAPH_*` | | See §19 |
| `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` | - | Backend only. **Never give the service-role key to Streamlit.** |
| `SUPABASE_STORAGE_BUCKET` | `documents` | |
| `MAX_UPLOAD_SIZE_MB` | `25` | |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `650` / `100` | Max chunk size and overlap after semantic grouping |
| `SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE` | `0.25` | Prefer splits where consecutive sentence similarity is at or below this document percentile |
| `INGESTION_STALE_AFTER_SECONDS` | `900` | After this, a `processing` job counts as crashed and can be reclaimed |
| `AUTH_MODE` / `DEV_OWNER_ID` | `development` / fixed UUID | Development identity. Rejected when `APP_ENV=production`. |
| `APP_ENV`, `LOG_LEVEL`, `API_HOST`, `API_PORT` | | |
| `STREAMLIT_API_URL` | `http://localhost:8000` | Frontend only |

Missing credentials are checked when the service that needs them is initialised. In
`development` the API still starts and reports `503 service_unavailable` for features that
can't run. In `production` startup fails fast.

## 4. Creating a Supabase project

1. Create a project at <https://supabase.com/dashboard>.
2. **Project Settings -> Database -> Connection string**: copy the *URI*.
   - Direct connection (`db.<ref>.supabase.co:5432`) is best for migrations.
   - Pooler (`...pooler.supabase.com:6543`) also requires setting `DATABASE_USE_PGBOUNCER=true`.
3. **Project Settings -> API**: copy the project URL into `SUPABASE_URL` and the
   `service_role` key into `SUPABASE_SERVICE_ROLE_KEY`.

## 5. Enabling pgvector

The initial migration runs `CREATE EXTENSION IF NOT EXISTS vector`. The Supabase `postgres`
role is allowed to do this. If your role isn't, enable it first under **Database -> Extensions
-> vector** in the dashboard, or run as a privileged role:

```sql
create extension if not exists vector;
```

**Vector index:** the migration creates an **HNSW** index (`vector_cosine_ops`, `m=16`,
`ef_construction=64`). This requires pgvector >= 0.5, which Supabase has. Unlike IVFFlat, HNSW
works on an empty table and needs no retraining. Things to know:

- Recall versus speed is tuned at query time with `SET hnsw.ef_search = 40..200`.
- Index builds use memory: raise `maintenance_work_mem` for large re-indexes.
- An HNSW index on `vector` is limited to 2000 dimensions.
- Each index is tied to one distance operator. Queries must use `<=>` (cosine) to use it.

## 6. Creating the Storage bucket

**Storage -> New bucket**: name it `documents` (or your `SUPABASE_STORAGE_BUCKET`), leave
**Public** turned off, and optionally set a 25 MB file size limit. If you restrict MIME types,
allow `application/pdf`, `image/png`, and `image/jpeg`. The backend uses the service-role key,
so no RLS policies are needed for it. Objects are
stored as `{owner_id}/{document_id}.{pdf|png|jpg}`, so user-supplied filenames never appear in paths.

## 7. Database migrations

Alembic reads `DATABASE_URL` from settings. Run it from `backend/`:

```bash
cd backend
alembic upgrade head          # apply
alembic current               # show the applied revision
alembic check                 # verify the models match the migrations
alembic downgrade base        # drop the schema (keeps the vector extension)
# new migration after a model change:
alembic revision --autogenerate -m "describe change"
```

The app never calls `create_all()`.

## 8. LiteLLM (optional embeddings gateway)

The backend uses the OpenAI Python SDK. To route embeddings through [LiteLLM](https://docs.litellm.ai/) instead of OpenAI directly:

1. Start LiteLLM (example — adjust to your `config.yaml` / providers):

   ```bash
   pip install 'litellm[proxy]'
   litellm --model openai/text-embedding-3-small --port 4000
   ```

   Or run the LiteLLM proxy you already use and note its URL (often port **4000**).

2. In `backend/.env`:

   ```env
   LITELLM_API_BASE=http://localhost:4000
   LITELLM_API_KEY=sk-your-litellm-master-key
   OPENAI_EMBEDDING_MODEL=openai/text-embedding-3-small
   OPENAI_EMBEDDING_DIMENSIONS=1536
   ```

   If your proxy accepts the same key as OpenAI, you can set only `OPENAI_API_KEY` and leave `LITELLM_API_KEY` empty.

3. Restart the FastAPI app. Logs will include `embedding_client_litellm` with the base URL (never the key).

Leave `LITELLM_API_BASE` empty to call OpenAI at `api.openai.com` as before.

## 9. Running FastAPI

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r backend/requirements.txt -r frontend/requirements.txt
cd backend && uvicorn app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

## 10. Running Streamlit

```bash
export STREAMLIT_API_URL=http://localhost:8000      # or: set -a; source frontend/.env; set +a
streamlit run frontend/app.py
```

## 11. Unit tests

Run from the repository root. These need no credentials and no network apart from tiktoken's
one-time download of its encoding file:

```bash
pytest                         # unit + frontend tests; DB tests are skipped
ruff check . && ruff format --check .
```

`backend/tests/conftest.py` clears `DATABASE_URL`, `OPENAI_API_KEY`, and the Supabase variables
before the app is imported. This means the tests can never reach your real services.

## 12. Database (integration) tests

`backend/tests/db/` covers the API, ownership, ingestion, retries, concurrency, and the Alembic
migration. It runs against a **disposable** Postgres+pgvector database given by
`TEST_DATABASE_URL`. A guard refuses any host that isn't local and any database name that
doesn't contain `test`. OpenAI and Supabase are replaced with in-memory fakes.

```bash
# Option A: Docker
docker compose up -d postgres            # also creates aitwin_test with the vector extension
TEST_DATABASE_URL=postgresql://aitwin:aitwin@localhost:54329/aitwin_test pytest

# Option B: no Docker or root (pgserver bundles Postgres 16 + pgvector)
uv pip install pgserver
python -c "import pgserver; s=pgserver.get_server('.pgdata', cleanup_mode=None); s.psql('create database aitwin_test')"
TEST_DATABASE_URL="postgresql://postgres@/aitwin_test?host=$PWD/.pgdata" pytest
```

The compose file provides **only Postgres + pgvector**. It does not emulate Supabase Storage, so
running the app end to end still needs a real Supabase project, or your own fake. Tests marked
`integration` against real Supabase or OpenAI are reserved and excluded by default
(`-m 'not integration'`).

## 13. API examples

```bash
API=http://localhost:8000/api/v1
curl $API/health
curl $API/health/ready                       # 503 + per-check status when not ready

curl -X POST $API/documents -F file=@paper.pdf -F visibility=global -F title="My paper"
# 202 {"document_id": "...", "status": "uploaded", "message": "...", "duplicate": false, "document": {...}}

curl "$API/documents?limit=20&offset=0&status=ready&visibility=global"
curl $API/documents/<id>
curl $API/documents/<id>/status
curl -X POST $API/documents/<id>/retry       # failed, queued (uploaded), or stale processing
curl -X DELETE $API/documents/<id>           # 204
```

Errors always have the form `{"error": {"code": "...", "message": "..."}}`. Raw SQL errors,
stack traces, and storage paths are never returned.

| Case | Status |
|---|---|
| Not PDF/PNG/JPEG, or MIME type does not match the extension | 415 |
| Larger than `MAX_UPLOAD_SIZE_MB` | 413 |
| Empty, corrupted, encrypted, or no PDF signature | 422 `invalid_pdf` |
| Empty, corrupted, or not a real PNG/JPEG | 422 `invalid_image` |
| Duplicate (same owner + SHA-256 + visibility) | 200, `duplicate: true` |
| Duplicate with different visibility, or original being deleted | 409 |
| Missing document or another owner's document | 404 (the two are indistinguishable) |
| Storage failure | 502 |
| Dependency not configured | 503 |

**Deduplication policy:** uniqueness is per owner on `(owner_id, file_hash)`. Re-uploading the
same bytes with the same visibility returns the existing document and doesn't re-process it.
Different owners never share or overwrite documents. To change a document's visibility, delete
it and re-upload.

## 14. Ingestion workflow

1. **Upload** (request path):
   - Validate the name and MIME type.
   - Stream the file to a temp file while hashing and enforcing the size limit (oversized
     `Content-Length` is rejected up front).
   - Check magic bytes, then inspect: PyMuPDF for PDFs (rejects encrypted and zero-page files),
     Pillow for PNG/JPEG.
   - Apply deduplication.
   - Upload to Storage, then insert a row with status `uploaded`. If the insert fails, the
     stored object is deleted; if that deletion also fails, the orphaned path is logged.
   - Enqueue the job and return 202.
2. **Claim:** one atomic `UPDATE ... WHERE status IN ('uploaded','failed') OR stale processing
   RETURNING`. The returned `processing_started_at` is a *fencing token*. Concurrent runs see no
   row and skip.
3. **Download, extract, chunk, embed**, with **no open transaction**. PyMuPDF, Pillow/Tesseract,
   and tiktoken run in `asyncio.to_thread`. Images are treated as a single page of OCR text.
   OpenAI calls use explicit timeouts, retries are bounded with backoff and honour `Retry-After`,
   and identical texts are embedded once.
4. **Finalize** (one transaction): lock the row, check it is still `processing` with the *same*
   fencing token, delete existing chunks, bulk-insert the new chunks and embeddings, then set
   `ready` with `page_count`, `chunk_count`, `embedding_model`, `embedding_dimensions`, and
   `processed_at`. A document is never `ready` before its chunks are committed.
5. **Failure:** status becomes `failed` with a sanitized, user-safe `error_message`. This is
   guarded by the same fencing token, so a superseded run can't overwrite a newer one.
   **Retry** resets the document to `uploaded`, and the next run replaces any earlier chunks,
   so duplicates are impossible.

**Delete:**

1. Mark the document `deleting` and commit.
2. Delete the Storage object (a missing object counts as success).
3. Delete the row; chunks are removed by `ON DELETE CASCADE`.

If the storage step fails, the API returns 502, the row stays `deleting`, and calling DELETE
again resumes the cleanup. An ingestion run that finishes during deletion aborts without
writing.

**Chunking:**

- Pages are split into paragraphs, then sentences (units for embedding only).
- Each sentence is embedded; low consecutive cosine similarity (local minima at or below
  `SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE`) marks topic boundaries.
- Segments are merged or split to respect `CHUNK_SIZE_TOKENS`; overlap uses trailing whole
  units up to `CHUNK_OVERLAP_TOKENS`.
- `page_start` and `page_end` are the minimum and maximum source pages in the chunk.
- `token_count` is measured on the final text with the model's tiktoken encoding.

## 15. Known limitations

- **BackgroundTasks is not durable.** Jobs run in the API process after the response is sent.
  A restart or crash loses queued and in-flight work. Those documents stay `uploaded` (re-queue
  them with `POST /retry`, the "Re-queue ingestion" button) or `processing` (retryable after
  `INGESTION_STALE_AFTER_SECONDS`). Long CPU work also competes with request handling.
- PostgreSQL and Storage aren't transactional together. The ordering above keeps failures
  recoverable, but a crash at exactly the wrong moment can leave an orphaned object. Orphans are
  logged as `orphaned_storage_object`.
- **PDF images:** pages with little extractable text are OCR'd from a rendered page bitmap;
  on text-heavy pages, embedded raster images above a size threshold are OCR'd separately
  (requires Tesseract, same as PNG/JPEG). Vector graphics, tables, and complex layouts are
  **Tables** on pages with detectable grid structure are exported as markdown via
  PyMuPDF ``find_tables`` (borderless or scanned tables may still be missed).
  Standalone PNG/JPEG files are OCR'd as a single page.
- The whole PDF is processed in one pass. Very large documents (thousands of pages) would
  benefit from streaming chunk persistence.
- A single development identity, with no auth UI.
- Changing the embedding model or dimensions requires a migration and re-embedding.
  `embedding_model` and `embedding_dimensions` are stored per document so that mixed vectors can
  be detected.
- Streamlit's own upload cap is 200 MB. The backend enforces `MAX_UPLOAD_SIZE_MB`.
- **RAG chat:** no LangGraph, no automatic query rewriting, no Playground over private docs.
  Retrieval uses recent user turns concatenated with the current question. LiteLLM must expose
  OpenAI-compatible chat or responses endpoints for answers (embeddings alone are not enough).

## 16. Development identity, and replacing it

`backend/app/dependencies/auth.py:get_current_principal` returns a `Principal(owner_id)`. With
`AUTH_MODE=development`, every request belongs to `DEV_OWNER_ID`. **Nothing the client sends
can change the owner.** Any other mode returns 401 (fail closed), and development mode is
rejected when `APP_ENV=production`.

To add real authentication (for example, Supabase Auth):

1. Have Streamlit sign the user in and send `Authorization: Bearer <access_token>`.
2. Replace `get_current_principal` so it verifies the JWT and uses the `sub` claim as `owner_id`.
   Verify against the project's JWKS (or JWT secret) and check `aud`, `exp`, and `iss`.
3. Routes, services, and repository queries already depend only on `Principal.owner_id`, so
   nothing else changes. Existing development data can be re-owned with a one-off `UPDATE`.

## 17. Replacing BackgroundTasks with a durable worker

Routes depend only on `IngestionJobRunner.enqueue(document_id)`.

1. Implement, for example, `ArqJobRunner.enqueue`, which calls
   `await redis.enqueue_job("ingest", str(document_id))`.
2. In the worker, build `IngestionService` (session factory, storage, chunker, embedder) and
   call `await ingestion.run(document_id)`. It is already idempotent, claim-based, and fenced, so
   at-least-once delivery is safe.
3. Return the new runner from `get_job_runner` in `backend/app/dependencies/services.py`.
4. Optionally add a periodic sweeper that re-enqueues `uploaded` documents and stale
   `processing` documents.

## 18. RAG chat (Subtask 2)

After `alembic upgrade head` (migration `0002` adds `conversations` and `messages`):

1. Upload documents on **Documents**, set visibility to **global**, wait until **ready**.
2. Open **AI Twin Chat**, start or select a conversation, ask a question grounded in those docs.
3. Assistant replies include **Sources** expanders with excerpts and page hints.

**API** (`/api/v1/conversations`): create/list/get/delete conversations; list messages;
`POST .../messages` with `{ "content", "client_message_id"? }` returns user + assistant rows.
Wrong-owner IDs return **404**. Generation failures return **502**; the user message is kept.

**Implementation:** `ChatService` commits the user message, then runs the LangGraph workflow
(see §19). Streamlit never POSTs on render; `client_message_id` supports idempotent retries.

**LiteLLM:** chat uses the same `LITELLM_API_BASE` / key as embeddings. Your proxy must expose
OpenAI-compatible **chat** or **responses** routes; verify with your proxy operator if answers fail with 401/404.

## 19. LangGraph chat workflow (Subtask 3)

Each `POST .../messages` turn runs a bounded **StateGraph** under `backend/app/workflows/`:

1. **Query plan** (structured OpenAI output): standalone query, `documents` / `conversation_history` sources, optional clarification.
2. **Parallel retrieval** (separate DB sessions): pgvector document search + bounded conversation excerpts.
3. **Merge evidence**, **assess sufficiency**, optional **retrieval retry** (rewrite, broaden, PostgreSQL full-text, neighbor chunks) up to `LANGGRAPH_MAX_RETRIEVAL_ATTEMPTS`.
4. **Generate** (existing `OpenAIGenerationService`), **validate** citations deterministically, optional **repair** up to `LANGGRAPH_MAX_ANSWER_REPAIRS`.
5. **Abstain** or **502 retrieval/generation error** when appropriate (retrieval failures are not treated as “no documents”).

| Variable | Default | Purpose |
|---|---|---|
| `LANGGRAPH_MAX_RETRIEVAL_ATTEMPTS` | `2` | Cap retrieval rounds |
| `LANGGRAPH_MAX_ANSWER_REPAIRS` | `1` | Cap citation/claim repair |
| `LANGGRAPH_ENABLE_DEBUG_METADATA` | `false` | Optional `operational_metadata` on chat responses + Streamlit debug expander |
| `LANGGRAPH_CHECKPOINTING_ENABLED` | `false` | Postgres checkpointer when `langgraph-checkpoint-postgres` is installed (see §21) |
| `LANGGRAPH_GRAPH_VERSION` | `twin-v4` | Bump when graph topology changes; incompatible checkpoints are not resumed |
| `LANGGRAPH_RECOVERY_ENABLED` | `true` | Retry transient workflow failures; resume when checkpointing is on |
| `LANGGRAPH_MAX_RECOVERY_ATTEMPTS` | `2` | Extra invocations after transient errors |
| `LANGGRAPH_WORKFLOW_TIMEOUT_SECONDS` | `120` | Reserved for future hard timeout |
| `LANGGRAPH_MAX_EVIDENCE_ITEMS` | `24` | Cap merged evidence count |

## 20. Memory and conversation summaries (Subtask 4)

After `alembic upgrade head` (migration **`0003`**: `owner_memories`, `conversation_summaries`, `workflow_executions`):

- **Memories:** owner-scoped pgvector rows extracted from **user** turns (background job after chat). Never stores private document verbatim text.
- **Summaries:** one rolling summary per conversation (incremental LLM updates).
- **Context:** the LangGraph workflow loads summary + semantic memories before planning; generation prepends a labeled **untrusted** context block (citations remain document-only).

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_ENABLED` | `true` | Semantic memory retrieval in the graph |
| `MEMORY_EXTRACTION_ENABLED` | `true` | Background extraction after chat |
| `MEMORY_EXTRACTION_INTERVAL_MESSAGES` | `6` | Run extraction every N messages |
| `MEMORY_RETRIEVAL_TOP_K` | `8` | Memories injected into context |
| `CONVERSATION_SUMMARY_ENABLED` | `true` | Load/update summaries |
| `CONVERSATION_SUMMARY_TRIGGER_MESSAGES` | `12` | Update summary every N messages |

**API:** `GET /api/v1/memories`, `PATCH /api/v1/memories/{id}` (deactivate); `GET /api/v1/conversations/{id}/summary`; `GET /api/v1/conversations/{id}/workflows/{workflow_id}` (sanitized execution status).

Streamlit: **Memories** page; chat sidebar **Conversation summary** expander; debug metadata may show `used_persistent_context`.

## 21. Workflow recovery and checkpoints

Each chat turn creates a **`workflow_executions`** row (id = internal `request_id`). With **`client_message_id`**, a completed execution plus existing assistant message short-circuits duplicate graph runs.

When **`LANGGRAPH_CHECKPOINTING_ENABLED=true`**, the API runs LangGraph’s **`AsyncPostgresSaver.setup()`** at startup (same database as the app; separate checkpoint tables). **`thread_id`** is `{owner_id}:{conversation_id}` (server-only). Checkpoints complement but do **not** replace the `messages` table as the user-visible source of truth.

Install: `langgraph-checkpoint-postgres` (see `backend/requirements.txt`). On first enable, restart the API so `setup()` creates checkpoint tables.

## 22. Playground (Subtask 6)

The **Playground** (`/api/v1/playground/*` and Streamlit **Playground** nav) supports private PDF uploads, scoped retrieval, LangGraph queries (without Twin memory/summary nodes), prompt templates, experiments, evaluation runs, and audited **private → global** promotion.

| Scope | Retrieval |
|-------|-----------|
| `global_only` | Same as AI Twin (owner’s global, ready docs) |
| `private_only` | Owner’s private, ready docs |
| `combined` | Owner’s global + private, ready |
| `selected` | Only listed document IDs (validated server-side) |

- **Upload:** `POST /api/v1/playground/documents` always creates **private** documents.
- **Promote:** `POST /api/v1/playground/documents/{id}/promote` (owner, `ready` only) writes `document_promotion_audits`.
- **Query:** `POST /api/v1/playground/query` with optional `prompt_version_id`.
- **Eval:** `POST /api/v1/playground/evaluations/run` against `evaluation/datasets/*.jsonl`.

Apply migration **0004** after pulling: `cd backend && alembic upgrade head`.

## 23. Evaluation and performance (Subtask 5)

Offline-first evaluation lives in [`evaluation/`](evaluation/) (not in production API routes).

```bash
# Smoke (CI-safe, no OpenAI)
python -m evaluation --dataset evaluation/datasets/core.jsonl --limit 5 --offline

# Full offline suite (35 synthetic cases)
python -m evaluation --dataset evaluation/datasets/core.jsonl --offline

# Opt-in live LLM + DB baseline path
python -m evaluation --dataset evaluation/datasets/core.jsonl --live-llm --mode baseline

# Compare two JSON reports
python -m evaluation.compare --baseline reports/run-A.json --candidate reports/run-B.json
```

- **Dataset:** [`evaluation/datasets/core.jsonl`](evaluation/datasets/core.jsonl) — see [`evaluation/datasets/README.md`](evaluation/datasets/README.md) for labeling rules.
- **Metrics:** Recall/Precision@K, MRR, nDCG (when grades exist), key-fact recall, abstention, citation checks, source-selection (agentic offline mode). LLM-judge scores are optional (`--live-llm`) and labeled as estimates.
- **Reports:** JSON + Markdown under `reports/` (gitignored). See [`evaluation/reports/OPTIMIZATION_LOG.md`](evaluation/reports/OPTIMIZATION_LOG.md) for optimization decisions.
- **Pricing:** [`evaluation/pricing/models.yaml`](evaluation/pricing/models.yaml) — set verified per-million token rates; unlisted models show `cost_available: false`.
- **Instrumentation:** [`backend/app/observability/`](backend/app/observability/) records LLM/embedding usage metadata (no prompts). LangGraph nodes record `node_timings` in debug metadata.
- **CI:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs ruff, pytest, and a 5-case offline smoke eval.

## Repository layout

```
backend/   app/{api,core,db,dependencies,repositories,schemas,services,observability}, migrations/, tests/{unit,db,evaluation}
evaluation/ datasets/, metrics/, runners/, fixtures/, pricing/, reports/
frontend/  app.py, pages/, components/, services/api_client.py, tests/
docker-compose.yml   local Postgres + pgvector (dev and tests only)
pyproject.toml       ruff + pytest configuration
```
