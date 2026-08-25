# DocuIntel Architecture

Modules 0 through 13 and Evaluation E1 through E5 are implemented and manually accepted for the current portfolio scope; asynchronous jobs, authentication, and later product modules remain outside this repository boundary.

The following is the target architecture. Modules 0 and 1 implement the FastAPI shell, configuration, logging, health check, PDF upload, and native extraction. Module 2 adds selective OCR and basic document structure extraction. Module 3 adds PostgreSQL/pgvector persistence, Alembic migrations, and document-management APIs. Module 4 adds structure-aware chunking and local embeddings. Module 5 adds SQL-side semantic, keyword, and hybrid search. Module 6 adds cross-encoder reranking, Module 7 adds grounded single-question RAG through Ollama, Module 8 adds durable conversation sessions with bounded multi-turn RAG, and Module 9 observes those existing services through a read-only evaluation layer.

```text
Streamlit Frontend [Module 13 complete; HTTP only]
   |
FastAPI
   |
   +-- Document Processing
   |      +-- PDF Parser [Module 1 implemented: PyMuPDF native extraction]
   |      +-- OCR [Module 2 implemented: Tesseract fallback]
   |      +-- Layout Parser [Module 2 implemented: heuristic native structure]
   |      +-- Chunker [Module 4 implemented]
   |
   +-- Embedding Service [Module 4 implemented: Sentence Transformers]
   |
   +-- Retrieval Service [Modules 5–6 implemented]
   |      +-- Vector Search [pgvector]
   |      +-- Keyword Search [PostgreSQL FTS + GIN]
   |      +-- Hybrid Search [RRF]
   |      +-- Candidate Pool -> CrossEncoder Reranker [Module 6]
   |
   +-- RAG Service [Module 7 implemented]
   |      +-- Context Manager
   |      +-- Grounded Prompt Builder
   |      +-- Source/Citation Validation
   |      +-- Conversation Orchestrator [Module 8 implemented]
   |             +-- Persisted Sessions and Ordered Messages
   |             +-- Bounded History Builder
   |             +-- Follow-up Query Rewriter
   |
   +-- LLM Provider [Modules 7–11 implemented]
          +-- Ollama local HTTP client

   +-- Evaluation Layer [Module 9 implemented]
          +-- JSON/JSONL dataset loader
          +-- Retrieval evaluator -> SearchService
          +-- RAG evaluator -> RAGService
          +-- Metrics, baselines, quality gates, JSON reports

   +-- Document Analysis [Module 11 implemented]
          +-- Ordered PostgreSQL chunk loading
          +-- Bounded hierarchical summarizer
          +-- Caller-constrained JSON classifier

   +-- Structured Information [Module 12.2 implemented]
          +-- One-document bounded field extraction
          +-- Evidence/type/source validation
          +-- Existing JSONB table inventory
          +-- Constrained plan -> deterministic table executor

   +-- Privacy Workflow [Module 12.4 implemented]
          +-- Deterministic high-confidence PII detectors
          +-- Native PyMuPDF word-coordinate resolver
          +-- Explicit selection -> PyMuPDF apply_redactions()
          +-- Controlled generated artifact download

PostgreSQL + pgvector [Modules 3–5 implemented; Modules 6, 9, and 11 consume data through services]
```

## Layers

### Frontend

Module 10 implements a Streamlit presentation layer under `streamlit_app/`. Module 11 adds the Analyze page, Module 12.2 extends that page minimally, Module 12.3 adds a minimal Compare section, and Module 12.4 adds a dedicated Privacy page. The frontend has a centralized `ApiClient` for JSON GET/POST, binary artifact downloads, multipart PDF upload, DELETE, transport errors, HTTP errors, timeouts, and safe user-facing messages. Page functionality covers Home/backend status, documents and structure inspection, search with existing filters/reranking, grounded Ask, persistent conversations, grounded summaries, caller-labeled classification, structured extraction, table inventory/preview, table queries, explicit two-document comparison, and review-first PII redaction. Components display source excerpts, ranks, timing metadata, loading states, empty states, and explicit confirmation for destructive actions.

The frontend imports only its own API adapters and Streamlit components. It does not import or call SQLAlchemy, PostgreSQL, repositories, SearchService, RAGService, ConversationService, AnalysisService, embedding models, the CrossEncoder, OCR, or Ollama. All business logic and infrastructure access remain behind FastAPI. `DOCUINTEL_API_BASE_URL` defaults to `http://127.0.0.1:8001`; the app does not add CORS or start backend services.

### FastAPI

The API layer exposes liveness/readiness endpoints, PDF upload, document-management, indexing, chunk, and search endpoints. It validates requests and responses with Pydantic models and coordinates services. Routes remain thin and do not contain PDF-processing, embedding, retrieval, or SQL business logic.

### Document Processing

Module 1 implements PDF upload validation, UUID-based local storage, PyMuPDF native page-level text extraction, basic PDF metadata extraction, and OCR-candidate flags. Module 2 adds selective Tesseract OCR, OCR confidence, PyMuPDF text-block layout elements, heuristic headings/paragraphs/list items, and native table extraction. Scanned table understanding and ML layout models remain planned.

### Persistence and document management

Module 3 keeps generated PDF binaries in `data/processed/uploads/` and stores authoritative document metadata, pages, OCR state, layout elements, chunks, tables, and version-chain records in PostgreSQL. The existing `document_tables.headers` and `document_tables.rows` JSONB columns preserve structured table data, with each table linked to its page; Module 12.2 reuses this representation for inventory, preview, and querying without adding tables. Module 4 adds indexing state and chunk provenance/page ranges. Module 5 adds a generated English `tsvector` and GIN index for chunk text/headings. SQLAlchemy models are defined under `app/db/`, repository methods isolate data access, and services coordinate extraction, chunking, embedding, SQL retrieval, RRF fusion, and API projections. PostgreSQL's `vector` extension is created by the first migration; Module 4 fills the existing `vector(384)` column with normalized all-MiniLM-L6-v2 vectors.

### Embedding Service

The Module 4 embedding service lazily loads `sentence-transformers/all-MiniLM-L6-v2`, batches inputs, normalizes embeddings consistently, validates the configured dimension, and is reused for Module 5 query embeddings.

### Retrieval Service

The retrieval service keeps vector and PostgreSQL full-text SQL in the repository layer, uses explicit result projections that do not select raw vectors, and fuses candidate ranks with configurable RRF. When requested, the service expands and bounds the candidate pool, then passes the filtered candidates to the dedicated Python CrossEncoder service. That service loads `cross-encoder/ms-marco-MiniLM-L6-v2` lazily, scores query/chunk pairs in batches, applies deterministic score/base-rank tie ordering, and returns final top-k metadata while preserving the base scores and provenance.

The Module 6 request flow is:

```text
User query
    -> Search Service
    -> semantic / keyword / hybrid SQL retrieval
    -> bounded candidate pool
    -> CrossEncoder reranking service
    -> final top-k chunks
```

Reranking is CPU-compatible and does not write to PostgreSQL or change the schema. The raw relevance score is higher-is-more-relevant, not a calibrated probability. The Module 7 flow is final chunks -> bounded RAG context -> grounded prompt -> Ollama -> grounded answer plus source metadata. The context builder assigns stable S labels in final-rank order and never sends unbounded retrieved text. The grounded prompt prioritizes higher-ranked relevant evidence, preserves explicit facts and qualifiers, ignores irrelevant retrieved chunks, and requires concise answers without contradictions. Ollama generation uses the configurable `OLLAMA_TEMPERATURE` setting, defaulting to `0.1` for conservative local grounded QA.

The Module 8 conversational flow is:

```text
Conversation session
    -> persist current user message
    -> load bounded prior history
    -> Ollama follow-up query rewrite when history exists
    -> existing SearchService
    -> optional Module 6 reranking
    -> bounded document context
    -> history + original question + evidence prompt
    -> Ollama grounded answer
    -> persist assistant message
    -> answer, source labels, message IDs, and timings
```

Conversation history is stored in PostgreSQL through `ConversationRepository`. Sequence allocation is protected by a conversation row lock and a unique `(conversation_id, sequence_number)` constraint. Deleting a session cascades to its messages. The history builder selects the newest messages under both configured budgets and restores chronological order. The query rewriter receives history as delimited data and falls back to the original question on provider failure. It never persists system prompts or retrieved document context.

Conversational observability exposes phase timings for history loading, query rewriting, retrieval, reranking, and generation. `ConversationService` owns one outer wall-clock timer from conversation validation through assistant-message persistence and reports that elapsed value as `total_time_ms`; it does not sum potentially overlapping phase fields.

Module 12.1 extends the same flow for explicit multi-document scope without adding a second retrieval architecture:

```text
Question + typed document ID list
    -> validate ready/indexed scope and configured document-count bound
    -> existing SearchService per-document candidate retrieval
    -> deterministic deduplication and bounded combined candidate pool
    -> one optional CrossEncoder reranking pass
    -> bounded context with source/document provenance
    -> multi-document grounded prompt
    -> Ollama answer with validated S-label citations
```

An omitted `document_ids` filter means all indexed documents. A non-empty list is an explicit scope; selected IDs are deduplicated and validated before SQL retrieval. Each selected document receives a bounded retrieval opportunity, but only chunks that survive the shared ranking and context budgets reach Ollama. Multi-document prompts require independent evidence assessment, preserve disagreement instead of inventing reconciliation, and do not treat irrelevant selected documents as supporting evidence. Response metadata distinguishes selected, retrieved, and final-source document IDs. Conversation callers send the same filter on each turn; the existing conversation schema remains unchanged.

### RAG Service

`RAGService` calls the existing `SearchService` with the question, search mode, rerank flag, top-k, and filters. It does not query PostgreSQL directly. For an explicit multi-document filter, `SearchService` validates the scope, retrieves bounded candidates per selected document, merges them, and reranks the combined pool once when requested. RAG then builds a character-bounded context from final results, sends a maintainable grounded system/user prompt to `OllamaClient`, validates returned source labels, and projects answer/source/scope/timing metadata through `POST /api/v1/ask`. Empty retrieval results return without an Ollama call.

For Module 8, `RAGService.ask_conversational()` uses the same search and context path with a rewritten retrieval query while keeping the original question in the answer prompt. `ConversationService` owns persistence and bounded history; routes remain thin. The stateless `/api/v1/ask` contract remains unchanged.

### LLM Provider

`OllamaClient` uses asynchronous HTTP to `/api/generate` with `stream=false`, the configured `llama3.2:3b` default, conservative configurable temperature, and a bounded timeout. It supports ordinary text generation for RAG/summaries and Ollama JSON mode for constrained classification. It handles connection, timeout, missing-model, HTTP, and malformed-response failures as controlled service errors. DocuIntel never starts an Ollama process; CPU/GPU selection remains an Ollama runtime concern. No cloud provider is implemented.

### Document analysis

`AnalysisService` loads one document and all persisted chunks through `DocumentRepository.get_document_with_chunks()`, preserving chunk sequence order without N+1 content queries. `DocumentSummarizer` renders deterministic source blocks, batches them under `SUMMARY_BATCH_MAX_CHARS`, generates partial summaries, verifies each partial with `GroundingVerifier`, and performs one final synthesis bounded by `SUMMARY_FINAL_MAX_CHARS` when more than one batch exists. The final synthesis is verified again against the original chunks. One failed verification gets one structured repair and a second verification; malformed, timed-out, inconclusive, or repeatedly unsafe output receives a deterministic extractive fallback. A conservative lexical validator runs after verifier approval, and `AnalysisService` applies the same final boundary immediately before API projection, including the one-batch/one-chunk path where final synthesis is intentionally skipped. `DocumentClassifier` uses the same bounded chunk context, caller-supplied labels, Ollama JSON mode, Pydantic validation, and one retry for an unknown label. Generated results are returned with source metadata and timings but are not persisted, so Module 11 requires no migration.

Both analysis prompt families explicitly treat document content as untrusted data, ignore embedded instructions, prohibit outside knowledge, and preserve important factual qualifiers. Summary prompts additionally require every factual statement to be directly supported, prohibit relationships inferred from proximity, and require relevant absent details to be reported as not specified; hierarchical synthesis treats partial summaries as derived evidence and does not create relationships across them. `GroundingVerifier` uses structured JSON claims, exact source-label validation, exact supporting-evidence quote validation, bounded repair, and conservative fallback. A conservative deterministic post-generation boundary rejects new lexical content, unsupported actors/numbers/identifiers, and unsupported relational or evaluative language even when the local verifier says clean. Classification is constrained selection, not a calibrated probability model; OCR/extraction quality and caller label quality affect the result.

### Structured extraction and table querying

`StructuredExtractionService` loads one indexed document's ordered chunks through `DocumentRepository.get_document_with_chunks_and_tables()`. A deterministic token-overlap selector chooses relevant chunks under `STRUCTURED_EXTRACTION_MAX_CONTEXT_CHARS` while preserving source order and page/chunk metadata. `OllamaClient.generate_json()` receives a schema-focused prompt in which document text is delimited as untrusted data. The response is validated with strict Pydantic models, requested-field equality, source-label membership, type coercion, exact/explicit evidence checks, and conservative ambiguity handling. One provider repair attempt is bounded; if validation still fails, the service returns safe not-found values rather than unsafe provider claims. No extraction result is persisted.

The same repository projection exposes `DocumentTableRecord` as `PersistedTableRecord` with headers, rows, page, document, filename, and table ID. `TableQueryService` returns an inventory and bounded preview, optionally asks Ollama for a finite `TableQueryPlan`, validates operations and exact headers, and executes select/filter/min/max/sum/average/count/sort/top-N in ordinary Python. Numeric parsing is conservative and preserves original display strings. Query answers are derived from the structured result and cite `T1` table provenance including selected row indices. Model-generated plans are never treated as SQL, Python, shell commands, or expressions; malicious table cells remain data only.

### Evaluation Layer

Module 9 is deliberately outside the request-serving path. The dataset loader reads JSON or JSONL cases with expected documents/pages/chunks, key facts, filters, tags, and explicit no-evidence expectations. `RetrievalEvaluator` calls the existing `SearchService`, so the evaluation path cannot silently drift to a second vector-search implementation. `RAGEvaluator` calls the existing `RAGService`, so context construction, reranking, filters, provider handling, and citation validation remain identical to the API path.

The evaluator records per-case results and aggregates Success@K, Recall@K, MRR, latency, rerank rank movement, key-fact coverage, citation/evidence checks, and no-evidence correctness. It can compare semantic, keyword, hybrid, and reranked configurations with a baseline and apply explicitly requested quality gates. Reports are JSON artifacts under the ignored `evaluation/results/` runtime directory. Evaluation is read-only with respect to application data: it creates no tables, writes no answers or conversations, and requires no migration. Real PostgreSQL/Ollama evaluation is opt-in; normal tests use an isolated PostgreSQL database where available and mocked Ollama HTTP.

#### E1 dataset foundation

Evaluation E1 sits beside, rather than inside, the Module 9 runtime evaluator. The `evaluation/` package normalizes source truth into `EvaluationDocument`, `EvaluationPage`, `EvaluationLayoutRegion`, `EvaluationEntity`, and `EvaluationQAPair` records. Adapters are selected through a registry and write deterministic JSONL manifests plus preparation metadata. The preparation CLI always requires a bounded limit and uses Hugging Face streaming for DocLayNet and FUNSD; DocVQA is intentionally local/manual and reports `DOCVQA_DATA_REQUIRED` when official files are absent.

The prepared corpus is stored under `data/evaluation/processed/`, separate from `data/processed/uploads/` and PostgreSQL. DocLayNet preserves its actual layout arrays and source PDF bytes where available, FUNSD preserves image-derived form pages and ClassLabel entities, and DocVQA preserves official question/answer records. `evaluation_inspect.py` validates manifest records, stable IDs, referenced PDFs, page counts, dataset/split consistency, and applicable ground-truth counts. E1 creates no database records, performs no model training, and makes no final metric claims; the later evaluation phase consumes its manifests.

#### E2 quantitative ingestion, OCR, and layout benchmark

Evaluation E2 consumes those manifests through `evaluation/e2/runner.py` and calls the existing `PDFIngestionService` with a local async-upload adapter. The benchmark intentionally stops before `DocumentManagementService`/repository persistence: PostgreSQL and Alembic are not involved. The same native extraction and selective Tesseract fallback used by the application produces the predictions.

FUNSD reference text is the E1 entity sequence in source order. CER/WER use Levenshtein distance with Unicode NFKC and line-ending normalization; the report also includes a whitespace-collapsed view. DocLayNet layout evaluation scales the source COCO 1025×1025 annotation space into the E1 PDF page dimensions when metadata supplies those dimensions, maps only compatible current layout types, and performs deterministic greedy one-to-one same-class matching at configurable IoU (0.5 by default). Unsupported source labels and prediction types remain visible in the artifacts and are excluded from comparable TP/FP/FN denominators.

Each run writes ignored `summary.json`, `per_document.jsonl`, `metrics.csv`, `report.md`, and `run_metadata.json` under `data/evaluation/results/e2/{run_id}/{dataset}/`. Reliability, machine-readable failure reasons, wall-clock mean/median/P95, text metrics, layout precision/recall/F1/mean matched IoU, metric definitions, OCR executable/version, and relevant settings are captured. The runner never logs or writes full document text to normal result artifacts. E2 is a bounded descriptive benchmark over prepared local data; it is not a claim of broad OCR/layout generalization. The presentation layer is documented separately below and does not alter E2.

#### E3 controlled DocVQA retrieval benchmark

Evaluation E3 is a separate benchmark runner, not a second retrieval implementation. It uses the E1 DocVQA adapter for bounded official question/document preparation, then sends each PDF through the existing `DocumentManagementService`, `PDFIngestionService`, `DocumentIndexingService`, `StructureAwareChunker`, `EmbeddingService`, and PostgreSQL repository. Retrieval is delegated unchanged to `SearchService`, which invokes the current PostgreSQL keyword `tsvector`/GIN search, pgvector cosine search, RRF hybrid fusion, and optional `CrossEncoderReranker`.

The benchmark scopes searches to the run's indexed document UUIDs so ordinary project documents cannot affect the comparison. It writes an explicit source-document/evaluation-ID/DocuIntel-UUID/chunk-ID mapping. Default cleanup verifies every mapped UUID against original filename, checksum, stored filename, and a run-local storage root before deleting; an identity mismatch fails closed. No database schema or migration is used for bookkeeping.

E3 derives binary relevance only after indexing: a chunk is relevant when it belongs to the question's true mapped document and contains one accepted DocVQA answer under the documented literal normalization rules. Missing answer text is `ANSWER_NOT_INDEXED`, not a retrieval failure hidden from coverage. The four methods share the same scorable questions, K values, and corpus filter. Reports contain Recall@1/3/5/10, Hit@K, MRR, document Hit@K, retrieval/reranking/total/wall timings, cold warm-up timings, candidate settings, and absolute/percentage-point deltas.

#### E4 end-to-end DocVQA RAG answer evaluation

Evaluation E4 is a separate read-only benchmark runner that reuses E3's official prepared manifest, deterministic question order, answer-bearing chunk classification, run-owned ingestion/indexing, and fail-closed cleanup. It invokes the production path `RAGService.ask()` twice per question: `SearchService` hybrid retrieval with `rerank=False`, then the same hybrid request with the configured `CrossEncoderReranker` enabled. The current `RAGContextBuilder`, grounded prompt, `OllamaClient`, source labels, and provider errors are therefore measured rather than reimplemented. No answers or benchmark bookkeeping are persisted in PostgreSQL.

E4's primary metrics are DocVQA-compatible ANLS and normalized EM, both reported end-to-end over all attempted questions and conditionally over E3-scorable questions. ANLS uses Unicode NFKC/lowercase character Levenshtein similarity with the official `0.5` normalized-distance cutoff and takes the maximum accepted-answer score; EM uses E3's conservative literal normalization. Controlled statuses preserve `ANSWERED`, `ABSTAINED`, `ANSWER_NOT_INDEXED`, `DOCUMENT_PROCESSING_FAILED`, `RETRIEVAL_NO_RELEVANT_EVIDENCE`, `GENERATION_FAILED`, and `GROUNDING_REJECTED` rather than deleting difficult cases.

Citation metrics are deterministic: emitted labels must resolve to the returned source list, at least one cited source must match the true DocVQA document, and gold-evidence support requires a cited chunk from E3's answer-bearing set. A bounded returned source excerpt is used only for the separate answer-support diagnostic. E4 never calls an LLM judge and does not call Module 11's summary verifier because that verifier is not part of the ordinary `/ask` response path. Retrieval/reranking/generation/total timings are aggregated as mean, median, and P95; unavailable context-build and grounding-verifier timers are explicitly null. One warm-up request per configuration is excluded from measured aggregates. Results are unique, non-overwriting artifacts under `data/evaluation/results/e4/{run_id}/`; E5 and the presentation layer are documented separately and do not alter this runner.

#### E4.1 timeout and answer-format diagnostic

E4.1 is a separate diagnostic runner under `evaluation/e4_1/`. It does not alter `OllamaClient`, the production 120-second default, RAG prompts, retrieval, reranking, context budgets, or model settings. The optional generation timeout is applied to a copied benchmark `Settings` object. The production timeout remains the `httpx.AsyncClient` timeout; there is no production retry or runner timeout. A benchmark-only safety ceiling is used only to prevent a host transport stall from making a diagnostic unbounded and is recorded separately.

E4.1 runs the existing `RAGService` path, records an explicit cold warm-up before any measured questions, and keeps raw production prose separate from a deterministic metric view that strips only supplied citation labels and narrow Markdown presentation markers. It writes `summary.json`, `per_question.jsonl`, `answers.jsonl`, `review_cases.jsonl`, `metrics.csv`, `report.md`, `run_metadata.json`, and `corpus_mapping.jsonl` under `data/evaluation/results/e4_1/{run_id}/`. The verified local diagnostic is controlled-blocked before the measured loop because the local CPU Ollama generation path timed out even for a minimal prompt; no E4.1 answer metrics are inferred from that block, while identical-subset production-timeout counts are projected from the preserved E4 artifacts. E5 and the presentation layer are documented separately and do not change this diagnostic.

#### E5 final benchmark consolidation

Evaluation E5 is a read-only portfolio layer. `evaluation/e5/baseline_manifest.json` explicitly selects the authoritative E2 FUNSD/DocLayNet summaries, E3 retrieval summary, E4 production RAG summary and answers artifact, and E4.1 controlled-blocked summary. `evaluation/e5/loader.py` fails closed on missing or malformed JSON, wrong module/schema/dataset/split/run identity, missing required metrics, and inconsistent blocked status; it never chooses a latest directory automatically.

`evaluation/e5/builder.py` loads metric values and denominators, calculates only explicit E3 reranking deltas, and writes scorecard JSON/CSV, metric provenance JSONL, pipeline-stage and retrieval comparison CSVs, limitations, portfolio claims, and Markdown reports under `data/evaluation/results/e5/{run_id}/`. `MEASURED` zeros remain distinct from `BLOCKED`, `NOT_MEASURED`, and `NOT_APPLICABLE`. The builder rejects unsupported numeric global-accuracy claims and does not invoke application services, databases, model runtimes, or any earlier benchmark runner. E5 is the final evaluation package for this scope; Module 13 consumes it read-only through the presentation layer.

### Engineering quality (Module 12)

Module 12 adds verification and delivery safeguards around the existing architecture rather than another runtime feature layer. Unit tests cover high-risk service, API, reporting, configuration, and HTTP-only Streamlit contracts. PostgreSQL integration tests are gated by `TEST_DATABASE_URL`, while tests requiring the external Ollama service are gated by explicit `RUN_REAL_*` variables. Those gates are documented intentional environment boundaries; they do not replace ordinary mocked-provider tests.

Pytest is configured with project-local `.pytest_tmp` and `.pytest_cache` paths so Windows verification does not depend on the user profile's system temporary directory. Coverage is available with `pytest-cov` using terminal-missing-lines, JSON, XML, and HTML reports. The deterministic performance harness measures chunking, a fixed embedding double, hybrid RRF fusion, and a fixed reranking double; it is a repeatability signal and not a claim about transformer model latency.

The delivery path includes compile checks, `pip check`, database migration verification, and full pytest coverage in GitHub Actions. The backend `Dockerfile` installs the runtime OCR/system libraries, keeps model downloads lazy, runs as a non-root user, exposes `/health` as a container healthcheck, and excludes local data, caches, tests, and secrets through `.dockerignore`. PostgreSQL remains the only service managed by the existing Compose file; Ollama remains an externally managed local HTTP service.

### Modules 10–12.2 request boundary

```text
Browser
   -> Streamlit pages/components
   -> streamlit_app/api/*
   -> FastAPI HTTP routes
   -> existing Module 0–11 services and data stores
```

The frontend is intentionally replaceable: the API remains the contract, and ordinary frontend tests use `httpx.MockTransport` so they do not need PostgreSQL, transformer models, Tesseract, or Ollama. Streamlit is a local presentation dependency, not a second application architecture.

### PostgreSQL and pgvector

The persistence layer stores document metadata, extracted elements, tables, chunks, version references, and citation-related page information. Database access is isolated behind SQLAlchemy repositories and Alembic is the schema lifecycle authority; application startup does not call `create_all()`.

### Module 12.3 comparison

`POST /api/v1/compare` accepts exactly one caller-selected base/target pair. The service loads existing ready indexed chunks and persisted JSONB tables through `DocumentRepository`, normalizes only repeated whitespace and extraction line wrapping, aligns exact text before applying bounded `SequenceMatcher` similarity plus token-overlap checks, and projects added/removed/modified/unchanged evidence with page/chunk provenance. Page number is not the primary alignment key, so moved content can remain unchanged; unrelated blocks are left unmatched rather than forced into a pair.

The same deterministic service aligns tables using header overlap, row identity, page proximity, and table order. It reports table/header/row additions and removals and changed cells, retaining both table IDs, pages, row indices, and columns. Meaningful PDF metadata (`title`, `author`, `subject`, and `keywords`) is compared; volatile database and processing timestamps are excluded. The existing Module 3 `document_versions` table is not used to infer chronology: version mode is explicitly directional from request Base to Target, and no new version-link migration is required.

Only bounded detected change records are sent to the configured Ollama model for an optional concise summary. Source labels are validated against the deterministic evidence set, summary vocabulary is checked against supplied evidence plus a small narrative allowlist, document text is treated as untrusted data, and unsafe/unavailable summaries fall back to deterministic descriptions. Comparison works without Ollama and does not implement visual pixel/redline rendering or async jobs.

### Module 12.4 privacy workflow

The privacy layer reuses the existing `DocumentRepository`, project-local upload storage, persisted page extraction state, and PyMuPDF. Existing native layout elements already persist block bounding boxes (`bbox_x0`, `bbox_y0`, `bbox_x1`, `bbox_y1`); the redaction resolver additionally reopens the untouched source PDF and uses exact PyMuPDF word boxes for substring-safe redaction. Existing OCR persistence stores OCR text and confidence but not word boxes, so OCR detections are reviewable and explicitly non-redactable instead of receiving guessed rectangles.

The flow is:

```text
ready indexed document
    -> deterministic email / phone / IBAN / Luhn-card detectors
    -> stable page/offset/type detection IDs
    -> exact native word-coordinate resolution
    -> user review and explicit detection-ID selection
    -> server-side re-detection and coordinate validation
    -> PyMuPDF redaction annotations + apply_redactions()
    -> new data/processed/redacted artifact
    -> reopen and verify selected text is not extractable
```

`POST /api/v1/documents/{document_id}/pii/detect` never calls an LLM. Phone matching requires a conservative structured format; IBAN candidates must satisfy country-length and mod-97 validation; card candidates must satisfy Luhn. Invoice IDs, dates, quantities, prices, UUIDs, names, and addresses are not broad-match PII in this module. `POST /api/v1/documents/{document_id}/pii/redact` accepts only server-issued detection IDs and never trusts client coordinates. The output is a new UUID-addressed PDF; originals are hashed before/after the operation and are never overwritten. `GET /api/v1/documents/{document_id}/pii/artifacts/{artifact_id}` exposes only generated artifacts under the controlled directory.

Privacy detection logs contain document IDs, counts, types, and pages only; raw matched values are not logged. The Privacy page keeps the workflow scan → review → select → redact → download and does not silently redact all detections. Module 12.4 does not implement names/addresses, probabilistic coverage, visual-only masking, cloud services, or async jobs.

### Module 13 presentation finish

Module 13 is a UI-only finish over the existing public API. `streamlit_app/app.py` provides a coherent top-level navigation of Home, Documents, Ask, Analyze, Compare, Privacy, and Evaluation. Ask groups the existing grounded Q&A, search evidence, and conversation experiences; it does not introduce a second retrieval path or alter request contracts. Shared state helpers clear transient document-specific results when the selected document changes, preventing stale analysis or evidence from being shown for a different document.

The Home page exposes the existing architecture and a main-page system-status panel. Status checks remain HTTP calls to `/health` and `/ready`; the UI does not probe, start, or select Ollama hardware. The configured `OLLAMA_MODEL` is displayed as an informational value. API failures are mapped to concise safe messages at the presentation boundary.

The Evaluation page in `streamlit_app/evaluation.py` is read-only. It loads the explicit `data/evaluation/results/e5/final_baseline_20260821_final/` package (or the validated `DOCUINTEL_E5_RESULTS_DIR` override), fails closed when required artifacts are missing or malformed, and renders stored E2 OCR/layout metrics, E3 retrieval/latency comparisons, reranker deltas, E4 reliability, limitations, and provenance. It uses the artifact's scorecard and CSV values rather than hardcoding benchmark results and never contacts PostgreSQL, transformer models, or Ollama. The page deliberately presents no generic total-system accuracy because E5 defines incompatible metric denominators.

Module 13 adds no database migration, dependency, backend route, OCR/retrieval/model/prompt behavior, conversational capability, authentication, or frontend framework. It is manually accepted as a presentation-only finish over the existing public API.
