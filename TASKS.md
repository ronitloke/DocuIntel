# DocuIntel Module Checklist

- [x] Module 0 - Project Bootstrap
- [x] Module 1 - PDF Ingestion
- [x] Module 2 - OCR & Document Structure
- [x] Module 3 - PostgreSQL, pgvector & Document Management
- [x] Module 4 - Structure-Aware Chunking & Embeddings
- [x] Module 5 - Semantic, Keyword & Hybrid Search
- [x] Module 6 - Cross-Encoder Reranking
- [x] Module 7 - RAG, Ollama & Page-Level Citations
- [x] Module 8 - Conversation Sessions & Multi-Turn RAG Memory
- [x] Module 9 - RAG Evaluation & Quality Benchmarking
- [x] Module 10 - Streamlit Web Application
- [x] Module 11 - Document Summarization & Classification
- [x] Module 12 - Engineering Quality
- [x] Module 12.1 - Explicit Multi-Document Q&A Semantics
- [x] Module 12.2 - Structured Information Extraction & Structured Table Querying
- [x] Module 12.2.1 - Streamlit Structured Analysis UI Hotfix
- [x] Module 12.2.2 - Table Query Plan Reliability Hotfix
- [x] Module 12.3 - PDF and Version Comparison + Change Detection
- [x] Module 12.4 - High-Confidence PII Detection & PDF Redaction
- [x] Evaluation E1 - Dataset Foundation & Reproducible Benchmark Harness
- [x] Evaluation E2 - Quantitative Ingestion, OCR & Layout Benchmark
- [x] Evaluation E3 - Controlled DocVQA Retrieval/Reranking Benchmark Infrastructure
- [x] Evaluation E4 - End-to-End DocVQA RAG Answer Accuracy & Grounding
- [x] Evaluation E4.1 - Timeout & Answer-Format Diagnostic (real run controlled-blocked by local Ollama generation stall)
- [x] Evaluation E5 - Final Benchmark Consolidation
- [x] Module 13 - Streamlit UI & Portfolio Finish (complete and manually accepted)

## Module 0 completed

- Created the Python package and future-module directory scaffold.
- Added typed FastAPI application factory and `GET /health` endpoint.
- Added environment-based settings, `.env.example`, and basic startup/shutdown logging.
- Added lightweight runtime and test dependency manifests.
- Added pytest coverage for application creation and the health endpoint.
- Added project specification, architecture, development rules, README, and this checklist.
- Created the Python 3.12 virtual environment and installed Module 0 dependencies.
- Initialized Git with `main` as the primary branch and no remote.

## Module 1 completed

- Added `POST /api/v1/documents/upload` with multipart PDF uploads and Swagger documentation.
- Added extension, MIME type, PDF signature, empty-file, size-limit, corruption, and encryption validation.
- Added UUID-based storage under `data/processed/uploads/` without exposing absolute paths.
- Added PyMuPDF native page text extraction, metadata mapping, page numbering, and character counts.
- Added configurable OCR-candidate detection without calling OCR or Tesseract.
- Added controlled application errors, ingestion logging, cleanup, and generated-PDF test coverage.
- Added PyMuPDF and `python-multipart` as the only new dependencies.

## Module 2 completed

- Added configurable Tesseract resolution, OCR language, and rendering DPI settings.
- Added selective OCR fallback for image-containing pages with insufficient native text.
- Added OCR success/failure state, word-level confidence, and unresolved-page statistics.
- Added heuristic native headings, paragraphs, list items, bounding boxes, font sizes, and bold flags.
- Added PyMuPDF native table detection with headers, rows, bounding boxes, and safe failure handling.
- Added generated native, scanned, mixed, and layout/table sample PDF utility.
- Added mocked and local Tesseract OCR tests while preserving all Module 0/1 tests.

## Module 3 completed

- Added PostgreSQL/pgvector Docker Compose configuration, environment settings, SQLAlchemy models/repository layer, and Alembic migration.
- Added atomic upload persistence for metadata, pages, OCR state, layout elements, and tables, plus SHA-256 duplicate rejection and safe local-file cleanup.
- Added paginated document/page retrieval, detail, page detail, delete, and PostgreSQL readiness APIs.
- Added PostgreSQL-gated integration tests for migration, pgvector, persistence, retrieval, deletion, duplicates, OCR state, and transaction rollback.
- Verified Docker PostgreSQL/pgvector health, Alembic upgrade/downgrade/upgrade, and the isolated integration database.
- Verified 29 tests pass, including five real PostgreSQL integration tests; `pip check` reports no broken requirements.
- Embeddings, vector search, semantic search, RAG, and all later modules remain unimplemented.

## Module 4 completed

- Added configurable structure-aware chunking that preserves heading context, paragraph/list boundaries, page ranges, OCR provenance, table structure, overlap, and deterministic fingerprints.
- Added lazy local `sentence-transformers/all-MiniLM-L6-v2` embedding generation with batching, normalized 384-dimensional vector validation, and clear model errors.
- Added an idempotent transactional document indexing workflow, document indexing metadata, chunk pagination/detail APIs, and a new Alembic migration.
- Verified fake-model endpoint/re-index behavior and a real all-MiniLM-L6-v2 embedding persisted in local pgvector; the full suite reports 37 passing tests.
- Reranking, RAG, Ollama, datasets, and frontend functionality remain unimplemented.

## Module 5 completed

- Added semantic search using the reused all-MiniLM-L6-v2 query embedding service and PostgreSQL/pgvector cosine similarity ranking.
- Added PostgreSQL English full-text search with generated `tsvector` content, `websearch_to_tsquery`, `ts_rank_cd`, and a GIN index in migration `0003_module5_search`.
- Added hybrid retrieval with configurable candidate pools and deterministic Reciprocal Rank Fusion, plus SQL-side document/content/OCR/page filters.
- Added `POST /api/v1/search`, constrained modes, score/provenance response fields, timing/logging, and no raw vector exposure.
- Verified fake-model search/filter/API behavior and real semantic, keyword, and hybrid searches against the local development PostgreSQL/pgvector database.
- Cross-encoder reranking was implemented as an opt-in second stage using `cross-encoder/ms-marco-MiniLM-L6-v2`, bounded candidate pools, lazy reuse, raw relevance scores, base-rank preservation, and retrieval/reranking timing fields.
- Verified semantic, keyword, and hybrid API reranking, filters, failure behavior, and a real CrossEncoder against indexed PostgreSQL/pgvector chunks; the complete database-backed suite contains 53 passing tests.
- RAG, Ollama, datasets, and frontend functionality remain unimplemented.

## Module 7 completed

- Added `POST /api/v1/ask` for grounded single-question answers using the existing SearchService and optional Module 6 reranking.
- Added asynchronous Ollama `/api/generate` client with configurable base URL, model, timeout, non-streaming generation, and controlled provider errors.
- Added deterministic S-labeled context building, character budgeting, grounded prompt templates, prompt-injection protection, source metadata, citation validation, and retrieval/rerank/generation timing.
- Added no-result handling that does not call Ollama, mocked provider/unit/API tests, and PostgreSQL-backed mocked RAG integration coverage.
- Real Ollama verification is environment-gated because ordinary tests must not require the external local service; agents, tools, cloud providers, authentication, and frontend functionality remain unimplemented.

## Module 8 completed

- Added Alembic migration `0004_module8_conversations` with PostgreSQL-backed `conversations` and ordered `messages` tables, user/assistant roles, foreign-key cascade deletion, and sequence/index constraints.
- Added `ConversationRepository` and thin APIs for create, list, get, ordered messages, delete, and `POST /api/v1/conversations/{conversation_id}/ask`.
- Added bounded newest-history selection with chronological prompt order, configurable message/character limits, deterministic first-question titles, and durable turn semantics where user persistence precedes retrieval and assistant persistence follows successful generation.
- Reused Module 5 SearchService and Module 6 reranking through `RAGService.ask_conversational`; added safe Ollama follow-up query rewriting with original-question fallback and no new model/dependency.
- Added explicit history/current-question/evidence prompt delimiters, prompt-injection protection, source/citation validation, response message IDs, retrieval query, and history/rewrite/RAG timings while preserving stateless `/api/v1/ask`.
- Added mocked unit/API tests, PostgreSQL persistence and multi-turn integration tests, and an opt-in real CPU-mode Ollama acceptance test.

## Module 9 completed

- Added human-editable JSON/JSONL evaluation datasets with stable cases, expected evidence labels, key facts, filters, tags, and explicit no-evidence expectations.
- Added read-only retrieval and RAG evaluators that reuse the existing SearchService, Module 6 reranking, RAGService, context builder, Ollama client, and citation validation.
- Added Success@K, Recall@K, MRR, latency mean/median, no-evidence behavior, rerank rank movement, key-fact coverage, citation/source/evidence checks, baseline comparison, and optional quality gates.
- Added `python -m app.evaluation.cli` retrieval, configuration comparison, and RAG commands with ignored JSON reports under `evaluation/results/`.
- Added deterministic unit coverage, isolated PostgreSQL integration coverage, mocked-Ollama RAG coverage, and an opt-in real local evaluation path. Module 10 and later functionality remain unimplemented.

## Module 10 completed

- Added a professional Streamlit presentation layer under `streamlit_app/` that communicates exclusively with the existing FastAPI HTTP API.
- Added centralized HTTP handling for health/readiness, document upload/detail/index/delete, pages, chunks, search, grounded RAG, and persistent conversations.
- Added Home, Documents, Search, Ask Documents, and Conversations pages with backend status, loading states, safe API error messages, empty states, evidence/source display, filters, timings, and explicit destructive-action confirmation.
- Added mocked HTTP unit coverage for request contracts, multipart upload, API errors, timeouts, connection failures, search/RAG payloads, indexing/deletion, and conversation turns.
- Added `DOCUINTEL_API_BASE_URL` and `DOCUINTEL_API_TIMEOUT_SECONDS`; no direct database, model, OCR, or Ollama imports are used by the frontend.
- Corrected project-root `.env` loading and documented the ignored `.env.example` copy step, stable PostgreSQL host port `55432`, normal FastAPI port `8001`, and health/readiness startup checks. Added conservative configurable Ollama temperature (`0.1`) and prompt rules that preserve explicit evidence, ignore irrelevant chunks, and prevent contradictory grounded answers.

## Module 11 completed

- Added grounded `POST /api/v1/documents/{document_id}/summary` with `brief`, `detailed`, and `bullet_points` styles over existing ordered PostgreSQL chunks.
- Added bounded hierarchical summarization with configurable `SUMMARY_BATCH_MAX_CHARS` and `SUMMARY_FINAL_MAX_CHARS`, partial summaries, final synthesis, source metadata, and content/generation/total timings.
- Added constrained `POST /api/v1/documents/{document_id}/classify` using caller-supplied labels, Ollama JSON mode, Pydantic output validation, one invalid-label retry, rationale, source metadata, and timings without fake confidence scores.
- Added explicit document-data prompt-injection protection, no-content/provider error handling, deterministic evidence labels, mocked unit/API coverage, and PostgreSQL-backed integration coverage.
- Added the HTTP-only Streamlit Analyze page for transient summary/classification display. No generated analysis is persisted and no Alembic migration was required.
- Added the Module 11.1 evidence-grounding patch: summary styles prohibit unsupported relationships, applicability assumptions, “standard” interpretations, and a `Grounded Assumptions` section; hierarchical synthesis preserves the same evidence-only rule and reports relevant missing details as not specified. A narrow deterministic safeguard separates unsupported source-identifier relationships without acting as a general factual verifier.
- Added Module 11.2 claim-level grounding verification: structured Ollama JSON claims, exact source-label validation, bounded repair and re-verification, conservative extractive fallback, partial-summary verification before final synthesis, separate grounding timings, detailed-style omissions, and one-bullet-per-line Markdown normalization. Classification remains unchanged and no migration is required.
- Added Module 11.2.1 grounding correctness hotfix: supported verifier claims now require exact cited supporting evidence, a conservative lexical safety boundary rejects new content/actors/numbers/identifiers/relationships/evaluative language, verifier timeout/malformed output fails closed, and `AnalysisService` applies one final safe-summary boundary before API projection. Single-chunk summaries are covered even when final synthesis is skipped; Streamlit receives newline-separated Markdown bullets from the safe result.

## Module 12 completed

- Established a clean baseline and preserved all existing Module 0–11.2.1 behavior and tests.
- Added high-value readiness, reporting, configuration, Streamlit status, and deterministic hot-path tests without requiring PostgreSQL, transformer downloads, or Ollama for ordinary unit execution.
- Added `pytest-cov`, project-local performance benchmarking, terminal/HTML/XML coverage commands, and explicit `performance` / `real_ollama` pytest markers.
- Added a non-root runtime `Dockerfile`, focused `.dockerignore`, Compose default-port alignment at `55432`, and GitHub Actions CI with pgvector, Alembic, compile, coverage, and `pip check` stages.
- Documented intentional environment-gated skips, local Windows pytest directories, coverage interpretation, benchmark limitations, Docker usage, and the Module 12 boundary.
- Module 13 was outside the Module 12 scope; its implementation is recorded below.

## Module 12.1 completed

- Added bounded explicit multi-document scope through the existing typed `SearchFilters.document_ids` contract, including duplicate removal, empty/maximum validation, and ready/indexed repository validation.
- Reused `SearchService` for per-document semantic, keyword, or hybrid retrieval, deterministic candidate merging, and one combined optional CrossEncoder reranking pass.
- Added grounded multi-document prompt rules for independent evidence, agreement, disagreement, irrelevant selected documents, source attribution, and retained prompt-injection protection.
- Added selected/retrieved/source document provenance metadata to RAG responses while preserving stable source labels and single-document behavior.
- Added multi-selection controls to Streamlit Search, Ask, and Conversations; conversation requests carry the selected scope on every follow-up without adding a migration.
- Added deterministic unit tests for multi-document retrieval, reranking, validation, prompt behavior, and provenance. No new dependencies or Alembic migration were required.

## Module 12.2 completed

- Confirmed the existing `document_tables` persistence is structured JSONB headers/rows linked to pages; reused it through repository projections without a schema migration.
- Added bounded single-document structured extraction at `POST /api/v1/documents/{document_id}/extract` with safe field definitions, typed values, found/not-found/ambiguous status, stable source labels, page/chunk provenance, exact evidence validation, one bounded JSON repair attempt, and fail-closed output.
- Added table inventory, bounded preview, and query endpoints. Natural-language plans are finite Pydantic data validated against real headers; deterministic Python execution supports select/filter, min/max, sum, average, count, sort, and top-N without SQL, `eval`, or `exec`.
- Added conservative numeric/currency parsing while preserving original table cell strings and `T1` table/page/row provenance in every query response.
- Added minimal HTTP-only Streamlit Analyze controls for line-oriented extraction fields, structured results/provenance, table selection/preview, natural-language query, deterministic result, and timings.
- Added deterministic unit/API/adapter tests for extraction types, missing and ambiguous values, malformed/unsafe model output, prompt injection, identifier/date confusion, table arithmetic/filtering/sorting, invalid plans/columns, malicious cells, provenance, and UI payloads.
- No new dependency or Alembic migration was required; Alembic remains `0004_module8_conversations`.
- Module 12.3 was outside that earlier checkpoint; Module 13 is recorded below.

## Module 12.2.1 completed

- Confirmed the Streamlit extraction adapter exposes `analysis_api.extract()` and sends the existing Module 12.2 request contract to FastAPI.
- Added scoped UI error handling for extraction, table inventory/preview, and table-query validation failures without exposing raw tracebacks or HTTP `Not Found` text.
- Preserved the normal empty state, `No structured tables were detected for this document.`, for documents with no persisted tables.
- Added adapter URL/query/payload, response, empty-state, and error-message regression tests. No backend contract, dependency, or migration change was required.

## Module 12.2.2 completed

- Resolved obvious count, sum, average, min, max, and safe quantity-alias intents deterministically before Ollama planning, so trivial queries do not depend on small-model JSON formatting.
- Added strict prompt examples and retained finite Pydantic plan validation for non-obvious questions; arithmetic and row selection remain deterministic Python execution.
- Added conservative unknown-column detection that rejects phrases such as `employee salary` without defaulting to the first label column or inventing a field.
- Added count grammar, one-row/header-exclusion, sum, min/max, alias, malformed-provider, unsupported-plan, provenance, and unknown-column regression coverage. No dependency or migration was added.

## Module 12.3 completed

- Added `POST /api/v1/compare` for exactly two explicitly selected ready indexed documents, with generic document and directional version modes.
- Added deterministic whitespace normalization, exact matching, conservative similarity-based modified pairing, added/removed/unchanged categories, meaningful metadata comparison, bounded evidence limits, page/chunk provenance, and timing metadata.
- Reused persisted JSONB table headers/rows for conservative table alignment and deterministic header, row, and cell change detection without asking Ollama to calculate differences.
- Added optional bounded Ollama summaries over detected changes only. Source-label validation, evidence-token validation, prompt-injection rules, and deterministic fallback keep the structured diff usable when Ollama is unavailable or unsafe.
- Added a minimal HTTP-only Streamlit Compare section, safe synthetic fixture generator, deterministic/API/adapter tests, and Module 11.2.1/12.1/12.2 regression coverage. No dependency or migration was added.

## Module 12.4 completed

- Inspected and reused persisted page extraction, native layout bounding boxes, PyMuPDF source PDFs, project-local upload storage, repository loading, FastAPI routing, and HTTP-only Streamlit adapters. Native redaction resolves exact PyMuPDF word coordinates; existing OCR text remains detectable but non-redactable because exact OCR word boxes are not persisted.
- Added deterministic high-confidence detectors for email, conservative phone numbers, checksum-validated IBANs, and Luhn-validated credit cards. Names and addresses remain explicitly out of scope; invoice IDs, dates, quantities, prices, page numbers, and UUIDs are false-positive controls.
- Added `POST /api/v1/documents/{document_id}/pii/detect`, explicit-ID `POST /api/v1/documents/{document_id}/pii/redact`, and controlled generated-artifact download. Privacy uses ready persisted pages and the original PDF, so retrieval indexing is not required. The server re-detects selected IDs, ignores client coordinates, applies PyMuPDF redactions, reopens the output, and verifies selected text is no longer extractable.
- Added project-local `data/processed/redacted/` output with UUID-safe filenames. Original PDFs are never overwritten; no database migration, new dependency, or new environment variable was required.
- Added the review-first Streamlit Privacy page: document/type selection, scan, counts, detection review, redactable status, explicit selection, redaction, and download with safe error messages.
- Added a fictional `module12_4_pii.pdf` fixture and focused detector, coordinate, redaction, API, security, and Streamlit adapter tests. Module 13 is recorded below.

## Evaluation E1 completed

- Added a separate root `evaluation/` preparation package with normalized Pydantic schemas for documents, pages, layout regions, entities, and QA pairs; deterministic evaluation IDs; JSONL manifest writing; preparation metadata; and manifest validation/inspection statistics.
- Added bounded registry-backed adapters for streaming `docling-project/DocLayNet-v1.2`, streaming `nielsr/funsd`, and official local/manual DocVQA data. The adapters preserve source ground truth and materialize local PDFs without asserting that annotations are model predictions.
- Added `scripts/evaluation_prepare.py` and `scripts/evaluation_inspect.py`. Preparation requires `--limit`, never performs an unlimited dataset download, writes under `data/evaluation/processed/`, and reports controlled `DOCVQA_DATA_REQUIRED` when official local files are absent.
- Added isolated `requirements-eval.txt` containing `datasets`; production runtime requirements were not expanded. Raw and processed evaluation payloads are ignored by Git, with empty-directory placeholders retained.
- Added deterministic unit tests for schemas, IDs, manifests, malformed/missing references, adapter normalization, limits, and missing DocVQA behavior. E1 does not ingest the prepared corpus into PostgreSQL or calculate final benchmark metrics.

## Evaluation E2 completed

- Added an isolated benchmark runner under `evaluation/e2/` and `scripts/evaluation_run_e2.py`. It calls the existing `PDFIngestionService` directly, exercises the current PyMuPDF/native extraction and Tesseract OCR path, and never inserts evaluation records into PostgreSQL.
- Added deterministic reliability metrics with attempted/successful/failed/skipped counts, success rate, machine-readable failure reasons, total processing mean/median/P95, conservative FUNSD CER/WER, and one-to-one DocLayNet IoU matching at a configurable threshold.
- Layout evaluation explicitly maps `heading`→`Section-header`, `paragraph`→`Text`, `list_item`→`List-item`, and `table`→`Table`; unsupported source/prediction labels are counted and excluded rather than silently mapped. DocLayNet COCO coordinates are scaled to the prepared PDF page dimensions using the source metadata preserved by E1.
- Each run writes `summary.json`, `per_document.jsonl`, `metrics.csv`, `report.md`, and `run_metadata.json` under ignored `data/evaluation/results/e2/{run_id}/{dataset}/`. Metadata records metric definitions, OCR executable/version, settings relevant to the run, platform, and command without document contents or secrets.
- Added focused tests for normalization, edit distance, empty-reference behavior, deterministic class-aware matching, unsupported labels, and complete artifact writing. The bounded real acceptance run used five DocLayNet validation documents and five FUNSD test documents. Module 13 is outside the E2 scope.

## Evaluation E3 completed

- Added a separate `evaluation/e3/` runner and `scripts/evaluation_run_e3.py` that use the existing E1 DocVQA adapter and the production `DocumentManagementService`, `DocumentIndexingService`, `SearchService`, PostgreSQL full-text/vector/hybrid paths, and configured CrossEncoder.
- Added bounded `--document-limit`, `--question-limit`, and `--top-k` controls. The four fixed comparisons are keyword, semantic, hybrid, and hybrid+rereanker with the same corpus filter, question set, K values, and binary answer-bearing-chunk relevance definition.
- Added literal normalized answer matching, SCORABLE/ANSWER_NOT_INDEXED/DOCUMENT_PROCESSING_FAILED/INVALID_GROUND_TRUTH states, Recall@1/3/5/10, Hit@K, MRR, document Hit@K, latency mean/median/P95, cold/warm timing metadata, and explicit hybrid/reranker deltas.
- Added run-owned source→E1→DocuIntel document→chunk mapping, fail-closed cleanup identity checks, unique non-overwriting result directories, and ignored JSON/JSONL/CSV/Markdown artifacts under `data/evaluation/results/e3/`.
- Added focused metric, question-selection, controlled-state, serialization, and cleanup-safety tests. The verified repository contains the official bounded DocVQA preparation used for the real E3 Retrieval Baseline V1; E3 artifacts remain historical and are not overwritten.

## Evaluation E4 completed

- Added `evaluation/e4/` and `scripts/evaluation_run_e4.py` for bounded real DocVQA end-to-end answer accuracy and grounding evaluation.
- Reused E3's manifest, deterministic question order, answer-bearing chunk ground truth, production ingestion/indexing/search/RAG/Ollama path, and fail-closed run-owned cleanup. No production retrieval, prompt, model, or migration change was made.
- Added DocVQA-compatible ANLS, normalized EM, end-to-end versus scorable metrics, abstention/coverage/failure statuses, citation/source-label/document/gold-evidence metrics, bounded excerpt support diagnostics, latency aggregates, warm-up metadata, reranker deltas, and required JSONL/CSV/Markdown artifacts under `data/evaluation/results/e4/{run_id}/`.
- Added seven focused unit tests for ANLS threshold/accepted answers/normalization, citation parsing, answer/failure accounting, response projection, reranking deltas, and controlled artifact serialization. Ordinary tests do not call Ollama.
- E4 does not implement Evaluation E5 or Module 13.

## Evaluation E4.1 completed

- Added `evaluation/e4_1/` and `scripts/evaluation_run_e4_1.py` for timeout-path inspection, benchmark-only generation-timeout overrides, explicit cold warm-up metadata, conservative raw/metric answer preservation, deterministic review buckets, and auditable diagnostic artifacts.
- Production `OLLAMA_TIMEOUT_SECONDS=120`, model, prompts, retrieval, reranking, context, grounding rules, and E4 artifacts remain unchanged. The optional diagnostic timeout is applied only to a copied benchmark settings object; no dependency or Alembic migration was added.
- Added focused synthetic tests for citation/Markdown cleanup, raw versus metric answers, timeout-setting isolation, completion/timeout accounting, review classification, and controlled artifact serialization.
- The real local diagnostic is controlled-blocked before its measured question loop: Ollama metadata is reachable, but minimal and grounded generation did not complete within the diagnostic window. The artifact records “not measurable” E4.1 answer metrics and preserves same-subset production-timeout counts from the immutable E4 run. E5 and Module 13 are documented in their own sections.

## Evaluation E5 completed

- Added the explicit `evaluation/e5/baseline_manifest.json` and fail-closed loader for the authoritative E2 FUNSD/DocLayNet, E3, E4, and E4.1 artifacts.
- Added the read-only `evaluation/e5/` builder and `scripts/evaluation_build_final_report.py`. It preserves metric provenance, dataset/split/run identity, denominators, measured zeros, blocked states, limitations, calculated E3 reranker deltas, portfolio claims, and chart-ready CSV data.
- Generated the real package under `data/evaluation/results/e5/final_baseline_20260821_final/` without rerunning E2, E3, E4, or E4.1. No production algorithm, model, prompt, setting, dependency, schema, or migration changed. A generic total-system accuracy number is intentionally not reported. Module 13 consumes this package read-only.

## Module 13 implementation and manual acceptance complete

- Refined the existing HTTP-only Streamlit presentation layer into Home, Documents, Ask, Analyze, Compare, Privacy, and Evaluation navigation. Ask groups grounded Q&A, search evidence, and persistent conversations without changing backend contracts.
- Added the polished Home overview with capability cards, the actual pipeline, API/database status, and configured Ollama model information. The UI never starts Ollama or accesses backend infrastructure directly.
- Added read-only E5 artifact loading and Evaluation presentation from the fixed `data/evaluation/results/e5/final_baseline_20260821_final/` directory, including document-understanding metrics, retrieval/latency charts, reranker impact, RAG reliability, limitations, and provenance. Missing/malformed artifacts fail closed.
- Added deterministic tests for E5 loading/formatting, blocked-state display, source metadata, document-selection state reset, safe API error copy, and navigation. No backend algorithm, model, prompt, dependency, migration, or future-module functionality was added.
- Module 13 was manually accepted through the local Streamlit process for Home/status, Documents, Ask, Analyze, Compare, Privacy, and Evaluation. No later module is implemented.
