# DocuIntel Project Specification

## Objective

DocuIntel is an intelligent document processing (IDP) and retrieval-augmented generation (RAG) platform. The implemented system turns native and scanned PDFs into searchable, structured knowledge and provides grounded answers with page-level citations.

Module 0 provides the repository structure, configuration foundations, a minimal health endpoint, and test scaffolding. Module 1 adds validated PDF uploads, UUID-based local storage, PyMuPDF native page extraction, basic PDF metadata, and configurable OCR-candidate detection. Module 2 adds selective Tesseract OCR, OCR confidence, heuristic layout structure, and native table extraction. Module 3 adds PostgreSQL/pgvector persistence and document management. Module 4 adds structure-aware chunks and local embeddings. Module 5 adds semantic, PostgreSQL full-text, and hybrid retrieval. Module 6 adds optional cross-encoder reranking. Module 7 adds grounded single-question RAG through the local Ollama HTTP API. Module 8 adds persisted conversation sessions, bounded multi-turn history, follow-up query rewriting, and conversational grounded RAG. Module 9 adds a read-only evaluation and quality-benchmarking layer. Module 10 adds a Streamlit presentation layer over the existing FastAPI HTTP API. Module 11 adds grounded document summarization and caller-constrained classification. Module 12.1 adds explicit bounded multi-document Q&A semantics while preserving single-document behavior. Module 12.2 adds evidence-grounded single-document structured extraction and deterministic structured table querying. Module 12.3 adds explicit two-document comparison, directional version comparison, deterministic text/table change detection, provenance, and optional grounded summaries. Module 12.4 adds local high-confidence structured PII detection and explicit coordinate-based PDF redaction; agents, async jobs, and later platform capabilities remain future work.

## Implemented Module 1 decisions

- PDF parsing uses PyMuPDF (`fitz`) with native text extraction only.
- Uploads default to a 25 MB maximum and are streamed to `data/processed/uploads/`.
- Physical filenames are generated from UUIDs; user filenames are metadata only.
- Pages with fewer than 20 meaningful native-text characters are flagged for Module 2 OCR fallback.
- Password-protected PDFs are rejected rather than unlocked.

## Implemented Module 2 decisions

- Tesseract is resolved from `TESSERACT_CMD`, PATH, and common Windows installation paths in that order.
- OCR is applied only to insufficient-native-text pages with visual image content; native pages bypass OCR.
- OCR uses PyMuPDF rendering at configurable DPI and `pytesseract.image_to_data()` for word-level confidence.
- Native layout classification is heuristic and uses relative font size, boldness, short labels, and list prefixes.
- Native tables use PyMuPDF table detection; table failures are logged and do not fail the whole document.

## Implemented Module 3 decisions

- PostgreSQL with pgvector is the authoritative metadata store; PDF binaries remain under the existing project-local upload directory.
- SQLAlchemy 2-style models and repositories persist documents, pages, OCR state, layout elements, native tables, future chunks, and document versions.
- Alembic owns schema creation and enables the PostgreSQL `vector` extension through the first migration.
- Exact PDF duplicates are identified by SHA-256 checksum and return HTTP 409 without creating a second database or file record.
- `/health` remains lightweight; `/ready` reports PostgreSQL readiness without exposing credentials.
- The nullable chunk embedding column is `vector(384)` and is populated by Module 4 indexing.

## Implemented Module 4 decisions

- `sentence-transformers/all-MiniLM-L6-v2` is loaded lazily and generates normalized 384-dimensional vectors in configurable batches.
- Chunking groups persisted paragraphs and list items under detected headings, prefers paragraph boundaries, uses sentence/word fallback for oversized units, and applies overlap only within the same section.
- Chunks preserve start/end pages, heading context, content type, OCR provenance, character/token counts, and a deterministic SHA-256 fingerprint.
- Persisted tables become row-oriented table chunks. Index replacement generates all chunks and vectors before deleting old chunks, then commits the replacement and document indexing metadata atomically.
- Chunk APIs expose text and metadata but never raw embedding arrays. RAG answers are generated separately and are not persisted in this module.

## Implemented Module 5 decisions

- Semantic search reuses the Module 4 all-MiniLM-L6-v2 embedding service and performs cosine similarity ordering in PostgreSQL/pgvector; the public score is higher-is-more-similar cosine similarity.
- Keyword search uses a generated English `tsvector` containing chunk text and section heading, `websearch_to_tsquery`, `ts_rank_cd`, and a PostgreSQL GIN index. It does not use `LIKE` as its primary retrieval method.
- Hybrid search retrieves configurable candidate pools and combines semantic and keyword ranks with configurable Reciprocal Rank Fusion. Duplicate chunk IDs are merged deterministically.
- Search filters are applied in SQL for document IDs, content type, OCR provenance, and overlapping page ranges. Search responses exclude raw vectors and return document/chunk/page/section provenance.
- `POST /api/v1/search` defaults to hybrid mode, validates query length, mode, and top-k limits, and accepts opt-in `rerank: true` for Module 6 second-stage scoring. Multilingual search remains future work.

## Implemented Module 6 decisions

- Reranking is a second stage after semantic, keyword, or hybrid SQL retrieval; it does not replace embeddings or move model scoring into PostgreSQL.
- The local `cross-encoder/ms-marco-MiniLM-L6-v2` model is loaded lazily through Sentence Transformers, reused between requests, and scored in configurable batches on CPU or an automatically available device.
- Reranked retrieval uses `max(RERANK_CANDIDATE_COUNT, top_k * RERANK_CANDIDATE_MULTIPLIER)` candidates with a server-side maximum, preserves SQL filters and all Module 5 diagnostic/provenance fields, and returns `base_rank`, final `rank`, raw `rerank_score`, and timing fields.
- Reranking is opt-in by request and unavailable-model failures return a controlled 503; there is no silent fallback that claims reranking occurred.
- Model input is deterministically bounded to the configured maximum representation while stored/API chunk text remains unchanged. No database migration is needed for this module.

## Implemented Module 7 decisions

- `POST /api/v1/ask` accepts one question, optional top-k, existing search mode/filter fields, and an opt-in rerank flag that defaults to true. It delegates retrieval to the existing SearchService.
- The RAG service assigns stable source labels in final-rank order, builds a character-bounded context using `RAG_MAX_CONTEXT_CHARS`, and preserves document/chunk/page/section/rank/reranker metadata in the response.
- The local asynchronous Ollama client calls `/api/generate` with `stream=false`, uses configurable `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT_SECONDS`, and validated `OLLAMA_TEMPERATURE`, and never starts an Ollama process. The default model is `llama3.2:3b` and the default temperature is `0.1`.
- The grounded prompt limits answers to supplied sources, prioritizes higher-ranked relevant evidence, preserves explicit facts and qualifiers, ignores irrelevant retrieved chunks, requires source labels, discloses insufficient evidence, and treats retrieved document text as untrusted data. Unknown source labels are rejected; missing labels are reported with `citations_valid: false`; answers must not contradict explicit evidence.
- Empty retrieval results return a no-evidence answer without calling Ollama. Retrieval, reranking, generation, and total timings are exposed. No answer or conversation history is persisted, and no schema migration is required.

## Implemented Module 8 decisions

- `POST /api/v1/conversations` creates a session, while list, metadata, message-history, delete, and `POST /api/v1/conversations/{conversation_id}/ask` endpoints manage one durable conversation without introducing users or authentication.
- Alembic migration `0004_module8_conversations` adds `conversations` and `messages`. Messages use `user`/`assistant` roles, transactionally allocated positive sequence numbers, a conversation foreign key with `ON DELETE CASCADE`, and indexes for ordered history.
- Conversation turns persist the user question before retrieval and persist the assistant answer only after successful RAG generation. System prompts, retrieved document context, hidden reasoning, and provider payloads are never persisted.
- History selection is deterministic and bounded by `RAG_HISTORY_MAX_MESSAGES` (default 10) and `RAG_HISTORY_MAX_CHARS` (default 6000). The newest selected messages are restored to chronological order before prompting.
- Follow-up questions are rewritten into standalone retrieval queries only when prior history exists. The rewrite uses the configured Ollama model, treats history as untrusted data, and falls back to the original question if rewriting fails. Retrieval still goes through the existing SearchService and Module 6 reranker.
- Conversational prompts keep prior history, the original current question, and retrieved document evidence in separate delimiters. Documents and history are data rather than instructions; only retrieved sources support factual claims and citations remain validated against stable `S1`, `S2`, ... labels.
- The conversational response returns answer, model, source metadata, citations, rewritten retrieval query, message IDs, and history/rewrite/RAG timing fields. No-result retrieval returns the existing controlled no-evidence answer without sending an empty context to answer generation.
- Ordinary tests mock Ollama. Real PostgreSQL/Ollama conversation verification is gated by `RUN_REAL_CONVERSATION_TEST=1`; the local CPU-mode Ollama runtime remains external to DocuIntel.

## Implemented Module 9 decisions

- Evaluation datasets are human-editable JSON or JSONL files containing stable case IDs, questions, expected document/page/chunk labels, expected key facts, filters, tags, and explicit no-evidence cases.
- Retrieval evaluation reuses `SearchService` for semantic, keyword, hybrid, and optional reranked configurations. It reports Success@K, Recall@K, MRR, no-evidence correctness, latency mean/median, and rerank rank movement without changing runtime search behavior.
- RAG evaluation reuses `RAGService`, including its existing context builder, grounded prompt, Ollama client, citation validation, filters, and no-result path. It reports key-fact coverage, citation presence/validity, expected-document citation, evidence support, no-evidence correctness, and generation/total latency.
- The `python -m app.evaluation.cli` runner writes JSON reports under the ignored `evaluation/results/` directory, supports mode comparisons and optional baseline/quality-gate checks, and never persists answers, conversations, or evaluation rows.
- Evaluation reports preserve per-case evidence, final/base ranks, rank deltas, source metadata, errors, and configuration labels. Character-level excerpts are used for lightweight evidence checks; the evaluator does not claim human-level answer quality.
- Module 9 adds no database tables and no Alembic migration. PostgreSQL integration tests use an isolated test database, while Ollama unit/integration tests use mocked HTTP; real local evaluation is explicitly opt-in.

## Implemented Module 10 decisions

- The Streamlit app lives under `streamlit_app/` and is a presentation layer only. It communicates with FastAPI through HTTP and does not access PostgreSQL, SQLAlchemy, repositories, search/RAG/conversation services, transformer models, OCR, or Ollama directly.
- `ApiClient` centralizes relative-path GET/POST JSON requests, PDF multipart upload, DELETE, response decoding, timeout/connection handling, HTTP error details, and safe user-facing failures. API adapters preserve the existing document, search, RAG, and conversation contracts.
- The UI provides Home/status, document upload and processing inspection, page/chunk details, indexing/deletion confirmation, filtered semantic/keyword/hybrid search, grounded Ask with source metadata, and persistent multi-turn conversations.
- `DOCUINTEL_API_BASE_URL` defaults to `http://127.0.0.1:8001`; `DOCUINTEL_API_TIMEOUT_SECONDS` defaults to 180 seconds for local CPU-mode generation. No CORS, authentication, frontend database, cloud LLM, or Ollama process management is added.
- Mocked HTTP tests cover successful calls, multipart upload, API detail errors, timeout/connection failures, search/RAG payload propagation, document actions, and conversation turns. Module 10 adds no database schema or Alembic migration.

## Implemented Module 11 decisions

- `POST /api/v1/documents/{document_id}/summary` analyzes only existing indexed PostgreSQL chunks and supports `brief`, `detailed`, and `bullet_points` styles.
- Summaries use deterministic chunk/page order, approximate character-bounded batches, partial Ollama summaries, and one final synthesis for multi-batch documents. `SUMMARY_BATCH_MAX_CHARS` and `SUMMARY_FINAL_MAX_CHARS` control the budgets.
- `POST /api/v1/documents/{document_id}/classify` accepts two to twenty caller-supplied unique labels, uses Ollama JSON mode, validates `selected_label` and `rationale` with Pydantic, and retries once for an unknown label. No fake confidence score is returned.
- Analysis responses include document identity, pages/chunks represented, stable source metadata, model, content-loading/generation/total timings, and no generated-result persistence. No migration is required.
- Summary and classification prompts use only supplied evidence and explicitly treat PDF text as untrusted data rather than instructions. Summary prompts must not create relationships or assumptions that are not explicitly supported by the indexed document evidence; they preserve ambiguity and report relevant absent details as not specified. Module 11.2 adds structured claim-level verification against original chunks, one bounded repair/re-verification cycle, and a deterministic extractive fallback; malformed/unknown verifier source labels are never accepted. Module 11.2.1 additionally requires exact supporting evidence spans for supported verifier claims, applies a conservative lexical safety boundary after verifier approval, fails closed on verifier timeout/malformed/inconclusive output, and applies a final safe-summary boundary immediately before API projection so one-chunk early returns cannot bypass grounding. Verification timing is returned separately from generation timing. OCR/extraction errors, CPU Ollama latency, long-document batching, and caller label quality remain limitations.
- Ordinary tests mock Ollama and PostgreSQL-backed API tests use the isolated test database. The Streamlit Analyze page communicates only through the public FastAPI HTTP endpoints and keeps results in temporary session state.

## Implemented Module 12.1 decisions

- `POST /api/v1/ask` and the existing conversation ask endpoint use the typed `SearchFilters.document_ids` list for explicit document scope. Omitted IDs mean all indexed documents; a non-empty list means only the selected documents; empty and over-limit lists return controlled validation errors.
- Explicit scopes are bounded by `RAG_MAX_SELECTED_DOCUMENTS` (default 20). The repository validates that every selected document exists, is `ready`, is indexed, and has indexed chunks before retrieval begins. Duplicate IDs are removed deterministically.
- Selected-document retrieval calls the existing `SearchService` once per document with bounded candidates, merges unique chunks, and applies one combined Module 6 reranking pass when requested. Existing semantic, keyword, hybrid, metadata filters, and single-document behavior remain unchanged.
- RAG responses expose `document_scope`, selected document IDs, document IDs represented by retrieved results, and document IDs represented by final source blocks. Stable `S1`, `S2`, ... labels continue to map exactly to returned source metadata.
- Multi-document grounded prompts require independent document assessment, explicit conflict reporting, no assumed agreement, no fabricated support from irrelevant selected documents, and no claim that all selected documents agree unless supplied evidence supports it. Retrieved text remains untrusted data and prompt-injection protection is retained.
- The Streamlit Search, Ask, and Conversations views provide All documents or multiple selected indexed documents. Conversation scope is sent with every turn, so a selected scope is not silently broadened during follow-up questions; no conversation-schema migration is required.
- Module 12.1 adds deterministic unit coverage for scope validation, per-document retrieval, combined reranking, agreement/conflict prompt rules, provenance, and scope propagation. It adds no dependency and no Alembic migration.

## Implemented Module 12.2 decisions

- `POST /api/v1/documents/{document_id}/extract` accepts one ready, indexed document and a bounded list of caller-defined fields. Field definitions contain a safe name, optional description, and `string`, `integer`, `number`, `boolean`, `date`, or `list[string]` type. `STRUCTURED_EXTRACTION_MAX_FIELDS` defaults to 10.
- Extraction evidence comes from the existing ordered indexed chunks and is bounded by `STRUCTURED_EXTRACTION_MAX_CONTEXT_CHARS` (default 12000). Results are transient and include typed values, `found`/`not_found`/`ambiguous` status, stable `S1` source labels, document/chunk/page provenance, and loading/generation/validation/total timings.
- Ollama JSON mode is constrained by strict Pydantic models with forbidden extra keys. Unknown fields/source labels, invalid types, unsupported values, missing evidence, and unsupported ambiguity are rejected; one bounded repair request is allowed. If safe recovery is impossible, all affected fields fail closed to null/not-found rather than exposing guessed data. Document text remains untrusted data and cannot override the schema.
- The existing `document_tables.headers` and `document_tables.rows` JSONB representation is sufficient. `GET /api/v1/documents/{document_id}/tables` inventories tables; the table preview and query routes preserve table/page/document provenance. No migration was required and Alembic remains `0004_module8_conversations`.
- `POST /api/v1/documents/{document_id}/tables/{table_id}/query` accepts a natural-language question or an already constrained plan. Allowed operations are select, filter, min, max, sum, average, count, sort, and top-N. Plans are validated against exact headers and execute only through ordinary Python logic; no SQL, `eval`, `exec`, shell command, or model-generated expression is executed.
- Numeric parsing accepts obvious integer/decimal/currency forms such as `1,200`, `1200.50`, and `€1,200.50`; original display strings remain in returned rows. Table answers derive from deterministic results and cite `T1` with table ID, page, and row indices.
- The Analyze page adds minimal line-oriented Structured extraction and Table query sections over HTTP-only adapters. Normal tests mock Ollama and cover type validation, missing/ambiguous values, injection text, table arithmetic, invalid plans, malicious cells, API contracts, and provenance. Module 12.3 comparison is documented below; Module 13 is documented in its own section.

## Implemented Module 12.4 decisions

- Module 12.4 supports only deterministic high-confidence `email`, conservative `phone_number`, checksum-validated `iban`, and Luhn-validated `credit_card` detections. Names and addresses are intentionally outside this initial scope; ordinary dates, quantities, invoice references, page numbers, and UUIDs are not treated as PII.
- `POST /api/v1/documents/{document_id}/pii/detect` scans one ready document locally and returns stable detection IDs, matched values for review, page/source metadata, offsets, validation method, redactability, verified coordinates where available, counts, and timings. Privacy processing does not require retrieval indexing or an LLM.
- Native PDF detections resolve exact consecutive PyMuPDF word coordinates. Existing persisted layout elements remain the stored structural coordinate source, while redaction uses exact source-PDF word boxes. OCR text is detected but is marked non-redactable because earlier OCR persistence stores text/confidence, not exact word boxes; the service fails closed rather than guessing.
- `POST /api/v1/documents/{document_id}/pii/redact` accepts only explicit server-issued detection IDs, re-runs detection, ignores client coordinates, applies PyMuPDF redaction annotations and `apply_redactions()`, saves a new project-local artifact under `data/processed/redacted/`, reopens it, and verifies selected text is no longer extractable. Original PDFs are never overwritten.
- `GET /api/v1/documents/{document_id}/pii/artifacts/{artifact_id}` serves only a UUID-addressed generated artifact under the controlled redaction directory. No arbitrary filesystem download exists, and no database migration or configuration variable was required.
- The Privacy Streamlit page performs scan → review → explicit selection → redaction → download. It displays matched values because users must review what will be removed, disables non-redactable detections, and maps backend failures to concise messages without tracebacks.
- The synthetic fixture contains fictional email, phone, valid GB IBAN, Luhn-valid test card, invoice reference, quantity, price, and date controls. Normal tests cover checksum validation, false positives, duplicate/fake selections, OCR coordinate failure, prompt-injection text as data, source preservation, real redaction verification, and adapter/API contracts.

## Main planned use cases

- Upload and process native or scanned PDF documents.
- Extract text, layout, tables, metadata, and document structure.
- Search one or more documents with keyword, semantic, and hybrid retrieval.
- Ask grounded questions and receive answers with page-level citations.
- Summarise, classify, compare, and extract structured information from documents.
- Detect and redact personally identifiable information (PII).

Capabilities outside the implemented upload, extraction, OCR, structure, persistence, chunking, embedding, retrieval, reranking, grounded answers, explicit multi-document Q&A, persisted multi-turn conversation slice, evaluation/benchmarking, summarization, classification, structured extraction/table querying, comparison, high-confidence local PII review/redaction, and Streamlit presentation layers remain future work.

## Module roadmap

Modules 0 through 13 and Evaluation E1 through E5 are complete for the current
portfolio scope. Later asynchronous processing, authentication, agents, and
other platform capabilities remain future work.

## Planned technology stack

- Python 3.12
- FastAPI and Uvicorn for the API
- Pydantic Settings for environment-based configuration
- PyMuPDF for the implemented Module 1 native PDF extraction
- pytesseract and Pillow for the implemented Module 2 OCR fallback
- Tesseract OCR as the local Module 2 OCR executable
- PostgreSQL with pgvector for document, chunk, and vector storage
- Sentence Transformers for Module 4 embeddings and Module 6 cross-encoder reranking
- Ollama for Modules 7–11 local grounded answer generation and analysis
- pytest and httpx for automated testing
- Docker for reproducible deployment and development environments

The existing Sentence Transformers dependency supports both Module 4 embeddings and Module 6 CrossEncoder reranking; Module 7 uses the existing `httpx` dependency for Ollama HTTP. No RAG framework or paid/cloud SDK is required.

## Planned AI models and tools

The following model integrations are documented by module:

1. Embeddings (Module 4): `sentence-transformers/all-MiniLM-L6-v2`
2. Reranker (Module 6): `cross-encoder/ms-marco-MiniLM-L6-v2`
3. Local generative LLM (Modules 7–8): `llama3.2:3b` through Ollama

Tesseract OCR is a locally installed executable used by Module 2, not a downloaded model.

## Evaluation E1 dataset foundation

Evaluation Module E1 prepares a bounded, normalized corpus and ground truth outside the normal DocuIntel database. The root `evaluation/` package contains Pydantic schemas, deterministic IDs, JSONL manifest writing, manifest validation, and dataset adapters. `scripts/evaluation_prepare.py` requires an explicit positive `--limit` and supports streaming DocLayNet (`docling-project/DocLayNet-v1.2`) and FUNSD (`nielsr/funsd`) access through the isolated `requirements-eval.txt` dependency. DocVQA is local/manual only: official JSON/JSONL files and referenced images must be placed under `data/evaluation/raw/docvqa/`; missing data returns `DOCVQA_DATA_REQUIRED` rather than using an unofficial substitute.

DocLayNet source PDF bytes are preserved when exposed by the live schema; otherwise the source page image is materialized into a same-sized PDF. Its actual parallel `bboxes`, `category_id`, `pdf_cells`, and metadata fields are normalized as ground-truth layout regions. FUNSD images become same-sized PDFs and its words, boxes, and live `ner_tags` ClassLabel names are preserved as entities. DocVQA questions and accepted answers are normalized and grouped by source image/document. No adapter claims these annotations are DocuIntel predictions.

Prepared output is deterministic JSONL under `data/evaluation/processed/{dataset}/manifest.jsonl`, with `preparation_metadata.json` recording source, split, bound, command, timestamp, and failures. Raw and processed dataset payloads are ignored by Git while empty directory placeholders remain versionable. E1 does not ingest evaluation files into PostgreSQL, does not create a migration, and does not train models.

## Evaluation E2 quantitative ingestion, OCR, and layout benchmark

Evaluation E2 runs the prepared DocLayNet and FUNSD manifests through the existing `PDFIngestionService` in a persistence-free direct runner. It measures reliability (processed, attempted, successful, failed, skipped, success rate, machine-readable reasons), wall-clock processing mean/median/inclusive P95, FUNSD text accuracy, and DocLayNet layout accuracy. It does not modify application behavior, add a database table, add an Alembic migration, or implement Module 13.

FUNSD CER/WER compare the source-order E1 entity text with the actual current page output using Levenshtein edit distance. The normalization policy is Unicode NFKC plus CRLF/CR-to-LF conversion; the benchmark does not lowercase, remove punctuation/numbers, spell-correct, or use semantic matching. Both strict and whitespace-collapsed views are reported. Empty reference and empty prediction score zero; empty reference with non-empty prediction scores one.

DocLayNet uses IoU threshold 0.5 by default and deterministic same-label greedy one-to-one matching. Source COCO coordinates are transformed to the E1 PDF page dimensions from preserved source metadata. Only explicit compatible mappings (`heading`→`Section-header`, `paragraph`→`Text`, `list_item`→`List-item`, `table`→`Table`) contribute to TP/FP/FN; unsupported labels are counted and excluded from the comparable denominator. No confidence scores are invented and no mAP is claimed.

Run artifacts are under ignored `data/evaluation/results/e2/{run_id}/{dataset}/`: `summary.json`, `per_document.jsonl`, `metrics.csv`, `report.md`, and `run_metadata.json`. Metadata records the run command, platform, relevant OCR settings, actual Tesseract executable/version, coordinate and metric definitions, and no document contents or secrets. The CLI supports only bounded prepared DocLayNet and FUNSD manifests; it is not a replacement for Module 9 retrieval/RAG evaluation. E2 results on five-document acceptance subsets are descriptive and not statistically strong generalization claims.

## Evaluation goals

Module 9 provides the evaluation mechanism for measuring:

- Success@K, Recall@K, and mean reciprocal rank (MRR)
- Citation presence and validity, expected-document citation, and evidence support
- Key-fact coverage and explicit no-evidence behavior
- Retrieval, reranking, generation, and total latency, including cold/warm comparisons when run locally
- Baseline deltas and optional quality-gate failures

Dataset-specific results are written by the evaluation CLI and are not hard-coded into the application. The repository contains a small sample dataset for smoke checks; meaningful benchmark claims require a representative human-reviewed dataset.

## Evaluation E3 controlled retrieval baseline

Evaluation E3 measures the current production retrieval behavior on bounded official DocVQA validation data. It does not alter OCR/layout, chunking, embedding, hybrid fusion, candidate limits, reranker model, or API behavior. E3 uses the existing E1 adapter, current PDF ingestion, persistence, indexing, and `SearchService` path. The four configurations are keyword, semantic, hybrid, and hybrid plus the configured CrossEncoder reranker.

Both document and question limits are bounded. The same run-owned indexed corpus, question set, document UUID filter, K values (1, 3, 5, 10), and relevance definition are used for every method. A relevant chunk must belong to the true DocVQA document and contain an accepted answer using Unicode NFKC normalization, casefolding, whitespace collapse, trimming, and literal boundary-aware matching. There is no LLM judgement, embedding-based ground truth, or fuzzy matching.

E3 distinguishes `SCORABLE`, `ANSWER_NOT_INDEXED`, `DOCUMENT_PROCESSING_FAILED`, and `INVALID_GROUND_TRUTH`. Retrieval metrics are calculated primarily over scorable questions while answer-indexability rate and all unscorable reasons remain visible. Reports include Recall@K, Hit@K, MRR, document Hit@K, retrieval/reranking/total/wall latency mean/median/P95, cold/warm notes, candidate settings, and explicit absolute/percentage-point deltas.

Run artifacts are unique and non-overwriting under `data/evaluation/results/e3/{run_id}/`. `corpus_mapping.jsonl` connects source document/image ID, E1 evaluation ID, local PDF, indexed DocuIntel UUID, and chunk IDs. By default E3 removes only run-owned documents after verifying exact identity and run-local storage; `--keep-indexed` is explicit. No migration or evaluation bookkeeping table is added.

The current checkout contains the bounded official DocVQA validation preparation used by the verified E3 baseline. E3 artifacts are historical and must not be overwritten. Evaluation E4 consumes the same manifest in a new run-owned corpus and does not alter the E3 baseline.

## Evaluation E4 end-to-end RAG answer accuracy and grounding

E4 evaluates the existing production `RAGService.ask()` path on official DocVQA validation questions. The two primary configurations are hybrid retrieval with `rerank=false` and hybrid retrieval with the existing CrossEncoder reranker enabled. The configurations share the same run-owned documents, question ordering, accepted answers, filters, top-k, prompts, Ollama model, temperature, and source-label validation. E4 introduces no production retrieval or generation algorithm and normally requires no migration.

The real acceptance command is `python scripts/evaluation_run_e4.py --split validation --document-limit 25 --question-limit 100 --top-k 5`. It writes unique artifacts under `data/evaluation/results/e4/{run_id}/`: summary, per-question status, generated-answer/source records, flattened metrics, Markdown report, run metadata, and corpus mapping. E3's identity-checked cleanup is reused, so ordinary project documents are not deleted.

The primary answer metric is DocVQA-compatible ANLS. After Unicode NFKC and lowercase preprocessing, character Levenshtein distance `d` is divided by `max(len(prediction), len(answer))`; if the normalized distance is `< 0.5`, the score is `1 - normalized_distance`, otherwise it is zero. The question score is the maximum over accepted answers. Empty controlled failures score zero. Normalized EM is reported separately using E3's Unicode NFKC, casefolding, whitespace-collapse, and trimming policy. Both end-to-end/all-attempted and conditional/E3-scorable aggregates are required; conditional metrics must never be presented as overall accuracy.

E4 preserves controlled coverage statuses for answered, abstained, answer-not-indexed, document-processing-failed, retrieval-no-evidence, generation-failed, and grounding-rejected cases. It measures citation presence, source-label reference validity, true-document citation, all-citations true-document rate, answer-bearing gold-chunk citation hit/precision, and a bounded excerpt support diagnostic. It does not use an LLM judge or label citation support as a hallucination rate. Existing AskResponse timings provide retrieval, reranking, generation, and total latency; unavailable context-build and Module 11 grounding-verification timings are null. Mean, median, P95, one warm-up exclusion, failures, and hybrid-versus-reranker deltas are reported. E4 is descriptive over the bounded 25-document/100-question target and does not implement E5 or Module 13.

## Evaluation E4.1 timeout and answer-format diagnostic

E4.1 adds no production behavior. It documents the actual E4 timeout path: `OllamaClient` passes `OLLAMA_TIMEOUT_SECONDS` directly to `httpx.AsyncClient` for the local `/api/generate` request. The production default remains 120 seconds, with no retry, separate generation timeout, or runner timeout. A benchmark-only timeout override may be supplied through `scripts/evaluation_run_e4_1.py --generation-timeout-seconds`; it is applied to a copied settings object and recorded as `NON-PRODUCTION TIMEOUT DIAGNOSTIC`.

The diagnostic selects the first deterministic 20 E3-scorable questions from the bounded 25-document/100-question corpus, warms each configuration before measurement, and preserves `raw_response` alongside a conservative deterministic `metric_answer`. Only provided citation labels, narrow Markdown presentation markers, and whitespace are removed. Explanatory prose is not semantically shortened, and no gold answer, fuzzy matching, LLM judge, or production prompt change is permitted. Deterministic review cases distinguish format mismatch, valid-citation wrong answer, no citation, grounding rejection, empty answer, and unclassified outcomes. If the warm-up cannot complete, the run fails closed before measured questions and reports answer/completion metrics as not measurable rather than fabricating failures.

The controlled local result is `data/evaluation/results/e4_1/e4_1_real_timeout_20260821_blocked/`: Ollama model metadata was reachable, but a minimal direct generation probe timed out after 90.362 seconds and the 300-second RAG warm-up did not complete. The preserved E4 production run supplies same-subset timeout/completion counts for comparison; E4.1 did not overwrite E4 artifacts, change production defaults, or add a migration. E5 and Module 13 remain unimplemented.

## Evaluation E5 final benchmark consolidation

E5 is a read-only consolidation of authoritative existing artifacts. Its explicit `evaluation/e5/baseline_manifest.json` selects E2 FUNSD and DocLayNet, E3 real DocVQA retrieval, E4 production-config RAG, and E4.1 controlled-blocked diagnostic runs by exact repository-relative paths and run IDs. The loader validates artifact existence, JSON shape, module schema, dataset, split, run identity, required metrics, and blocked status; it never substitutes a newer directory automatically.

`scripts/evaluation_build_final_report.py` reads only those artifacts and writes a new `data/evaluation/results/e5/{run_id}/` package containing scorecards, CSV comparison data, metric-level provenance and denominators, limitations, portfolio claims, and Markdown reports. E5 calculates only explicit reranker deltas from loaded E3 values. It preserves measured zero values separately from `BLOCKED`, `NOT_MEASURED`, and `NOT_APPLICABLE` values, and reports no generic total-system accuracy because OCR, layout, retrieval, generation completion, ANLS/EM, and citation metrics are not commensurate. E5 does not invoke Ollama, embeddings, OCR, PostgreSQL, ingestion, or any earlier benchmark runner, and adds no dependency or migration.

## Implemented Module 13 decisions

Module 13 is a presentation-only finish over the existing Streamlit HTTP client. Navigation is organized as Home, Documents, Ask, Analyze, Compare, Privacy, and Evaluation. The Ask page groups existing grounded single/multi-document Q&A, evidence search, and Module 8 conversations in tabs; no backend route or retrieval implementation is duplicated. Home describes the actual extraction/OCR → chunking/embedding → PostgreSQL/pgvector → retrieval/reranking → Ollama pipeline and reports existing API/database status.

The Evaluation page is read-only and consumes only the explicit E5 package at `data/evaluation/results/e5/final_baseline_20260821_final/`, with `DOCUINTEL_E5_RESULTS_DIR` as an optional fixed-directory override. It validates required artifacts, preserves `MEASURED`/`BLOCKED` states, and renders document-understanding, retrieval, reranking-impact, RAG-reliability, limitations, and provenance views from stored values. It does not rerun E5, access application services or PostgreSQL, call Ollama, or calculate a generic system accuracy.

Module 13 uses deterministic pure helpers for document-selection state reset, compact source metadata, E5 metric formatting, and safe API error copy. The UI remains Streamlit-only and HTTP-only. No new dependency, Alembic migration, AI capability, model, prompt, retrieval algorithm, OCR behavior, authentication, frontend framework, or Module 14 functionality is introduced. Implementation is complete and manually accepted.
