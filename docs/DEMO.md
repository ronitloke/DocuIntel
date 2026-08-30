# DocuIntel Demo

This walkthrough uses the reviewed synthetic PDFs already tracked under
`data/sample_pdfs/`. The bootstrap uploads them through the normal FastAPI
ingestion and indexing endpoints; it never writes directly to PostgreSQL and
never deletes existing documents.

## Prerequisites

- Docker Desktop with the Linux container engine running.
- A local virtual environment with the project dependencies installed.
- Ollama with `llama3.2:3b` available for Q&A, summary, and classification.
- Tesseract available in the API container for the OCR demo.

## Start deployment

From the repository root:

```powershell
docker compose up --build -d
```

The default published URLs are `http://localhost:8001` for FastAPI and
`http://localhost:8501` for Streamlit. If `.env` changes either host port,
pass matching `--api-url` and `--frontend-url` values to the bootstrap command.

## Check deployment

```powershell
python scripts/check_deployment.py
```

PostgreSQL readiness and the API/frontend are required for the demo. Ollama is
optional for startup but is needed by the language-model-backed workflows.

## Bootstrap demo corpus

```powershell
python scripts/bootstrap_demo.py
```

The command reports each fixture as added, already present, or failed. It
uses SHA-256 checksums returned by the existing document-list API to make
repeated runs idempotent. Existing documents are never deleted or recreated.
Upload/extraction is synchronous in the current API, so the script indexes a
successful upload immediately through `POST /api/v1/documents/{id}/index`.

## Open application

Open <http://localhost:8501>, then use the prepared workflows below. If a
fixture is already present, the second bootstrap reports it as already
present rather than creating a duplicate.

## Prepared workflows

### Grounded Q&A

Page: **Ask AI → Grounded Q&A**

Document: `module9-evaluation.pdf`

Settings: Hybrid search, CrossEncoder reranking enabled.

Question: **What is the notice period for resignation?**

The indexed evidence says employees must provide thirty days of written notice
before resignation. The generated answer should cite its returned source; the
exact natural-language wording is model-dependent.

### Search evidence

Page: **Ask AI → Search evidence**

Document: `module9-evaluation.pdf`

Query: **notice period**; mode: **Hybrid**.

Evidence containing the thirty-day resignation notice should rank prominently.
This workflow does not require Ollama.

### Summary

Page: **Analyze → Summary**

Document: `module9-evaluation.pdf`; style: **Brief**.

The result should preserve the document's supported employment notice
information. It should not add unsupported employee or invoice details.

### Classification

Page: **Analyze → Classification**

Document: `module9-evaluation.pdf`; labels: `Employment Policy`, `Expense
Policy`, `Invoice`, `Other`.

The supported expected label is **Employment Policy**.

### Structured extraction

Page: **Analyze → Extraction**

Document: `module9-evaluation.pdf`.

Suggested fields:

| Field | Type | Description | Supported result |
| --- | --- | --- | --- |
| `notice_period` | string | Required resignation notice period | `thirty days` |
| `invoice_reference` | string | Invoice/reference identifier | not found in this policy fixture |
| `employee_name` | string | Employee name | not found |

The not-found results demonstrate fail-closed extraction rather than
hallucinating absent data. The separate privacy fixture contains the synthetic
invoice control `INV-2026-0043`; the current public fixture set does not contain
the originally named combined policy PDF with both that value and the notice
period.

### Table intelligence

Page: **Analyze → Tables**

Document: `layout_table_sample.pdf`.

Ask:

- Which product has the highest quantity? → **Laptop, quantity 3**.
- What is the total price? → **2400**.
- How many rows are there? → **1**.

### Document comparison

Page: **Compare**

Base: `module12_3_base.pdf`; target: `module12_3_target.pdf`.

The accepted comparison fixtures cover the thirty-day to forty-five-day notice
change, remote-work policy change, removed annual training, added expense
claims statement, Laptop quantity/price changes, and the added Keyboard row.
Use the accepted comparison configuration; do not recalculate these results in
the bootstrap.

### Privacy and redaction

Page: **Privacy & Redaction**

Document: `module12_4_pii.pdf`.

Scan for email, phone number, IBAN, and credit card. The fictional fixture
contains one of each accepted test value:

- `privacy.test@example.com`
- `+1 (202) 555-0147`
- `GB82 WEST 1234 5698 7654 32`
- `4111 1111 1111 1111`

Select only email and credit card for redaction. The generated PDF should
remove those two values while preserving the phone and IBAN; the original
fixture remains unchanged. Generated redacted PDFs are runtime artifacts and
must not be committed.

### OCR

Page: **Documents**

Document: `scanned_text_sample.pdf`.

The safe scanned fixture contains the text “Scanned sample recovered through
Tesseract OCR.” Its document detail should report OCR processing. This is an
optional feature-specific check and does not change the core deployment
readiness contract.

### Evaluation

Page: **Evaluation → Retrieval**.

This is a read-only view of the existing bounded authoritative evaluation. The
bootstrap does not run benchmarks or add new evaluation calculations. Existing
reported metrics, including approximately 51.2% Hybrid Recall@1 and the
measured reranking latency cost, remain historical evaluation results.

## Limitations

The corpus is synthetic and intentionally small. Model outputs and CPU
latencies vary. Ollama is an external local service, and the bootstrap does
not start it, download models, or manage hardware selection.
