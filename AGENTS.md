# DocuIntel Development Rules

## Before changing the project

- Read `PROJECT_SPEC.md`, `ARCHITECTURE.md`, and `TASKS.md` before making major changes.
- Implement one requested module at a time.
- Do not implement future modules early.
- Preserve existing passing tests.

## Engineering standards

- Python code must use type hints.
- Keep FastAPI routes thin; business logic belongs in services.
- Database access must eventually be separated into repository/data-access layers.
- Use Pydantic models for request and response schemas.
- Never hardcode API keys or secrets; use environment variables.
- Add meaningful logging.
- Handle errors explicitly and do not silently swallow exceptions.
- Prefer clear, simple engineering over unnecessary abstractions.

## Verification and documentation

- Write meaningful pytest tests for new functionality.
- Run tests before declaring a task complete.
- Update documentation when architecture or configuration changes.
- Do not claim planned or scaffolded functionality as implemented.
