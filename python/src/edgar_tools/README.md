# edgar_tools

SEC EDGAR filing acquisition, content-addressed storage, and item-level section extraction.

- **Design**: `.claude/skills/sec-filing-db-coding/SKILL.md` and its `reference/` docs.
- **Build plan**: `.claude/tasks/README.md` (task 01 of 19 — scaffold — is complete; see that
  directory for the full sequence).
- **Status**: skeleton only. The public API (`Store`) lands starting task 03.

Run the module's own gates (scoped, see `.claude/tasks/README.md` for why):

```bash
uv run ruff check src/edgar_tools tests/edgar_tools
uv run ruff format --check src/edgar_tools tests/edgar_tools
uv run pytest tests/edgar_tools
uv run mypy src/edgar_tools
```
