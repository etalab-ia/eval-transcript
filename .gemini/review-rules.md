# Gemini pull request review rules

Review only the pull request diff you are given. Do not request unrelated rewrites or architectural migrations.

## Priorities

1. Correctness: flag bugs that can change scoring, provider output, or persisted benchmark artifacts.
2. Safety: flag secret exposure, unsafe shell usage, missing provider timeouts, or behavior that can leak local benchmark data.
3. CLI ergonomics: expected user/configuration failures should be actionable and should not produce raw tracebacks.
4. Test coverage: ask for targeted tests when behavior changes, especially for scoring, normalization, report rendering, save paths, and provider error parsing.
5. Maintainability: prefer small helpers and clear module boundaries over broad abstractions.

## Project-specific checks

- Python code should remain compatible with Python `>=3.12` and the locked `uv` environment.
- If dependencies change, `pyproject.toml` and `uv.lock` must stay synchronized.
- Do not approve changes that commit real files under `data/audio/`, `data/source_truth/`, or `data/transcriptions/` beyond `.gitkeep` placeholders.
- Provider calls should have explicit timeout/error handling appropriate to the adapter.
- OpenAI-compatible transcription providers should preserve expected response-format behavior where segment metadata is required.
- Scoring changes should preserve aggregate edit-count semantics rather than averaging per-sample WER unless explicitly intended.

## Review format

Return a concise Markdown review with:

- `Summary`: one or two sentences.
- `Findings`: bullet list of actionable findings, each with severity (`blocking`, `important`, or `nit`) and file/path context when possible.
- `Tests`: note what evidence is present or missing.

If there are no actionable findings, say so clearly. Do not invent line numbers or APIs that are not visible in the diff.
