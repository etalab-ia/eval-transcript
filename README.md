# eval-transcript

Simple benchmarking service for audio transcription for the French administration.

Initial scope:

- manage benchmark audio files under `data/audio/`
- keep ground-truth transcriptions under `data/ground_truth/`
- store model outputs under `data/transcriptions/`
- compare transcription quality for target models:
  - Whisper via WhisperX
  - Voxtral
  - Kyutai STT (`kyutai/stt-1b-en_fr`)
  - Cohere Transcribe
  - Scribe v2

## Getting started

This project uses [uv](https://docs.astral.sh/uv/) with a Python `src/` layout.

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management and command execution.
- Git for source control and pre-commit hooks.
- The configured Gitleaks hook is installed by `pre-commit` in its managed environment. A standalone `gitleaks` binary is only needed if you want to run full repository scans or use the `gitleaks` CLI directly outside `pre-commit`.

### Development setup

```bash
uv sync
uv run eval-transcript --help
```

Install the pre-commit hook to scan staged changes for secrets with Gitleaks:

```bash
uv run pre-commit install
```

You can also run the staged-changes secret scan manually:

```bash
uv run pre-commit run gitleaks
```

To scan the existing repository contents and history, install the standalone Gitleaks binary and run a direct repository scan:

```bash
gitleaks git --redact --verbose
```

The CLI loads a `.env` file from the current working directory before reading provider configuration. Explicit environment variables already set in the process take precedence, and `--base-url` / `--api-key` flags override both. For local development, copy `.env.example` to `.env` and fill in the secrets:

```bash
cp .env.example .env
```

### oMLX provider

If a local [oMLX](https://github.com/lamalab-org/omlx) server is running with its OpenAI-compatible API on `http://localhost:8000/v1`, set `OMLX_API_KEY` and list available models:

```bash
uv run eval-transcript omlx models
```

Transcribe one audio file through a model alias exposed by `/v1/models`:

```bash
uv run eval-transcript omlx transcribe data/audio/sample.wav \
  --model whisper-large-v3-asr-fp16 \
  --language fr
```

The transcribe command prints text only by default for quick visual comparison against ground-truth transcripts. Use `--json` to print the raw response with segment metadata.

For Cohere Transcribe on Apple Silicon, use the original Cohere model with oMLX's `mlx-audio` STT loader:

```text
CohereLabs/cohere-transcribe-03-2026
```

A smoke test on oMLX `0.3.12` loaded this model and successfully transcribed short English and French WAV files. oMLX exposes the downloaded model as:

```text
cohere-transcribe-03-2026
```

The converted MLX 8-bit candidates are currently not reliable oMLX targets. The `beshkenadze` 8-bit conversion is discovered and loaded, but fails during transcription with a convolution shape mismatch:

```text
beshkenadze/cohere-transcribe-03-2026-mlx-8bit
```

The `mlx-community` mirror is useful for `mlx-speech`, but is not currently a drop-in oMLX candidate:

```text
mlx-community/cohere-transcribe-03-2026-mlx-8bit
```

It stores its runnable files under `mlx-int8/`, so the current oMLX discovery fails to recognize it as a downloaded model. Moving or symlinking those files to the repository root makes oMLX discover the alias, but a smoke test on oMLX `0.3.12` failed during transcription with the same convolution shape mismatch. Treat it as incompatible with oMLX until the upstream conversion or loader changes.

After downloading a candidate locally and restarting or refreshing oMLX model discovery, check the model alias exposed by the local server:

```bash
uv run eval-transcript omlx models
```

Then pass that exact alias to the transcription command. If oMLX exposes the repository name as the alias, the command is:

```bash
uv run eval-transcript omlx transcribe data/audio/sample.wav \
  --model cohere-transcribe-03-2026
```

If oMLX exposes a different alias, use the alias printed by `omlx models` instead. Cohere Transcribe supports French, but on oMLX `0.3.12` the `--language fr` option is currently broken because oMLX maps `fr` to `french` before calling the Cohere loader. Omit `--language` for now; the smoke test transcribed French correctly without a language hint.

To save the text output for later comparison, use `--save`. The file is written to `data/transcriptions/<audio-stem>/omlx__<model>.txt`:

```bash
uv run eval-transcript omlx transcribe data/audio/sample.wav \
  --model whisper-large-v3-asr-fp16 \
  --language fr \
  --save
```

### WhisperX (local, the Transcript production engine)

Transcript (La Suite's meeting transcription) runs **WhisperX** in production: faster-whisper `large-v2` behind a pyannote VAD, served as an OpenAI-compatible HTTP server ([`suitenumerique/meet-whisperx`](https://github.com/suitenumerique/meet-whisperx)). For benchmarking you only need the two components that affect WER — the **ASR model + VAD** — so alignment and diarization can be skipped (they relocate words and add speaker labels, but do not change the transcript text).

WhisperX is not a CLI dependency; it is declared as an optional `whisperx` dependency group in `pyproject.toml`, pinned to the production version (`whisperx==3.8.5`, as used by `suitenumerique/meet-whisperx`). CTranslate2 is **CPU-only on Apple Silicon** (no Metal) and requires **Python < 3.13**, so run the group with `--python 3.12`.

Save this minimal transcription script as `transcribe.py` — VAD = pyannote default (same as prod), no alignment/diarization:

```python
import sys, whisperx
model = whisperx.load_model("large-v2", device="cpu", compute_type="float32", language="fr")
audio = whisperx.load_audio(sys.argv[1])
result = model.transcribe(audio, batch_size=16)
print(" ".join(s["text"].strip() for s in result["segments"]))
```

Save the output as a benchmark column under `data/transcriptions/<audio-stem>/whisperx__large-v2.txt`:

```bash
uv run --python 3.12 --group whisperx python transcribe.py data/audio/sample.mp3 \
  > data/transcriptions/sample/whisperx__large-v2.txt
```

Use `large-v3` to test a model upgrade. WhisperX's default VAD model is hosted by the WhisperX maintainers and loads without a Hugging Face token; pass `vad_method="silero"` to `load_model` if you hit any pyannote gating.

### Kyutai STT (local, via the oMLX provider)

[Kyutai STT](https://kyutai.org/stt) ships `kyutai/stt-1b-en_fr` (English/French, with built-in semantic VAD) and `kyutai/stt-2.6b-en`. It is **local-only**: there is no hosted or OpenAI-compatible HTTP endpoint. File transcription runs through the `moshi` (PyTorch) or `moshi_mlx` (Apple Silicon) packages, and the only server Kyutai ships is a Rust WebSocket streaming server. See [`kyutai-labs/delayed-streams-modeling`](https://github.com/kyutai-labs/delayed-streams-modeling) for the inference scripts.

To benchmark Kyutai alongside the other models, run it behind a small local OpenAI-compatible server that wraps `moshi`/`moshi_mlx` and exposes `GET /v1/models` plus `POST /v1/audio/transcriptions` returning `{"text": ...}` on `http://localhost:8000/v1`, then transcribe through the generic [oMLX provider](#omlx-provider):

```bash
uv run eval-transcript omlx transcribe data/audio/sample.mp3 \
  --model kyutai/stt-1b-en_fr \
  --language fr \
  --save
```

This reuses the existing oMLX OpenAI-compatible client, so no Kyutai-specific provider code is needed. Use `kyutai/stt-1b-en_fr` for French.

#### Native long-form (Apple Silicon, MLX)

For long files, transcribe directly with the `stt_from_file_mlx.py` script from [`kyutai-labs/delayed-streams-modeling`](https://github.com/kyutai-labs/delayed-streams-modeling) (`scripts/`). It uses the MLX Mimi tokenizer and runs the streaming model end to end, letting the model pick segment boundaries with its built-in semantic VAD — no manual chunking. The script is not part of this repo; download it first, then run it with `uv run --script` (its PEP 723 header pins `moshi_mlx`, so it runs in an isolated env):

```bash
curl -O https://raw.githubusercontent.com/kyutai-labs/delayed-streams-modeling/main/scripts/stt_from_file_mlx.py
# Upstream declares --max-steps without type=int, so "8000" arrives as a string and
# crashes in an MLX tensor shape. Patch it to an int before transcribing long files:
perl -pi -e 's/--max-steps", default=4096/--max-steps", type=int, default=4096/' stt_from_file_mlx.py
uv run --script stt_from_file_mlx.py data/audio/sample.mp3 --max-steps 8000
```

Notes:
- The script appends ~2 s of zero padding, so set `--max-steps` to about `ceil((duration_seconds + 2) * 12.5)` plus a small margin (the default `4096` truncates around 5.5 min). It must be an integer — hence the `type=int` patch above; without it, `--max-steps 8000` is passed as a string and crashes.
- Avoid `python -m moshi_mlx.run_inference` for long files: its `rustymimi` tokenizer caps around ~11 minutes of audio.
- The transcript is printed on stdout after the `starting inference ...` line; redirect it and drop the leading log lines to build `data/transcriptions/<audio-stem>/kyutai-native__stt-1b-en_fr.txt`.

### ElevenLabs provider

Set `ELEVENLABS_API_KEY` to use ElevenLabs Speech to Text with Scribe v2. The optional `ELEVENLABS_BASE_URL` can point to a regional ElevenLabs API base URL if needed.

List documented ElevenLabs speech-to-text models:

```bash
uv run eval-transcript elevenlabs models
```

Transcribe one audio or video file through Scribe v2:

```bash
uv run eval-transcript elevenlabs transcribe data/audio/sample.wav \
  --model scribe_v2 \
  --language fr
```

ElevenLabs accepts either ISO-639-1 or ISO-639-3 language hints, so `--language fr` and `--language fra` are both valid French hints. The transcribe command prints text only by default. Use `--json` to print the serialized SDK response with metadata such as words and timestamps, or `--save` to write `data/transcriptions/<audio-stem>/elevenlabs__<model>.txt`.

Optional Scribe v2 controls include `--timestamps-granularity none|word|character`, `--diarize`, `--num-speakers`, `--temperature`, `--seed`, `--no-verbatim`, and `--no-tag-audio-events`.

### Albert API provider

Set `ALBERT_API_KEY` and `ALBERT_BASE_URL` (for example `https://albert.api.etalab.gouv.fr/v1`) to use Albert API's audio transcription endpoint. List available models:

```bash
uv run eval-transcript albert models
```

Transcribe one audio file with Albert's Whisper model:

```bash
uv run eval-transcript albert transcribe data/audio/sample.wav \
  --model openai/whisper-large-v3 \
  --language fr
```

The transcribe command prints text only by default. Use `--json` to print the raw response, or `--save` to write `data/transcriptions/<audio-stem>/albert__<model>.txt`.


### Scaleway provider

Scaleway Generative APIs expose Voxtral through an OpenAI-compatible chat completions endpoint. Set `SCW_SECRET_KEY` and `SCW_DEFAULT_PROJECT_ID`; the CLI derives the project-scoped Generative APIs URL from `SCW_DEFAULT_PROJECT_ID`. Both `scaleway models` and `scaleway transcribe` also accept `--api-key` and `--project-id` to override these without a `.env` (useful in a worktree or CI).

List Voxtral models available through Scaleway Generative APIs:

```bash
uv run eval-transcript scaleway models
```

The `models` command queries the same Generative APIs endpoint used for transcription, so every listed ID can be passed directly to `--model`.

Transcribe one local MP3 or WAV file through Voxtral:

```bash
uv run eval-transcript scaleway transcribe data/audio/sample.mp3 \
  --model voxtral-small-24b-2507 \
  --language fr
```

Voxtral follows the language of its prompt, so the CLI sends a French prompt that explicitly forbids translation. Use `--language` to pin a different target language (it shapes the prompt), or `--prompt` to override the prompt entirely. The transcribe command prints text only by default. Use `--json` to print the raw chat completion response, or `--save` to write `data/transcriptions/<audio-stem>/scaleway__<model>.txt`.

## Data layout

The repository tracks the directory structure only. Audio and generated transcript artifacts are gitignored by default.

```text
data/
├── manifest.md        # benchmark index generated from local data files
├── audio/             # input audio files
├── ground_truth/      # human/ground-truth transcripts
└── transcriptions/    # model-generated transcripts
```


The ground-truth directory was previously named `data/source_truth/`. Existing checkouts can migrate the local directory with:

```bash
uv run eval-transcript data migrate
```

The legacy `--source-truth-dir` scoring flag is kept as a hidden deprecated alias for one release; prefer `--ground-truth-dir` in new scripts.

Generate or refresh the global benchmark manifest after adding local data files:

```bash
uv run eval-transcript manifest sync
```

### Scoring transcripts

Score generated transcripts against ground truth with the jiwer-backed scoring engine:

```bash
uv run eval-transcript score all
```

Score all generated outputs for one sample:

```bash
uv run eval-transcript score sample sample
```

The scorer matches `data/ground_truth/<sample-id>.md` (or `.txt`) with `data/transcriptions/<sample-id>/*.txt` and reports WER, CER, substitution/deletion/insertion counts, and the reference token count. Aggregate WER is computed from total edit counts across all scored transcripts, not by averaging per-transcript WER values. Text, Markdown, and JSON outputs also include provider/model grouped WER summaries for model comparison.

Use `--json` for machine-readable output, or `--normalization raw` to score exact text after Unicode normalization only. The default `standard` normalization is conservative for French: it normalizes Unicode, casing, apostrophe variants, punctuation/symbols, and whitespace while preserving accents.

Use `--normalization standard_numbers` to additionally fold numbers to a canonical form so spelled-out and digit notations match (`cinq`/`5`, `premier`/`1er`, `deux mille cinq cents`/`2 500`). This avoids penalizing a model only for writing numbers differently than the reference; it is useful on number-heavy material (budgets, statistics).

Text output includes top substitutions, insertions, and deletions by default. Use `--top-errors 0` to hide these summaries, or `--align` to append normalized `REF` / `HYP` / `ERR` alignment blocks for each scored transcript.

Use `--format markdown` or `--format csv` for report-friendly output, and `--output PATH` to write the rendered scoring report to a file. `--json` remains available as a shortcut for `--format json`.

`data/manifest.md` uses Markdown with YAML frontmatter to index samples, ground-truth paths, generated outputs, and placeholder metadata such as language, duration, domain, runtime, and real-time factor.

Ground-truth transcripts are matched to a sample by basename and may be either `.txt` or `.md` (for example `data/ground_truth/sample.txt` for `data/audio/sample.wav`).

### Judging semantic severity (LLM-as-a-judge)

WER counts wrong words but is blind to whether an error *changes the meaning* of the meeting (a negation added, a name hallucinated, a whole passage lost). The `judge` command adds a qualitative layer on top of WER: an LLM served by Albert API compares each generated transcript to the ground truth and reports only the divergences that change the sense, graded on a severity scale.

```bash
# Judge every transcript of one sample
uv run eval-transcript judge sample --output data/reports/judge_sample.md

# Judge the whole corpus
uv run eval-transcript judge
```

Severity scale (G0 cosmetic differences are out of scope, already neutralized by WER):

- **G3 — critical**: meaning/polarity inversion (negation added or removed, success↔failure, rhetorical question turned into an assertion), hallucination of a fact/number/person, or `effondrement` (a whole passage lost or replaced by gibberish).
- **G2 — major**: substantial information lost, or a key term garbled into a different referent.
- **G1 — minor**: still recoverable from context (a misspelled but identifiable name, a deformed technical term).

Each finding quotes both sides **verbatim**; a finding whose extract is not a literal substring of both the ground truth and the hypothesis is flagged `non-verbatim` (guards against the judge hallucinating divergences). The report ranks transcripts by a continuous **gravity score** (`G1×1, G2×2, G3×6, effondrement×12`) normalized per 1000 reference words, so audios of different difficulty stay comparable; a coarse verdict (`fidele` / `alterations_mineures` / `sens_degrade` / `inexploitable`) is derived from that score.

Options: `--judge-model` (default `mistral-medium-2508`; the rubric calibration is tuned for it — `openai/gpt-oss-120b` under-detects polarity inversions), `--passes N` for self-consistency (keeps only G3 findings stable across passes), `--hide-g1` to drop minor findings from the detail, `--output PATH` to write the Markdown report. Requires `ALBERT_API_KEY`.
