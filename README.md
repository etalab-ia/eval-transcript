# eval-transcript

Simple benchmarking service for audio transcription for the French administration.

Initial scope:

- manage benchmark audio files under `data/audio/`
- keep source-of-truth transcriptions under `data/source_truth/`
- store model outputs under `data/transcriptions/`
- compare transcription quality for target models:
  - Whisper via WhisperX
  - Voxtral
  - Parakeet

## Development

This project uses [uv](https://docs.astral.sh/uv/) with a Python `src/` layout.

```bash
uv sync
uv run eval-transcript --help
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

The transcribe command prints text only by default for quick visual comparison against source-of-truth transcripts. Use `--json` to print the raw response with segment metadata.

To save the text output for later comparison, use `--save`. The file is written to `data/transcriptions/<audio-stem>/omlx__<model>.txt`:

```bash
uv run eval-transcript omlx transcribe data/audio/sample.wav \
  --model whisper-large-v3-asr-fp16 \
  --language fr \
  --save
```

### Hugging Face Inference Providers

The CLI includes a guarded Hugging Face provider command for Parakeet so the benchmark has a clear path once hosted inference is restored:

```bash
uv run eval-transcript huggingface transcribe data/audio/sample.flac \
  --model nvidia/parakeet-tdt-0.6b-v3
```

Set `HF_TOKEN` to a token with Inference Providers permission. As of the current tested `huggingface_hub` release, this command intentionally fails with a detailed upstream explanation for `nvidia/parakeet-tdt-0.6b-v3`: the Hub model page advertises Together ASR availability, but `huggingface_hub` removed Together `automatic-speech-recognition` support in `v1.16.1` after a multipart upload dependency regression. The relevant upstream PRs are:

- `huggingface/huggingface_hub#4164`: added Together ASR support in `v1.16.0`
- `huggingface/huggingface_hub#4248`: removed Together ASR support in `v1.16.1`, with a note that it should be re-added later

Until Hugging Face publishes a patch release that restores Together ASR for this model, use the local oMLX Parakeet path instead:

```bash
uv run eval-transcript omlx transcribe data/audio/sample.wav \
  --model parakeet-tdt-0.6b-v3 \
  --language fr
```

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

Scaleway Generative APIs expose Voxtral through an OpenAI-compatible chat completions endpoint. Set `SCW_SECRET_KEY` and `SCW_DEFAULT_PROJECT_ID`; the CLI derives the project-scoped Generative APIs URL from `SCW_DEFAULT_PROJECT_ID`.

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
├── source_truth/      # human/source-of-truth transcripts
└── transcriptions/    # model-generated transcripts
```

Generate or refresh the global benchmark manifest after adding local data files:

```bash
uv run eval-transcript manifest sync
```

`data/manifest.md` uses Markdown with YAML frontmatter to index samples, source-truth paths, generated outputs, and placeholder metadata such as language, duration, domain, runtime, and real-time factor.

Source-of-truth transcripts are matched to a sample by basename and may be either `.txt` or `.md` (for example `data/source_truth/sample.txt` for `data/audio/sample.wav`).
