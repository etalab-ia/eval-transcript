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

## Data layout

The repository tracks the directory structure only. Audio and generated transcript artifacts are gitignored by default.

```text
data/
├── audio/             # input audio files
├── source_truth/      # human/source-of-truth transcripts
└── transcriptions/    # model-generated transcripts
```
