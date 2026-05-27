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
uv run eval-transcript
```

## Data layout

The repository tracks the directory structure only. Audio and generated transcript artifacts are gitignored by default.

```text
data/
├── audio/             # input audio files
├── source_truth/      # human/source-of-truth transcripts
└── transcriptions/    # model-generated transcripts
```
