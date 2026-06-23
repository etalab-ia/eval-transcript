# Local wrapper servers (Kyutai, Cohere)

Minimal OpenAI-compatible HTTP servers wrapping two **local** ASR models so they
can be benchmarked through the generic `omlx` provider of `eval-transcript`
(same `POST /v1/audio/transcriptions` contract). They are referenced by the
"Kyutai STT" and "Cohere Transcribe" sections of the main `README.md`.

Both expose:
- `GET  /v1/models` → `{"data": [{"id": ...}]}`
- `POST /v1/audio/transcriptions` → `{"text": ...}` (multipart: `file`, `model`, `language`)

| Server | Modèle | Port conseillé | Notes |
| --- | --- | ---: | --- |
| `kyutai_server.py` | `kyutai/stt-1b-en_fr` (variante `-trfs`) | 8000 | Découpage manuel ~30 s aligné sur les silences + `generate()` neuf par segment (la fenêtre de contexte de Kyutai s'effondre au-delà de ~4 min en passe continue). |
| `cohere_server.py` | `CohereLabs/cohere-transcribe-03-2026` | 8001 | Long-form géré par le modèle (le feature extractor découpe, `decode` réassemble via `audio_chunk_index`). |

Les deux servent sur des **ports distincts** → ils peuvent tourner simultanément.

## Lancer

Dépendances : voir `pyproject.toml` (transformers ≥ 5.3 — requis par la classe
native Cohere ; validé sur 5.9 —, torch, librosa, soundfile, sentencepiece,
fastapi, uvicorn, python-multipart).

```bash
# Kyutai (port 8000)
uv run uvicorn kyutai_server:app --host 127.0.0.1 --port 8000

# Cohere (port 8001)
uv run uvicorn cohere_server:app --host 127.0.0.1 --port 8001
```

Puis transcrire via le provider `omlx` (pointer `OMLX_BASE_URL` sur le bon port) :

```bash
OMLX_BASE_URL="http://127.0.0.1:8000/v1" uv run eval-transcript omlx transcribe \
  data/audio/<id>.mp3 --model kyutai/stt-1b-en_fr --language fr --save

OMLX_BASE_URL="http://127.0.0.1:8001/v1" uv run eval-transcript omlx transcribe \
  data/audio/<id>.mp3 --model cohere-transcribe-03-2026 --language fr --save
```

## Note sur le chargement de Cohere

`cohere_server.py` charge le modèle via la **classe native**
`CohereAsrForConditionalGeneration` (intégrée nativement à `transformers ≥ 5.x`),
et **non** via le chemin remote-code (`AutoModelForSpeechSeq2Seq` +
`trust_remote_code=True`). Sous transformers 5.9, le chemin remote-code applique
mal la `generation_config` (`decoder_start_token_id`) : le modèle génère alors
du texte multilingue aberrant **en ignorant l'audio**. La classe native applique
correctement la config et produit la transcription attendue. `float32` reste
obligatoire (le masque d'attention à `-1e9` déborde la plage de `float16`).
