"""Serveur HTTP OpenAI-compatible minimal exposant Cohere Transcribe.

Runtime : transformers / PyTorch (MPS sur Apple Silicon). Le modèle
`CohereLabs/cohere-transcribe-03-2026` (model_type `cohere_asr`) se charge via
la classe NATIVE `CohereAsrForConditionalGeneration` (transformers >= 5.3), et
NON via le remote-code (`AutoModelForSpeechSeq2Seq` + `trust_remote_code`), dont
le chargement est cassé sous transformers 5.9 (cf. note dans `get_model`).

Particularité vs Kyutai : Cohere gère le long-form TOUT SEUL. Le feature
extractor découpe l'audio au-delà de `max_audio_clip_s`, et `processor.decode`
réassemble les chunks via `audio_chunk_index`. Donc PAS de découpage manuel ici.

On le sert sur un port distinct (8001 par défaut) pour coexister avec le
serveur Kyutai (8000) ; pointer `OMLX_BASE_URL=http://127.0.0.1:8001/v1` côté
eval-transcript pour scorer Cohere.

Endpoints attendus par le provider `omlx` de eval-transcript :
  - GET  /v1/models                -> {"data": [{"id": ...}]}
  - POST /v1/audio/transcriptions  -> {"text": ...}   (multipart: file, model, language)

Lancer :  uv run uvicorn cohere_server:app --host 127.0.0.1 --port 8001
"""
from __future__ import annotations

import os

# Repli CPU op-par-op pour les ops non implémentées sur MPS. AVANT import torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import tempfile
import threading
from pathlib import Path

import librosa
import torch
from fastapi import FastAPI, File, Form, UploadFile
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

MODEL_REPO = "CohereLabs/cohere-transcribe-03-2026"
SERVED_NAME = "cohere-transcribe-03-2026"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# Cohere attend de l'audio à 16 kHz (cf. model card).
SR = 16000
# Langue par défaut des transcriptions (notre corpus est FR).
DEFAULT_LANGUAGE = "fr"
# Tokens générés par chunk interne (le modèle re-découpe le long-form lui-même).
MAX_NEW_TOKENS = 256

app = FastAPI(title="cohere-server")
_model = None
_processor = None
# Sérialise le lazy-load : deux requêtes concurrentes pendant le chargement
# initial chargeraient sinon le modèle 2x (pic mémoire → risque d'OOM).
_load_lock = threading.Lock()


def get_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    with _load_lock:
        if _model is None:
            # `cohere_asr` est intégré nativement à transformers >= 5.x : on
            # utilise la classe native plutôt que le remote-code
            # (`trust_remote_code` + AutoModelForSpeechSeq2Seq), dont le chemin
            # de chargement est cassé sous transformers 5.9
            # (decoder_start_token_id mal appliqué → génération multilingue
            # aberrante en ignorant l'audio). La classe native applique
            # correctement la generation_config.
            _processor = AutoProcessor.from_pretrained(MODEL_REPO)
            # float32 obligatoire : le code Cohere masque l'attention avec -1e9,
            # qui déborde la plage du float16 (max ~65504) → "value cannot be
            # converted to type c10::Half without overflow". Plus lent mais correct.
            _model = CohereAsrForConditionalGeneration.from_pretrained(
                MODEL_REPO, dtype=torch.float32
            ).to(DEVICE)
            _model.eval()
    return _model, _processor


def transcribe_path(path: str, language: str) -> str:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    model, processor = get_model()
    # Le feature extractor découpe lui-même si l'audio dépasse max_audio_clip_s.
    inputs = processor(
        audio=audio, sampling_rate=SR, return_tensors="pt", language=language
    )
    audio_chunk_index = inputs.get("audio_chunk_index")
    inputs = inputs.to(model.device, dtype=model.dtype)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    # `language` est obligatoire au decode dès qu'un audio_chunk_index est
    # présent (réassemblage long-form) ; on le passe systématiquement.
    decoded = processor.decode(
        outputs,
        skip_special_tokens=True,
        audio_chunk_index=audio_chunk_index,
        language=language,
    )
    # decode renvoie une liste (un élément par item du batch).
    text = decoded[0] if isinstance(decoded, (list, tuple)) else decoded
    return text.strip()


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": SERVED_NAME, "object": "model", "owned_by": "cohere"}],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    # `model` et `response_format` sont acceptés pour la compat OpenAI mais
    # ignorés : ce serveur n'expose qu'un seul modèle et ne renvoie que du texte.
    model: str | None = Form(None),
    language: str | None = Form(None),
    response_format: str | None = Form(None),
):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = transcribe_path(tmp_path, language or DEFAULT_LANGUAGE)
    finally:
        os.unlink(tmp_path)
    return {"text": text}
