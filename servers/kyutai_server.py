"""Serveur HTTP OpenAI-compatible minimal exposant Kyutai STT.

Runtime : transformers / PyTorch (MPS sur Apple Silicon), via la variante
`-trfs` du modèle (classes KyutaiSpeechToText* de transformers >= 4.53).
On évite l'API streaming de moshi : la variante transformers se charge et
s'inférence comme un modèle HF classique.

Endpoints attendus par le provider `omlx` de eval-transcript :
  - GET  /v1/models                -> {"data": [{"id": ...}]}
  - POST /v1/audio/transcriptions  -> {"text": ...}   (multipart: file, model, language)

Lancer :  uv run uvicorn kyutai_server:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import os

# Certaines ops du modèle ne sont pas implémentées sur MPS : on autorise le
# repli CPU op-par-op plutôt que de planter. À définir AVANT l'import de torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import tempfile
import threading
from pathlib import Path

import librosa
import torch
from fastapi import FastAPI, File, Form, UploadFile
from transformers import (
    KyutaiSpeechToTextForConditionalGeneration,
    KyutaiSpeechToTextProcessor,
)

MODEL_REPO = "kyutai/stt-1b-en_fr-trfs"
SERVED_NAME = "kyutai/stt-1b-en_fr"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# La variante transformers attend de l'audio à 24 kHz.
SR = 24000
# Découpage manuel SANS overlap aligné sur les silences, comme parakeet-server :
# borne la conso mémoire/temps de `generate` sur des discours longs (jusqu'à
# ~30 min) et évite les coupures en plein mot. Kyutai étant un modèle de
# streaming, chaque segment isolé se transcrit proprement.
SEGMENT_S = 30.0
MIN_SEGMENT_S = 0.3

app = FastAPI(title="kyutai-server")
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
            _processor = KyutaiSpeechToTextProcessor.from_pretrained(MODEL_REPO)
            # float16 sur MPS : ~2x plus rapide que float32, sortie identique sur
            # nos tests FR. Repli float32 si MPS indispo (CPU).
            dtype = torch.float16 if DEVICE == "mps" else torch.float32
            _model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
                MODEL_REPO, torch_dtype=dtype
            ).to(DEVICE)
            _model.eval()
    return _model, _processor


def _silence_aligned_cuts(audio) -> list[int]:
    """Bornes de coupe (~30 s) alignées sur le passage le plus calme proche."""
    seg = int(SEGMENT_S * SR)
    cuts = [0]
    pos = seg
    while pos < len(audio):
        window = audio[max(0, pos - 2 * SR):min(len(audio), pos + 2 * SR)]
        if len(window):
            quietest = int(librosa.feature.rms(y=window, frame_length=1024, hop_length=512).argmin())
            offset = quietest * 512 - len(window) // 2
            pos = pos + offset
        cuts.append(pos)
        pos += seg
    cuts.append(len(audio))
    return cuts


def transcribe_path(path: str) -> str:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    model, processor = get_model()
    min_seg = int(MIN_SEGMENT_S * SR)
    cuts = _silence_aligned_cuts(audio)

    parts: list[str] = []
    for start, end in zip(cuts, cuts[1:]):
        chunk = audio[start:end]
        if len(chunk) < min_seg:
            continue
        inputs = processor(audio=chunk, sampling_rate=SR, return_tensors="pt")
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            output_tokens = model.generate(**inputs)
        text = processor.batch_decode(output_tokens, skip_special_tokens=True)[0].strip()
        if text:
            parts.append(text)
    return " ".join(parts)


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": SERVED_NAME, "object": "model", "owned_by": "kyutai"}],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    # `model`, `language` et `response_format` sont acceptés pour la compat
    # OpenAI mais ignorés : un seul modèle servi, langue gérée par le modèle,
    # sortie texte uniquement.
    model: str | None = Form(None),
    language: str | None = Form(None),
    response_format: str | None = Form(None),
):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = transcribe_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    return {"text": text}
