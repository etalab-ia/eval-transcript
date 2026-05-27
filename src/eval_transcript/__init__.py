from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_transcript.omlx import DEFAULT_API_KEY_ENV, OmlxClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark French administration audio transcription models.")
    subparsers = parser.add_subparsers(dest="command")

    omlx = subparsers.add_parser("omlx", help="Interact with a local oMLX OpenAI-compatible API")
    omlx_subparsers = omlx.add_subparsers(dest="omlx_command")

    omlx_models = omlx_subparsers.add_parser("models", help="List models exposed by the local oMLX API")
    omlx_models.add_argument("--base-url", default=None, help="OpenAI-compatible oMLX base URL")
    omlx_models.add_argument("--api-key", default=None, help=f"oMLX API key; defaults to ${DEFAULT_API_KEY_ENV}")

    omlx_transcribe = omlx_subparsers.add_parser("transcribe", help="Transcribe one audio file through oMLX")
    omlx_transcribe.add_argument("audio", type=Path, help="Audio file to transcribe")
    omlx_transcribe.add_argument("--model", required=True, help="Model alias exposed by /v1/models")
    omlx_transcribe.add_argument("--language", default=None, help="Optional language hint, e.g. fr")
    omlx_transcribe.add_argument("--base-url", default=None, help="OpenAI-compatible oMLX base URL")
    omlx_transcribe.add_argument("--api-key", default=None, help=f"oMLX API key; defaults to ${DEFAULT_API_KEY_ENV}")

    args = parser.parse_args()

    if args.command == "omlx" and args.omlx_command == "models":
        client = OmlxClient(base_url=args.base_url, api_key=args.api_key)
        for model in client.list_models():
            print(model.id)
        return

    if args.command == "omlx" and args.omlx_command == "transcribe":
        client = OmlxClient(base_url=args.base_url, api_key=args.api_key)
        result = client.transcribe(model=args.model, audio_path=args.audio, language=args.language)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    parser.print_help()
