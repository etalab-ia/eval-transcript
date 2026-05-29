from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from eval_transcript.albert import (
    DEFAULT_API_KEY_ENV as ALBERT_API_KEY_ENV,
    DEFAULT_TRANSCRIPTION_MODEL as ALBERT_DEFAULT_TRANSCRIPTION_MODEL,
    AlbertClient,
    AlbertError,
)
from eval_transcript.manifest import DEFAULT_MANIFEST_PATH, discover_samples, render_manifest
from eval_transcript.omlx import DEFAULT_API_KEY_ENV as OMLX_API_KEY_ENV, OmlxClient, OmlxError
from eval_transcript.scaleway import (
    DEFAULT_MODEL as SCALEWAY_DEFAULT_MODEL,
    DEFAULT_PROMPT as SCALEWAY_DEFAULT_PROMPT,
    ScalewayClient,
    ScalewayError,
    transcription_text as scaleway_transcription_text,
)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Benchmark French administration audio transcription models.")
    subparsers = parser.add_subparsers(dest="command")

    manifest = subparsers.add_parser("manifest", help="Manage the benchmark manifest")
    manifest_subparsers = manifest.add_subparsers(dest="manifest_command")
    manifest_sync = manifest_subparsers.add_parser("sync", help="Write data/manifest.md from current benchmark files")
    manifest_sync.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Manifest path to write")

    albert = subparsers.add_parser("albert", help="Interact with Albert API")
    albert_subparsers = albert.add_subparsers(dest="albert_command")
    albert_models = albert_subparsers.add_parser("models", help="List models exposed by Albert API")
    albert_models.add_argument("--base-url", default=None, help="Albert API base URL")
    albert_models.add_argument("--api-key", default=None, help=f"Albert API key; defaults to ${ALBERT_API_KEY_ENV}")
    albert_transcribe = albert_subparsers.add_parser("transcribe", help="Transcribe one audio file through Albert API")
    albert_transcribe.add_argument("audio", type=Path, help="Audio file to transcribe")
    albert_transcribe.add_argument("--model", default=ALBERT_DEFAULT_TRANSCRIPTION_MODEL, help="Model ID to use")
    albert_transcribe.add_argument("--language", default=None, help="Optional language hint, e.g. fr")
    albert_transcribe.add_argument("--prompt", default=None, help="Optional transcription prompt")
    albert_transcribe.add_argument("--temperature", type=float, default=None, help="Optional sampling temperature between 0 and 1")
    albert_transcribe.add_argument("--base-url", default=None, help="Albert API base URL")
    albert_transcribe.add_argument("--api-key", default=None, help=f"Albert API key; defaults to ${ALBERT_API_KEY_ENV}")
    albert_transcribe.add_argument("--json", action="store_true", help="Print the raw transcription JSON response")
    albert_transcribe.add_argument("--save", action="store_true", help="Write text output to data/transcriptions/<audio-stem>/albert__<model>.txt")
    albert_transcribe.add_argument("--output-dir", type=Path, default=None, help="Directory for saved text output; defaults to data/transcriptions and implies --save")

    scaleway = subparsers.add_parser("scaleway", help="Interact with Scaleway Generative APIs")
    scaleway_subparsers = scaleway.add_subparsers(dest="scaleway_command")
    scaleway_models = scaleway_subparsers.add_parser("models", help="List Scaleway inference models")
    scaleway_models.add_argument("--name", default="voxtral", help="Optional model name filter")
    scaleway_models.add_argument("--region", default=None, help="Scaleway region; defaults to $SCW_DEFAULT_REGION or fr-par")
    scaleway_transcribe = scaleway_subparsers.add_parser("transcribe", help="Transcribe one audio file through Scaleway Voxtral")
    scaleway_transcribe.add_argument("audio", type=Path, help="Audio file to transcribe (.mp3 or .wav)")
    scaleway_transcribe.add_argument("--model", default=SCALEWAY_DEFAULT_MODEL, help="Scaleway model ID to use")
    scaleway_transcribe.add_argument("--prompt", default=SCALEWAY_DEFAULT_PROMPT, help="Prompt sent with the audio input")
    scaleway_transcribe.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    scaleway_transcribe.add_argument("--max-tokens", type=int, default=2048, help="Maximum output tokens")
    scaleway_transcribe.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling value")
    scaleway_transcribe.add_argument("--api-key", default=None, help="Scaleway secret key; defaults to $SCW_SECRET_KEY")
    scaleway_transcribe.add_argument("--json", action="store_true", help="Print the raw chat completion JSON response")
    scaleway_transcribe.add_argument("--save", action="store_true", help="Write text output to data/transcriptions/<audio-stem>/scaleway__<model>.txt")
    scaleway_transcribe.add_argument("--output-dir", type=Path, default=None, help="Directory for saved text output; defaults to data/transcriptions and implies --save")

    omlx = subparsers.add_parser("omlx", help="Interact with a local oMLX OpenAI-compatible API")
    omlx_subparsers = omlx.add_subparsers(dest="omlx_command")
    omlx_models = omlx_subparsers.add_parser("models", help="List models exposed by the local oMLX API")
    omlx_models.add_argument("--base-url", default=None, help="OpenAI-compatible oMLX base URL")
    omlx_models.add_argument("--api-key", default=None, help=f"oMLX API key; defaults to ${OMLX_API_KEY_ENV}")
    omlx_transcribe = omlx_subparsers.add_parser("transcribe", help="Transcribe one audio file through oMLX")
    omlx_transcribe.add_argument("audio", type=Path, help="Audio file to transcribe")
    omlx_transcribe.add_argument("--model", required=True, help="Model alias exposed by /v1/models")
    omlx_transcribe.add_argument("--language", default=None, help="Optional language hint, e.g. fr")
    omlx_transcribe.add_argument("--base-url", default=None, help="OpenAI-compatible oMLX base URL")
    omlx_transcribe.add_argument("--api-key", default=None, help=f"oMLX API key; defaults to ${OMLX_API_KEY_ENV}")
    omlx_transcribe.add_argument("--json", action="store_true", help="Print the raw transcription JSON response")
    omlx_transcribe.add_argument("--save", action="store_true", help="Write text output to data/transcriptions/<audio-stem>/omlx__<model>.txt")
    omlx_transcribe.add_argument("--output-dir", type=Path, default=None, help="Directory for saved text output; defaults to data/transcriptions and implies --save")

    args = parser.parse_args()

    try:
        if args.command == "manifest" and args.manifest_command == "sync":
            manifest_text = render_manifest(discover_samples())
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(manifest_text, encoding="utf-8")
            print(args.manifest)
            return

        if args.command == "albert" and args.albert_command == "models":
            client = AlbertClient(base_url=args.base_url, api_key=args.api_key)
            for model in client.list_models():
                print(model.id)
            return

        if args.command == "albert" and args.albert_command == "transcribe":
            client = AlbertClient(base_url=args.base_url, api_key=args.api_key)
            response_format = "json" if args.json else None
            result = client.transcribe(model=args.model, audio_path=args.audio, language=args.language, prompt=args.prompt, response_format=response_format, temperature=args.temperature)
            text = result.get("text") or ""
            print_or_save_result(result=result, text=text, json_output=args.json, save=args.save, output_dir=args.output_dir, audio_path=args.audio, provider="albert", model=args.model)
            return

        if args.command == "scaleway" and args.scaleway_command == "models":
            client = ScalewayClient(region=args.region)
            for model in client.list_models(name=args.name):
                print(model.name)
            return

        if args.command == "scaleway" and args.scaleway_command == "transcribe":
            client = ScalewayClient(secret_key=args.api_key)
            result = client.transcribe(audio_path=args.audio, model=args.model, prompt=args.prompt, temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p)
            text = scaleway_transcription_text(result)
            print_or_save_result(result=result, text=text, json_output=args.json, save=args.save, output_dir=args.output_dir, audio_path=args.audio, provider="scaleway", model=args.model)
            return

        if args.command == "omlx" and args.omlx_command == "models":
            client = OmlxClient(base_url=args.base_url, api_key=args.api_key)
            for model in client.list_models():
                print(model.id)
            return

        if args.command == "omlx" and args.omlx_command == "transcribe":
            client = OmlxClient(base_url=args.base_url, api_key=args.api_key)
            response_format = "verbose_json" if args.json else None
            result = client.transcribe(model=args.model, audio_path=args.audio, language=args.language, response_format=response_format)
            text = result.get("text") or ""
            print_or_save_result(result=result, text=text, json_output=args.json, save=args.save, output_dir=args.output_dir, audio_path=args.audio, provider="omlx", model=args.model)
            return
    except (FileNotFoundError, AlbertError, ScalewayError, OmlxError, httpx.HTTPError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.command == "manifest":
        manifest.print_help()
    elif args.command == "albert":
        albert.print_help()
    elif args.command == "scaleway":
        scaleway.print_help()
    elif args.command == "omlx":
        omlx.print_help()
    else:
        parser.print_help()


def print_or_save_result(*, result: dict, text: str, json_output: bool, save: bool, output_dir: Path | None, audio_path: Path, provider: str, model: str) -> None:
    if save or output_dir:
        output_path = transcription_output_path(output_dir=output_dir or Path("data/transcriptions"), audio_path=audio_path, provider=provider, model=model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        if json_output:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        print(output_path)
        return
    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(text)


def transcription_output_path(*, output_dir: Path, audio_path: Path, provider: str, model: str) -> Path:
    return output_dir / safe_filename(audio_path.stem) / f"{safe_filename(provider)}__{safe_filename(model)}.txt"


def safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value).strip("._")
