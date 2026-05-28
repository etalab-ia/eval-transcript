from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

from eval_transcript.manifest import DEFAULT_MANIFEST_PATH, discover_samples, render_manifest
from eval_transcript.omlx import DEFAULT_API_KEY_ENV, OmlxClient, OmlxError


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark French administration audio transcription models.")
    subparsers = parser.add_subparsers(dest="command")

    manifest = subparsers.add_parser("manifest", help="Manage the benchmark manifest")
    manifest_subparsers = manifest.add_subparsers(dest="manifest_command")

    manifest_sync = manifest_subparsers.add_parser("sync", help="Write data/manifest.md from current benchmark files")
    manifest_sync.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Manifest path to write")

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
    omlx_transcribe.add_argument("--json", action="store_true", help="Print the raw transcription JSON response")
    omlx_transcribe.add_argument(
        "--save",
        action="store_true",
        help="Write text output to data/transcriptions/<audio-stem>/omlx__<model>.txt",
    )
    omlx_transcribe.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saved text output; defaults to data/transcriptions and implies --save",
    )

    args = parser.parse_args()

    try:
        if args.command == "manifest" and args.manifest_command == "sync":
            manifest_text = render_manifest(discover_samples())
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(manifest_text, encoding="utf-8")
            print(args.manifest)
            return

        if args.command == "omlx" and args.omlx_command == "models":
            client = OmlxClient(base_url=args.base_url, api_key=args.api_key)
            for model in client.list_models():
                print(model.id)
            return

        if args.command == "omlx" and args.omlx_command == "transcribe":
            client = OmlxClient(base_url=args.base_url, api_key=args.api_key)
            response_format = "verbose_json" if args.json else None
            result = client.transcribe(
                model=args.model,
                audio_path=args.audio,
                language=args.language,
                response_format=response_format,
            )
            text = result.get("text") or ""
            if args.save or args.output_dir:
                output_path = transcription_output_path(
                    output_dir=args.output_dir or Path("data/transcriptions"),
                    audio_path=args.audio,
                    provider="omlx",
                    model=args.model,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(text, encoding="utf-8")
                if args.json:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                    return
                print(output_path)
                return
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return
            print(text)
            return
    except (FileNotFoundError, OmlxError, httpx.HTTPError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.command == "manifest":
        manifest.print_help()
    elif args.command == "omlx":
        omlx.print_help()
    else:
        parser.print_help()


def transcription_output_path(*, output_dir: Path, audio_path: Path, provider: str, model: str) -> Path:
    return output_dir / safe_filename(audio_path.stem) / f"{safe_filename(provider)}__{safe_filename(model)}.txt"


def safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value).strip("._")
