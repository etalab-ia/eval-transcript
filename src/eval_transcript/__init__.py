from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from eval_transcript.data_migrate import DataMigrationError, migrate_source_truth_to_ground_truth
from eval_transcript.albert import (
    DEFAULT_API_KEY_ENV as ALBERT_API_KEY_ENV,
    DEFAULT_TRANSCRIPTION_MODEL as ALBERT_DEFAULT_TRANSCRIPTION_MODEL,
    AlbertClient,
    AlbertError,
)
from eval_transcript.elevenlabs import (
    DEFAULT_API_KEY_ENV as ELEVENLABS_API_KEY_ENV,
    DEFAULT_BASE_URL_ENV as ELEVENLABS_BASE_URL_ENV,
    DEFAULT_MODEL as ELEVENLABS_DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS as ELEVENLABS_DEFAULT_TIMEOUT_SECONDS,
    ElevenLabsClient,
    ElevenLabsError,
    elevenlabs_transcription_text,
)
from eval_transcript.judge import DEFAULT_JUDGE_MODEL, JudgeError
from eval_transcript.judge_cli import (
    JudgeCliError,
    render_markdown as render_judge_markdown,
    run_judge,
    write_or_print_report as write_or_print_judge_report,
)
from eval_transcript.openrouter import (
    DEFAULT_JUDGE_MODEL as OPENROUTER_DEFAULT_JUDGE_MODEL,
    OpenRouterClient,
)
from eval_transcript.panel import (
    parse_judge_spec,
    render_panel_markdown,
    run_panel,
)
from eval_transcript.manifest import DEFAULT_MANIFEST_PATH, discover_samples, render_manifest
from eval_transcript.omlx import DEFAULT_API_KEY_ENV as OMLX_API_KEY_ENV, OmlxClient, OmlxError
from eval_transcript.scaleway import (
    DEFAULT_MODEL as SCALEWAY_DEFAULT_MODEL,
    ScalewayClient,
    ScalewayError,
    build_prompt as scaleway_build_prompt,
    transcription_text as scaleway_transcription_text,
)
from eval_transcript.scoring import NormalizationMode
from eval_transcript.scoring_cli import (
    DEFAULT_GROUND_TRUTH_DIR as SCORING_DEFAULT_GROUND_TRUTH_DIR,
    DEFAULT_TRANSCRIPTIONS_DIR as SCORING_DEFAULT_TRANSCRIPTIONS_DIR,
    ScoringError,
    render_scores_output,
    score_all_outputs,
    score_sample_outputs,
    write_or_print_score_output,
)
from eval_transcript.transcriptions import TranscriptionOutput, print_transcription_output, transcription_text


DEPRECATED_SOURCE_TRUTH_FLAG_MESSAGE = (
    "Warning: --source-truth-dir is deprecated; use --ground-truth-dir instead."
)


def resolve_ground_truth_dir(args: argparse.Namespace) -> Path:
    source_truth_dir = getattr(args, "source_truth_dir", None)
    if source_truth_dir is not None:
        print(DEPRECATED_SOURCE_TRUTH_FLAG_MESSAGE, file=sys.stderr)
        return source_truth_dir
    return args.ground_truth_dir


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Benchmark French administration audio transcription models.")
    subparsers = parser.add_subparsers(dest="command")

    manifest = subparsers.add_parser("manifest", help="Manage the benchmark manifest")
    manifest_subparsers = manifest.add_subparsers(dest="manifest_command")
    manifest_sync = manifest_subparsers.add_parser("sync", help="Write data/manifest.md from current benchmark files")
    manifest_sync.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Manifest path to write")

    data = subparsers.add_parser("data", help="Manage local benchmark data")
    data_subparsers = data.add_subparsers(dest="data_command")
    data_migrate = data_subparsers.add_parser("migrate", help="Migrate data/source_truth to data/ground_truth")
    data_migrate.add_argument("--source", type=Path, default=Path("data/source_truth"), help="Legacy source_truth directory to move")
    data_migrate.add_argument("--target", type=Path, default=Path("data/ground_truth"), help="New ground_truth directory to create")
    data_migrate.add_argument("--force", action="store_true", help="Merge into an existing ground_truth directory")

    score_parent = argparse.ArgumentParser(add_help=False)
    score_parent.add_argument("--ground-truth-dir", type=Path, default=SCORING_DEFAULT_GROUND_TRUTH_DIR, help="Directory containing ground truth .md or .txt files")
    score_parent.add_argument("--source-truth-dir", type=Path, default=None, help=argparse.SUPPRESS)
    score_parent.add_argument("--transcriptions-dir", type=Path, default=SCORING_DEFAULT_TRANSCRIPTIONS_DIR, help="Directory containing generated transcript outputs")
    score_parent.add_argument("--normalization", choices=[mode.value for mode in NormalizationMode], default=NormalizationMode.STANDARD.value, help="Normalization mode used before scoring")
    score_parent.add_argument("--json", action="store_true", help="Print machine-readable scoring JSON")
    score_parent.add_argument("--format", choices=["text", "json", "markdown", "csv"], default="text", help="Output format; --json is a shortcut for --format json")
    score_parent.add_argument("--output", type=Path, default=None, help="Write scoring output to this path instead of stdout")
    score_parent.add_argument("--align", action="store_true", help="Show normalized REF/HYP/ERR alignment blocks in text output")
    score_parent.add_argument("--top-errors", type=int, default=10, help="Number of top substitutions, insertions, and deletions to show in text output; use 0 to hide")

    score = subparsers.add_parser("score", help="Score generated transcripts against ground truth")
    score_subparsers = score.add_subparsers(dest="score_command")
    score_sample = score_subparsers.add_parser("sample", parents=[score_parent], help="Score all generated transcripts for one sample")
    score_sample.add_argument("sample_id", help="Sample ID matching data/ground_truth/<sample-id>.md")
    score_subparsers.add_parser("all", parents=[score_parent], help="Score all ground truth/generated transcript pairs")

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
    albert_transcribe.add_argument("--timeout", type=float, default=None, help="HTTP timeout in seconds; raise it for long audio (default: 120)")
    albert_transcribe.add_argument("--json", action="store_true", help="Print the raw transcription JSON response")
    albert_transcribe.add_argument("--save", action="store_true", help="Write text output to data/transcriptions/<audio-stem>/albert__<model>.txt")
    albert_transcribe.add_argument("--output-dir", type=Path, default=None, help="Directory for saved text output; defaults to data/transcriptions and implies --save")

    scaleway = subparsers.add_parser("scaleway", help="Interact with Scaleway Generative APIs")
    scaleway_subparsers = scaleway.add_subparsers(dest="scaleway_command")
    scaleway_models = scaleway_subparsers.add_parser("models", help="List Scaleway Generative APIs models")
    scaleway_models.add_argument("--name", default="voxtral", help="Optional model name filter")
    scaleway_models.add_argument("--api-key", default=None, help="Scaleway secret key; defaults to $SCW_SECRET_KEY")
    scaleway_models.add_argument("--project-id", default=None, help="Scaleway project ID; defaults to $SCW_DEFAULT_PROJECT_ID")
    scaleway_transcribe = scaleway_subparsers.add_parser("transcribe", help="Transcribe one audio file through Scaleway Voxtral")
    scaleway_transcribe.add_argument("audio", type=Path, help="Audio file to transcribe (.mp3 or .wav)")
    scaleway_transcribe.add_argument("--model", default=SCALEWAY_DEFAULT_MODEL, help="Scaleway model ID to use")
    scaleway_transcribe.add_argument("--language", default=None, help="Optional language hint, e.g. fr; shapes the default prompt")
    scaleway_transcribe.add_argument("--prompt", default=None, help="Override the generated transcription prompt")
    scaleway_transcribe.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    scaleway_transcribe.add_argument("--max-tokens", type=int, default=2048, help="Maximum output tokens")
    scaleway_transcribe.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling value")
    scaleway_transcribe.add_argument("--api-key", default=None, help="Scaleway secret key; defaults to $SCW_SECRET_KEY")
    scaleway_transcribe.add_argument("--project-id", default=None, help="Scaleway project ID; defaults to $SCW_DEFAULT_PROJECT_ID")
    scaleway_transcribe.add_argument("--timeout", type=float, default=None, help="HTTP timeout in seconds; raise it for long audio (default: 120)")
    scaleway_transcribe.add_argument("--json", action="store_true", help="Print the raw chat completion JSON response")
    scaleway_transcribe.add_argument("--save", action="store_true", help="Write text output to data/transcriptions/<audio-stem>/scaleway__<model>.txt")
    scaleway_transcribe.add_argument("--output-dir", type=Path, default=None, help="Directory for saved text output; defaults to data/transcriptions and implies --save")

    elevenlabs = subparsers.add_parser("elevenlabs", help="Interact with ElevenLabs Speech to Text")
    elevenlabs_subparsers = elevenlabs.add_subparsers(dest="elevenlabs_command")
    elevenlabs_models = elevenlabs_subparsers.add_parser("models", help="List documented ElevenLabs speech-to-text models")
    elevenlabs_models.add_argument("--api-key", default=None, help=f"ElevenLabs API key; defaults to ${ELEVENLABS_API_KEY_ENV}")
    elevenlabs_models.add_argument("--base-url", default=None, help=f"Optional ElevenLabs API base URL; defaults to ${ELEVENLABS_BASE_URL_ENV} if set")
    elevenlabs_transcribe = elevenlabs_subparsers.add_parser("transcribe", help="Transcribe one audio file through ElevenLabs Speech to Text")
    elevenlabs_transcribe.add_argument("audio", type=Path, help="Audio or video file to transcribe")
    elevenlabs_transcribe.add_argument("--model", default=ELEVENLABS_DEFAULT_MODEL, help="ElevenLabs speech-to-text model ID to use")
    elevenlabs_transcribe.add_argument("--language", default=None, help="Optional language hint, e.g. fr or fra")
    elevenlabs_transcribe.add_argument("--timestamps-granularity", choices=["none", "word", "character"], default=None, help="Optional timestamp granularity returned by ElevenLabs")
    elevenlabs_transcribe.add_argument("--diarize", action="store_true", default=None, help="Enable speaker diarization")
    elevenlabs_transcribe.add_argument("--num-speakers", type=int, default=None, help="Optional maximum speaker count for diarization")
    elevenlabs_transcribe.add_argument("--no-tag-audio-events", action="store_false", dest="tag_audio_events", default=None, help="Disable audio event tags such as laughter or footsteps")
    elevenlabs_transcribe.add_argument("--temperature", type=float, default=None, help="Optional transcription temperature between 0.0 and 2.0")
    elevenlabs_transcribe.add_argument("--seed", type=int, default=None, help="Optional deterministic sampling seed")
    elevenlabs_transcribe.add_argument("--no-verbatim", action="store_true", default=None, help="Remove filler words, false starts, and disfluencies; only supported with scribe_v2")
    elevenlabs_transcribe.add_argument("--api-key", default=None, help=f"ElevenLabs API key; defaults to ${ELEVENLABS_API_KEY_ENV}")
    elevenlabs_transcribe.add_argument("--base-url", default=None, help=f"Optional ElevenLabs API base URL; defaults to ${ELEVENLABS_BASE_URL_ENV} if set")
    elevenlabs_transcribe.add_argument("--timeout", type=float, default=None, help=f"HTTP timeout in seconds; raise it for long audio (default: {ELEVENLABS_DEFAULT_TIMEOUT_SECONDS:g})")
    elevenlabs_transcribe.add_argument("--json", action="store_true", help="Print the raw transcription JSON response")
    elevenlabs_transcribe.add_argument("--save", action="store_true", help="Write text output to data/transcriptions/<audio-stem>/elevenlabs__<model>.txt")
    elevenlabs_transcribe.add_argument("--output-dir", type=Path, default=None, help="Directory for saved text output; defaults to data/transcriptions and implies --save")

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
    omlx_transcribe.add_argument("--timeout", type=float, default=None, help="HTTP timeout in seconds; raise it for long audio (default: 120)")
    omlx_transcribe.add_argument("--json", action="store_true", help="Print the raw transcription JSON response")
    omlx_transcribe.add_argument("--save", action="store_true", help="Write text output to data/transcriptions/<audio-stem>/omlx__<model>.txt")
    omlx_transcribe.add_argument("--output-dir", type=Path, default=None, help="Directory for saved text output; defaults to data/transcriptions and implies --save")

    judge = subparsers.add_parser("judge", help="LLM-as-a-judge: gravité sémantique des transcripts (Albert API ou OpenRouter)")
    judge.add_argument("sample_id", nargs="?", default=None, help="Sample ID; omettre pour juger tout le corpus")
    judge.add_argument("--ground-truth-dir", type=Path, default=SCORING_DEFAULT_GROUND_TRUTH_DIR, help="Directory containing ground truth .md or .txt files")
    judge.add_argument("--source-truth-dir", type=Path, default=None, help=argparse.SUPPRESS)
    judge.add_argument("--transcriptions-dir", type=Path, default=SCORING_DEFAULT_TRANSCRIPTIONS_DIR, help="Directory containing generated transcript outputs")
    judge.add_argument("--judge-provider", choices=["albert", "openrouter"], default="albert", help="Fournisseur du modèle juge (défaut: albert). 'openrouter' permet un juge tiers non-Mistral, sans biais de famille.")
    judge.add_argument("--judge-model", default=None, help=f"Modèle juge. Défaut selon le provider : albert→{DEFAULT_JUDGE_MODEL}, openrouter→{OPENROUTER_DEFAULT_JUDGE_MODEL}")
    judge.add_argument("--passes", type=int, default=1, help="Nombre de passes self-consistency (>1 ne garde que les G3 stables)")
    judge.add_argument("--output", type=Path, default=None, help="Écrire le rapport markdown ici au lieu de stdout")
    judge.add_argument("--hide-g1", action="store_true", help="Masquer les écarts mineurs (G1) dans le détail")

    panel = subparsers.add_parser("panel", help="Panel multi-juges: compare plusieurs juges et/ou calcule un consensus G3")
    panel.add_argument("sample_id", nargs="?", default=None, help="Sample ID; omettre pour juger tout le corpus")
    panel.add_argument("--ground-truth-dir", type=Path, default=SCORING_DEFAULT_GROUND_TRUTH_DIR, help="Directory containing ground truth .md or .txt files")
    panel.add_argument("--source-truth-dir", type=Path, default=None, help=argparse.SUPPRESS)
    panel.add_argument("--transcriptions-dir", type=Path, default=SCORING_DEFAULT_TRANSCRIPTIONS_DIR, help="Directory containing generated transcript outputs")
    panel.add_argument("--judge", action="append", dest="judges", metavar="PROVIDER[:MODEL]", help="Juge à inclure, répétable. Ex: --judge albert --judge openrouter:anthropic/claude-sonnet-4.5 (défaut: albert + openrouter)")
    panel.add_argument("--mode", choices=["compare", "consensus", "both"], default="both", help="compare (tableau côte à côte), consensus (panel G3), ou both (défaut)")
    panel.add_argument("--consensus-min", type=int, default=None, help="Nb min de juges devant s'accorder pour retenir un G3 (défaut: majorité stricte)")
    panel.add_argument("--passes", type=int, default=1, help="Passes self-consistency PAR juge")
    panel.add_argument("--output", type=Path, default=None, help="Écrire le rapport markdown ici au lieu de stdout")
    panel.add_argument("--hide-g1", action="store_true", help="Masquer les écarts mineurs (G1) dans le détail du consensus")

    args = parser.parse_args()

    try:
        if args.command == "manifest" and args.manifest_command == "sync":
            manifest_text = render_manifest(discover_samples())
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(manifest_text, encoding="utf-8")
            print(args.manifest)
            return

        if args.command == "judge":
            if args.output is not None and args.output.is_dir():
                raise JudgeCliError(f"Output path must be a file, not a directory: {args.output}")
            if args.judge_provider == "openrouter":
                judge_client = OpenRouterClient()
                judge_provider_label = "OpenRouter"
                judge_model = args.judge_model or OPENROUTER_DEFAULT_JUDGE_MODEL
            else:
                judge_client = AlbertClient()
                judge_provider_label = "Albert API"
                judge_model = args.judge_model or DEFAULT_JUDGE_MODEL
            results = run_judge(
                args.sample_id,
                ground_truth_dir=resolve_ground_truth_dir(args),
                transcriptions_dir=args.transcriptions_dir,
                judge_model=judge_model,
                passes=args.passes,
                client=judge_client,
            )
            report = render_judge_markdown(
                results, include_g1=not args.hide_g1, judge_provider=judge_provider_label
            )
            write_or_print_judge_report(report, output_path=args.output)
            return

        if args.command == "panel":
            if args.output is not None and args.output.is_dir():
                raise JudgeCliError(f"Output path must be a file, not a directory: {args.output}")
            raw_specs = args.judges or ["albert", "openrouter"]
            try:
                specs = [parse_judge_spec(s) for s in raw_specs]
            except ValueError as exc:
                raise JudgeCliError(str(exc)) from exc
            results_by_spec = run_panel(
                args.sample_id,
                specs,
                ground_truth_dir=resolve_ground_truth_dir(args),
                transcriptions_dir=args.transcriptions_dir,
                passes=args.passes,
            )
            report = render_panel_markdown(
                results_by_spec,
                mode=args.mode,
                min_agree=args.consensus_min,
                include_g1=not args.hide_g1,
            )
            write_or_print_judge_report(report, output_path=args.output)
            return

        if args.command == "data" and args.data_command == "migrate":
            result = migrate_source_truth_to_ground_truth(
                source_dir=args.source,
                target_dir=args.target,
                force=args.force,
            )
            print(result.message)
            return

        if args.command == "score" and args.score_command == "sample":
            scored = score_sample_outputs(
                args.sample_id,
                ground_truth_dir=resolve_ground_truth_dir(args),
                transcriptions_dir=args.transcriptions_dir,
                normalization=args.normalization,
            )
            write_or_print_score_output(
                render_scores_output(
                    scored,
                    output_format="json" if args.json else args.format,
                    show_alignment=args.align,
                    top_errors=args.top_errors,
                ),
                output_path=args.output,
            )
            return

        if args.command == "score" and args.score_command == "all":
            scored = score_all_outputs(
                ground_truth_dir=resolve_ground_truth_dir(args),
                transcriptions_dir=args.transcriptions_dir,
                normalization=args.normalization,
            )
            write_or_print_score_output(
                render_scores_output(
                    scored,
                    output_format="json" if args.json else args.format,
                    show_alignment=args.align,
                    top_errors=args.top_errors,
                ),
                output_path=args.output,
            )
            return

        if args.command == "albert" and args.albert_command == "models":
            client = AlbertClient(base_url=args.base_url, api_key=args.api_key)
            for model in client.list_models():
                print(model.id)
            return

        if args.command == "albert" and args.albert_command == "transcribe":
            client = AlbertClient(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
            response_format = "json" if args.json else None
            result = client.transcribe(model=args.model, audio_path=args.audio, language=args.language, prompt=args.prompt, response_format=response_format, temperature=args.temperature)
            text = transcription_text(result.get("text"))
            print_transcription_output(
                TranscriptionOutput(
                    result=result,
                    text=text,
                    json_output=args.json,
                    save=args.save,
                    output_dir=args.output_dir,
                    audio_path=args.audio,
                    provider="albert",
                    model=args.model,
                )
            )
            return

        if args.command == "scaleway" and args.scaleway_command == "models":
            client = ScalewayClient(secret_key=args.api_key, project_id=args.project_id)
            for model_id in client.list_models(name=args.name):
                print(model_id)
            return

        if args.command == "scaleway" and args.scaleway_command == "transcribe":
            client = ScalewayClient(secret_key=args.api_key, project_id=args.project_id, timeout=args.timeout)
            prompt = args.prompt if args.prompt is not None else scaleway_build_prompt(args.language)
            result = client.transcribe(audio_path=args.audio, model=args.model, prompt=prompt, temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p)
            text = scaleway_transcription_text(result)
            print_transcription_output(
                TranscriptionOutput(
                    result=result,
                    text=text,
                    json_output=args.json,
                    save=args.save,
                    output_dir=args.output_dir,
                    audio_path=args.audio,
                    provider="scaleway",
                    model=args.model,
                )
            )
            return

        if args.command == "elevenlabs" and args.elevenlabs_command == "models":
            client = ElevenLabsClient(api_key=args.api_key, base_url=args.base_url, require_api_key=False)
            for model in client.list_models():
                print(model.id)
            return

        if args.command == "elevenlabs" and args.elevenlabs_command == "transcribe":
            client = ElevenLabsClient(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
            result = client.transcribe(
                audio_path=args.audio,
                model=args.model,
                language=args.language,
                tag_audio_events=args.tag_audio_events,
                num_speakers=args.num_speakers,
                timestamps_granularity=args.timestamps_granularity,
                diarize=args.diarize,
                temperature=args.temperature,
                seed=args.seed,
                no_verbatim=args.no_verbatim,
            )
            text = elevenlabs_transcription_text(result)
            print_transcription_output(
                TranscriptionOutput(
                    result=result,
                    text=text,
                    json_output=args.json,
                    save=args.save,
                    output_dir=args.output_dir,
                    audio_path=args.audio,
                    provider="elevenlabs",
                    model=args.model,
                )
            )
            return

        if args.command == "omlx" and args.omlx_command == "models":
            client = OmlxClient(base_url=args.base_url, api_key=args.api_key)
            for model in client.list_models():
                print(model.id)
            return

        if args.command == "omlx" and args.omlx_command == "transcribe":
            client = OmlxClient(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
            response_format = "verbose_json" if args.json else None
            result = client.transcribe(model=args.model, audio_path=args.audio, language=args.language, response_format=response_format)
            text = transcription_text(result.get("text"))
            print_transcription_output(
                TranscriptionOutput(
                    result=result,
                    text=text,
                    json_output=args.json,
                    save=args.save,
                    output_dir=args.output_dir,
                    audio_path=args.audio,
                    provider="omlx",
                    model=args.model,
                )
            )
            return
    except (FileNotFoundError, DataMigrationError, ScoringError, JudgeCliError, JudgeError, AlbertError, ScalewayError, ElevenLabsError, OmlxError, httpx.HTTPError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.command == "manifest":
        manifest.print_help()
    elif args.command == "data":
        data.print_help()
    elif args.command == "score":
        score.print_help()
    elif args.command == "albert":
        albert.print_help()
    elif args.command == "scaleway":
        scaleway.print_help()
    elif args.command == "elevenlabs":
        elevenlabs.print_help()
    elif args.command == "omlx":
        omlx.print_help()
    else:
        parser.print_help()
