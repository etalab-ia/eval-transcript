"""CLI pour le LLM-as-a-judge de gravité sémantique (cf. judge.py)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from eval_transcript.albert import AlbertClient
from eval_transcript.judge import (
    DEFAULT_JUDGE_MODEL,
    SEVERITIES,
    JudgeError,
    JudgeResult,
    judge_pair,
)
from eval_transcript.manifest import (
    DEFAULT_GROUND_TRUTH_DIR,
    DEFAULT_TRANSCRIPTIONS_DIR,
    GROUND_TRUTH_SUFFIXES,
    discover_sample_ids,
    find_ground_truth_path,
    find_output_paths,
    parse_output_name,
)


class JudgeCliError(RuntimeError):
    """Raised when judge inputs cannot be discovered or read."""


@dataclass(frozen=True)
class JudgePair:
    sample_id: str
    provider: str
    model: str
    ground_truth_path: Path
    transcription_path: Path


def discover_pairs(
    sample_id: str | None,
    *,
    ground_truth_dir: Path = DEFAULT_GROUND_TRUTH_DIR,
    transcriptions_dir: Path = DEFAULT_TRANSCRIPTIONS_DIR,
) -> list[JudgePair]:
    if sample_id is not None:
        sample_ids = [sample_id]
    else:
        sample_ids = discover_sample_ids(
            audio_dir=Path("__missing_audio_dir__"),
            ground_truth_dir=ground_truth_dir,
            transcriptions_dir=transcriptions_dir,
        )

    pairs: list[JudgePair] = []
    for sid in sample_ids:
        truth = find_ground_truth_path(ground_truth_dir, sid)
        if truth is None:
            if sample_id is not None:
                expected = " or ".join(
                    (ground_truth_dir / f"{sid}{suffix}").as_posix() for suffix in GROUND_TRUTH_SUFFIXES
                )
                raise JudgeCliError(f"Missing ground truth for sample {sid}: expected {expected}")
            continue
        outputs = find_output_paths(transcriptions_dir, sid)
        if not outputs:
            if sample_id is not None:
                raise JudgeCliError(
                    f"No transcription outputs for sample {sid}: expected {transcriptions_dir / sid}/*.txt"
                )
            continue
        for out in outputs:
            provider, model = parse_output_name(out)
            pairs.append(
                JudgePair(
                    sample_id=sid,
                    provider=provider,
                    model=model,
                    ground_truth_path=truth,
                    transcription_path=out,
                )
            )
    return pairs


def run_judge(
    sample_id: str | None,
    *,
    ground_truth_dir: Path = DEFAULT_GROUND_TRUTH_DIR,
    transcriptions_dir: Path = DEFAULT_TRANSCRIPTIONS_DIR,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    passes: int = 1,
    client: AlbertClient | None = None,
    progress: bool = True,
) -> list[JudgeResult]:
    pairs = discover_pairs(
        sample_id,
        ground_truth_dir=ground_truth_dir,
        transcriptions_dir=transcriptions_dir,
    )
    if not pairs:
        raise JudgeCliError("Aucun couple (référence, hypothèse) trouvé.")
    client = client or AlbertClient()
    results: list[JudgeResult] = []
    failures = 0
    for i, pair in enumerate(pairs, 1):
        if progress:
            # Logs sur stderr : stdout est réservé au rapport Markdown (pipe-safe).
            print(
                f"[{i}/{len(pairs)}] juge {pair.sample_id} :: {pair.provider}/{pair.model} …",
                file=sys.stderr,
                flush=True,
            )
        reference = pair.ground_truth_path.read_text(encoding="utf-8")
        hypothesis = pair.transcription_path.read_text(encoding="utf-8")
        try:
            result = judge_pair(
                reference=reference,
                hypothesis=hypothesis,
                sample_id=pair.sample_id,
                provider=pair.provider,
                model=pair.model,
                client=client,
                judge_model=judge_model,
                passes=passes,
            )
        except JudgeError as exc:
            failures += 1
            print(f"    ⚠️  {pair.sample_id} :: {pair.provider}/{pair.model} — {exc}", file=sys.stderr, flush=True)
            continue
        results.append(result)
    # Ne pas réussir silencieusement (exit 0 + rapport vide) si tout a échoué :
    # une clé Albert invalide doit faire échouer la commande en CI / script.
    if not results and failures:
        raise JudgeCliError(
            f"Aucun jugement produit : les {failures} couple(s) ont tous échoué (voir les erreurs ci-dessus)."
        )
    return results


def write_or_print_report(content: str, *, output_path: Path | None = None) -> None:
    """Écrit le rapport sur disque, ou sur stdout si aucun chemin (pipe-safe)."""
    if output_path is None:
        print(content)
        return
    if output_path.is_dir():
        raise JudgeCliError(f"Output path must be a file, not a directory: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"Rapport écrit : {output_path}", file=sys.stderr)


def render_markdown(results: list[JudgeResult], *, include_g1: bool = True) -> str:
    lines: list[str] = []
    lines.append("# Gravité sémantique (LLM-as-a-judge)\n")
    if results:
        lines.append(
            f"> Juge : `{results[0].judge_model}` (Albert API). Hors scope : G0 (ponctuation, euh, répétitions)."
        )
        lines.append(
            "> **Score** = somme pondérée (G3=6, G2=2, G1=1, effondrement=12) normalisée pour 1000 mots "
            "de référence. Un `effondrement` applique un plancher de verdict `sens_degrade` (jamais "
            "`fidele`/`alterations_mineures`). Trié du plus dégradé au plus fidèle.\n"
        )

    # Tableau de synthèse, trié par score décroissant (le pire en haut)
    lines.append("## Synthèse\n")
    lines.append("| Échantillon | Modèle | G3 | G2 | G1 | Effond. | Score /1k | Verdict |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in sorted(results, key=lambda x: (x.sample_id, -x.score_per_1k)):
        c = r.counts
        lines.append(
            f"| {r.sample_id} | {r.provider}/{r.model} | {c['G3']} | {c['G2']} | {c['G1']} "
            f"| {r.collapse_count} | {r.score_per_1k:.1f} | {r.verdict} |"
        )
    lines.append("")

    # Détail par modèle
    lines.append("## Détail\n")
    severities_kept = SEVERITIES if include_g1 else ("G2", "G3")
    for r in sorted(results, key=lambda x: (x.sample_id, x.provider, x.model)):
        lines.append(f"### {r.sample_id} — {r.provider}/{r.model}\n")
        shown = [d for d in r.divergences if d.gravite in severities_kept]
        if not shown:
            lines.append("_Aucun écart de sens remonté._\n")
            continue
        for d in sorted(shown, key=lambda x: x.gravite, reverse=True):
            flag = "" if d.verbatim_ok else " ⚠️ non-verbatim (à vérifier)"
            lines.append(f"- **{d.gravite}** · `{d.type}`{flag}")
            lines.append(f"  - réf : « {d.extrait_reference} »")
            lines.append(f"  - hyp : « {d.extrait_hypothese} »")
            lines.append(f"  - impact : {d.impact_sens}")
        lines.append("")
    return "\n".join(lines)
