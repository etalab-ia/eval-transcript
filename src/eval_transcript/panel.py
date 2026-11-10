"""Panel multi-juges pour la gravité sémantique (cf. judge.py).

Deux usages, à partir des mêmes résultats :

1. **Comparer** plusieurs juges (ex. `mistral-medium-2508` via Albert vs un
   modèle tiers via OpenRouter) : un tableau donne, pour chaque transcript, le
   score /1000 mots de chaque juge + l'écart max (signale les désaccords).
2. **Consensus** (panel) : pour chaque transcript, on ne retient un G3 que s'il
   est signalé par au moins K juges (majorité stricte par défaut). Cela neutralise
   le biais d'un juge isolé (ex. un Mistral indulgent envers un transcript Mistral).

⚠️ La rubrique G1/G2/G3 est calibrée sur `mistral-medium-2508`. Les comptages
bruts ne sont pas comparables entre juges : on compare les CLASSEMENTS, et on
ne met au consensus que les G3 (les divergences G1/G2 du consensus proviennent
du premier juge listé — y mettre le juge calibré).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from eval_transcript.albert import AlbertClient
from eval_transcript.judge import (
    DEFAULT_JUDGE_MODEL,
    SEVERITIES,
    Divergence,
    JudgeResult,
    _normalize,
    _verdict_from,
)
from eval_transcript.judge_cli import run_judge
from eval_transcript.openrouter import (
    DEFAULT_JUDGE_MODEL as OPENROUTER_DEFAULT_JUDGE_MODEL,
    OpenRouterClient,
)

PROVIDERS = ("albert", "openrouter")


@dataclass(frozen=True)
class JudgeSpec:
    """Un juge du panel : un fournisseur + un modèle."""

    provider: str
    model: str

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def make_client(self) -> AlbertClient:
        if self.provider == "openrouter":
            return OpenRouterClient()
        return AlbertClient()


def parse_judge_spec(raw: str) -> JudgeSpec:
    """Parse "provider" ou "provider:model" (modèle facultatif → défaut du provider)."""
    raw = raw.strip()
    provider, _, model = raw.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in PROVIDERS:
        raise ValueError(
            f"Provider de juge inconnu: '{provider}' (attendu: {', '.join(PROVIDERS)})"
        )
    if not model:
        model = OPENROUTER_DEFAULT_JUDGE_MODEL if provider == "openrouter" else DEFAULT_JUDGE_MODEL
    return JudgeSpec(provider=provider, model=model)


def _transcript_key(result: JudgeResult) -> tuple[str, str, str]:
    """Identité du TRANSCRIPT jugé (pas du juge) : (échantillon, provider, modèle)."""
    return (result.sample_id, result.provider, result.model)


def run_panel(
    sample_id: str | None,
    specs: list[JudgeSpec],
    *,
    ground_truth_dir,
    transcriptions_dir,
    passes: int = 1,
    progress: bool = True,
) -> list[tuple[str, list[JudgeResult]]]:
    """Fait tourner chaque juge sur le corpus ; renvoie [(label_juge, résultats)]."""
    out: list[tuple[str, list[JudgeResult]]] = []
    for spec in specs:
        if progress:
            print(f"\n=== Juge {spec.label} ===", file=sys.stderr, flush=True)
        results = run_judge(
            sample_id,
            ground_truth_dir=ground_truth_dir,
            transcriptions_dir=transcriptions_dir,
            judge_model=spec.model,
            passes=passes,
            client=spec.make_client(),
            progress=progress,
        )
        out.append((spec.label, results))
    return out


def _cluster_g3_by_overlap(results: list[JudgeResult]) -> list[dict]:
    """Regroupe les G3 de plusieurs juges par RECOUVREMENT d'extrait de référence.

    Deux juges signalent souvent la même erreur en citant des empans légèrement
    différents (« Nat sera content » vs « comme ça Nat sera content ») : une égalité
    stricte les compterait comme deux écarts distincts et ne verrait jamais l'accord.
    On regroupe donc dès qu'un extrait normalisé en contient un autre (containment).

    Chaque cluster : {judges: set d'indices de juges, divs: list de Divergence}.
    """
    clusters: list[dict] = []
    for idx, r in enumerate(results):
        for d in r.divergences:
            if d.gravite != "G3":
                continue
            norm = _normalize(d.extrait_reference)
            if not norm:
                continue
            target = None
            for cl in clusters:
                if any(norm in m or m in norm for m in cl["norms"]):
                    target = cl
                    break
            if target is None:
                target = {"norms": set(), "judges": set(), "divs": []}
                clusters.append(target)
            target["norms"].add(norm)
            target["judges"].add(idx)
            target["divs"].append(d)
    return clusters


def consensus_for_transcript(
    results: list[JudgeResult], *, panel_size: int, min_agree: int | None = None
) -> tuple[JudgeResult, dict[str, int], int, int]:
    """Consensus d'un transcript jugé par plusieurs juges.

    `panel_size` = nombre total de juges du panel ; `results` ne contient que ceux
    qui ont effectivement produit un résultat pour CE transcript (un juge peut
    échouer sur un couple). Le seuil d'accord est calculé sur `panel_size`, PAS sur
    les résultats présents : sinon, un transcript jugé par un seul juge verrait tous
    ses G3 « passer » le consensus (seuil 1/1), ce qui viderait le panel de son sens.

    - **G3** : mis au vote. Un G3 est retenu s'il est signalé par ≥ seuil juges
      DISTINCTS, l'accord étant établi par recouvrement d'extrait (cf.
      `_cluster_g3_by_overlap`), pas par égalité stricte. Le représentant retenu
      est l'extrait le plus précis (le plus court) du cluster.
    - **G1/G2** : NON comparables entre juges (calibration propre à chacun), donc
      pas mis au vote. On prend ceux du **juge primaire = 1er de `results`** (mettre
      le juge calibré en tête). Choix explicite, pas un effet de bord.

    Renvoie (résultat consensus, {ref_normalisée du représentant -> nb de juges
    d'accord}, couverture = nb de juges présents, seuil retenu).
    """
    resolved = min_agree if min_agree is not None else (panel_size // 2 + 1)
    n = len(results)

    consensus_g3: list[Divergence] = []
    counts: dict[str, int] = {}
    for cl in _cluster_g3_by_overlap(results):
        agree = len(cl["judges"])
        if agree < resolved:
            continue
        # Représentant = extrait de référence le plus précis (le plus court).
        rep = min(cl["divs"], key=lambda d: len(d.extrait_reference))
        consensus_g3.append(rep)
        counts[_normalize(rep.extrait_reference)] = agree

    # G1/G2 du juge primaire (1er listé) ; Divergence est frozen -> partage sûr.
    primary = results[0]
    lower = [d for d in primary.divergences if d.gravite in ("G1", "G2")]

    merged = JudgeResult(
        sample_id=primary.sample_id,
        provider=primary.provider,
        model=primary.model,
        judge_model="panel",
        verdict="",
        divergences=consensus_g3 + lower,
        reference_word_count=primary.reference_word_count,
    )
    merged.verdict = _verdict_from(merged)
    return merged, counts, n, resolved


def render_panel_markdown(
    results_by_spec: list[tuple[str, list[JudgeResult]]],
    *,
    mode: str = "both",
    min_agree: int | None = None,
    include_g1: bool = True,
) -> str:
    specs = [label for label, _ in results_by_spec]
    indexed = {
        label: {_transcript_key(r): r for r in results} for label, results in results_by_spec
    }
    all_keys = sorted({k for d in indexed.values() for k in d})

    lines: list[str] = []
    lines.append("# Panel multi-juges — gravité sémantique\n")
    lines.append(f"> Juges : {', '.join(f'`{s}`' for s in specs)}.")
    lines.append(
        "> ⚠️ Rubrique G1/G2/G3 calibrée sur `mistral-medium-2508` : comparer les "
        "**classements** entre juges, pas les totaux absolus.\n"
    )

    if mode in ("compare", "both"):
        lines.extend(_render_compare(specs, indexed, all_keys))
    if mode in ("consensus", "both"):
        lines.extend(_render_consensus(specs, indexed, all_keys, min_agree, include_g1))
    return "\n".join(lines)


def _render_compare(specs, indexed, all_keys) -> list[str]:
    lines: list[str] = []
    lines.append("## Comparaison des juges\n")
    lines.append("> Cellule = score /1k mots (G3). « Écart max » = plus grand écart de score entre juges (désaccord).\n")
    header = "| Échantillon | Transcript | " + " | ".join(specs) + " | Écart max |"
    sep = "|---|---|" + "---:|" * len(specs) + "---:|"
    lines.append(header)
    lines.append(sep)

    def mean_score(key) -> float:
        scores = [indexed[s][key].score_per_1k for s in specs if key in indexed[s]]
        return sum(scores) / len(scores) if scores else 0.0

    for key in sorted(all_keys, key=lambda k: (k[0], -mean_score(k))):
        sample, provider, model = key
        cells: list[str] = []
        scores: list[float] = []
        for s in specs:
            r = indexed[s].get(key)
            if r is None:
                cells.append("—")
                continue
            scores.append(r.score_per_1k)
            cells.append(f"{r.score_per_1k:.1f} ({r.counts['G3']})")
        spread = f"{max(scores) - min(scores):.1f}" if len(scores) >= 2 else "—"
        lines.append(f"| {sample} | {provider}/{model} | " + " | ".join(cells) + f" | {spread} |")
    lines.append("")
    return lines


def _render_consensus(specs, indexed, all_keys, min_agree, include_g1) -> list[str]:
    lines: list[str] = []
    seuil_txt = f"≥ {min_agree} juges" if min_agree is not None else "majorité stricte des juges"
    lines.append(f"## Consensus (G3 retenu si {seuil_txt})\n")
    lines.append(
        "> Un G3 n'est retenu que s'il est signalé par assez de juges (sur le même "
        "extrait de référence). Les G1/G2 affichés proviennent du 1er juge listé "
        f"(`{specs[0]}`). Trié du plus dégradé au plus fidèle.\n"
    )

    # Calcul du consensus par transcript. Le seuil d'accord est fonction de la
    # taille du panel (len(specs)), pas du nombre de juges présents pour le couple.
    panel_size = len(specs)
    merged_by_key: dict[tuple[str, str, str], tuple[JudgeResult, dict[str, int], int, int]] = {}
    for key in all_keys:
        results = [indexed[s][key] for s in specs if key in indexed[s]]
        if not results:
            continue
        merged_by_key[key] = consensus_for_transcript(results, panel_size=panel_size, min_agree=min_agree)

    # Synthèse
    lines.append("| Échantillon | Transcript | G3 cons. | Score /1k | Verdict | Couverture |")
    lines.append("|---|---|---:|---:|---|---:|")
    for key in sorted(all_keys, key=lambda k: (k[0], -(merged_by_key[k][0].score_per_1k if k in merged_by_key else 0))):
        if key not in merged_by_key:
            continue
        merged, _counts, n, _seuil = merged_by_key[key]
        sample, provider, model = key
        coverage = f"{n}/{panel_size}" + (" ⚠️" if n < panel_size else "")
        lines.append(
            f"| {sample} | {provider}/{model} | {merged.counts['G3']} "
            f"| {merged.score_per_1k:.1f} | {merged.verdict} | {coverage} |"
        )
    lines.append("")
    if any(n < panel_size for _m, _c, n, _s in merged_by_key.values()):
        lines.append(
            "> ⚠️ Couverture partielle (`n/panel < 1`) : un ou plusieurs juges ont "
            "échoué sur ce transcript. Le seuil de consensus reste calculé sur le "
            "panel complet, donc un G3 d'un juge isolé n'est PAS retenu.\n"
        )

    # Détail des G3 consensus
    lines.append("### Détail des G3 consensus\n")
    severities_kept = SEVERITIES if include_g1 else ("G2", "G3")
    for key in sorted(merged_by_key):
        merged, counts, n, _seuil = merged_by_key[key]
        sample, provider, model = key
        lines.append(f"#### {sample} — {provider}/{model}\n")
        shown = [d for d in merged.divergences if d.gravite in severities_kept]
        if not shown:
            lines.append("_Aucun écart retenu au consensus._\n")
            continue
        for d in sorted(shown, key=lambda x: x.gravite, reverse=True):
            agree = ""
            if d.gravite == "G3":
                c = counts.get(_normalize(d.extrait_reference))
                if c is not None:
                    agree = f" — accord {c}/{panel_size} juges"
            flag = "" if d.verbatim_ok else " ⚠️ non-verbatim (à vérifier)"
            lines.append(f"- **{d.gravite}** · `{d.type}`{agree}{flag}")
            lines.append(f"  - réf : « {d.extrait_reference} »")
            lines.append(f"  - hyp : « {d.extrait_hypothese} »")
            lines.append(f"  - impact : {d.impact_sens}")
        lines.append("")
    return lines
