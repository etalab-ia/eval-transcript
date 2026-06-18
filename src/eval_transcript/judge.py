"""LLM-as-a-judge: gravité sémantique des erreurs de transcription.

Couche qualitative complémentaire au WER. Au lieu de compter les mots faux,
on demande à un LLM (servi par Albert API) si les écarts entre la vérité
verbatim (référence) et un transcript généré (hypothèse) *changent le sens*
de la réunion : inversion de polarité, hallucination de fait/personne,
perte d'information substantielle, terme-clé changé de référent.

Taxonomie de gravité (validée dans la note de pilote du vault) :
  G3 — Critique : inversion de sens/polarité, hallucination de fait ou de
        personne, ou effondrement (perte massive de contenu).
  G2 — Majeure : perte d'une info substantielle, ou mot-clé rendu
        incompréhensible / changé de référent.
  G1 — Mineure : déformation récupérable au contexte (nom propre, terme
        technique), atténuation de ton.
  G0 — Cosmétique : euh, répétitions, ponctuation, casse → HORS SCOPE.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from eval_transcript.albert import AlbertClient, AlbertError


DEFAULT_JUDGE_MODEL = "mistral-medium-2508"

SEVERITIES = ("G1", "G2", "G3")
DIVERGENCE_TYPES = (
    "inversion_polarite",
    "hallucination_fait",
    "hallucination_personne",
    "perte_info",
    "terme_change",
    "nom_deforme",
    "effondrement",
)
VERDICTS = ("fidele", "alterations_mineures", "sens_degrade", "inexploitable")

# Poids de gravité pour le score continu. Une inversion de sens vaut ~6× une
# déformation récupérable ; un effondrement (passage entier perdu) ~2× une
# inversion. Ancrés sur le corpus pilote (3 réunions), heuristiques.
SEVERITY_WEIGHTS = {"G1": 1, "G2": 2, "G3": 6}
COLLAPSE_TYPE = "effondrement"
EFFONDREMENT_WEIGHT = 12
# Bornes du verdict, exprimées en score pondéré pour 1000 mots de référence
# (neutralise le régime de l'audio : un hebdo dense ≠ un entretien calme).
VERDICT_BANDS = (
    (0.0, "fidele"),
    (8.0, "alterations_mineures"),
    (30.0, "sens_degrade"),
)  # au-delà de la dernière borne → "inexploitable"
# Plancher : dès qu'un passage entier est perdu, le transcript ne peut pas être
# jugé "fidèle" ni "altérations mineures", quel que soit le score.
COLLAPSE_VERDICT_FLOOR = "sens_degrade"

SYSTEM_PROMPT = """Tu es évaluateur de transcription automatique de réunions en français.

On te donne une RÉFÉRENCE (vérité verbatim, vérifiée à l'oreille) et une \
HYPOTHÈSE (transcript produit par un modèle automatique). Ta mission : repérer \
UNIQUEMENT les écarts qui CHANGENT LE SENS pour quelqu'un qui lirait le \
compte-rendu de la réunion.

IGNORE totalement (n'est PAS un écart) : ponctuation, casse, accents, « euh », \
hésitations, répétitions, faux départs, et toute reformulation qui préserve le \
sens.

## Rubrique de gravité (applique-la STRICTEMENT, sans surclasser)

G3 — Critique. RÉSERVÉ aux cas où le sens est INVERSÉ ou FAUSSÉ pour le lecteur :
- inversion de polarité : négation ajoutée ou retirée ; antonyme ; \
question muée en affirmation (ou l'inverse) ; succès ↔ échec ; \
attribution inversée (qui paie/qui décide/qui fait l'action).
- hallucination d'un FAIT, d'un CHIFFRE/MONTANT ou d'une PERSONNE que le \
lecteur prendrait pour vrai (un prénom réel remplacé par un autre prénom \
plausible ; un montant changé ; un événement inventé).
- effondrement : un passage entier perdu, ou remplacé par du charabia / une \
boucle / une autre langue (type = "effondrement").

G2 — Majeure. Le sens local est dégradé mais PAS inversé :
- perte d'une information substantielle (mais pas un passage entier).
- mot-clé / nom propre transformé en un AUTRE mot porteur de sens, au point \
qu'on ne retrouve plus le référent (ex. « Anthropic » → « philanthropique »).

G1 — Mineure. RÉCUPÉRABLE grâce au contexte, le référent reste identifiable :
- nom propre/sigle mal orthographié mais reconnaissable (« DINUM » → « DINU »).
- terme technique déformé mais devinable, atténuation de ton légère.

## Règles de NON-surclassement (impératives)

- Une différence de NOMBRE (singulier/pluriel), d'orthographe, d'accent ou de \
casse qui NE CHANGE PAS le référent → G1 maximum, JAMAIS G3 \
(ex. « le Premier ministre » → « les premiers ministres » = G1).
- Un nom propre déformé mais encore identifiable au contexte → G1 ; il ne \
devient G2 que si le référent est perdu, et G3 (hallucination_personne) que \
s'il est remplacé par une AUTRE personne crédible.
- Dans le doute entre deux niveaux, choisis le PLUS BAS. Ne remonte en G3 que \
si tu peux nommer l'inversion ou l'invention précise dans `impact_sens`.

## Types

inversion_polarite, hallucination_fait, hallucination_personne, perte_info, \
terme_change, nom_deforme, effondrement.

## Règle anti-invention (impérative)

Pour chaque écart, recopie VERBATIM l'extrait exact de la RÉFÉRENCE et \
l'extrait exact de l'HYPOTHÈSE (copier-coller, sans rien modifier). Pour un \
effondrement, cite le début du passage de référence concerné et mets \
"[passage perdu]" côté hypothèse. Si tu ne peux pas citer littéralement, ne \
remonte pas l'écart.

Réponds STRICTEMENT en JSON, sans aucun texte autour, avec ce format :
{
  "divergences": [
    {
      "extrait_reference": "...",
      "extrait_hypothese": "...",
      "type": "inversion_polarite",
      "gravite": "G3",
      "impact_sens": "une phrase nommant l'inversion/invention précise"
    }
  ]
}
"""

USER_TEMPLATE = """RÉFÉRENCE (vérité verbatim) :
\"\"\"
{reference}
\"\"\"

HYPOTHÈSE (transcript {provider} / {model}) :
\"\"\"
{hypothesis}
\"\"\"

Liste les écarts de sens en JSON."""


@dataclass(frozen=True)
class Divergence:
    extrait_reference: str
    extrait_hypothese: str
    type: str
    gravite: str
    impact_sens: str
    verbatim_ok: bool = True


@dataclass
class JudgeResult:
    sample_id: str
    provider: str
    model: str
    judge_model: str
    verdict: str
    divergences: list[Divergence] = field(default_factory=list)
    reference_word_count: int = 0
    raw: str = ""

    @property
    def counts(self) -> dict[str, int]:
        tally = {s: 0 for s in SEVERITIES}
        for d in self.divergences:
            if d.gravite in tally:
                tally[d.gravite] += 1
        return tally

    @property
    def collapse_count(self) -> int:
        return sum(1 for d in self.divergences if d.type == COLLAPSE_TYPE)

    @property
    def has_collapse(self) -> bool:
        return self.collapse_count > 0

    @property
    def weighted_score(self) -> int:
        total = 0
        for d in self.divergences:
            if d.type == COLLAPSE_TYPE:
                total += EFFONDREMENT_WEIGHT
            else:
                total += SEVERITY_WEIGHTS.get(d.gravite, 0)
        return total

    @property
    def score_per_1k(self) -> float:
        """Score pondéré normalisé pour 1000 mots de référence (régime-neutre)."""
        if self.reference_word_count <= 0:
            return float(self.weighted_score)
        return self.weighted_score * 1000.0 / self.reference_word_count


class JudgeError(RuntimeError):
    """Raised when the judge call or its parsing fails."""


def _normalize(text: str) -> str:
    """Casse, accents et espaces neutralisés, pour la vérification verbatim souple."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    # Les ligatures FR ne sont pas décomposées par NFD : les expanser avant le
    # filtre ASCII, sinon « bœuf » devient « b uf » et casse la comparaison
    # avec une hypothèse qui écrit « boeuf ».
    text = text.replace("œ", "oe").replace("æ", "ae")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def _is_verbatim(extract: str, source: str) -> bool:
    norm_extract = _normalize(extract)
    if not norm_extract:
        return False
    return norm_extract in _normalize(source)


def build_messages(reference: str, hypothesis: str, *, provider: str, model: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                reference=reference.strip(),
                hypothesis=hypothesis.strip(),
                provider=provider or "?",
                model=model or "?",
            ),
        },
    ]


def _extract_json(content: str) -> dict[str, Any]:
    content = content.strip()
    # Tolère un éventuel bloc ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
    if fence:
        content = fence.group(1)
    else:
        first, last = content.find("{"), content.rfind("}")
        if first != -1 and last != -1 and last > first:
            content = content[first : last + 1]
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"Réponse du juge non parsable en JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise JudgeError(f"JSON du juge inattendu (pas un objet): {type(data).__name__}")
    return data


def parse_judge_response(
    content: str,
    *,
    reference: str,
    hypothesis: str,
    sample_id: str,
    provider: str,
    model: str,
    judge_model: str,
) -> JudgeResult:
    data = _extract_json(content)
    raw_divs = data.get("divergences", [])
    divergences: list[Divergence] = []
    if isinstance(raw_divs, list):
        for item in raw_divs:
            if not isinstance(item, dict):
                continue
            ref_extract = str(item.get("extrait_reference", "")).strip()
            hyp_extract = str(item.get("extrait_hypothese", "")).strip()
            gravite = str(item.get("gravite", "")).strip().upper()
            if gravite not in SEVERITIES:
                continue
            div_type = str(item.get("type", "")).strip() or "?"
            # Pour un effondrement, le côté hypothèse est volontairement un
            # marqueur ("[passage perdu]") absent du texte : on n'exige le
            # verbatim que sur la référence, sinon faux flag systématique.
            verbatim_ok = _is_verbatim(ref_extract, reference) and (
                div_type == COLLAPSE_TYPE or _is_verbatim(hyp_extract, hypothesis)
            )
            divergences.append(
                Divergence(
                    extrait_reference=ref_extract,
                    extrait_hypothese=hyp_extract,
                    type=div_type,
                    gravite=gravite,
                    impact_sens=str(item.get("impact_sens", "")).strip(),
                    verbatim_ok=verbatim_ok,
                )
            )
    word_count = len(reference.split())
    result = JudgeResult(
        sample_id=sample_id,
        provider=provider,
        model=model,
        judge_model=judge_model,
        verdict="",
        divergences=divergences,
        reference_word_count=word_count,
        raw=content,
    )
    # Le verdict du modèle est ignoré : on le dérive du score continu (caveat 2).
    result.verdict = _verdict_from(result)
    return result


def _verdict_from(result: JudgeResult) -> str:
    """Verdict catégoriel dérivé du score pondéré /1k mots.

    L'effondrement est déjà fortement pénalisé dans le score (poids 12) ; il
    n'impose donc pas un verdict binaire, mais applique un plancher : un
    passage perdu interdit "fidele"/"alterations_mineures".
    """
    score = result.score_per_1k
    verdict = "inexploitable"
    for threshold, label in VERDICT_BANDS:
        if score <= threshold:
            verdict = label
            break
    if result.has_collapse and verdict in ("fidele", "alterations_mineures"):
        verdict = COLLAPSE_VERDICT_FLOOR
    return verdict


def judge_pair(
    *,
    reference: str,
    hypothesis: str,
    sample_id: str,
    provider: str,
    model: str,
    client: AlbertClient | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    temperature: float = 0.0,
    passes: int = 1,
) -> JudgeResult:
    """Évalue un couple (référence, hypothèse) via Albert API.

    passes > 1 : self-consistency simple — on relance N fois et on ne garde que
    les divergences G3 stables (extrait de référence vu dans >50 % des passes).
    À température 0, les passes seraient déterministes (donc identiques et
    inutiles) : on relève automatiquement la température dans ce cas.
    """
    if passes > 1 and temperature == 0.0:
        temperature = 0.7
    client = client or AlbertClient()
    messages = build_messages(reference, hypothesis, provider=provider, model=model)

    results: list[JudgeResult] = []
    for _ in range(max(1, passes)):
        try:
            content = client.chat_completion_text(
                model=judge_model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except AlbertError as exc:
            raise JudgeError(f"Appel juge Albert échoué: {exc}") from exc
        results.append(
            parse_judge_response(
                content,
                reference=reference,
                hypothesis=hypothesis,
                sample_id=sample_id,
                provider=provider,
                model=model,
                judge_model=judge_model,
            )
        )

    if len(results) == 1:
        return results[0]
    return _merge_passes(results)


def _merge_passes(results: list[JudgeResult], *, min_agree: int | None = None) -> JudgeResult:
    """Fusionne plusieurs JudgeResult du MÊME couple (réf, hyp).

    Sert à deux usages structurellement identiques :
    - self-consistency d'un seul juge (N passes), `min_agree=None` → majorité
      stricte (un G3 vu dans > 50 % des passes est conservé) ;
    - panel multi-juges (cf. `panel.py`), `min_agree=K` → un G3 est conservé
      s'il est signalé par au moins K juges.

    Les divergences non-G3 (G1/G2) proviennent du 1er résultat listé (`base`) :
    leur calibration n'est pas comparable entre juges, donc on n'en fait pas le
    consensus — d'où l'intérêt de lister le juge calibré (mistral) en premier.
    """
    base = results[0]
    n = len(results)
    # G3 : ne garder que les stables (accord d'assez de passes / de juges)
    g3_keys: dict[str, int] = {}
    for r in results:
        seen: set[str] = set()
        for d in r.divergences:
            if d.gravite == "G3":
                seen.add(_normalize(d.extrait_reference))
        for k in seen:
            g3_keys[k] = g3_keys.get(k, 0) + 1
    if min_agree is not None:
        stable_g3 = {k for k, c in g3_keys.items() if c >= min_agree}
    else:
        threshold = n / 2.0
        stable_g3 = {k for k, c in g3_keys.items() if c > threshold}

    # Représentant de chaque G3 stable, collecté sur TOUTES les passes (pas
    # seulement la passe 0) : un G3 stable manqué à la passe 0 mais présent
    # aux passes suivantes doit quand même être conservé.
    stable_g3_divs: dict[str, Divergence] = {}
    for r in results:
        for d in r.divergences:
            if d.gravite != "G3":
                continue
            norm_ref = _normalize(d.extrait_reference)
            if norm_ref in stable_g3 and norm_ref not in stable_g3_divs:
                stable_g3_divs[norm_ref] = d

    merged: list[Divergence] = []
    seen_keys: set[str] = set()
    for d in stable_g3_divs.values():
        key = (d.gravite, _normalize(d.extrait_reference))
        seen_keys.add(key)
        merged.append(d)
    # Les divergences non-G3 restent celles de la passe de référence.
    for d in base.divergences:
        if d.gravite == "G3":
            continue
        key = (d.gravite, _normalize(d.extrait_reference))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(d)
    base.divergences = merged
    base.verdict = _verdict_from(base)
    return base
