from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval_transcript.albert import AlbertError
from eval_transcript.judge import (
    EFFONDREMENT_WEIGHT,
    SEVERITY_WEIGHTS,
    Divergence,
    JudgeResult,
    _extract_json,
    _is_verbatim,
    JudgeError,
    _merge_passes,
    _verdict_from,
    judge_pair,
    parse_judge_response,
)
from eval_transcript.judge_cli import JudgeCliError, run_judge, write_or_print_report


class _FailingClient:
    """Client factice dont chaque appel juge échoue (clé invalide simulée)."""

    def chat_completion_text(self, **kwargs: object) -> str:
        raise AlbertError("403 Forbidden - Invalid API key.")


def _result(divs: list[tuple[str, str]], word_count: int = 1000) -> JudgeResult:
    """Construit un JudgeResult à partir de couples (gravité, type)."""
    divergences = [Divergence("", "", typ, grav, "") for grav, typ in divs]
    result = JudgeResult("s", "p", "m", "judge", "", divergences, word_count)
    result.verdict = _verdict_from(result)
    return result


class ExtractJsonTests(unittest.TestCase):
    def test_parses_plain_object(self) -> None:
        self.assertEqual(_extract_json('{"divergences": []}'), {"divergences": []})

    def test_tolerates_fenced_json_block(self) -> None:
        content = 'Voici le résultat :\n```json\n{"divergences": []}\n```\n'
        self.assertEqual(_extract_json(content), {"divergences": []})

    def test_strips_text_around_object(self) -> None:
        self.assertEqual(_extract_json('blabla {"a": 1} fin'), {"a": 1})


class VerbatimGuardrailTests(unittest.TestCase):
    def test_substring_match_is_case_and_accent_insensitive(self) -> None:
        self.assertTrue(_is_verbatim("Premier Ministre", "le premier ministre veut"))

    def test_absent_extract_is_not_verbatim(self) -> None:
        self.assertFalse(_is_verbatim("phrase inventée", "le texte réel"))

    def test_empty_extract_is_not_verbatim(self) -> None:
        self.assertFalse(_is_verbatim("", "du texte"))

    def test_french_ligatures_match_their_expansion(self) -> None:
        # « bœuf » dans la référence doit matcher « boeuf » dans l'hypothèse.
        self.assertTrue(_is_verbatim("le bœuf et la sœur", "alors le boeuf et la soeur"))
        self.assertTrue(_is_verbatim("ex æquo", "ils sont ex aequo"))


class ScoringTests(unittest.TestCase):
    def test_weights_sum_correctly(self) -> None:
        result = _result([("G3", "inversion_polarite"), ("G2", "perte_info"), ("G1", "nom_deforme")])
        expected = SEVERITY_WEIGHTS["G3"] + SEVERITY_WEIGHTS["G2"] + SEVERITY_WEIGHTS["G1"]
        self.assertEqual(result.weighted_score, expected)

    def test_effondrement_uses_heavy_weight_not_g3(self) -> None:
        result = _result([("G3", "effondrement")])
        self.assertEqual(result.weighted_score, EFFONDREMENT_WEIGHT)
        self.assertEqual(result.collapse_count, 1)

    def test_score_normalized_per_1k_words(self) -> None:
        result = _result([("G2", "perte_info")], word_count=2000)
        self.assertAlmostEqual(result.score_per_1k, SEVERITY_WEIGHTS["G2"] * 1000.0 / 2000)


class VerdictTests(unittest.TestCase):
    def test_perfect_transcript_is_fidele(self) -> None:
        self.assertEqual(_result([]).verdict, "fidele")

    def test_single_local_collapse_is_not_inexploitable(self) -> None:
        # Un effondrement isolé sur un transcript par ailleurs propre reste "sens_degrade"
        # (plancher), pas "inexploitable".
        self.assertEqual(_result([("G3", "effondrement")]).verdict, "sens_degrade")

    def test_multiple_collapses_push_to_inexploitable(self) -> None:
        result = _result([("G3", "effondrement")] * 3 + [("G2", "perte_info")] * 5, word_count=1500)
        self.assertEqual(result.verdict, "inexploitable")

    def test_collapse_floor_blocks_alterations_mineures(self) -> None:
        # Score faible mais effondrement présent -> ne peut pas être "alterations_mineures".
        result = _result([("G3", "effondrement")], word_count=5000)
        self.assertIn(result.verdict, ("sens_degrade", "inexploitable"))
        self.assertNotIn(result.verdict, ("fidele", "alterations_mineures"))


class ParseJudgeResponseTests(unittest.TestCase):
    def _parse(self, payload: dict, reference: str, hypothesis: str) -> JudgeResult:
        return parse_judge_response(
            json.dumps(payload),
            reference=reference,
            hypothesis=hypothesis,
            sample_id="s",
            provider="p",
            model="m",
            judge_model="judge",
        )

    def test_keeps_divergence_and_flags_non_verbatim(self) -> None:
        payload = {
            "divergences": [
                {
                    "extrait_reference": "absent de la ref",
                    "extrait_hypothese": "Donald",
                    "type": "hallucination_personne",
                    "gravite": "G3",
                    "impact_sens": "nom inventé",
                }
            ]
        }
        result = self._parse(payload, reference="Nat sera content", hypothesis="Donald sera content")
        self.assertEqual(len(result.divergences), 1)
        self.assertFalse(result.divergences[0].verbatim_ok)

    def test_effondrement_is_exempt_from_hypothesis_verbatim(self) -> None:
        # Le marqueur "[passage perdu]" n'est pas dans l'hypothèse : un
        # effondrement dont la référence est verbatim ne doit PAS être flaggé.
        payload = {
            "divergences": [
                {
                    "extrait_reference": "tout le monde se souvient de lui",
                    "extrait_hypothese": "[passage perdu]",
                    "type": "effondrement",
                    "gravite": "G3",
                    "impact_sens": "passage entier perdu",
                }
            ]
        }
        result = self._parse(
            payload,
            reference="alors tout le monde se souvient de lui salut",
            hypothesis="tu vois parce que merci",
        )
        self.assertEqual(len(result.divergences), 1)
        self.assertTrue(result.divergences[0].verbatim_ok)

    def test_drops_invalid_severity(self) -> None:
        payload = {"divergences": [{"extrait_reference": "a", "extrait_hypothese": "b", "gravite": "G9"}]}
        result = self._parse(payload, reference="a", hypothesis="b")
        self.assertEqual(result.divergences, [])

    def test_reference_word_count_is_computed(self) -> None:
        result = self._parse({"divergences": []}, reference="un deux trois quatre", hypothesis="x")
        self.assertEqual(result.reference_word_count, 4)
        self.assertEqual(result.verdict, "fidele")


class MergePassesTests(unittest.TestCase):
    def _res(self, divs: list[Divergence]) -> JudgeResult:
        return JudgeResult("s", "p", "m", "judge", "", divs, 1000)

    def test_stable_g3_missed_in_first_pass_is_kept(self) -> None:
        # G3 "alpha" vu dans les 3 passes ; G3 "beta" vu seulement aux passes 2
        # et 3 (manqué à la passe 0). Les deux sont stables (>50 %) et doivent
        # survivre, même si "beta" est absent de la passe de référence.
        alpha = Divergence("alpha", "a", "inversion_polarite", "G3", "")
        beta = Divergence("beta", "b", "inversion_polarite", "G3", "")
        results = [
            self._res([alpha]),
            self._res([alpha, beta]),
            self._res([alpha, beta]),
        ]
        merged = _merge_passes(results)
        refs = {d.extrait_reference for d in merged.divergences}
        self.assertIn("alpha", refs)
        self.assertIn("beta", refs)

    def test_unstable_g3_is_dropped(self) -> None:
        # G3 vu dans une seule passe sur trois (<50 %) -> écarté.
        stable = Divergence("stable", "s", "inversion_polarite", "G3", "")
        flaky = Divergence("flaky", "f", "inversion_polarite", "G3", "")
        results = [
            self._res([stable, flaky]),
            self._res([stable]),
            self._res([stable]),
        ]
        merged = _merge_passes(results)
        refs = {d.extrait_reference for d in merged.divergences}
        self.assertIn("stable", refs)
        self.assertNotIn("flaky", refs)


class RunJudgeCliTests(unittest.TestCase):
    def _corpus(self, root: Path) -> None:
        (root / "ground_truth").mkdir()
        (root / "transcriptions" / "sample").mkdir(parents=True)
        (root / "ground_truth" / "sample.txt").write_text("la vérité de référence", encoding="utf-8")
        (root / "transcriptions" / "sample" / "albert__m.txt").write_text("une hypothèse", encoding="utf-8")

    def test_all_failures_raise_instead_of_exit_zero(self) -> None:
        # Tous les couples échouent (clé invalide) -> ne doit pas réussir
        # silencieusement avec un rapport vide, mais lever JudgeCliError.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._corpus(root)
            with self.assertRaises(JudgeCliError):
                run_judge(
                    "sample",
                    ground_truth_dir=root / "ground_truth",
                    transcriptions_dir=root / "transcriptions",
                    client=_FailingClient(),
                    progress=False,
                )

    def test_output_to_directory_raises_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(JudgeCliError):
                write_or_print_report("# rapport", output_path=Path(tmp))


class JudgePairErrorWrappingTests(unittest.TestCase):
    def test_error_message_uses_client_provider_name(self) -> None:
        # Le wrapper d'erreur de judge_pair doit nommer le bon fournisseur,
        # pas « Albert » en dur, quand le juge tourne via OpenRouter.
        class _OpenRouterFailing:
            provider_name = "OpenRouter"

            def chat_completion_text(self, **kwargs: object) -> str:
                raise AlbertError("401 Unauthorized")

        with self.assertRaises(JudgeError) as ctx:
            judge_pair(
                reference="réf",
                hypothesis="hyp",
                sample_id="s",
                provider="p",
                model="m",
                client=_OpenRouterFailing(),
            )
        self.assertIn("OpenRouter", str(ctx.exception))
        self.assertNotIn("Albert", str(ctx.exception))

    def test_falls_back_when_model_rejects_response_format(self) -> None:
        # Un modèle qui refuse le JSON mode (400 response_format) doit déclencher
        # un second appel SANS response_format, pas un échec.
        calls: list[dict] = []

        class _NoJsonMode:
            provider_name = "OpenRouter"

            def chat_completion_text(self, **kwargs):
                calls.append(kwargs)
                if "response_format" in kwargs and kwargs["response_format"]:
                    raise AlbertError(
                        "OpenRouter request failed: 400 Bad Request for / - "
                        "response_format is not supported by this model"
                    )
                return '{"divergences": []}'

        result = judge_pair(
            reference="réf",
            hypothesis="hyp",
            sample_id="s",
            provider="p",
            model="m",
            client=_NoJsonMode(),
        )
        self.assertEqual(len(calls), 2)  # 1er avec JSON mode (400), 2e sans
        self.assertNotIn("response_format", calls[1])
        self.assertEqual(result.divergences, [])

    def test_does_not_retry_on_auth_error(self) -> None:
        # Une 401 n'est PAS un rejet de response_format -> pas de retry, échec direct.
        calls: list[dict] = []

        class _Unauthorized:
            provider_name = "OpenRouter"

            def chat_completion_text(self, **kwargs):
                calls.append(kwargs)
                raise AlbertError("OpenRouter request failed: 401 Unauthorized for /")

        with self.assertRaises(JudgeError):
            judge_pair(
                reference="r", hypothesis="h", sample_id="s",
                provider="p", model="m", client=_Unauthorized(),
            )
        self.assertEqual(len(calls), 1)  # pas de retry


if __name__ == "__main__":
    unittest.main()
