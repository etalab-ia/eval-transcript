from __future__ import annotations

import unittest

from eval_transcript.albert import AlbertClient
from eval_transcript.judge import Divergence, JudgeResult, _verdict_from
from eval_transcript.openrouter import OpenRouterClient
from eval_transcript.panel import (
    JudgeSpec,
    consensus_for_transcript,
    parse_judge_spec,
    render_panel_markdown,
)


def _result(judge: str, g3_refs: list[str], *, sample="s", provider="voxtral", model="m") -> JudgeResult:
    divs = [Divergence(ref, "[h]", "inversion_polarite", "G3", "x") for ref in g3_refs]
    r = JudgeResult(sample, provider, model, judge, "", divs, 1000)
    r.verdict = _verdict_from(r)
    return r


class ParseSpecTests(unittest.TestCase):
    def test_provider_only_uses_default_model(self) -> None:
        self.assertEqual(parse_judge_spec("albert"), JudgeSpec("albert", "mistral-medium-2508"))
        spec = parse_judge_spec("openrouter")
        self.assertEqual(spec.provider, "openrouter")
        self.assertTrue(spec.model.startswith("anthropic/"))

    def test_provider_and_model(self) -> None:
        # Un id de modèle OpenRouter contient un « / » mais pas de « : ».
        spec = parse_judge_spec("openrouter:openai/gpt-5")
        self.assertEqual(spec, JudgeSpec("openrouter", "openai/gpt-5"))

    def test_unknown_provider_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_spec("groq:llama")

    def test_make_client_type(self) -> None:
        self.assertIsInstance(parse_judge_spec("albert").make_client(), AlbertClient)
        self.assertIsInstance(parse_judge_spec("openrouter").make_client(), OpenRouterClient)


class ConsensusTests(unittest.TestCase):
    def test_majority_keeps_only_agreed_g3(self) -> None:
        # 3 juges : "alpha" vu par les 3, "beta" par 2, "gamma" par 1.
        results = [
            _result("j1", ["alpha", "beta", "gamma"]),
            _result("j2", ["alpha", "beta"]),
            _result("j3", ["alpha"]),
        ]
        merged, counts, n, seuil = consensus_for_transcript(results, panel_size=3)
        self.assertEqual(n, 3)
        self.assertEqual(seuil, 2)  # majorité stricte de 3
        kept = {d.extrait_reference for d in merged.divergences if d.gravite == "G3"}
        self.assertEqual(kept, {"alpha", "beta"})  # gamma (1 juge) écarté
        self.assertEqual(counts["alpha"], 3)
        self.assertEqual(counts["gamma"], 1)

    def test_min_agree_override(self) -> None:
        results = [
            _result("j1", ["alpha", "beta"]),
            _result("j2", ["alpha"]),
        ]
        # Unanimité exigée (2/2) : seul "alpha" survit.
        merged, _counts, _n, seuil = consensus_for_transcript(results, panel_size=2, min_agree=2)
        self.assertEqual(seuil, 2)
        kept = {d.extrait_reference for d in merged.divergences if d.gravite == "G3"}
        self.assertEqual(kept, {"alpha"})

    def test_partial_coverage_does_not_auto_pass(self) -> None:
        # Panel de 2 juges, mais un seul a produit un résultat pour ce couple
        # (l'autre a échoué). Le seuil doit rester celui du panel (2), donc le
        # G3 du juge isolé n'est PAS retenu — sinon le consensus est bidon.
        results = [_result("j1", ["alpha"])]
        merged, _counts, n, seuil = consensus_for_transcript(results, panel_size=2)
        self.assertEqual(n, 1)  # couverture partielle remontée
        self.assertEqual(seuil, 2)  # seuil sur le panel complet, pas sur n=1
        kept = {d.extrait_reference for d in merged.divergences if d.gravite == "G3"}
        self.assertEqual(kept, set())  # rien retenu : 1 juge < seuil 2

    def test_does_not_mutate_inputs(self) -> None:
        r1 = _result("j1", ["alpha"])
        r2 = _result("j2", [])
        before = len(r1.divergences)
        consensus_for_transcript([r1, r2], panel_size=2)
        self.assertEqual(len(r1.divergences), before)  # original intact


class RenderTests(unittest.TestCase):
    def test_render_has_compare_and_consensus(self) -> None:
        by_spec = [
            ("albert:mistral-medium-2508", [_result("albert:mistral-medium-2508", ["alpha", "beta"])]),
            ("openrouter:anthropic/claude-sonnet-4.5", [_result("openrouter:anthropic/claude-sonnet-4.5", ["alpha"])]),
        ]
        md = render_panel_markdown(by_spec, mode="both")
        self.assertIn("## Comparaison des juges", md)
        self.assertIn("## Consensus", md)
        self.assertIn("Écart max", md)
        self.assertIn("voxtral/m", md)

    def test_compare_only(self) -> None:
        by_spec = [("albert", [_result("albert", ["alpha"])])]
        md = render_panel_markdown(by_spec, mode="compare")
        self.assertIn("## Comparaison des juges", md)
        self.assertNotIn("## Consensus", md)


if __name__ == "__main__":
    unittest.main()
