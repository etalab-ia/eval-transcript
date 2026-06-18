from __future__ import annotations

import os
import unittest
from unittest import mock

from eval_transcript.albert import AlbertClient
from eval_transcript.openrouter import (
    DEFAULT_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    OpenRouterClient,
)


class OpenRouterClientTests(unittest.TestCase):
    def test_is_albert_subclass(self) -> None:
        # judge_pair est typé `AlbertClient | None` et catche `AlbertError` ;
        # la sous-classe garantit la compatibilité de duck-typing.
        self.assertTrue(issubclass(OpenRouterClient, AlbertClient))

    def test_defaults_and_auth_header(self) -> None:
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "k-123"}, clear=True):
            client = OpenRouterClient()
        self.assertEqual(client.base_url, DEFAULT_BASE_URL)
        self.assertEqual(client.provider_name, "OpenRouter")
        self.assertEqual(client.headers["Authorization"], "Bearer k-123")
        # Attribution par défaut (X-Title), pas de Referer sans URL configurée.
        self.assertEqual(client.headers["X-Title"], "eval-transcript")
        self.assertNotIn("HTTP-Referer", client.headers)

    def test_attribution_headers(self) -> None:
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}, clear=True):
            client = OpenRouterClient(app_title="bench", app_url="https://ex.test")
        self.assertEqual(client.headers["HTTP-Referer"], "https://ex.test")
        self.assertEqual(client.headers["X-Title"], "bench")

    def test_missing_key_does_not_leak_albert_key(self) -> None:
        # Sans clé OpenRouter, on ne doit PAS retomber sur $ALBERT_API_KEY.
        with mock.patch.dict(os.environ, {"ALBERT_API_KEY": "albert-secret"}, clear=True):
            client = OpenRouterClient()
        self.assertIsNone(client.api_key)
        self.assertNotIn("Authorization", client.headers)

    def test_default_judge_model_is_not_mistral(self) -> None:
        # Le défaut OpenRouter doit éviter la famille Mistral (anti-biais).
        self.assertNotIn("mistral", DEFAULT_JUDGE_MODEL.lower())


if __name__ == "__main__":
    unittest.main()
