from __future__ import annotations

import unittest

from eval_transcript.scaleway import (
    DEFAULT_PROMPT,
    ScalewayClient,
    build_prompt,
    generative_api_base_url,
    transcription_text,
)


class BuildPromptTests(unittest.TestCase):
    def test_no_language_uses_default_anti_translation_prompt(self) -> None:
        self.assertEqual(build_prompt(None), DEFAULT_PROMPT)

    def test_blank_language_is_treated_as_absent(self) -> None:
        self.assertEqual(build_prompt("   "), DEFAULT_PROMPT)

    def test_known_language_code_is_expanded_to_french_name(self) -> None:
        prompt = build_prompt("fr")
        self.assertIn("en français", prompt)
        self.assertIn("sans le traduire", prompt)

    def test_unknown_language_is_used_verbatim(self) -> None:
        self.assertIn("en occitan", build_prompt("occitan"))


class ListModelsTests(unittest.TestCase):
    def _client(self, payload: object) -> ScalewayClient:
        client = ScalewayClient(secret_key="secret", project_id="project")
        client._request_json = lambda method, path, **kwargs: {"data": payload}  # type: ignore[method-assign]
        return client

    def test_filters_non_string_ids_before_lowercasing(self) -> None:
        client = self._client(
            [
                {"id": "voxtral-small-24b-2507"},
                {"id": None},
                {"id": 123},
                {"id": ""},
                {"no_id": "ignored"},
                "not-a-dict",
            ]
        )
        self.assertEqual(client.list_models(), ["voxtral-small-24b-2507"])

    def test_name_filter_is_case_insensitive(self) -> None:
        client = self._client([{"id": "Voxtral-Small"}, {"id": "whisper-large"}])
        self.assertEqual(client.list_models(name="voxtral"), ["Voxtral-Small"])

    def test_non_list_payload_returns_empty(self) -> None:
        client = self._client({"unexpected": "shape"})
        self.assertEqual(client.list_models(), [])


class HelpersTests(unittest.TestCase):
    def test_base_url_is_derived_from_project_id(self) -> None:
        self.assertEqual(
            generative_api_base_url("abc-123"),
            "https://api.scaleway.ai/abc-123/v1",
        )

    def test_base_url_is_empty_without_project_id(self) -> None:
        self.assertEqual(generative_api_base_url(None), "")

    def test_transcription_text_reads_first_choice_content(self) -> None:
        result = {"choices": [{"message": {"content": "bonjour le monde"}}]}
        self.assertEqual(transcription_text(result), "bonjour le monde")

    def test_transcription_text_handles_missing_fields(self) -> None:
        self.assertEqual(transcription_text({}), "")

    def test_timeout_defaults_when_none(self) -> None:
        from eval_transcript.scaleway import DEFAULT_TIMEOUT_SECONDS

        self.assertEqual(ScalewayClient(secret_key="s", project_id="p").timeout, DEFAULT_TIMEOUT_SECONDS)

    def test_timeout_override_is_honored(self) -> None:
        self.assertEqual(ScalewayClient(secret_key="s", project_id="p", timeout=600.0).timeout, 600.0)


if __name__ == "__main__":
    unittest.main()
