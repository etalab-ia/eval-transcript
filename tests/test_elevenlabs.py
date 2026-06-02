from __future__ import annotations

import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from elevenlabs.core import ApiError

from eval_transcript.elevenlabs import (
    ElevenLabsClient,
    ElevenLabsError,
    elevenlabs_error_detail,
    elevenlabs_transcription_text,
    serializable_response,
)


@dataclass(frozen=True)
class DummyWord:
    text: str
    start: float


class DummyPydanticResponse:
    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        return {"text": "bonjour", "words": [DummyWord("bonjour", 0.0)], "mode": mode}


class NonCallableDumpAttributes:
    model_dump = "not callable"
    dict = "not callable"


class FakeSpeechToText:
    def __init__(self, response: object) -> None:
        self.response = response
        self.request: dict[str, object] | None = None

    def convert(self, **kwargs: object) -> object:
        self.request = kwargs
        return self.response


class FakeSdkClient:
    def __init__(self, response: object) -> None:
        self.speech_to_text = FakeSpeechToText(response)


class ElevenLabsClientTests(unittest.TestCase):
    def test_constructor_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ElevenLabsError, "ELEVENLABS_API_KEY"):
                ElevenLabsClient()

    def test_transcribe_passes_scribe_parameters_and_serializes_response(self) -> None:
        fake_client = FakeSdkClient(DummyPydanticResponse())
        with TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "sample.wav"
            audio_path.write_bytes(b"audio")

            result = ElevenLabsClient(api_key="test-key", sdk_client=fake_client).transcribe(
                audio_path=audio_path,
                model="scribe_v2",
                language="fr",
                timestamps_granularity="word",
                diarize=True,
                num_speakers=2,
                temperature=0.0,
                seed=123,
                no_verbatim=True,
            )

        self.assertEqual(result["text"], "bonjour")
        self.assertEqual(result["mode"], "json")
        request = fake_client.speech_to_text.request
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request["model_id"], "scribe_v2")
        self.assertEqual(request["language_code"], "fr")
        self.assertEqual(request["timestamps_granularity"], "word")
        self.assertEqual(request["diarize"], True)
        self.assertEqual(request["num_speakers"], 2)
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["seed"], 123)
        self.assertEqual(request["no_verbatim"], True)

    def test_transcription_text_from_single_transcript(self) -> None:
        self.assertEqual(elevenlabs_transcription_text({"text": "bonjour"}), "bonjour")

    def test_transcription_text_from_multichannel_transcripts(self) -> None:
        result = {"transcripts": [{"text": "canal un"}, {"text": "canal deux"}, {"text": None}]}

        self.assertEqual(elevenlabs_transcription_text(result), "canal un\ncanal deux")

    def test_serializable_response_handles_dicts_lists_and_dataclasses(self) -> None:
        response = {"words": [DummyWord("bonjour", 0.0)]}

        self.assertEqual(serializable_response(response), {"words": [{"text": "bonjour", "start": 0.0}]})

    def test_serializable_response_ignores_non_callable_dump_attributes(self) -> None:
        value = NonCallableDumpAttributes()

        self.assertIs(serializable_response(value), value)

    def test_serializable_response_converts_sets_to_lists(self) -> None:
        self.assertEqual(sorted(serializable_response({"values": {"a", "b"}})["values"]), ["a", "b"])

    def test_api_errors_are_summarized_without_headers(self) -> None:
        error = ApiError(
            status_code=401,
            body={"detail": {"status": "detected_unusual_activity", "message": "Free Tier access has been disabled."}},
        )

        self.assertEqual(elevenlabs_error_detail(error), "status 401 - Free Tier access has been disabled.")


if __name__ == "__main__":
    unittest.main()
