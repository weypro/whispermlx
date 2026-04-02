"""Tests for whispermlx/schema.py TypedDict definitions."""

from whispermlx.schema import AlignedTranscriptionResult
from whispermlx.schema import SingleAlignedSegment
from whispermlx.schema import SingleCharSegment
from whispermlx.schema import SingleSegment
from whispermlx.schema import SingleWordSegment
from whispermlx.schema import TranscriptionResult


class TestSingleWordSegment:
    def test_required_fields_accepted(self):
        seg: SingleWordSegment = {"word": "hello", "start": 0.0, "end": 1.0, "score": 0.9}
        assert seg["word"] == "hello"
        assert seg["start"] == 0.0
        assert seg["end"] == 1.0
        assert seg["score"] == 0.9

    def test_is_dict_subtype(self):
        seg: SingleWordSegment = {"word": "hi", "start": 0.0, "end": 0.5, "score": 0.8}
        assert isinstance(seg, dict)

    def test_optional_speaker_field(self):
        seg: SingleWordSegment = {"word": "hi", "start": 0.0, "end": 0.5, "score": 0.8}
        seg["speaker"] = "SPEAKER_00"
        assert seg["speaker"] == "SPEAKER_00"

    def test_dict_access(self):
        seg: SingleWordSegment = {"word": "test", "start": 1.0, "end": 2.0, "score": 0.5}
        assert list(seg.keys()) == ["word", "start", "end", "score"]


class TestSingleCharSegment:
    def test_required_fields(self):
        seg: SingleCharSegment = {"char": "h", "start": 0.0, "end": 0.1, "score": 0.95}
        assert seg["char"] == "h"
        assert isinstance(seg, dict)


class TestSingleSegment:
    def test_required_fields(self):
        seg: SingleSegment = {"start": 0.0, "end": 5.0, "text": "hello world"}
        assert seg["text"] == "hello world"

    def test_optional_avg_logprob_absent(self):
        seg: SingleSegment = {"start": 0.0, "end": 5.0, "text": "test"}
        assert "avg_logprob" not in seg

    def test_optional_avg_logprob_present(self):
        seg: SingleSegment = {"start": 0.0, "end": 5.0, "text": "test", "avg_logprob": -0.5}
        assert seg["avg_logprob"] == -0.5

    def test_is_dict(self):
        seg: SingleSegment = {"start": 1.0, "end": 2.0, "text": "hi"}
        assert isinstance(seg, dict)


class TestSingleAlignedSegment:
    def test_has_words(self):
        seg: SingleAlignedSegment = {
            "start": 0.0,
            "end": 3.0,
            "text": "hi there",
            "words": [
                {"word": "hi", "start": 0.0, "end": 1.0, "score": 0.9},
                {"word": "there", "start": 1.0, "end": 3.0, "score": 0.8},
            ],
            "chars": None,
        }
        assert len(seg["words"]) == 2
        assert seg["chars"] is None

    def test_chars_can_be_list(self):
        seg: SingleAlignedSegment = {
            "start": 0.0,
            "end": 1.0,
            "text": "hi",
            "words": [],
            "chars": [{"char": "h", "start": 0.0, "end": 0.5, "score": 0.9}],
        }
        assert len(seg["chars"]) == 1


class TestTranscriptionResult:
    def test_has_segments_and_language(self):
        result: TranscriptionResult = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "test"}],
            "language": "en",
        }
        assert result["language"] == "en"
        assert len(result["segments"]) == 1

    def test_is_dict(self):
        result: TranscriptionResult = {"segments": [], "language": "fr"}
        assert isinstance(result, dict)


class TestAlignedTranscriptionResult:
    def test_has_word_segments(self):
        result: AlignedTranscriptionResult = {
            "segments": [],
            "word_segments": [{"word": "hi", "start": 0.0, "end": 1.0, "score": 0.9}],
        }
        assert len(result["word_segments"]) == 1

    def test_optional_speaker_embeddings(self):
        result: AlignedTranscriptionResult = {"segments": [], "word_segments": []}
        result["speaker_embeddings"] = {"SPEAKER_00": [0.1, 0.2, 0.3]}
        assert "speaker_embeddings" in result
