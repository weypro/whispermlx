"""Tests for whispermlx/SubtitlesProcessor.py."""

import pytest

from whispermlx.SubtitlesProcessor import SubtitlesProcessor
from whispermlx.SubtitlesProcessor import format_timestamp
from whispermlx.SubtitlesProcessor import normal_round


class TestNormalRound:
    def test_rounds_down_below_half(self):
        assert normal_round(1.4) == 1

    def test_rounds_up_at_half(self):
        assert normal_round(1.5) == 2

    def test_integer_unchanged(self):
        assert normal_round(3.0) == 3

    def test_rounds_down_for_negative_fraction(self):
        assert normal_round(2.3) == 2

    def test_zero(self):
        assert normal_round(0.0) == 0


class TestFormatTimestamp:
    def test_zero_srt(self):
        result = format_timestamp(0.0, is_vtt=False)
        assert result == "00:00:00,000"

    def test_zero_vtt(self):
        result = format_timestamp(0.0, is_vtt=True)
        assert result == "00:00:00.000"

    def test_one_hour_srt(self):
        result = format_timestamp(3600.0, is_vtt=False)
        assert result.startswith("01:00:00")
        assert "," in result

    def test_one_hour_vtt(self):
        result = format_timestamp(3600.0, is_vtt=True)
        assert result.startswith("01:00:00")
        assert "." in result

    def test_negative_raises(self):
        with pytest.raises(AssertionError):
            format_timestamp(-1.0)

    def test_milliseconds_preserved(self):
        result = format_timestamp(1.001)
        assert "001" in result

    def test_one_minute_thirty(self):
        result = format_timestamp(90.0)
        assert "01:30" in result


class TestEstimateTimestampForWord:
    def test_has_prev_end_and_next_start(self):
        words = [
            {"word": "hello", "start": 0.0, "end": 1.0},
            {"word": "unknown"},  # missing timestamps
            {"word": "world", "start": 2.0, "end": 3.0},
        ]
        sp = SubtitlesProcessor([], "en")
        sp.estimate_timestamp_for_word(words, 1)
        assert words[1]["start"] == pytest.approx(1.0)
        assert words[1]["end"] == pytest.approx(2.0)

    def test_has_prev_end_no_next_start(self):
        words = [
            {"word": "hello", "start": 0.0, "end": 1.0},
            {"word": "world"},
        ]
        sp = SubtitlesProcessor([], "en")
        sp.estimate_timestamp_for_word(words, 1)
        assert words[1]["start"] == pytest.approx(1.0)
        assert words[1]["end"] > 1.0

    def test_has_next_start_no_prev_end(self):
        words = [
            {"word": "hello"},
            {"word": "world", "start": 1.5, "end": 2.5},
        ]
        sp = SubtitlesProcessor([], "en")
        sp.estimate_timestamp_for_word(words, 0)
        assert words[0]["end"] == pytest.approx(1.5)

    def test_isolated_word_with_no_neighbors(self):
        words = [{"word": "alone"}]
        sp = SubtitlesProcessor([], "en")
        sp.estimate_timestamp_for_word(words, 0)
        assert words[0]["start"] == 0
        assert words[0]["end"] == 0

    def test_isolated_word_with_next_segment_time(self):
        words = [{"word": "alone"}]
        sp = SubtitlesProcessor([], "en")
        sp.estimate_timestamp_for_word(words, 0, next_segment_start_time=5.0)
        assert words[0]["start"] == pytest.approx(4.0)
        assert words[0]["end"] == pytest.approx(4.5)


class TestDetermineAdvancedSplitPoints:
    def _make_words(self, texts):
        """Fake word dicts with dummy timestamps."""
        t = 0.0
        words = []
        for text in texts:
            words.append({"word": text, "start": t, "end": t + 0.5})
            t += 0.5
        return words

    def test_short_text_no_split_points(self):
        words = self._make_words(["Hi", "there"])
        segment = {"start": 0.0, "end": 1.0, "text": "Hi there", "words": words}
        sp = SubtitlesProcessor([segment], "en")
        splits = sp.determine_advanced_split_points(segment)
        assert splits == []

    def test_long_text_produces_split(self):
        # Create text longer than max_line_length (45 chars)
        word_texts = ["word"] * 20  # 20 * 5 = 100 chars (> 45)
        words = self._make_words(word_texts)
        segment = {"start": 0.0, "end": 10.0, "text": " ".join(word_texts), "words": words}
        sp = SubtitlesProcessor([segment], "en")
        splits = sp.determine_advanced_split_points(segment)
        assert len(splits) >= 1

    def test_complex_script_uses_shorter_threshold(self):
        sp_ja = SubtitlesProcessor([], "ja")
        sp_en = SubtitlesProcessor([], "en")
        assert sp_ja.max_line_length < sp_en.max_line_length
        assert sp_ja.min_char_length_splitter < sp_en.min_char_length_splitter


class TestGenerateSubtitlesFromSplitPoints:
    def _segment_with_words(self, word_texts):
        t = 0.0
        words = []
        for text in word_texts:
            words.append({"word": text, "start": t, "end": t + 0.5})
            t += 0.5
        return {
            "start": 0.0,
            "end": t,
            "text": " ".join(word_texts),
            "words": words,
        }

    def test_no_split_points_produces_one_subtitle(self):
        segment = self._segment_with_words(["hello", "world"])
        sp = SubtitlesProcessor([segment], "en")
        subtitles = sp.generate_subtitles_from_split_points(segment, [])
        assert len(subtitles) == 1
        assert "hello" in subtitles[0]["text"]

    def test_single_split_produces_two_subtitles(self):
        segment = self._segment_with_words(["hello", "world", "how", "are", "you"])
        sp = SubtitlesProcessor([segment], "en")
        # split after index 1 → ["hello", "world"] and ["how", "are", "you"]
        subtitles = sp.generate_subtitles_from_split_points(segment, [1])
        assert len(subtitles) == 2

    def test_string_words_fallback(self):
        """When words are plain strings (no dicts), use proportional time."""
        segment = {
            "start": 0.0,
            "end": 4.0,
            "text": "hello world how are you",
            "words": ["hello", "world", "how", "are", "you"],
        }
        sp = SubtitlesProcessor([segment], "en")
        subtitles = sp.generate_subtitles_from_split_points(segment, [1])
        assert len(subtitles) == 2
        # Times should be proportional
        assert subtitles[0]["start"] == pytest.approx(0.0)

    def test_subtitle_has_start_end_text(self):
        segment = self._segment_with_words(["hi", "there"])
        sp = SubtitlesProcessor([segment], "en")
        subtitles = sp.generate_subtitles_from_split_points(segment, [])
        for sub in subtitles:
            assert "start" in sub
            assert "end" in sub
            assert "text" in sub


class TestProcessSegments:
    def test_returns_list(self):
        segment = {
            "start": 0.0,
            "end": 2.0,
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0.0, "end": 1.0},
                {"word": "world", "start": 1.0, "end": 2.0},
            ],
        }
        sp = SubtitlesProcessor([segment], "en")
        result = sp.process_segments(advanced_splitting=True)
        assert isinstance(result, list)

    def test_non_advanced_returns_one_per_segment(self):
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "hello",
                "words": [{"word": "hello", "start": 0.0, "end": 2.0}],
            },
            {
                "start": 3.0,
                "end": 5.0,
                "text": "world",
                "words": [{"word": "world", "start": 3.0, "end": 5.0}],
            },
        ]
        sp = SubtitlesProcessor(segments, "en")
        result = sp.process_segments(advanced_splitting=False)
        assert len(result) == 2

    def test_output_dicts_have_required_keys(self):
        segment = {
            "start": 0.0,
            "end": 2.0,
            "text": "hi",
            "words": [{"word": "hi", "start": 0.0, "end": 2.0}],
        }
        sp = SubtitlesProcessor([segment], "en")
        result = sp.process_segments()
        for item in result:
            assert "start" in item
            assert "end" in item
            assert "text" in item


class TestSave:
    def test_writes_srt_file(self, tmp_path):
        segment = {
            "start": 0.0,
            "end": 2.0,
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0.0, "end": 1.0},
                {"word": "world", "start": 1.0, "end": 2.0},
            ],
        }
        sp = SubtitlesProcessor([segment], "en")
        filename = str(tmp_path / "test.srt")
        count = sp.save(filename=filename, advanced_splitting=True)
        assert count >= 1
        content = (tmp_path / "test.srt").read_text()
        assert "-->" in content

    def test_writes_vtt_file(self, tmp_path):
        segment = {
            "start": 0.0,
            "end": 2.0,
            "text": "hello",
            "words": [{"word": "hello", "start": 0.0, "end": 2.0}],
        }
        sp = SubtitlesProcessor([segment], "en", is_vtt=True)
        filename = str(tmp_path / "test.vtt")
        sp.save(filename=filename)
        content = (tmp_path / "test.vtt").read_text()
        assert "WEBVTT" in content

    def test_returns_subtitle_count(self, tmp_path):
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "hello",
                "words": [{"word": "hello", "start": 0.0, "end": 2.0}],
            },
            {
                "start": 3.0,
                "end": 5.0,
                "text": "world",
                "words": [{"word": "world", "start": 3.0, "end": 5.0}],
            },
        ]
        sp = SubtitlesProcessor(segments, "en")
        filename = str(tmp_path / "test.srt")
        count = sp.save(filename=filename)
        assert count == 2
