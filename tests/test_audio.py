"""Tests for whispermlx/audio.py."""

import subprocess

import numpy as np
import pytest
import torch

from whispermlx.audio import N_SAMPLES
from whispermlx.audio import SAMPLE_RATE
from whispermlx.audio import load_audio
from whispermlx.audio import log_mel_spectrogram
from whispermlx.audio import pad_or_trim


class TestLoadAudio:
    def test_load_audio_success(self, tmp_path, mocker):
        """mock ffmpeg subprocess returning int16 PCM bytes."""
        pcm_data = np.array([1000, -1000, 500, -500], dtype=np.int16).tobytes()
        mock_result = mocker.MagicMock()
        mock_result.stdout = pcm_data
        mocker.patch("whispermlx.audio.subprocess.run", return_value=mock_result)

        audio_file = str(tmp_path / "audio.wav")
        result = load_audio(audio_file)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        # values normalised to [-1, 1]
        assert result.max() <= 1.0
        assert result.min() >= -1.0
        assert len(result) == 4

    def test_load_audio_ffmpeg_failure(self, tmp_path, mocker):
        """CalledProcessError from ffmpeg should become RuntimeError."""
        err = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"No such file")
        mocker.patch("whispermlx.audio.subprocess.run", side_effect=err)

        with pytest.raises(RuntimeError, match="Failed to load audio"):
            load_audio(str(tmp_path / "missing.wav"))

    def test_load_audio_empty_output(self, tmp_path, mocker):
        """Empty bytes should produce an empty float32 array without crashing."""
        mock_result = mocker.MagicMock()
        mock_result.stdout = b""
        mocker.patch("whispermlx.audio.subprocess.run", return_value=mock_result)

        result = load_audio(str(tmp_path / "silent.wav"))
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 0

    def test_load_audio_normalizes_values(self, tmp_path, mocker):
        """int16 max value should map to ~1.0."""
        pcm_data = np.array([32767], dtype=np.int16).tobytes()
        mock_result = mocker.MagicMock()
        mock_result.stdout = pcm_data
        mocker.patch("whispermlx.audio.subprocess.run", return_value=mock_result)

        result = load_audio(str(tmp_path / "loud.wav"))
        assert abs(result[0] - 32767 / 32768.0) < 1e-4


class TestPadOrTrim:
    def test_trim_numpy_array(self):
        audio = np.ones(N_SAMPLES + 1000, dtype=np.float32)
        result = pad_or_trim(audio)
        assert result.shape == (N_SAMPLES,)

    def test_pad_numpy_array(self):
        audio = np.ones(1000, dtype=np.float32)
        result = pad_or_trim(audio)
        assert result.shape == (N_SAMPLES,)
        # trailing zeros
        assert result[1000:].sum() == 0.0

    def test_trim_torch_tensor(self):
        audio = torch.ones(N_SAMPLES + 500)
        result = pad_or_trim(audio)
        assert result.shape == (N_SAMPLES,)

    def test_pad_torch_tensor(self):
        audio = torch.ones(500)
        result = pad_or_trim(audio)
        assert result.shape == (N_SAMPLES,)
        assert result[500:].sum().item() == 0.0

    def test_already_correct_length_numpy(self):
        audio = np.ones(N_SAMPLES, dtype=np.float32)
        result = pad_or_trim(audio)
        assert result.shape == (N_SAMPLES,)
        np.testing.assert_array_equal(result, audio)

    def test_already_correct_length_tensor(self):
        audio = torch.ones(N_SAMPLES)
        result = pad_or_trim(audio)
        assert result.shape == (N_SAMPLES,)

    def test_custom_length(self):
        audio = np.ones(100, dtype=np.float32)
        result = pad_or_trim(audio, length=50)
        assert result.shape == (50,)

    def test_pad_custom_length(self):
        audio = np.ones(30, dtype=np.float32)
        result = pad_or_trim(audio, length=100)
        assert result.shape == (100,)
        assert result[30:].sum() == 0.0


class TestLogMelSpectrogram:
    def test_returns_correct_shape_from_numpy(self):
        """30-second audio at 16kHz should produce (80, 3000) mel spectrogram."""
        from whispermlx.audio import N_FRAMES
        from whispermlx.audio import N_SAMPLES

        audio = np.zeros(N_SAMPLES, dtype=np.float32)
        result = log_mel_spectrogram(audio, n_mels=80)
        assert result.shape == (80, N_FRAMES)

    def test_returns_tensor(self):
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        result = log_mel_spectrogram(audio, n_mels=80)
        assert torch.is_tensor(result)

    def test_accepts_tensor_input(self):
        from whispermlx.audio import N_FRAMES
        from whispermlx.audio import N_SAMPLES

        audio = torch.zeros(N_SAMPLES)
        result = log_mel_spectrogram(audio, n_mels=80)
        assert result.shape == (80, N_FRAMES)

    def test_padding_extends_output(self):
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        result_no_pad = log_mel_spectrogram(audio, n_mels=80, padding=0)
        result_padded = log_mel_spectrogram(audio, n_mels=80, padding=SAMPLE_RATE)
        # padded version should have more frames
        assert result_padded.shape[-1] > result_no_pad.shape[-1]
