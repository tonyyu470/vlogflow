"""Characterize the audio-decoding boundary used by the installed ASR version."""

import numpy as np
import pytest
import soundfile as sf


def test_installed_asr_can_decode_24khz_reference_to_16khz(tmp_path):
    pytest.importorskip("faster_whisper")
    from faster_whisper.audio import decode_audio

    path = tmp_path / "reference.wav"
    sf.write(path, np.sin(2 * np.pi * 220 * np.arange(24000) / 24000) * 0.1, 24000)
    decoded = decode_audio(str(path), sampling_rate=16000)
    assert len(decoded) == 16000 and np.isfinite(decoded).all()
