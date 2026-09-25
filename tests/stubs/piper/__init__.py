"""Small Piper stand-in for offline TaleTwinkle tests."""

from types import SimpleNamespace

import numpy as np


class PiperVoice:
    """Return deterministic speech-like samples without downloading an ONNX model."""

    @classmethod
    def load(cls, model_path, config_path, use_cuda=False):
        """Match Piper's loader signature."""
        del config_path, use_cuda
        voice = cls()
        voice.model_path = model_path
        return voice

    def synthesize(self, text):
        """Yield one audio chunk shaped like Piper's real output."""
        sample_rate = 22050
        sample_count = sample_rate * max(1, len(str(text).split()) // 3)
        time_axis = np.arange(sample_count, dtype=np.float32) / sample_rate
        voice_offset = sum(ord(character) for character in self.model_path) % 70
        samples = (0.12 * np.sin(2 * np.pi * (170 + voice_offset) * time_axis)).astype(np.float32)
        yield SimpleNamespace(audio_float_array=samples, sample_rate=sample_rate)
