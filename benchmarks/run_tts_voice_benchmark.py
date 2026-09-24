"""Benchmark ten child-facing personas from the selected local HF TTS model."""

# Import Part

import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIRECTORY))

import app  # noqa: E402


# Function Part

OUTPUT_PATH = Path(__file__).resolve().parent / "tts_voice_persona_results.csv"
BENCHMARK_STORY = (
    "Once upon a time, a small fox found a silver star beside a friendly tree. "
    "The fox asked an owl for help. Together they carried the star across a bright "
    "meadow and placed it safely in the sky. Every friend cheered, and the little "
    "fox skipped home beneath its warm glow."
)


def analyze_wav_bytes(audio_bytes: bytes) -> dict[str, float]:
    """Measure duration, loudness, clipping, and a transparent quality proxy."""
    audio_waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    peak_amplitude = float(np.max(np.abs(audio_waveform)))
    root_mean_square = float(np.sqrt(np.mean(np.square(audio_waveform))))
    clipping_ratio = float(np.mean(np.abs(audio_waveform) >= 0.999))
    duration_seconds = len(audio_waveform) / float(sample_rate)
    quality_proxy = 100.0
    if not 0.01 <= root_mean_square <= 0.35:
        quality_proxy -= 25.0
    if peak_amplitude > 0.99:
        quality_proxy -= 25.0
    if clipping_ratio > 0.001:
        quality_proxy -= 25.0
    if duration_seconds <= 1.0:
        quality_proxy -= 25.0
    return {
        "duration_seconds": duration_seconds,
        "peak_amplitude": peak_amplitude,
        "root_mean_square": root_mean_square,
        "clipping_ratio": clipping_ratio,
        "technical_quality_proxy_percent": quality_proxy,
    }


def main() -> None:
    """Generate and analyze the first ten requested voice-persona choices."""
    benchmark_results = []
    selected_voice_names = list(app.VOICE_PERSONAS.keys())[:10]
    for voice_name in selected_voice_names:
        start_time = time.perf_counter()
        audio_bytes = app.create_narrated_story_audio(
            BENCHMARK_STORY,
            voice_name,
            1.00,
            "Fairy Tale 🏰",
        )
        runtime_seconds = time.perf_counter() - start_time
        audio_metrics = analyze_wav_bytes(audio_bytes)
        result = {
            "voice_persona": voice_name,
            "runtime_seconds": runtime_seconds,
            "audio_size_bytes": len(audio_bytes),
            **audio_metrics,
        }
        benchmark_results.append(result)
        print(
            f"{voice_name} | runtime={runtime_seconds:.2f}s | "
            f"quality_proxy={audio_metrics['technical_quality_proxy_percent']:.0f}%",
            flush=True,
        )

    pd.DataFrame(benchmark_results).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(benchmark_results)} voice-persona results.", flush=True)


if __name__ == "__main__":
    main()
