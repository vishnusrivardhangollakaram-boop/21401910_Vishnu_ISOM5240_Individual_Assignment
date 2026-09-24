"""Generate original, lightweight background music loops for TaleTwinkle."""

from pathlib import Path

import numpy as np
from scipy.io import wavfile


SAMPLE_RATE = 22_050
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "assets" / "music"


THEME_SETTINGS = {
    "fairy_tale": {"tempo": 92, "notes": [60, 64, 67, 72], "wave": "sine"},
    "space_quest": {"tempo": 78, "notes": [48, 55, 60, 67], "wave": "triangle"},
    "gentle_mystery": {"tempo": 72, "notes": [57, 60, 64, 63], "wave": "sine"},
    "jungle_adventure": {"tempo": 108, "notes": [55, 62, 67, 69], "wave": "triangle"},
    "silly_poem": {"tempo": 118, "notes": [60, 67, 64, 72], "wave": "sine"},
    "ocean_magic": {"tempo": 68, "notes": [53, 60, 65, 69], "wave": "sine"},
}


def midi_note_to_frequency(midi_note: int) -> float:
    """Convert a MIDI note number into its frequency in hertz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def create_tone(frequency: float, duration_seconds: float, wave_type: str) -> np.ndarray:
    """Create one softly faded musical tone."""
    sample_count = int(SAMPLE_RATE * duration_seconds)
    timeline = np.arange(sample_count, dtype=np.float32) / SAMPLE_RATE
    phase = 2.0 * np.pi * frequency * timeline

    if wave_type == "triangle":
        waveform = 2.0 * np.abs(2.0 * ((frequency * timeline) % 1.0) - 1.0) - 1.0
    else:
        waveform = np.sin(phase)

    fade_sample_count = max(1, int(SAMPLE_RATE * min(0.08, duration_seconds / 4.0)))
    envelope = np.ones(sample_count, dtype=np.float32)
    envelope[:fade_sample_count] = np.linspace(0.0, 1.0, fade_sample_count)
    envelope[-fade_sample_count:] = np.linspace(1.0, 0.0, fade_sample_count)
    return waveform.astype(np.float32) * envelope


def create_theme_loop(tempo: int, notes: list[int], wave_type: str) -> np.ndarray:
    """Create a quiet sixteen-bar loop from a short child-friendly motif."""
    beat_duration = 60.0 / tempo
    sequence = notes + notes[::-1] + notes[1:] + notes[-2::-1]
    melody_parts = []

    for index, midi_note in enumerate(sequence * 2):
        note_duration = beat_duration * (1.5 if index % 4 == 3 else 1.0)
        root_tone = create_tone(midi_note_to_frequency(midi_note), note_duration, wave_type)
        harmony_tone = create_tone(
            midi_note_to_frequency(midi_note - 12), note_duration, "sine"
        )
        melody_parts.append((0.72 * root_tone) + (0.28 * harmony_tone))

    complete_loop = np.concatenate(melody_parts)
    complete_loop /= max(np.max(np.abs(complete_loop)), 1e-6)
    return complete_loop * 0.18


def save_theme_music(theme_name: str, settings: dict[str, object]) -> None:
    """Generate and save one theme loop as a mono WAV file."""
    audio_waveform = create_theme_loop(
        tempo=int(settings["tempo"]),
        notes=list(settings["notes"]),
        wave_type=str(settings["wave"]),
    )
    output_path = OUTPUT_DIRECTORY / f"{theme_name}.wav"
    wavfile.write(output_path, SAMPLE_RATE, (audio_waveform * 32767).astype(np.int16))
    print(f"Created {output_path.name}: {len(audio_waveform) / SAMPLE_RATE:.2f} seconds")


def main() -> None:
    """Generate all TaleTwinkle theme music files."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for theme_name, settings in THEME_SETTINGS.items():
        save_theme_music(theme_name, settings)


if __name__ == "__main__":
    main()
