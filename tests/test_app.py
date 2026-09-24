"""Fast unit tests for TaleTwinkle's deterministic processing functions."""

# Import Part

import io
from pathlib import Path

import numpy as np
from PIL import Image

import app


# Function Part

def test_prepare_uploaded_image_resizes_and_converts() -> None:
    """A large uploaded image should become RGB and fit the model limit."""
    image_buffer = io.BytesIO()
    Image.new("RGBA", (1800, 1200), (20, 80, 140, 180)).save(image_buffer, format="PNG")
    prepared_image = app.prepare_uploaded_image(image_buffer.getvalue())
    assert prepared_image.mode == "RGB"
    assert max(prepared_image.size) == app.MAXIMUM_IMAGE_EDGE


def test_story_length_matches_slider_target() -> None:
    """Length fitting should preserve the exact requested word count."""
    draft_story = "A kind fox found a glowing path. Friends followed the stars together."
    fitted_story = app.fit_story_to_requested_length(
        draft_story,
        "Everyone reached home smiling",
        75,
    )
    assert app.count_story_words(fitted_story) == 75


def test_story_length_is_clamped_to_assignment_range() -> None:
    """Unexpected values must still respect the assignment's 50–100 word range."""
    draft_story = "A tiny bird shared a bright idea with every friend."
    short_story = app.fit_story_to_requested_length(draft_story, "They all cheered", 10)
    long_story = app.fit_story_to_requested_length(draft_story, "They all cheered", 500)
    assert app.count_story_words(short_story) == 50
    assert app.count_story_words(long_story) == 100


def test_repeated_sentences_are_removed() -> None:
    """Exact repeated model sentences should appear only once."""
    repeated_story = "A fox smiled. A fox smiled. The moon sparkled."
    cleaned_story = app.remove_repeated_sentences(repeated_story)
    assert cleaned_story.count("A fox smiled") == 1


def test_repeated_caption_words_are_collapsed() -> None:
    """A looping caption should retain one meaningful occurrence."""
    looping_caption = "jungle jungle jungle with bright bright leaves"
    assert app.remove_consecutive_word_repetition(looping_caption) == "jungle with bright leaves"


def test_unsuitable_language_filter() -> None:
    """The safety filter should catch a blocked term without false casing issues."""
    assert app.contains_unsuitable_language("A WEAPON appeared in the tale.") is True
    assert app.contains_unsuitable_language("A friendly rabbit shared a carrot.") is False


def test_voice_catalog_contains_requested_regions_and_personas() -> None:
    """Default, parrot, Tom, US, UK, and AU choices should all be available."""
    voice_names = " ".join(app.VOICE_PERSONAS)
    assert list(app.VOICE_PERSONAS)[0].startswith("🌙 Luna")
    assert "Polly" in voice_names and "Tom" in voice_names
    assert "US" in voice_names and "UK" in voice_names and "AU" in voice_names
    assert len(app.VOICE_PERSONAS) == 21


def test_music_mixing_adds_a_tail() -> None:
    """The mixed waveform should continue five seconds beyond narration."""
    sample_rate = 22050
    narration = np.zeros(sample_rate, dtype=np.float32)
    music_path = Path(app.MUSIC_DIRECTORY) / "fairy_tale.wav"
    mixed_audio = app.mix_narration_with_theme_music(
        narration,
        sample_rate,
        music_path,
        music_tail_seconds=2.0,
    )
    assert len(mixed_audio) == sample_rate * 3


# Main Part

if __name__ == "__main__":
    raise SystemExit("Run these checks with: pytest -q")
