"""TaleTwinkle: turn one uploaded picture into a narrated children's story.

The application is intentionally function-based. Each input, processing, and
output responsibility is separated so the flow is readable, testable, and easy
to maintain without classes.
"""

# =============================================================================
# Import Part
# =============================================================================

import base64
import hashlib
import html
import io
import re
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import torch
from PIL import Image, ImageOps
from transformers import pipeline, set_seed


# =============================================================================
# Function Part: Constants and Configuration
# =============================================================================

APP_DIRECTORY = Path(__file__).resolve().parent
BACKGROUND_DIRECTORY = APP_DIRECTORY / "assets" / "backgrounds"
MUSIC_DIRECTORY = APP_DIRECTORY / "assets" / "music"

CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
STORY_MODEL_NAME = "roneneldan/TinyStories-33M"
TTS_MODEL_NAME = "facebook/mms-tts-eng"
MINIMUM_STORY_WORDS = 50
MAXIMUM_STORY_WORDS = 100
DEFAULT_STORY_WORDS = 75
MAXIMUM_IMAGE_EDGE = 1024

THEME_SETTINGS = {
    "Fairy Tale 🏰": {
        "key": "Fairy Tale",
        "background": "fairy_tale.png",
        "music": "fairy_tale.wav",
        "accent": "#8d4de8",
        "hook": "A warm sparkle danced through the friendly kingdom",
        "closing": "From then on, kindness kept the whole kingdom glowing happily",
    },
    "Space Quest 🚀": {
        "key": "Space Quest",
        "background": "space_quest.png",
        "music": "space_quest.wav",
        "accent": "#3f75e8",
        "hook": "A tiny starship blinked hello beneath the silver stars",
        "closing": "Together they zoomed home, proud of their bright discovery",
    },
    "Gentle Mystery 🔎": {
        "key": "Gentle Mystery",
        "background": "gentle_mystery.png",
        "music": "gentle_mystery.wav",
        "accent": "#8a5a3b",
        "hook": "A curious clue waited quietly where everyone could look",
        "closing": "The friendly mystery ended with cheers, smiles, and one last clue",
    },
    "Jungle Adventure 🦜": {
        "key": "Jungle Adventure",
        "background": "jungle_adventure.png",
        "music": "jungle_adventure.wav",
        "accent": "#2f8b57",
        "hook": "Bright leaves rustled as a friendly adventure began",
        "closing": "They returned safely, carrying happy memories from their adventure",
    },
    "Silly Poem 🎵": {
        "key": "Silly Poem",
        "background": "silly_poem.png",
        "music": "silly_poem.wav",
        "accent": "#e05b9d",
        "hook": "A bouncy rhyme skipped in with a wiggle and a grin",
        "closing": "They giggled in rhythm, then danced home before tea",
    },
    "Ocean Magic 🐠": {
        "key": "Ocean Magic",
        "background": "ocean_magic.png",
        "music": "ocean_magic.wav",
        "accent": "#159bb5",
        "hook": "A trail of bubbles shimmered through the gentle blue sea",
        "closing": "The reef sparkled brighter as every sea friend celebrated together",
    },
}

# The local Hugging Face voice is transformed through pitch and pace processing.
# Region labels are playful character inspiration, not a claim of a literal
# demographic speaker or guaranteed regional accent.
VOICE_PERSONAS = {
    "🌙 Luna — Story Lady (Default)": {"pitch": 0.0, "pace": 1.00},
    "🦜 Polly — Parrot Pal": {"pitch": 4.0, "pace": 1.08},
    "🧢 Tom — Tale Buddy": {"pitch": -1.8, "pace": 1.03},
    "🇺🇸 Sunny Stella — US-Inspired Lady": {"pitch": 0.4, "pace": 1.00},
    "🇺🇸 Captain Max — US-Inspired Man": {"pitch": -2.5, "pace": 0.98},
    "🇺🇸 Grandpa Oak — US-Inspired Grandfather": {"pitch": -3.2, "pace": 0.86},
    "🇺🇸 Grandma Rose — US-Inspired Grandmother": {"pitch": -0.8, "pace": 0.88},
    "🇺🇸 Buddy Ben — US-Inspired Boy Style": {"pitch": 2.5, "pace": 1.07},
    "🇺🇸 Giggle Grace — US-Inspired Girl Style": {"pitch": 3.2, "pace": 1.05},
    "🇬🇧 Lady Poppy — UK-Inspired Lady": {"pitch": 0.2, "pace": 0.95},
    "🇬🇧 Sir Finn — UK-Inspired Man": {"pitch": -2.4, "pace": 0.94},
    "🇬🇧 Grandpa Alfie — UK-Inspired Grandfather": {"pitch": -3.0, "pace": 0.85},
    "🇬🇧 Grandma Bea — UK-Inspired Grandmother": {"pitch": -0.7, "pace": 0.87},
    "🇬🇧 Clever Charlie — UK-Inspired Boy Style": {"pitch": 2.4, "pace": 1.03},
    "🇬🇧 Merry Millie — UK-Inspired Girl Style": {"pitch": 3.0, "pace": 1.02},
    "🇦🇺 Coral Chloe — AU-Inspired Lady": {"pitch": 0.6, "pace": 1.03},
    "🇦🇺 Ranger Jack — AU-Inspired Man": {"pitch": -2.3, "pace": 1.02},
    "🇦🇺 Grandpa Wally — AU-Inspired Grandfather": {"pitch": -3.0, "pace": 0.89},
    "🇦🇺 Grandma Mabel — AU-Inspired Grandmother": {"pitch": -0.6, "pace": 0.91},
    "🇦🇺 Koala Kai — AU-Inspired Boy Style": {"pitch": 2.5, "pace": 1.09},
    "🇦🇺 Matilda — AU-Inspired Girl Style": {"pitch": 3.1, "pace": 1.07},
}

UNSUITABLE_STORY_WORDS = {
    "blood",
    "burn",
    "burned",
    "burning",
    "dead",
    "death",
    "die",
    "gun",
    "hate",
    "horror",
    "kill",
    "killed",
    "nightmare",
    "shoot",
    "shooting",
    "battle",
    "fight",
    "fighting",
    "scared",
    "trick",
    "tricked",
    "fooled",
    "weapon",
}

SUPPORTING_SENTENCES = [
    "Everyone paused, listened carefully, and shared one bright idea",
    "The smallest helper noticed a cheerful clue hiding nearby",
    "Working together made the tricky moment feel much easier",
    "Soon their brave little hearts felt lighter and their smiles grew wider",
    "A kind word gave every new friend the courage to keep trying",
]


# =============================================================================
# Function Part: User Interface Helpers
# =============================================================================

def read_file_as_base64(file_path: Path) -> str:
    """Return a local image as Base64 text for dynamic Streamlit styling."""
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def apply_page_design(theme_name: str) -> None:
    """Apply the selected theme's background, colors, and accessible layout."""
    selected_theme = THEME_SETTINGS[theme_name]
    background_path = BACKGROUND_DIRECTORY / selected_theme["background"]
    background_base64 = read_file_as_base64(background_path)
    accent_color = selected_theme["accent"]
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(246, 250, 255, 0.74), rgba(255, 247, 253, 0.82)),
                url("data:image/png;base64,{background_base64}");
            background-position: center;
            background-size: cover;
            background-attachment: fixed;
        }}
        .block-container {{
            max-width: 1280px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }}
        h1, h2, h3 {{ color: #30234d; }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: rgba(255, 255, 255, 0.88);
            border: 2px solid rgba(255, 255, 255, 0.95);
            border-radius: 24px;
            box-shadow: 0 12px 35px rgba(48, 35, 77, 0.12);
            backdrop-filter: blur(8px);
        }}
        div[data-testid="stFileUploaderDropzone"] {{
            background: rgba(255, 255, 255, 0.92);
            border: 3px dashed {accent_color};
            border-radius: 22px;
        }}
        .taletwinkle-title {{
            text-align: center;
            font-size: clamp(2.3rem, 5vw, 4.2rem);
            font-weight: 900;
            color: #30234d;
            text-shadow: 0 3px 0 white;
            margin-bottom: 0;
        }}
        .taletwinkle-subtitle {{
            text-align: center;
            color: #51436f;
            font-size: 1.08rem;
            font-weight: 650;
            margin-bottom: 1.2rem;
        }}
        .story-card {{
            background: rgba(255, 255, 255, 0.94);
            border-left: 8px solid {accent_color};
            border-radius: 20px;
            color: #2f2841;
            font-size: 1.16rem;
            line-height: 1.72;
            padding: 1.2rem 1.35rem;
            box-shadow: 0 10px 28px rgba(48, 35, 77, 0.12);
        }}
        .tiny-note {{ color: #5d5371; font-size: 0.88rem; }}
        .stButton button, div[data-baseweb="select"] > div {{ border-radius: 14px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_story_card(story_text: str, heading: str = "Your Story") -> None:
    """Display escaped story text in a high-contrast child-friendly card."""
    safe_story_text = html.escape(story_text).replace("\n", "<br>")
    st.subheader(f"✨ {heading}")
    st.markdown(
        f'<div class="story-card">{safe_story_text}</div>',
        unsafe_allow_html=True,
    )


def display_initial_preview(theme_name: str) -> None:
    """Show an image, story placeholder, and audio control before upload."""
    selected_theme = THEME_SETTINGS[theme_name]
    preview_image_path = BACKGROUND_DIRECTORY / selected_theme["background"]
    preview_music_path = MUSIC_DIRECTORY / selected_theme["music"]
    st.image(preview_image_path, caption="Your picture will appear here", use_container_width=True)
    display_story_card(
        "Drop in a picture and TaleTwinkle will create a brand-new story automatically.",
        heading="Your Story Will Sparkle Here",
    )
    st.caption("🎧 Theme music preview")
    st.audio(preview_music_path.read_bytes(), format="audio/wav", autoplay=True, loop=True)


# =============================================================================
# Function Part: Image Input and Caption Processing
# =============================================================================

def prepare_uploaded_image(image_bytes: bytes) -> Image.Image:
    """Validate orientation, color mode, and size for reliable model inference.

    Args:
        image_bytes: Original bytes supplied by the Streamlit upload control.

    Returns:
        A correctly oriented RGB image no larger than 1024 pixels per edge.
    """
    uploaded_image = Image.open(io.BytesIO(image_bytes))
    uploaded_image = ImageOps.exif_transpose(uploaded_image).convert("RGB")
    uploaded_image.thumbnail((MAXIMUM_IMAGE_EDGE, MAXIMUM_IMAGE_EDGE), Image.Resampling.LANCZOS)
    return uploaded_image


def remove_consecutive_word_repetition(text_value: str) -> str:
    """Collapse caption-model loops such as 'jungle jungle jungle'."""
    text_tokens = text_value.split()
    cleaned_tokens = []
    previous_normalized_token = ""
    for text_token in text_tokens:
        normalized_token = re.sub(r"[^a-z0-9]", "", text_token.lower())
        if normalized_token and normalized_token == previous_normalized_token:
            continue
        cleaned_tokens.append(text_token)
        previous_normalized_token = normalized_token
    return " ".join(cleaned_tokens)


@st.cache_resource(show_spinner=False)
def load_image_caption_pipeline():
    """Load and reuse the benchmark-selected BLIP image-caption pipeline."""
    return pipeline(
        "image-to-text",
        model=CAPTION_MODEL_NAME,
        device=0 if torch.cuda.is_available() else -1,
    )


@st.cache_data(show_spinner=False, max_entries=40)
def generate_image_caption(image_bytes: bytes) -> str:
    """Generate and clean a concise factual description of the uploaded image."""
    uploaded_image = prepare_uploaded_image(image_bytes)
    caption_pipeline = load_image_caption_pipeline()
    caption_output = caption_pipeline(uploaded_image, max_new_tokens=40)
    raw_caption = str(caption_output[0]["generated_text"]).strip()
    cleaned_caption = remove_consecutive_word_repetition(
        re.sub(r"\s+", " ", raw_caption).strip(" .")
    )
    return cleaned_caption or "a bright and interesting scene"


# =============================================================================
# Function Part: Story Generation and Safety
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_story_generation_pipeline():
    """Load the compact children's-story model selected by the benchmark."""
    return pipeline(
        "text-generation",
        model=STORY_MODEL_NAME,
        device=0 if torch.cuda.is_available() else -1,
    )


def count_story_words(story_text: str) -> int:
    """Count readable words while allowing apostrophes and hyphens."""
    return len(re.findall(r"\b[\w'’-]+\b", story_text, flags=re.UNICODE))


def build_story_opening(image_caption: str, theme_name: str) -> str:
    """Create an image-grounded opening that steers the small story model."""
    selected_theme = THEME_SETTINGS[theme_name]
    cleaned_caption = image_caption.strip().rstrip(".!?")
    return (
        f"Once upon a time, the picture came alive: {cleaned_caption}. "
        f"{selected_theme['hook']}. Everyone chose to help"
    )


def remove_repeated_sentences(story_text: str) -> str:
    """Remove exact or near-exact repeated sentences from model output."""
    sentence_parts = re.split(r"(?<=[.!?])\s+", story_text)
    unique_sentences = []
    normalized_sentences = set()
    for sentence in sentence_parts:
        cleaned_sentence = re.sub(r"\s+", " ", sentence).strip()
        normalized_sentence = re.sub(r"[^a-z0-9 ]", "", cleaned_sentence.lower())
        if cleaned_sentence and normalized_sentence not in normalized_sentences:
            unique_sentences.append(cleaned_sentence)
            normalized_sentences.add(normalized_sentence)
    return " ".join(unique_sentences)


def contains_unsuitable_language(story_text: str) -> bool:
    """Return True when generated text contains a blocked child-safety term."""
    story_words = set(re.findall(r"[a-z']+", story_text.lower()))
    return bool(story_words & UNSUITABLE_STORY_WORDS)


def trim_text_to_word_limit(story_text: str, maximum_words: int) -> str:
    """Trim text to an exact maximum word count without splitting a word."""
    word_matches = list(re.finditer(r"\b[\w'’-]+\b", story_text, flags=re.UNICODE))
    if len(word_matches) <= maximum_words:
        return story_text.strip()
    final_word_end = word_matches[maximum_words - 1].end()
    return story_text[:final_word_end].rstrip(" ,;:-.!?") + "."


def build_exact_padding_sentences(required_word_count: int) -> str:
    """Build grammatical positive sentences containing an exact word count."""
    exact_sentences = {
        1: "Hooray.",
        2: "They smiled.",
        3: "Everyone felt proud.",
        4: "Their kind teamwork shone.",
        5: "The friends cheered with delight.",
        6: "Their happy plan worked at last.",
        7: "Every friendly helper felt brave and proud.",
        8: "Gentle laughter filled the air around them all.",
        9: "Together, the cheerful friends made one more wonderful memory.",
        10: "Everyone clapped because kindness had guided their brave little team.",
    }
    padding_sentences = []
    while required_word_count > 10:
        padding_sentences.append(exact_sentences[10])
        required_word_count -= 10
    if required_word_count > 0:
        padding_sentences.append(exact_sentences[required_word_count])
    return " ".join(padding_sentences)


def fit_story_to_requested_length(
    draft_story: str,
    closing_sentence: str,
    requested_word_count: int,
) -> str:
    """Fit a safe draft to the slider target while preserving a happy ending."""
    requested_word_count = max(
        MINIMUM_STORY_WORDS,
        min(MAXIMUM_STORY_WORDS, requested_word_count),
    )
    cleaned_draft = remove_repeated_sentences(re.sub(r"\s+", " ", draft_story).strip())
    closing_sentence = closing_sentence.strip().rstrip(".!?") + "."
    closing_word_count = count_story_words(closing_sentence)
    body_word_target = max(1, requested_word_count - closing_word_count)

    dangling_final_words = {
        "a",
        "an",
        "and",
        "at",
        "because",
        "but",
        "for",
        "from",
        "of",
        "or",
        "the",
        "to",
        "with",
    }
    candidate_sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+", cleaned_draft):
        cleaned_sentence = sentence.strip().rstrip(".!?")
        sentence_words = re.findall(r"[A-Za-z']+", cleaned_sentence.lower())
        is_internal_ending = cleaned_sentence.lower() in {"the end", "end"}
        if (
            cleaned_sentence
            and sentence_words
            and sentence_words[-1] not in dangling_final_words
            and not is_internal_ending
        ):
            candidate_sentences.append(f"{cleaned_sentence}.")
    candidate_sentences.extend(f"{sentence.rstrip('.')} ." for sentence in SUPPORTING_SENTENCES)

    selected_sentences = []
    selected_word_count = 0
    for candidate_sentence in candidate_sentences:
        candidate_word_count = count_story_words(candidate_sentence)
        if selected_word_count + candidate_word_count <= body_word_target:
            selected_sentences.append(candidate_sentence)
            selected_word_count += candidate_word_count

    remaining_word_count = body_word_target - selected_word_count
    if remaining_word_count > 0:
        selected_sentences.append(build_exact_padding_sentences(remaining_word_count))
    fitted_body = " ".join(selected_sentences).strip()
    final_story = f"{fitted_body} {closing_sentence}"
    return re.sub(r"\s+([.!?,])", r"\1", re.sub(r"\s+", " ", final_story)).strip()


@st.cache_data(show_spinner=False, max_entries=60)
def generate_children_story(
    image_caption: str,
    theme_name: str,
    requested_word_count: int,
    random_seed: int,
) -> str:
    """Generate a themed, image-relevant, safe story at the selected length."""
    story_pipeline = load_story_generation_pipeline()
    story_opening = build_story_opening(image_caption, theme_name)
    set_seed(random_seed)
    model_output = story_pipeline(
        story_opening,
        max_new_tokens=max(60, requested_word_count),
        do_sample=True,
        temperature=0.78,
        top_p=0.92,
        repetition_penalty=1.16,
        no_repeat_ngram_size=3,
        num_return_sequences=1,
        return_full_text=False,
        pad_token_id=50256,
    )
    generated_continuation = str(model_output[0]["generated_text"]).strip()
    generated_continuation = remove_repeated_sentences(generated_continuation)

    # A deterministic safe body is used if the open model generates a blocked term.
    if contains_unsuitable_language(generated_continuation):
        generated_continuation = " ".join(SUPPORTING_SENTENCES) + "."

    selected_theme = THEME_SETTINGS[theme_name]
    complete_draft = f"{story_opening} {generated_continuation}"
    fitted_story = fit_story_to_requested_length(
        complete_draft,
        selected_theme["closing"],
        requested_word_count,
    )
    return fitted_story


# =============================================================================
# Function Part: Text-to-Speech and Theme Audio
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_text_to_speech_pipeline():
    """Load and reuse the privacy-preserving local Hugging Face TTS model."""
    return pipeline(
        "text-to-speech",
        model=TTS_MODEL_NAME,
        device=0 if torch.cuda.is_available() else -1,
    )


@st.cache_data(show_spinner=False, max_entries=40)
def synthesize_huggingface_waveform(story_text: str) -> tuple[np.ndarray, int]:
    """Synthesize narration locally so story text is not sent to a TTS service."""
    speech_pipeline = load_text_to_speech_pipeline()
    speech_output = speech_pipeline(story_text)
    speech_waveform = np.asarray(speech_output["audio"], dtype=np.float32).squeeze()
    sample_rate = int(speech_output["sampling_rate"])
    if speech_waveform.ndim > 1:
        speech_waveform = np.mean(speech_waveform, axis=1)
    return speech_waveform, sample_rate


def apply_voice_persona(
    speech_waveform: np.ndarray,
    sample_rate: int,
    pitch_steps: float,
    requested_speed: float,
    persona_pace: float,
) -> np.ndarray:
    """Apply the selected playful pitch persona and user-controlled narration pace."""
    if abs(pitch_steps) < 0.01:
        pitched_waveform = speech_waveform
    else:
        pitched_waveform = librosa.effects.pitch_shift(
            y=speech_waveform,
            sr=sample_rate,
            n_steps=pitch_steps,
        )
    effective_speed = max(0.50, min(2.00, requested_speed * persona_pace))
    if abs(effective_speed - 1.00) < 0.01:
        adjusted_waveform = pitched_waveform
    else:
        adjusted_waveform = librosa.effects.time_stretch(
            y=pitched_waveform,
            rate=effective_speed,
        )
    return np.asarray(adjusted_waveform, dtype=np.float32)


def mix_narration_with_theme_music(
    narration_waveform: np.ndarray,
    sample_rate: int,
    music_path: Path,
    music_tail_seconds: float = 5.0,
) -> np.ndarray:
    """Duck music under narration, then resume it for a short ending tail."""
    music_waveform, music_sample_rate = sf.read(music_path, dtype="float32")
    if music_waveform.ndim > 1:
        music_waveform = np.mean(music_waveform, axis=1)
    if int(music_sample_rate) != sample_rate:
        music_waveform = librosa.resample(
            music_waveform,
            orig_sr=int(music_sample_rate),
            target_sr=sample_rate,
        )

    tail_sample_count = int(sample_rate * music_tail_seconds)
    total_sample_count = len(narration_waveform) + tail_sample_count
    repetitions_required = int(np.ceil(total_sample_count / max(len(music_waveform), 1)))
    looped_music = np.tile(music_waveform, repetitions_required)[:total_sample_count]

    mixed_waveform = looped_music * 0.15
    mixed_waveform[: len(narration_waveform)] *= 0.48
    mixed_waveform[: len(narration_waveform)] += narration_waveform * 0.94

    peak_amplitude = float(np.max(np.abs(mixed_waveform))) if mixed_waveform.size else 0.0
    if peak_amplitude > 0.98:
        mixed_waveform = mixed_waveform * (0.98 / peak_amplitude)
    return np.asarray(mixed_waveform, dtype=np.float32)


def encode_waveform_as_wav(audio_waveform: np.ndarray, sample_rate: int) -> bytes:
    """Encode a floating-point waveform as browser-compatible WAV bytes."""
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_waveform, sample_rate, format="WAV", subtype="PCM_16")
    return wav_buffer.getvalue()


@st.cache_data(show_spinner=False, max_entries=40)
def create_narrated_story_audio(
    story_text: str,
    voice_name: str,
    narration_speed: float,
    theme_name: str,
) -> bytes:
    """Create the chosen voice persona and mix it with theme music."""
    selected_voice = VOICE_PERSONAS[voice_name]
    selected_theme = THEME_SETTINGS[theme_name]
    narration_waveform, sample_rate = synthesize_huggingface_waveform(story_text)
    persona_waveform = apply_voice_persona(
        narration_waveform,
        sample_rate,
        float(selected_voice["pitch"]),
        narration_speed,
        float(selected_voice["pace"]),
    )
    music_path = MUSIC_DIRECTORY / selected_theme["music"]
    mixed_waveform = mix_narration_with_theme_music(
        persona_waveform,
        sample_rate,
        music_path,
    )
    return encode_waveform_as_wav(mixed_waveform, sample_rate)


# =============================================================================
# Main Part: Input, Process, and Output Flow
# =============================================================================

def main() -> None:
    """Run TaleTwinkle's complete Streamlit input-process-output workflow."""
    st.set_page_config(
        page_title="TaleTwinkle — Picture to Story",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # -------------------------------------------------------------------------
    # Input Part
    # -------------------------------------------------------------------------
    st.markdown('<p class="taletwinkle-title">✨ TaleTwinkle ✨</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="taletwinkle-subtitle">Drop in a picture. Hear a little world come alive.</p>',
        unsafe_allow_html=True,
    )

    left_controls, center_content, right_controls = st.columns([0.90, 1.85, 1.05], gap="large")
    with left_controls:
        with st.container(border=True):
            st.subheader("🎨 Pick a World")
            selected_theme_name = st.selectbox(
                "Story theme",
                options=list(THEME_SETTINGS.keys()),
                label_visibility="collapsed",
            )
            st.markdown("🛡️ **Kind-story shield:** on")
            st.caption("Stories are checked for frightening words before narration.")

    with right_controls:
        with st.container(border=True):
            st.subheader("🎛️ Story Controls")
            selected_story_length = st.slider(
                "Story length (words)",
                min_value=MINIMUM_STORY_WORDS,
                max_value=MAXIMUM_STORY_WORDS,
                value=DEFAULT_STORY_WORDS,
                step=5,
            )
            selected_voice_speed = st.slider(
                "Voice speed",
                min_value=0.50,
                max_value=2.00,
                value=1.00,
                step=0.05,
                format="%.2fx",
            )
            selected_voice_name = st.selectbox(
                "Choose a voice friend",
                options=list(VOICE_PERSONAS.keys()),
                index=0,
            )

    apply_page_design(selected_theme_name)

    with center_content:
        uploaded_file = st.file_uploader(
            "Drop a picture here",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload one JPG, PNG, or WebP picture. A new upload replaces the old story.",
        )

    # -------------------------------------------------------------------------
    # Process Part
    # -------------------------------------------------------------------------
    if uploaded_file is None:
        with center_content:
            display_initial_preview(selected_theme_name)
        return

    uploaded_image_bytes = uploaded_file.getvalue()
    prepared_image = prepare_uploaded_image(uploaded_image_bytes)
    selected_theme = THEME_SETTINGS[selected_theme_name]
    waiting_music_path = MUSIC_DIRECTORY / selected_theme["music"]

    with center_content:
        st.image(prepared_image, caption="Your story picture", use_container_width=True)
        waiting_music_placeholder = st.empty()
        waiting_music_placeholder.audio(
            waiting_music_path.read_bytes(),
            format="audio/wav",
            autoplay=True,
            loop=True,
        )

        processing_start_time = time.perf_counter()
        with st.status("✨ Sprinkling story magic...", expanded=True) as generation_status:
            st.write("👀 Looking closely at the picture...")
            generated_caption = generate_image_caption(uploaded_image_bytes)

            st.write("✍️ Weaving a safe, themed story...")
            deterministic_seed = int(hashlib.sha256(uploaded_image_bytes).hexdigest()[:8], 16)
            generated_story = generate_children_story(
                generated_caption,
                selected_theme_name,
                selected_story_length,
                deterministic_seed,
            )

            st.write("🎙️ Inviting your voice friend to narrate...")
            narrated_audio = create_narrated_story_audio(
                generated_story,
                selected_voice_name,
                selected_voice_speed,
                selected_theme_name,
            )
            generation_status.update(
                label="🌟 Your TaleTwinkle story is ready!",
                state="complete",
                expanded=False,
            )

        total_processing_seconds = time.perf_counter() - processing_start_time
        waiting_music_placeholder.empty()

        # ---------------------------------------------------------------------
        # Output Part
        # ---------------------------------------------------------------------
        display_story_card(generated_story, heading=selected_theme["key"])
        st.caption("🔊 Narration starts automatically. Use the player to pause or replay.")
        st.audio(narrated_audio, format="audio/wav", autoplay=True)

        result_metric_columns = st.columns(2)
        result_metric_columns[0].metric("Story words", f"{count_story_words(generated_story)}")
        result_metric_columns[1].metric("Processing time", f"{total_processing_seconds:.2f} sec")

        with st.expander("🧑‍🏫 Grown-up and assessor details"):
            st.write(f"**Image caption:** {generated_caption}")
            st.write(f"**Caption model:** `{CAPTION_MODEL_NAME}`")
            st.write(f"**Story model:** `{STORY_MODEL_NAME}`")
            st.write(f"**Text-to-speech model:** `{TTS_MODEL_NAME}`")
            st.write(
                "Voice labels are playful region-inspired sound personas created from a local "
                "Hugging Face voice plus pitch and pace effects. They are not literal demographic "
                "speakers or guaranteed regional accents."
            )
            st.write(f"**Measured end-to-end processing:** {total_processing_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
