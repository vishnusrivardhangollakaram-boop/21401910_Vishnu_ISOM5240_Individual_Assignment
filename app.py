"""
TaleTwinkle - picture-to-story app for children aged 3 to 10
===========================================================
ISOM5240 Deep Learning Business Applications with Python - Individual Assignment
Storytelling application using Hugging Face Transformers pipelines, deployed on Streamlit Cloud.

User flow (no button needed):
    1. The child drops or chooses a picture (centre of the screen). It appears straight away.
    2. Stage 1  IMAGE -> TEXT  : a Hugging Face image-captioning pipeline describes the picture.
    3. Stage 2  TEXT  -> STORY : a Hugging Face instruction model writes a 50-100 word story in the chosen
                                 theme (left) and length (right). The words appear on screen as they are written.
    4. Stage 3  STORY -> VOICE : each sentence is sent to the storyteller voice IN THE BACKGROUND as soon as it
                                 is finished, so the narration is ready moments after the story, and it plays by itself.

Speed/memory design: BLIP loads first and is released after captioning; Qwen and only the required Piper voice
then load together and are released after each run. Models use compact precision where supported, story words
stream live, generation stops near the requested length, and speech is produced sentence by sentence while Qwen writes.

Business requirement : an engaging, safe, fast storytelling experience that keeps young children interested.
User requirements    : users are 3-10 years old, so the screen is simple (one drop area, no required buttons),
                       words are simple, content is kid-safe, picture/story/audio are always visible,
                       and the response is as fast as possible (short attention span).

Model selection (10 candidates per stage, 10 test images, see README.md and benchmarks/):
    Stage 1 image captioning ("image-to-text"): Salesforce/blip-image-captioning-base (chosen: best accuracy-for-speed),
        Salesforce/blip-image-captioning-large, microsoft/git-base-coco, microsoft/git-large-coco, microsoft/git-base,
        microsoft/git-base-textcaps, microsoft/git-large-textcaps, microsoft/git-large,
        nlpconnect/vit-gpt2-image-captioning, ydshieh/vit-gpt2-coco-en
    Stage 2 story generation ("text-generation"): Qwen/Qwen3-0.6B (chosen), Qwen/Qwen2.5-0.5B-Instruct,
        Qwen/Qwen2.5-1.5B-Instruct,
        HuggingFaceTB/SmolLM2-135M-Instruct, HuggingFaceTB/SmolLM2-360M-Instruct, HuggingFaceTB/SmolLM2-1.7B-Instruct,
        TinyLlama/TinyLlama-1.1B-Chat-v1.0, google/flan-t5-base, google/flan-t5-small,
        roneneldan/TinyStories-Instruct-33M
    Stage 3 text-to-speech: gTTS UK/US/Australian/Indian English (chosen: familiar female voices, fast),
        rhasspy/piper-voices (chosen: sub-second local male and child-style narration), microsoft/speecht5_tts,
        suno/bark-small, hexgrad/Kokoro-82M, kakao-enterprise/vits-ljs, kakao-enterprise/vits-vctk,
        espnet/kan-bayashi_ljspeech_vits, facebook/fastspeech2-en-ljspeech, pyttsx3

Code structure (course conventions):
    IMPORT PART   : libraries, then settings/constants (model names, themes, voices) in one place.
    FUNCTION PART : small single-purpose functions (no classes), each with a docstring, grouped as
                    1 page style, 2 models, 3 picture, 4 Stage 1, 5 Stage 2, 6 Stage 3, 7 full run,
                    8 screen, 9 input/process/output helpers used by main().
    MAIN PART     : main() runs the INPUT -> PROCESS -> OUTPUT flow; started by `if __name__ == "__main__"`.
    All function calls pass values as parameter=argument; times are shown with 2 decimals (f"{value:.2f}").

Acknowledgement: generative AI (Claude and ChatGPT) was used as a coding assistant, as permitted by the course syllabus.
"""

# ==============================
# IMPORT PART
# ==============================
import base64
import ctypes
import gc
import hashlib
import html
import io
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from queue import Empty

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image, ImageOps
from scipy.io import wavfile
from scipy.signal import correlate, resample_poly
from transformers import StoppingCriteriaList, TextIteratorStreamer, pipeline
# Optional libraries are imported inside the only function that needs them, so the app still starts and the
# other voices keep working even if one of them is unavailable: gtts (Google voices), soundfile (MP3 decoding),
# huggingface_hub + piper (local Piper voices).

# ---- Settings and constants (change model names here to try other benchmark candidates) ----
APP_NAME = "TaleTwinkle"
APP_TAGLINE = "Drop in a picture. Hear a little world come alive."

IMAGE_CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
STORY_GENERATION_MODEL_NAME = "Qwen/Qwen3-0.6B"
LOCAL_SPEECH_MODEL_NAME = "rhasspy/piper-voices"
DEFAULT_PIPER_MODEL_FILE = "en/en_GB/alba/medium/en_GB-alba-medium.onnx"
SHRINK_MODELS_WITH_INT8 = True              # int8 weights: ~3x smaller and faster on Streamlit Cloud's CPU
KEEP_MODELS_LOADED = True                   # True: load BLIP + Qwen once at start-up and reuse them for every story
MEMORY_SAFETY_LIMIT_MB = 2300               # if the app uses more memory than this, models are released after use
DEFAULT_CPU_THREADS = 2                     # Streamlit Community Cloud gives each app about 2 CPU cores
LOGGER = logging.getLogger(APP_NAME)
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

MINIMUM_STORY_WORDS = 50
MAXIMUM_STORY_WORDS = 100
DEFAULT_STORY_WORDS = 75
STORY_WORD_STEP = 5
STORY_WORD_MARGIN = 10                      # the story may run up to 10 words over the slider value (max 100)
STORY_TOKENS_PER_WORD = 1.45                # token budget per word, so generation stops near the wanted length
STORY_EXTRA_TOKENS = 8
STORY_TEMPERATURE = 0.8                     # sampling settings: varied but still sensible stories
STORY_TOP_P = 0.9
STORY_REPETITION_PENALTY = 1.1
STORY_STREAM_TIMEOUT_SECONDS = 90
CAPTION_MAX_NEW_TOKENS = 30
CAPTION_REPETITION_PENALTY = 1.3
CAPTION_NO_REPEAT_NGRAM_SIZE = 3
MAXIMUM_SPEECH_CHUNK_CHARACTERS = 180       # long sentences are split so each speech request stays short
SENTENCE_PAUSE_SECONDS = 0.15
SPEECH_RESULT_TIMEOUT_SECONDS = 60
VOICE_SPEED_CHOICES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
DEFAULT_VOICE_SPEED = 1.0

PREFERRED_IMAGE_DISPLAY_SIZE = 460          # pixels on the longest side: small pictures are scaled up, big ones down
MODEL_INPUT_IMAGE_SIZE = 512                # pictures are shrunk to this before captioning (faster, same accuracy)
ACCEPTED_PICTURE_TYPES = ["jpg", "jpeg", "png", "webp"]
BACKGROUND_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "backgrounds")
NARRATION_PEAK_LEVEL = 0.95
GOOGLE_VOICE_PARALLEL_REQUESTS = 6

UNSAFE_STORY_WORDS = ["kill", "killed", "blood", "bloody", "dead", "die", "died", "death", "gun", "knife", "shoot",
                      "weapon", "murder", "hate", "stupid", "sexy", "drunk", "beer", "wine", "cigarette", "horror",
                      "nightmare", "bomb", "war", "burned", "scream", "screamed"]

STORY_THEMES = {
    "fairy_tale": {
        "label": "🏰 Fairy Tale", "selector_label": "🏰✨  Fairy Tale Land",
        "world_name": "Fairy Tale Land", "mascot": "🦄", "background_file": "fairy_tale.png",
        "story_instruction": "Make it a magical fairy tale with kind fairies, castles, wishes or talking animals.",
        "floating_emojis": ["✨", "🦄", "🌸", "👑", "🧚", "⭐"], "accent_colour": "#8d4de8",
    },
    "space_quest": {
        "label": "🚀 Space Quest", "selector_label": "🚀👽  Space Quest",
        "world_name": "Outer Space", "mascot": "👽", "background_file": "space_quest.png",
        "story_instruction": "Make it a friendly space adventure with rockets, twinkly stars, planets or cute friendly aliens.",
        "floating_emojis": ["⭐", "🪐", "🚀", "👽", "🛸", "🌟"], "accent_colour": "#3f75e8",
    },
    "gentle_mystery": {
        "label": "🔎 Gentle Mystery", "selector_label": "🔎🦉  Gentle Mystery",
        "world_name": "Clue Library", "mascot": "🦉", "background_file": "gentle_mystery.png",
        "story_instruction": "Make it a gentle, cosy mystery where the friends find clues and solve a small puzzle. Nothing scary.",
        "floating_emojis": ["🔍", "🗝️", "🦉", "❓", "📚", "🌙"], "accent_colour": "#8a5a3b",
    },
    "jungle_adventure": {
        "label": "🦜 Jungle Adventure", "selector_label": "🦁🦜  Jungle Adventure",
        "world_name": "Jungle Island", "mascot": "🦁", "background_file": "jungle_adventure.png",
        "story_instruction": "Make it an exciting jungle adventure with exploring, brave friends, animals and a treasure map.",
        "floating_emojis": ["🗺️", "🧭", "🌴", "🦜", "🦋", "💎"], "accent_colour": "#2f8b57",
    },
    "silly_poem": {
        "label": "🎵 Silly Poem", "selector_label": "🎵🐝  Silly Poem",
        "world_name": "Rhyme Garden", "mascot": "🐝", "background_file": "silly_poem.png",
        "story_instruction": "Write it as a short, silly rhyming poem with short lines that rhyme and a giggly surprise.",
        "floating_emojis": ["🎵", "🌈", "🐝", "🌻", "🎈", "🎶"], "accent_colour": "#e05b9d",
    },
    "ocean_magic": {
        "label": "🐠 Ocean Magic", "selector_label": "🐠🐙  Ocean Magic",
        "world_name": "Coral Reef", "mascot": "🐙", "background_file": "ocean_magic.png",
        "story_instruction": "Make it a magical underwater adventure with friendly fish, shiny shells and sparkly coral.",
        "floating_emojis": ["🫧", "🐠", "🐚", "🐬", "🪸", "⭐"], "accent_colour": "#159bb5",
    },
}
DEFAULT_THEME_KEY = "fairy_tale"
POEM_THEME_KEY = "silly_poem"

# Storyteller voices. Google supplies familiar regional voices. Piper supplies fast local male/female
# checkpoints hosted on Hugging Face. Kid choices are clearly labelled child-style pitch effects.
VOICE_PERSONAS = {
    "lily_british": {
        "label": "👩‍🏫 Story Lady Lily (British)", "avatar": "👩‍🏫", "about": "A warm British storyteller",
        "engine": "google", "accent_domain": "co.uk", "semitone_shift": 0.0, "tempo_factor": 1.0, "robot_effect": False,
    },
    "polly_parrot": {
        "label": "🦜 Polly the Parrot", "avatar": "🦜", "about": "A chirpy cartoon parrot",
        "engine": "google", "accent_domain": "us", "semitone_shift": 6.0, "tempo_factor": 1.08, "robot_effect": False,
    },
    "robo_beep": {
        "label": "🤖 Robo Beep", "avatar": "🤖", "about": "A friendly cartoon robot",
        "engine": "google", "accent_domain": "us", "semitone_shift": 1.0, "tempo_factor": 1.0, "robot_effect": True,
    },
    "giggle_grace": {
        "label": "👧 Giggle Grace (Kid-Style Girl)", "avatar": "👧", "about": "A lively peer-age girl-style voice",
        "engine": "piper", "accent_domain": "", "piper_model_file": "en/en_US/amy/medium/en_US-amy-medium.onnx",
        "semitone_shift": 3.0, "tempo_factor": 1.05, "robot_effect": False,
    },
    "captain_finn": {
        "label": "🧭 Captain Finn (American Male)", "avatar": "🧭", "about": "A bold, friendly adventure narrator",
        "engine": "piper", "accent_domain": "", "piper_model_file": "en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "semitone_shift": 0.0, "tempo_factor": 1.0, "robot_effect": False,
    },
    "chloe_australian": {
        "label": "👩 Aunty Chloe (Australian)", "avatar": "👩", "about": "A sunny Australian lady",
        "engine": "google", "accent_domain": "com.au", "semitone_shift": 0.0, "tempo_factor": 1.0, "robot_effect": False,
    },
    "sir_alan": {
        "label": "🎩 Sir Alan (British Male)", "avatar": "🎩", "about": "A gentle British bedtime narrator",
        "engine": "piper", "accent_domain": "", "piper_model_file": "en/en_GB/alan/medium/en_GB-alan-medium.onnx",
        "semitone_shift": 0.0, "tempo_factor": 0.96, "robot_effect": False,
    },
    "whiskers_kitten": {
        "label": "🐱 Whiskers the Kitten", "avatar": "🐱", "about": "A tiny, playful cartoon kitten",
        "engine": "google", "accent_domain": "com.au", "semitone_shift": 4.5, "tempo_factor": 1.05, "robot_effect": False,
    },
    "amy_american": {
        "label": "👩 Aunt Amy (American)", "avatar": "👩", "about": "A friendly American lady",
        "engine": "google", "accent_domain": "us", "semitone_shift": 0.0, "tempo_factor": 1.0, "robot_effect": False,
    },
    "priya_indian": {
        "label": "👩‍🏫 Teacher Priya (Indian English)", "avatar": "👩‍🏫",
        "about": "A kind teacher with an Indian English accent",
        "engine": "google", "accent_domain": "co.in", "semitone_shift": 0.0, "tempo_factor": 1.0, "robot_effect": False,
    },
}
DEFAULT_VOICE_KEY = "lily_british"


# ==============================
# FUNCTION PART
# ==============================

# ---------- 1. Page set-up and styling functions ----------
def configure_page():
    """
    Set the browser tab title, icon and wide layout. Must be the first Streamlit call.
    """
    st.set_page_config(page_title=f"{APP_NAME} Picture Stories for Kids", page_icon="🎈",
                       layout="wide", initial_sidebar_state="collapsed")


def initialise_session_state():
    """
    Create the session-state entries that remember the current story and narration between
    Streamlit reruns (so changing the voice does not re-write the story).
    """
    default_state_values = {
        "story_request_key": None, "story_result": None,
        "narration_request_key": None, "narration_result": None,
        "story_version": 0, "celebrate_new_story": False,
        "scroll_to_new_audio": False, "scroll_request_count": 0,
    }
    for state_name, default_value in default_state_values.items():
        if state_name not in st.session_state:
            st.session_state[state_name] = default_value


def find_background_file(background_file_name):
    """
    Find a theme background picture. Looks in assets/backgrounds first, then in a few other likely
    folders, and accepts .jpg, .jpeg or .png, so a slightly different upload layout does not break the app.

    Parameters:
        background_file_name (str): expected file name, e.g. "fairy_tale.png".
    Returns:
        str or None: full path of the picture, or None if it cannot be found.
    """
    app_folder = os.path.dirname(os.path.abspath(__file__))
    file_stem = os.path.splitext(background_file_name)[0]
    search_folders = [BACKGROUND_FOLDER, os.path.join(app_folder, "assets"), os.path.join(app_folder, "backgrounds"),
                      app_folder, os.path.join(app_folder, "test_images")]
    for search_folder in search_folders:
        for file_extension in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            candidate_path = os.path.join(search_folder, f"{file_stem}{file_extension}")
            if os.path.isfile(candidate_path):
                return candidate_path
    return None


@st.cache_data(show_spinner=False, max_entries=6)
def read_background_as_base64(background_file_name):
    """
    Read a theme background picture and return it as a CSS data URL.

    Parameters:
        background_file_name (str): file inside assets/backgrounds.
    Returns:
        str or None: "data:image/...;base64,..." text, or None if the picture is missing
                     (the page then uses a plain colour gradient instead of crashing).
    """
    background_path = find_background_file(background_file_name=background_file_name)
    if background_path is None:
        return None
    image_type = "png" if background_path.lower().endswith(".png") else "jpeg"
    with open(background_path, "rb") as background_file:
        return f"data:image/{image_type};base64,{base64.b64encode(background_file.read()).decode('utf-8')}"


def build_floating_emoji_html(floating_emojis):
    """
    Build the HTML for slowly floating theme emojis in the background (pure CSS animation).

    Parameters:
        floating_emojis (list[str]): emojis that match the theme.
    Returns:
        str: HTML snippet.
    """
    emoji_spans = []
    for emoji_index, emoji_symbol in enumerate(floating_emojis * 2):
        left_percent = (emoji_index * 37 + 5) % 96
        delay_seconds = (emoji_index * 1.9) % 14
        duration_seconds = 18 + (emoji_index * 3) % 11
        font_size_rem = 1.4 + (emoji_index % 3) * 0.6
        emoji_spans.append(
            f'<span class="floating-emoji" style="left:{left_percent}%; font-size:{font_size_rem:.2f}rem; '
            f'animation-delay:-{delay_seconds:.2f}s; animation-duration:{duration_seconds}s;">{emoji_symbol}</span>')
    return f'<div class="floating-layer">{"".join(emoji_spans)}</div>'


def apply_page_style(theme_settings):
    """
    Inject the kid-friendly CSS: illustrated theme background, floating emojis, rounded cards,
    big friendly fonts and theme-coloured controls.

    Parameters:
        theme_settings (dict): one entry of STORY_THEMES.
    """
    accent_colour = theme_settings["accent_colour"]
    background_data_url = read_background_as_base64(background_file_name=theme_settings["background_file"])
    background_css = (f'url("{background_data_url}")' if background_data_url
                      else f"linear-gradient(160deg, #fff6fb 0%, {accent_colour}33 55%, #e8f4ff 100%)")
    page_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&display=swap');
    .stApp {{ background-image: linear-gradient(rgba(255,255,255,0.30), rgba(255,255,255,0.45)),
        {background_css};
        background-size: cover; background-position: center; background-attachment: fixed; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .block-container, [data-testid="stMainBlockContainer"] {{ position: relative; z-index: 1; max-width: 1420px;
        padding-top: 0.45rem; padding-bottom: 1.5rem; }}
    h1, h2, h3, h4, p, label, li, .story-card, .taletwinkle-title, .taletwinkle-tagline {{ font-family: 'Fredoka', 'Comic Sans MS', sans-serif !important; }}
    .floating-layer {{ position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden; }}
    .floating-emoji {{ position: absolute; bottom: -12vh; opacity: 0.6; animation-name: floatUp;
        animation-timing-function: linear; animation-iteration-count: infinite; }}
    @keyframes floatUp {{ 0% {{ transform: translateY(0) rotate(0deg); }}
        50% {{ transform: translateY(-60vh) translateX(18px) rotate(160deg); }}
        100% {{ transform: translateY(-125vh) rotate(340deg); }} }}
    @keyframes bounceSoft {{ 0%, 100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-8px); }} }}
    @keyframes wiggle {{ 0%, 100% {{ transform: rotate(-6deg); }} 50% {{ transform: rotate(6deg); }} }}
    @keyframes selectedPop {{ 0%, 100% {{ transform: scale(1); }} 50% {{ transform: scale(1.035); }} }}
    .taletwinkle-title {{ text-align: center; font-size: clamp(3.4rem, 7vw, 6.2rem); line-height: 0.98;
        font-weight: 700; letter-spacing: -0.04em; color: #30234d; text-shadow: 0 0 18px white, 0 5px 0 white,
        0 12px 28px {accent_colour}55; margin: 0 0 0.15rem; animation: bounceSoft 3s ease-in-out infinite; }}
    .taletwinkle-tagline {{ text-align: center; font-size: clamp(1.05rem, 2vw, 1.4rem); color: #30234d;
        text-shadow: 0 0 10px white; margin-bottom: 0.55rem; font-weight: 600; }}
    .panel-card {{ background: linear-gradient(145deg, rgba(255,255,255,0.98), {accent_colour}18); border-radius: 20px;
        padding: 0.62rem 0.85rem; border: 1px solid {accent_colour}35; box-shadow: 0 8px 22px rgba(48,35,77,0.12);
        margin-bottom: 0.45rem; }}
    .panel-title {{ font-family: 'Fredoka', sans-serif; font-size: 1.3rem; font-weight: 600; color: #30234d; margin: 0; }}
    .mascot {{ font-size: 3rem; text-align: center; animation: wiggle 2.2s ease-in-out infinite; }}
    .world-name {{ text-align: center; font-family: 'Fredoka', sans-serif; font-size: 1.05rem; color: #30234d; }}
    [data-testid="stVerticalBlockBorderWrapper"]:has(.premium-panel-marker) {{
        background: linear-gradient(155deg, rgba(255,255,255,0.96), {accent_colour}25) !important;
        border: 2px solid {accent_colour}75 !important; border-radius: 28px !important;
        box-shadow: 0 16px 38px rgba(48,35,77,0.18), inset 0 1px 0 rgba(255,255,255,0.95) !important;
        backdrop-filter: blur(12px); padding: 0.85rem 0.9rem 0.7rem !important; position: relative; overflow: hidden; }}
    [data-testid="stVerticalBlockBorderWrapper"]:has(.premium-panel-marker)::before {{ content: ""; position: absolute;
        top: 0; left: 10%; right: 10%; height: 4px; border-radius: 0 0 8px 8px; background: {accent_colour}; }}
    .premium-panel-marker {{ height: 0; overflow: hidden; }}
    .st-key-live_story_box, [data-testid="stVerticalBlockBorderWrapper"]:has(.live-story-marker) {{ background: #ffffff !important;
        border: 2px solid {accent_colour}70 !important; border-radius: 26px !important;
        box-shadow: 0 12px 30px rgba(48,35,77,0.16) !important; padding: 1rem 1.3rem !important;
        min-height: 9rem; font-family: 'Fredoka', sans-serif; font-size: 1.28rem; line-height: 1.72; color: #2f2841; }}
    .st-key-live_story_box [data-testid="stMarkdownContainer"],
    .st-key-live_story_box [data-testid="stMarkdownContainer"] p {{ background: #ffffff !important; color: #2f2841 !important; }}
    .live-story-marker {{ height: 0; overflow: hidden; }}
    .caption-card {{ background: rgba(255,255,255,0.96); border: 2px solid {accent_colour}55; border-radius: 18px;
        padding: 0.62rem 0.95rem; margin: 0.45rem 0 0.75rem; color: #342b4d; font-size: 1.05rem;
        box-shadow: 0 7px 18px rgba(48,35,77,0.12); text-align: center; }}
    .placeholder-card {{ background: rgba(255,255,255,0.90); border: 3px dashed {accent_colour}; border-radius: 24px;
        text-align: center; padding: 1.4rem; font-family: 'Fredoka', sans-serif; font-size: 1.2rem; color: #51436f; margin-bottom: 0.8rem; }}
    .story-card {{ width: 100%; background: rgba(255,255,255,0.97); border-radius: 26px; padding: 1.15rem 1.45rem;
        border: 2px solid {accent_colour}60; border-left: 10px solid {accent_colour}; font-size: 1.3rem; line-height: 1.68;
        color: #2f2841; box-shadow: 0 12px 30px rgba(48,35,77,0.16); margin: 0.45rem 0 0.65rem; }}
    .word-chip {{ display: inline-block; background: {accent_colour}; color: white; border-radius: 999px;
        padding: 0.15rem 0.8rem; font-size: 0.95rem; margin: 0 0.4rem 0.3rem 0; font-family: 'Fredoka', sans-serif; }}
    .sound-label {{ font-family: 'Fredoka', sans-serif; font-size: 1.15rem; color: #30234d; margin: 0.2rem 0;
        background: rgba(255,255,255,0.88); border-radius: 14px; padding: 0.3rem 0.8rem; display: inline-block; }}
    [data-testid="stFileUploader"] section, [data-testid="stFileUploaderDropzone"] {{
        border: 4px dashed {accent_colour} !important; border-radius: 26px !important;
        background: rgba(255,255,255,0.93) !important; padding: 1.6rem !important; }}
    [data-testid="stFileUploader"] label p {{ font-size: 1.35rem !important; font-weight: 600; color: #30234d;
        background: rgba(255,255,255,0.88); border-radius: 12px; padding: 0.1rem 0.6rem; }}
    [data-testid="stImage"] img {{ border-radius: 22px; box-shadow: 0 8px 24px rgba(48,35,77,0.18); }}
    div[role="radiogroup"] label {{ background: rgba(255,255,255,0.92); border-radius: 18px; padding: 0.55rem 0.72rem;
        margin: 0.18rem 0; width: 100%; min-height: 3.05rem; border: 2px solid transparent;
        box-shadow: 0 3px 8px rgba(0,0,0,0.07); cursor: pointer; transition: transform .18s ease, box-shadow .18s ease; }}
    div[role="radiogroup"] label:hover {{ transform: translateX(5px) scale(1.02);
        box-shadow: 0 7px 16px {accent_colour}35; border-color: {accent_colour}75; }}
    div[role="radiogroup"] label:has(input:checked) {{ border: 2px solid {accent_colour};
        background: linear-gradient(90deg, white, {accent_colour}32); transform: translateX(5px);
        animation: selectedPop 2.4s ease-in-out infinite; }}
    div[role="radiogroup"] label p {{ font-size: 1.08rem !important; font-weight: 650 !important; }}
    [data-baseweb="select"] > div {{ border: 3px solid {accent_colour} !important; border-radius: 18px !important;
        font-size: 1.05rem; background: white; }}
    [data-testid="stSlider"] {{ background: linear-gradient(135deg, rgba(255,255,255,0.98), {accent_colour}20);
        border: 2px solid {accent_colour}55; border-radius: 19px; padding: 0.58rem 0.82rem 0.28rem;
        box-shadow: inset 0 1px 0 white, 0 5px 12px rgba(48,35,77,0.09); }}
    [data-testid="stSlider"] [data-baseweb="slider"] {{ min-height: 2.4rem; }}
    [role="slider"] {{ background-color: {accent_colour} !important; width: 1.35rem !important; height: 1.35rem !important;
        box-shadow: 0 0 0 5px white, 0 4px 11px {accent_colour}75 !important; cursor: grab !important; }}
    [role="slider"]:active {{ cursor: grabbing !important; transform: scale(1.12); }}
    .control-impact {{ margin: 0.28rem 0 0.55rem; padding: 0.38rem 0.65rem; text-align: center;
        border-radius: 999px; background: {accent_colour}18; color: #3c3155; font-weight: 600; font-size: 0.92rem; }}
    .character-card {{ background: linear-gradient(135deg, #ffffff, {accent_colour}25); border: 2px solid {accent_colour}65;
        border-radius: 20px; padding: 0.55rem 0.7rem; text-align: center; margin-top: 0.35rem;
        box-shadow: 0 7px 17px rgba(48,35,77,0.12); cursor: pointer; }}
    .character-card .mascot {{ font-size: 3.35rem; }}
    audio {{ width: 100%; }}
    .stButton > button {{ border-radius: 999px; border: 3px solid {accent_colour}; font-family: 'Fredoka', sans-serif;
        font-size: 1.1rem; background: white; }}
    [data-testid="stExpander"] {{ background: rgba(255,255,255,0.93); border-radius: 18px; }}
    @media (max-width: 900px) {{ .taletwinkle-title {{ font-size: clamp(3rem, 13vw, 4.5rem); }}
        [data-testid="stVerticalBlockBorderWrapper"]:has(.premium-panel-marker) {{ margin-bottom: 0.45rem; }}
        .story-card {{ font-size: 1.12rem; padding: 1rem; }} }}
    @media (prefers-reduced-motion: reduce) {{ .floating-emoji, .taletwinkle-title, .mascot,
        div[role="radiogroup"] label:has(input:checked) {{ animation: none !important; transition: none !important; }} }}
    </style>
    {build_floating_emoji_html(floating_emojis=theme_settings["floating_emojis"])}
    """
    # Remove the indentation: Markdown would show lines indented by 4+ spaces as a code block.
    unindented_page_css = "\n".join(css_line.strip() for css_line in page_css.splitlines() if css_line.strip())
    st.markdown(unindented_page_css, unsafe_allow_html=True)


def render_header():
    """
    Show the big bouncing app title and the friendly tagline.
    """
    st.markdown(f'<div class="taletwinkle-title">✨ {APP_NAME} ✨</div>'
                f'<div class="taletwinkle-tagline">{APP_TAGLINE}</div>', unsafe_allow_html=True)


# ---------- 2. Model loading functions (loaded AND warmed up once when the app starts) ----------
def get_hugging_face_token():
    """
    Read an optional Hugging Face access token from Streamlit secrets or the environment.

    Returns:
        str or None: the token, if one is configured.
    """
    try:
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return os.environ.get("HF_TOKEN")


def detect_available_cpu_cores():
    """
    Find how many CPU cores this app is really allowed to use. On Streamlit Cloud the container sees
    the whole host machine's cores but may only use about 2, so PyTorch would otherwise start far too
    many threads that slow each other down.

    Returns:
        int: number of usable CPU cores (at least 1).
    """
    try:
        # cgroup v2: "quota period", e.g. "200000 100000" means 2 cores.
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as cpu_limit_file:
            cpu_quota_text, cpu_period_text = cpu_limit_file.read().split()[:2]
        if cpu_quota_text != "max":
            return max(1, math.ceil(int(cpu_quota_text) / int(cpu_period_text)))
    except (OSError, ValueError):
        pass
    try:
        # cgroup v1: separate quota and period files.
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="utf-8") as quota_file:
            cpu_quota = int(quota_file.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="utf-8") as period_file:
            cpu_period = int(period_file.read().strip())
        if cpu_quota > 0:
            return max(1, math.ceil(cpu_quota / cpu_period))
    except (OSError, ValueError):
        pass
    return min(os.cpu_count() or 1, DEFAULT_CPU_THREADS)


@st.cache_resource(show_spinner=False)
def configure_cpu_threads():
    """
    Tell PyTorch to use exactly the usable number of CPU cores (done once per server).

    Returns:
        int: the number of PyTorch threads now in use.
    """
    usable_cpu_cores = detect_available_cpu_cores()
    torch.set_num_threads(usable_cpu_cores)
    LOGGER.info("PyTorch threads set to %d (machine reports %s cores)", torch.get_num_threads(), os.cpu_count())
    return torch.get_num_threads()


def get_process_memory_mb():
    """
    Read how much memory (RAM) this app is using right now.

    Returns:
        float or None: memory in MB, or None if it cannot be read on this system.
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as process_status_file:
            for status_line in process_status_file:
                if status_line.startswith("VmRSS:"):
                    return int(status_line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def log_memory_use(moment_description):
    """
    Write the current memory use to the log (visible in Streamlit Cloud: Manage app -> logs).

    Parameters:
        moment_description (str): what just happened, e.g. "after loading the story model".
    Returns:
        float or None: memory in MB.
    """
    memory_mb = get_process_memory_mb()
    if memory_mb is not None:
        LOGGER.info("Memory %s: %.2f MB (safety limit %d MB)", moment_description, memory_mb, MEMORY_SAFETY_LIMIT_MB)
    return memory_mb


def memory_is_too_high():
    """
    Check whether the app is close to Streamlit Cloud's memory limit.

    Returns:
        bool: True if memory use is above MEMORY_SAFETY_LIMIT_MB.
    """
    memory_mb = get_process_memory_mb()
    return memory_mb is not None and memory_mb > MEMORY_SAFETY_LIMIT_MB


def return_free_memory_to_system():
    """
    Collect unused Python objects and ask the C library to hand freed memory back to the system,
    so the memory figure really drops after a model is released.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def get_story_chat_template_settings(model_name):
    """
    Return model-specific chat-template controls for the Transformers pipeline.

    The text-generation pipeline automatically applies ``add_generation_prompt=True`` when the
    final chat message belongs to the user. Qwen3 additionally needs ``enable_thinking=False`` so
    it writes the short story immediately instead of spending time and tokens on a reasoning block.

    Parameters:
        model_name (str): Hugging Face story model id.
    Returns:
        dict: keyword arguments passed directly to ``pipeline(messages, ...)``.
    """
    if model_name.startswith("Qwen/Qwen3-"):
        return {"tokenizer_encode_kwargs": {"enable_thinking": False}}
    return {}


def shrink_model_to_int8(loaded_pipeline):
    """
    Convert the model's Linear layers to int8 (PyTorch dynamic quantization): about 3x less memory
    and faster CPU inference, with almost no change in output quality.

    Parameters:
        loaded_pipeline (Pipeline): a transformers pipeline.
    Returns:
        str: "int8" if the model was shrunk, otherwise "float32".
    """
    if not SHRINK_MODELS_WITH_INT8:
        return "float32"
    try:
        torch.ao.quantization.quantize_dynamic(loaded_pipeline.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        return "int8"
    except Exception:
        return "float32"


def clear_cached_model(cached_loader, loader_arguments=()):
    """
    Release one cached model entry (or the whole cache of that loader) so repeated use cannot
    accumulate model memory on Streamlit Cloud, then ask Python to reclaim unused memory.

    Parameters:
        cached_loader (function): a function decorated with st.cache_resource.
        loader_arguments (tuple): the arguments of the cached entry to release; empty = release all entries.
    """
    clear_function = getattr(cached_loader, "clear", None)
    if clear_function is not None:
        try:
            clear_function(*loader_arguments)
        except TypeError:
            clear_function()
    return_free_memory_to_system()


@st.cache_resource(show_spinner=False)
def load_caption_pipeline(model_name):
    """
    Load the image-to-text (captioning) pipeline once, shrink it and warm it up.

    Parameters:
        model_name (str): Hugging Face model id.
    Returns:
        dict: {"pipeline": Pipeline, "precision": str}.
    """
    warm_up_picture = Image.new("RGB", (64, 64), "white")
    caption_pipeline = None
    LOGGER.info("Loading caption model: %s", model_name)
    try:
        caption_pipeline = pipeline("image-to-text", model=model_name, device=-1, token=get_hugging_face_token())
        model_precision = shrink_model_to_int8(loaded_pipeline=caption_pipeline)
        caption_pipeline(warm_up_picture, generate_kwargs={"max_new_tokens": 5})
    except Exception as quantized_load_error:
        LOGGER.warning("Caption model int8 load failed; retrying in float32: %s", quantized_load_error)
        if caption_pipeline is not None:
            del caption_pipeline
        gc.collect()
        # If the shrunk model cannot run on this machine, use the normal full-precision model.
        try:
            caption_pipeline = pipeline("image-to-text", model=model_name, device=-1, token=get_hugging_face_token())
            model_precision = "float32"
            caption_pipeline(warm_up_picture, generate_kwargs={"max_new_tokens": 5})
        except Exception:
            LOGGER.exception("Caption model failed to load: %s", model_name)
            if caption_pipeline is not None:
                del caption_pipeline
            gc.collect()
            raise
    LOGGER.info("Caption model ready: %s (%s)", model_name, model_precision)
    return {"pipeline": caption_pipeline, "precision": model_precision}


@st.cache_resource(show_spinner=False)
def load_story_pipeline(model_name):
    """
    Load the text-generation (story) pipeline once and warm it up.

    Qwen3 uses bfloat16 directly. The focused benchmark found that converting Qwen3 from float32
    to legacy dynamic int8 increased peak memory sharply, while its native bfloat16 checkpoint was
    safer for Streamlit Cloud. Other candidates retain the int8-first fallback path for comparison.

    Parameters:
        model_name (str): Hugging Face model id.
    Returns:
        dict: {"pipeline": Pipeline, "precision": str}.
    """
    warm_up_messages = [{"role": "user", "content": "Say hi."}]
    qwen3_chat_settings = get_story_chat_template_settings(model_name=model_name)
    story_pipeline = None
    LOGGER.info("Loading story model: %s", model_name)
    if model_name.startswith("Qwen/Qwen3-"):
        try:
            story_pipeline = pipeline("text-generation", model=model_name, device=-1, torch_dtype=torch.bfloat16,
                                      token=get_hugging_face_token())
            model_precision = "bfloat16"
            story_pipeline(warm_up_messages, max_new_tokens=3, do_sample=False, **qwen3_chat_settings)
            LOGGER.info("Story model ready: %s (%s, thinking disabled)", model_name, model_precision)
            return {"pipeline": story_pipeline, "precision": model_precision}
        except Exception:
            LOGGER.exception("Qwen3 story model failed to load: %s", model_name)
            if story_pipeline is not None:
                del story_pipeline
            gc.collect()
            raise
    try:
        story_pipeline = pipeline("text-generation", model=model_name, device=-1, token=get_hugging_face_token())
        model_precision = shrink_model_to_int8(loaded_pipeline=story_pipeline)
        story_pipeline(warm_up_messages, max_new_tokens=3, do_sample=False, **qwen3_chat_settings)
    except Exception as quantized_load_error:
        LOGGER.warning("Story model int8 load failed; retrying in bfloat16: %s", quantized_load_error)
        if story_pipeline is not None:
            del story_pipeline
        gc.collect()
        # Fallback: half-size bfloat16 weights still fit Streamlit Cloud's memory.
        try:
            story_pipeline = pipeline("text-generation", model=model_name, device=-1, torch_dtype=torch.bfloat16,
                                      token=get_hugging_face_token())
            model_precision = "bfloat16"
            story_pipeline(warm_up_messages, max_new_tokens=3, do_sample=False, **qwen3_chat_settings)
        except Exception:
            LOGGER.exception("Story model failed to load: %s", model_name)
            if story_pipeline is not None:
                del story_pipeline
            gc.collect()
            raise
    LOGGER.info("Story model ready: %s (%s)", model_name, model_precision)
    return {"pipeline": story_pipeline, "precision": model_precision}


@st.cache_resource(show_spinner=False, max_entries=2)
def load_piper_voice(model_file):
    """
    Download and cache one fast Piper voice checkpoint from Hugging Face.

    Parameters:
        model_file (str): ONNX file path inside the Piper voices repository.
    Returns:
        PiperVoice: ready local CPU speech model.
    """
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice

    LOGGER.info("Loading Piper voice: %s", model_file)
    selected_voice = None
    try:
        model_path = hf_hub_download(repo_id=LOCAL_SPEECH_MODEL_NAME, filename=model_file)
        configuration_path = hf_hub_download(repo_id=LOCAL_SPEECH_MODEL_NAME, filename=f"{model_file}.json")
        selected_voice = PiperVoice.load(model_path=model_path, config_path=configuration_path, use_cuda=False)
        LOGGER.info("Piper voice ready: %s", model_file)
        return selected_voice
    except Exception:
        LOGGER.exception("Piper voice failed to load: %s", model_file)
        if selected_voice is not None:
            del selected_voice
        gc.collect()
        raise


@st.cache_resource(show_spinner=False)
def check_google_voice_available():
    """
    Warm up Google TTS once at start-up and check that it can be reached.

    Returns:
        bool: True if Google TTS answered.
    """
    try:
        speak_sentence_with_google(sentence_text="Hello.", accent_domain="co.uk")
        LOGGER.info("Google TTS availability check succeeded")
        return True
    except Exception as availability_error:
        LOGGER.warning("Google TTS unavailable; Piper fallback will be used: %s", availability_error)
        return False


@st.cache_resource(show_spinner=False)
def get_speech_lock():
    """
    One shared lock so each local Piper model speaks one sentence at a time (thread safety).

    Returns:
        threading.Lock: the lock.
    """
    return threading.Lock()


def load_story_and_voice_models(persona_settings):
    """
    Load Stage 2 (story model) and only the Stage 3 voice that is really needed, in parallel,
    after captioning has finished and the caption model has been released (keeps memory low).

    Parameters:
        persona_settings (dict): chosen storyteller voice.
    Returns:
        dict: {"story": story model dict, "local_speech": Piper voice or None, "google_voice_available": bool}.
    """
    google_voice_available = check_google_voice_available()
    voice_engine = choose_voice_engine(persona_settings=persona_settings,
                                       google_voice_available=google_voice_available)
    selected_piper_file = (select_local_model_file(persona_settings=persona_settings)
                           if voice_engine == "piper" else None)
    LOGGER.info("Loading Stage 2 and Stage 3 resources; voice engine=%s", voice_engine)
    with ThreadPoolExecutor(max_workers=2) as model_workers:
        story_job = model_workers.submit(load_story_pipeline, model_name=STORY_GENERATION_MODEL_NAME)
        piper_job = (model_workers.submit(load_piper_voice, model_file=selected_piper_file)
                     if selected_piper_file else None)
        return {
            "story": story_job.result(),
            "local_speech": piper_job.result() if piper_job else None,
            "google_voice_available": google_voice_available,
        }


def release_stage_two_and_three_models():
    """
    Clear the story model and Piper voice caches after a run so repeated use cannot accumulate model memory.
    """
    clear_cached_model(cached_loader=load_story_pipeline, loader_arguments=(STORY_GENERATION_MODEL_NAME,))
    clear_cached_model(cached_loader=load_piper_voice)
    LOGGER.info("Released Stage 2 and Stage 3 model caches")


def should_release_models():
    """
    Decide whether models must be released after use: always when KEEP_MODELS_LOADED is False,
    otherwise only when memory is above the safety limit (protects the app from being stopped).

    Returns:
        bool: True if models should be released now.
    """
    if not KEEP_MODELS_LOADED:
        return True
    if memory_is_too_high():
        LOGGER.warning("Memory above %d MB: releasing models after this step", MEMORY_SAFETY_LIMIT_MB)
        return True
    return False


def preload_story_models():
    """
    Load and warm up the caption and story models once when the app starts, so every story
    (including the first one after start-up) skips the ~25 s of model loading.

    Returns:
        bool: True if the models are loaded and kept in memory.
    """
    if not KEEP_MODELS_LOADED:
        return False
    load_caption_pipeline(model_name=IMAGE_CAPTION_MODEL_NAME)
    log_memory_use(moment_description="after loading the caption model")
    load_story_pipeline(model_name=STORY_GENERATION_MODEL_NAME)
    log_memory_use(moment_description="after loading the story model")
    return True


# ---------- 3. Picture functions ----------
def open_uploaded_picture(uploaded_picture_file):
    """
    Open the uploaded file as an RGB picture, fixing phone rotation and transparent backgrounds.

    Parameters:
        uploaded_picture_file (UploadedFile): file from st.file_uploader.
    Returns:
        PIL.Image.Image: RGB picture.
    """
    picture = Image.open(io.BytesIO(uploaded_picture_file.getvalue()))
    picture = ImageOps.exif_transpose(picture)
    if picture.mode in ("RGBA", "LA", "P"):
        picture = picture.convert("RGBA")
        white_background = Image.new("RGBA", picture.size, (255, 255, 255, 255))
        picture = Image.alpha_composite(white_background, picture)
    return picture.convert("RGB")


def resize_image_for_display(picture, longest_side_pixels):
    """
    Scale the picture so its longest side equals the preferred display size:
    small pictures are scaled UP and large pictures are scaled DOWN (aspect ratio kept).

    Parameters:
        picture (PIL.Image.Image): the picture.
        longest_side_pixels (int): preferred size of the longest side.
    Returns:
        PIL.Image.Image: resized picture.
    """
    scale_factor = longest_side_pixels / max(picture.size)
    new_size = (max(1, round(picture.width * scale_factor)), max(1, round(picture.height * scale_factor)))
    return picture.resize(new_size, Image.LANCZOS)


def prepare_image_for_model(picture, maximum_side_pixels):
    """
    Shrink very large pictures before captioning (never enlarges), which saves time.

    Parameters:
        picture (PIL.Image.Image): the picture.
        maximum_side_pixels (int): largest allowed side.
    Returns:
        PIL.Image.Image: picture ready for the captioning model.
    """
    if max(picture.size) <= maximum_side_pixels:
        return picture
    return resize_image_for_display(picture=picture, longest_side_pixels=maximum_side_pixels)


def calculate_picture_fingerprint(picture_bytes):
    """
    Create a short fingerprint of the picture so the same picture is not processed twice.

    Parameters:
        picture_bytes (bytes): raw uploaded bytes.
    Returns:
        str: MD5 hex digest.
    """
    return hashlib.md5(picture_bytes).hexdigest()


# ---------- 4. Stage 1: image-to-text functions ----------
def clean_caption(raw_caption):
    """
    Remove common captioning artefacts (e.g. "arafed", "there is", "illustration", "background")
    so that the story generator receives a clean description.

    Parameters:
        raw_caption (str): caption produced by the image-to-text model.
    Returns:
        str: cleaned caption in lower case.
    """
    cleaned_caption = raw_caption.lower().strip()
    unwanted_phrases = ["arafed", "araffe", "arafe", "there is ", "there are ", "this is ",
                        "an image of ", "a picture of ", "a photo of ", "an illustration of ", "illustration of ",
                        " illustration", "a cartoon of ", "cartoon of ", "a painting of ", "painting of ",
                        " background", "a drawing of "]
    for unwanted_phrase in unwanted_phrases:
        cleaned_caption = cleaned_caption.replace(unwanted_phrase, " ")
    # Remove immediately repeated words (some models stutter, e.g. "jungle jungle jungle").
    cleaned_caption = re.sub(r"\b(\w+)( \1\b)+", r"\1", cleaned_caption)
    cleaned_caption = re.sub(r"\s+", " ", cleaned_caption).strip(" ,.")
    return cleaned_caption or "a bright and happy scene"


def describe_picture(caption_pipeline, picture):
    """
    Stage 1: turn the picture into a short text description.

    Parameters:
        caption_pipeline (Pipeline): image-to-text pipeline.
        picture (PIL.Image.Image): the picture.
    Returns:
        tuple(str, float): cleaned caption and seconds taken.
    """
    start_time = time.perf_counter()
    caption_output = caption_pipeline(picture, generate_kwargs={"max_new_tokens": CAPTION_MAX_NEW_TOKENS,
                                                                "repetition_penalty": CAPTION_REPETITION_PENALTY,
                                                                "no_repeat_ngram_size": CAPTION_NO_REPEAT_NGRAM_SIZE})
    picture_caption = clean_caption(raw_caption=caption_output[0]["generated_text"])
    return picture_caption, time.perf_counter() - start_time


# ---------- 5. Stage 2: story generation functions ----------
def build_story_messages(image_caption, theme_instruction, target_word_count):
    """
    Build the chat messages that ask the instruction model for a short, kid-friendly story in the chosen theme.

    Parameters:
        image_caption (str): description of the picture.
        theme_instruction (str): theme-specific instruction from STORY_THEMES.
        target_word_count (int): story length chosen on the slider.
    Returns:
        list[dict]: system and user messages.
    """
    system_message = ("You are a kind storyteller for children aged 3 to 10. "
                      "Use short sentences and simple, happy words. "
                      "Never include anything scary, violent, sad or unsafe.")
    user_message = (f"Write a story for young children, about {target_word_count} words long, "
                    f"based on this picture: \"{image_caption}\". "
                    f"Include the things you can see in the picture. "
                    f"{theme_instruction} "
                    f"Start in a fun, surprising way and give it a happy ending. Write only the story, with no title.")
    return [{"role": "system", "content": system_message},
            {"role": "user", "content": user_message}]


def split_text_into_sentences(story_text):
    """
    Split text into sentences (or poem lines). Used for trimming and for speaking sentence by sentence.

    Parameters:
        story_text (str): any text.
    Returns:
        list[str]: sentences / lines, each under about 180 characters.
    """
    sentence_list = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", story_text) if sentence.strip()]
    speech_sentences = []
    for sentence in sentence_list:
        # Split very long sentences at a comma (or a space) so each speech request stays short.
        while len(sentence) > MAXIMUM_SPEECH_CHUNK_CHARACTERS:
            split_position = sentence.rfind(",", 0, MAXIMUM_SPEECH_CHUNK_CHARACTERS)
            split_position = (split_position if split_position > 40
                              else sentence.rfind(" ", 0, MAXIMUM_SPEECH_CHUNK_CHARACTERS))
            speech_sentences.append(sentence[:split_position + 1].strip())
            sentence = sentence[split_position + 1:].strip()
        if sentence:
            speech_sentences.append(sentence)
    return speech_sentences


def queue_finished_sentences(stream_progress, queue_sentence_for_speech):
    """
    Send every sentence that is already complete to the voice engine (runs in the background),
    so narration is being recorded while the rest of the story is still being written.

    Parameters:
        stream_progress (dict): shared progress ("text" so far, "queued_sentences" count).
        queue_sentence_for_speech (function): starts speech for one sentence.
    """
    sentences_so_far = split_text_into_sentences(story_text=stream_progress["text"])
    text_ends_sentence = bool(re.search(r"([.!?][\"'”’)]*|\n)\s*$", stream_progress["text"]))
    finished_sentences = sentences_so_far if text_ends_sentence else sentences_so_far[:-1]
    for finished_sentence in finished_sentences[stream_progress["queued_sentences"]:]:
        queue_sentence_for_speech(sentence_text=finished_sentence)
    stream_progress["queued_sentences"] = max(stream_progress["queued_sentences"], len(finished_sentences))


def stream_story_text(story_pipeline, story_messages, target_word_count, stream_progress,
                      queue_sentence_for_speech, use_early_stop):
    """
    Generate the story in a background thread and yield new words as soon as they are written
    (shown live by st.write_stream inside the white story box). Finished sentences are queued for speech;
    generation stops as soon as the story is long enough.

    Parameters:
        story_pipeline (Pipeline): text-generation pipeline.
        story_messages (list[dict]): chat messages.
        target_word_count (int): wanted length.
        stream_progress (dict): shared progress dictionary (text, timings, problems).
        queue_sentence_for_speech (function): starts speech for one sentence.
        use_early_stop (bool): stop generating once enough words exist.
    Yields:
        str: pieces of story text.
    """
    upper_word_limit = min(MAXIMUM_STORY_WORDS, target_word_count + STORY_WORD_MARGIN)
    stop_event = threading.Event()
    text_streamer = TextIteratorStreamer(story_pipeline.tokenizer, skip_prompt=True,
                                         skip_special_tokens=True, timeout=STORY_STREAM_TIMEOUT_SECONDS)

    def stop_when_story_is_long_enough(input_ids, scores, **keyword_arguments):
        """Stopping rule checked after every new token (True = stop generating)."""
        return torch.full((input_ids.shape[0],), stop_event.is_set(), dtype=torch.bool, device=input_ids.device)

    generation_settings = {"text_inputs": story_messages, "streamer": text_streamer,
                           "max_new_tokens": int(upper_word_limit * STORY_TOKENS_PER_WORD) + STORY_EXTRA_TOKENS,
                           "do_sample": True, "temperature": STORY_TEMPERATURE, "top_p": STORY_TOP_P,
                           "repetition_penalty": STORY_REPETITION_PENALTY,
                           "pad_token_id": story_pipeline.tokenizer.eos_token_id}
    generation_settings.update(get_story_chat_template_settings(model_name=STORY_GENERATION_MODEL_NAME))
    if use_early_stop:
        generation_settings["stopping_criteria"] = StoppingCriteriaList([stop_when_story_is_long_enough])

    def run_generation():
        """Run the pipeline and record any error (runs in the background thread)."""
        try:
            story_pipeline(**generation_settings)
        except Exception as generation_error:
            stream_progress["problems"].append(generation_error)
            text_streamer.end()

    generation_thread = threading.Thread(target=run_generation, daemon=True)
    generation_thread.start()
    try:
        for new_text_piece in text_streamer:
            if stream_progress["first_words_time"] is None and new_text_piece.strip():
                stream_progress["first_words_time"] = time.perf_counter()
            stream_progress["text"] += new_text_piece
            queue_finished_sentences(stream_progress=stream_progress, queue_sentence_for_speech=queue_sentence_for_speech)
            if len(stream_progress["text"].split()) > upper_word_limit:
                stop_event.set()
            yield new_text_piece
    except Empty:
        stream_progress["problems"].append(TimeoutError("The story model took too long."))
    stop_event.set()
    generation_thread.join(timeout=2)


def trim_story_to_word_limit(story_text, target_word_count, is_poem):
    """
    Trim the story to complete sentences (or poem lines) close to the target word count,
    staying inside 50-100 words.

    Parameters:
        story_text (str): generated story.
        target_word_count (int): length chosen on the slider.
        is_poem (bool): True for the poem theme (keep line breaks).
    Returns:
        tuple(str, int): trimmed story and its word count.
    """
    # Remove titles, markdown symbols and "Story:" labels the model may add.
    story_lines = [line for line in story_text.strip().splitlines()
                   if not re.match(r"^\s*(#|title\s*:|\*\*title)", line, flags=re.IGNORECASE)]
    cleaned_story_text = "\n".join(story_lines).replace("**", "").replace("Story:", "").strip()

    if is_poem and cleaned_story_text.count("\n") >= 2:
        story_units = [line.strip() for line in cleaned_story_text.splitlines() if line.strip()]
        unit_separator = "\n"
    else:
        flat_story_text = re.sub(r"\s+", " ", cleaned_story_text)
        story_units = [unit.strip() for unit in re.split(r"(?<=[.!?])\s+", flat_story_text) if unit.strip()]
        unit_separator = " "

    upper_word_limit = min(MAXIMUM_STORY_WORDS, target_word_count + STORY_WORD_MARGIN)
    kept_units = []
    kept_word_count = 0
    for story_unit in story_units:
        unit_word_count = len(story_unit.split())
        if kept_word_count + unit_word_count > upper_word_limit:
            break
        kept_units.append(story_unit)
        kept_word_count += unit_word_count

    # Drop an unfinished last sentence (prose only) if enough words remain.
    if unit_separator == " " and kept_units and not re.search(r"[.!?][\"'”’)]*$", kept_units[-1]):
        if kept_word_count - len(kept_units[-1].split()) >= MINIMUM_STORY_WORDS:
            kept_word_count -= len(kept_units[-1].split())
            kept_units = kept_units[:-1]

    trimmed_story = unit_separator.join(kept_units)
    if kept_word_count < MINIMUM_STORY_WORDS:
        # Sentences were too long to fit: cut by words instead.
        all_words = re.sub(r"\s+", " ", cleaned_story_text).split(" ")
        trimmed_story = " ".join(all_words[:target_word_count]).rstrip(",;:")
        if not re.search(r"[.!?]$", trimmed_story):
            trimmed_story += "."
    return trimmed_story, len(trimmed_story.split())


def story_is_kid_safe(story_text):
    """
    Check the story against a list of words that are not suitable for young children.

    Parameters:
        story_text (str): the story.
    Returns:
        bool: True if no unsafe word was found.
    """
    lower_story_text = story_text.lower()
    return not any(re.search(rf"\b{re.escape(unsafe_word)}\b", lower_story_text) for unsafe_word in UNSAFE_STORY_WORDS)


def make_backup_story(image_caption, theme_key, target_word_count):
    """
    Build a simple, always-safe story from a template. Used ONLY if the story model fails or writes
    something unsuitable, so the app never leaves the child without a story.

    Parameters:
        image_caption (str): description of the picture.
        theme_key (str): key of STORY_THEMES.
        target_word_count (int): wanted length.
    Returns:
        tuple(str, int): story and word count.
    """
    theme_middles = {
        "fairy_tale": "A kind fairy sprinkled golden sparkles, and every wish came true.",
        "space_quest": "A friendly little alien waved from a shiny rocket and shared a star cookie.",
        "gentle_mystery": "They followed tiny clues and found the lost key under a leaf.",
        "jungle_adventure": "They followed an old map past the waterfall and found a box of shiny treasure.",
        "silly_poem": "They sang a silly song, all the whole day long.",
        "ocean_magic": "A smiling fish showed them a shell that sparkled like a rainbow.",
    }
    backup_sentences = [
        f"Look at this: {image_caption}!",
        "Everyone was curious and ready for a wonderful day.",
        theme_middles.get(theme_key, theme_middles["fairy_tale"]),
        "They laughed, they helped each other, and they shared everything they had.",
        "Little friends came to join the fun, one by one.",
        "When the day was done, they all felt warm, happy and proud.",
        "It was the best day ever, and they could not wait to play again tomorrow.",
        "The end.",
    ]
    return trim_story_to_word_limit(story_text=" ".join(backup_sentences), target_word_count=target_word_count, is_poem=False)


def write_story_live(story_pipeline, image_caption, theme_key, target_word_count, story_slot, queue_sentence_for_speech):
    """
    Stage 2: write the story with words appearing live in story_slot (sentences are queued for speech
    while writing), then trim and safety-check it.

    Parameters:
        story_pipeline (Pipeline): text-generation pipeline.
        image_caption (str): description of the picture.
        theme_key (str): chosen theme.
        target_word_count (int): chosen length.
        story_slot (st.empty): placeholder where the story appears.
        queue_sentence_for_speech (function): starts speech for one sentence.
    Returns:
        dict: story, word_count, first_words_time, used_backup_story.
    """
    story_messages = build_story_messages(image_caption=image_caption,
                                          theme_instruction=STORY_THEMES[theme_key]["story_instruction"],
                                          target_word_count=target_word_count)
    stream_progress = {"text": "", "queued_sentences": 0, "first_words_time": None, "problems": []}
    # Try with the early-stop rule first; if this transformers version rejects it, try once without it.
    for use_early_stop in (True, False):
        stream_progress.update({"text": "", "queued_sentences": 0, "problems": []})
        try:
            with story_slot.container(border=True, key="live_story_box"):
                st.markdown('<div class="live-story-marker"></div>', unsafe_allow_html=True)
                st.markdown('<span class="word-chip">✍️ Writing your story...</span>', unsafe_allow_html=True)
                st.write_stream(stream_story_text(story_pipeline=story_pipeline, story_messages=story_messages,
                                                  target_word_count=target_word_count, stream_progress=stream_progress,
                                                  queue_sentence_for_speech=queue_sentence_for_speech,
                                                  use_early_stop=use_early_stop))
        except Exception as streaming_error:
            stream_progress["problems"].append(streaming_error)
        if stream_progress["text"].strip():
            break

    final_story, final_word_count = trim_story_to_word_limit(story_text=stream_progress["text"],
                                                             target_word_count=target_word_count,
                                                             is_poem=(theme_key == POEM_THEME_KEY))
    used_backup_story = False
    if final_word_count < MINIMUM_STORY_WORDS - 10 or not story_is_kid_safe(story_text=final_story):
        final_story, final_word_count = make_backup_story(image_caption=image_caption, theme_key=theme_key,
                                                          target_word_count=target_word_count)
        used_backup_story = True
    return {"story": final_story, "word_count": final_word_count,
            "first_words_time": stream_progress["first_words_time"], "used_backup_story": used_backup_story}


# ---------- 6. Stage 3: text-to-speech functions ----------
def decode_mp3_bytes(mp3_bytes):
    """
    Decode MP3 audio into float samples. Uses soundfile; falls back to ffmpeg if needed.

    Parameters:
        mp3_bytes (bytes): MP3 data.
    Returns:
        tuple(np.ndarray, int): mono float32 samples and sample rate.
    """
    try:
        import soundfile
        decoded_samples, sample_rate = soundfile.read(io.BytesIO(mp3_bytes), dtype="float32")
    except Exception:
        if shutil.which("ffmpeg") is None:
            raise
        sample_rate = 24000     # gTTS audio is 24 kHz; ask ffmpeg for raw 16-bit mono samples at that rate
        ffmpeg_result = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-acodec", "pcm_s16le",
                                        "-ac", "1", "-ar", str(sample_rate), "pipe:1"],
                                       input=mp3_bytes, capture_output=True, check=True)
        decoded_samples = np.frombuffer(ffmpeg_result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if decoded_samples.ndim > 1:
        decoded_samples = decoded_samples.mean(axis=1)
    return decoded_samples.astype(np.float32), sample_rate


def speak_sentence_with_google(sentence_text, accent_domain):
    """
    Speak one sentence with Google Text-to-Speech (gTTS) in the chosen English accent.

    Parameters:
        sentence_text (str): text to speak.
        accent_domain (str): "co.uk", "us", "com.au" or "co.in".
    Returns:
        tuple(np.ndarray, int): mono samples and sample rate.
    """
    from gtts import gTTS
    mp3_buffer = io.BytesIO()
    gTTS(text=sentence_text, lang="en", tld=accent_domain, timeout=10).write_to_fp(mp3_buffer)
    return decode_mp3_bytes(mp3_bytes=mp3_buffer.getvalue())


def speak_sentence_with_piper(sentence_text, model_file):
    """
    Speak one sentence locally with a fast Piper ONNX checkpoint from Hugging Face.

    Parameters:
        sentence_text (str): text to speak.
        model_file (str): selected Piper ONNX file inside ``rhasspy/piper-voices``.
    Returns:
        tuple(np.ndarray, int): mono float samples and model sample rate.
    """
    selected_voice = load_piper_voice(model_file=model_file)
    with get_speech_lock():
        generated_chunks = list(selected_voice.synthesize(sentence_text))
    speech_segments = [np.asarray(chunk.audio_float_array, dtype=np.float32).squeeze()
                       for chunk in generated_chunks if len(chunk.audio_float_array)]
    if not speech_segments:
        raise RuntimeError("Piper returned no speech audio.")
    return np.concatenate(speech_segments), int(generated_chunks[0].sample_rate)


def select_local_model_file(persona_settings):
    """
    Choose the Piper voice file: the persona's own voice, or a matching regional fallback
    (British voice for UK personas, American voice otherwise) when Google TTS is unavailable.

    Parameters:
        persona_settings (dict): chosen storyteller voice.
    Returns:
        str: Piper ONNX file path inside the rhasspy/piper-voices repository.
    """
    if persona_settings.get("piper_model_file"):
        return persona_settings["piper_model_file"]
    if persona_settings.get("accent_domain") == "co.uk":
        return DEFAULT_PIPER_MODEL_FILE
    return "en/en_US/amy/medium/en_US-amy-medium.onnx"


def choose_voice_engine(persona_settings, google_voice_available):
    """
    Decide which engine speaks: the persona's own engine, or local Piper
    if Google TTS cannot be reached.

    Parameters:
        persona_settings (dict): chosen storyteller voice.
        google_voice_available (bool): result of the start-up check.
    Returns:
        str: "google" or "piper".
    """
    if persona_settings["engine"] == "google" and google_voice_available:
        return "google"
    return "piper"


def speak_sentence(sentence_text, voice_engine, persona_settings, local_speech_resources):
    """
    Speak one sentence with the chosen engine (called in background threads).

    Parameters:
        sentence_text (str): text to speak.
        voice_engine (str): "google" or "piper".
        persona_settings (dict): accent, voice pack, and language choices.
        local_speech_resources: warmed local fallback kept alive by Streamlit's cache.
    Returns:
        tuple(np.ndarray, int): mono samples and sample rate.
    """
    if voice_engine == "google":
        return speak_sentence_with_google(sentence_text=sentence_text,
                                          accent_domain=persona_settings["accent_domain"])
    # local_speech_resources keeps the cached Piper voice alive during the run; the voice itself is
    # looked up through the cached loader inside speak_sentence_with_piper.
    return speak_sentence_with_piper(sentence_text=sentence_text,
                                     model_file=select_local_model_file(persona_settings=persona_settings))


def start_speech_queue(voice_engine, persona_settings, local_speech_resources):
    """
    Create a background worker pool and a function that queues one sentence for speech.
    Google requests run in parallel; the local model runs one sentence at a time.

    Parameters:
        voice_engine (str): "google" or "piper".
        persona_settings (dict): selected storyteller settings.
        local_speech_resources: warmed local Piper fallback.
    Returns:
        tuple(ThreadPoolExecutor, dict, function): the pool, sentence->future map, queue function.
    """
    worker_count = GOOGLE_VOICE_PARALLEL_REQUESTS if voice_engine == "google" else 1
    speech_workers = ThreadPoolExecutor(max_workers=worker_count)
    speech_jobs = {}

    def queue_sentence_for_speech(sentence_text):
        """Start speaking this sentence in the background (only once per sentence)."""
        if sentence_text not in speech_jobs:
            speech_jobs[sentence_text] = speech_workers.submit(speak_sentence, sentence_text=sentence_text,
                                                               voice_engine=voice_engine,
                                                               persona_settings=persona_settings,
                                                               local_speech_resources=local_speech_resources)

    return speech_workers, speech_jobs, queue_sentence_for_speech


def time_stretch_speech(audio_samples, sample_rate, stretch_rate):
    """
    Change speaking speed WITHOUT changing pitch, using WSOLA (waveform-similarity overlap-add).
    stretch_rate > 1 speaks faster, < 1 speaks slower.

    Parameters:
        audio_samples (np.ndarray): mono samples.
        sample_rate (int): samples per second.
        stretch_rate (float): speed factor.
    Returns:
        np.ndarray: time-stretched samples.
    """
    if abs(stretch_rate - 1.0) < 0.01 or len(audio_samples) < sample_rate // 10:
        return audio_samples
    frame_length = int(0.040 * sample_rate)
    synthesis_hop = frame_length // 2
    analysis_hop = synthesis_hop * stretch_rate
    search_tolerance = int(0.012 * sample_rate)
    fade_window = np.hanning(frame_length).astype(np.float32)
    padded_samples = np.pad(audio_samples, (search_tolerance, frame_length + search_tolerance + synthesis_hop))
    number_of_frames = int((len(audio_samples) - frame_length) / analysis_hop) + 1
    output_samples = np.zeros(number_of_frames * synthesis_hop + frame_length, dtype=np.float32)
    window_sum = np.zeros_like(output_samples)
    previous_position = search_tolerance
    for frame_index in range(max(number_of_frames, 1)):
        nominal_position = int(frame_index * analysis_hop) + search_tolerance
        if frame_index == 0:
            best_position = nominal_position
        else:
            # Find the frame that best continues the previous one (keeps the voice smooth).
            continuation_start = previous_position + synthesis_hop
            natural_continuation = padded_samples[continuation_start: continuation_start + frame_length]
            search_region = padded_samples[nominal_position - search_tolerance:
                                           nominal_position + search_tolerance + frame_length]
            similarity = correlate(search_region, natural_continuation, mode="valid", method="fft")
            best_position = nominal_position - search_tolerance + int(np.argmax(similarity))
        output_start = frame_index * synthesis_hop
        best_frame = padded_samples[best_position: best_position + frame_length]
        output_samples[output_start: output_start + frame_length] += best_frame * fade_window
        window_sum[output_start: output_start + frame_length] += fade_window
        previous_position = best_position
    return output_samples / np.maximum(window_sum, 1e-3)


def change_pitch_and_tempo(audio_samples, sample_rate, semitone_shift, tempo_factor):
    """
    Shift the voice pitch (in semitones) and set the speaking tempo in one step: time-stretch first,
    then resample. Used for the cartoon characters and the voice speed slider.

    Parameters:
        audio_samples (np.ndarray): mono samples.
        sample_rate (int): samples per second.
        semitone_shift (float): +12 = one octave higher, -12 = one octave lower.
        tempo_factor (float): 1.0 normal, 2.0 twice as fast, 0.5 half speed.
    Returns:
        np.ndarray: processed samples (same sample rate).
    """
    pitch_factor = 2 ** (semitone_shift / 12)
    stretched_samples = time_stretch_speech(audio_samples=audio_samples, sample_rate=sample_rate,
                                            stretch_rate=tempo_factor / pitch_factor)
    if abs(pitch_factor - 1.0) < 0.005:
        return stretched_samples
    resample_ratio = Fraction(1 / pitch_factor).limit_denominator(60)
    return resample_poly(stretched_samples, resample_ratio.numerator, resample_ratio.denominator).astype(np.float32)


def add_robot_effect(audio_samples, sample_rate):
    """
    Make a friendly robot voice with a fast "buzzy" amplitude modulation.

    Parameters:
        audio_samples (np.ndarray): mono samples.
        sample_rate (int): samples per second.
    Returns:
        np.ndarray: robot voice samples.
    """
    time_axis = np.arange(len(audio_samples)) / sample_rate
    return (audio_samples * (0.55 + 0.45 * np.sin(2 * np.pi * 60 * time_axis))).astype(np.float32)


def normalise_loudness(audio_samples, peak_level):
    """
    Scale audio so its loudest point equals peak_level.

    Parameters:
        audio_samples (np.ndarray): samples.
        peak_level (float): wanted peak (0-1).
    Returns:
        np.ndarray: scaled samples.
    """
    loudest_value = float(np.max(np.abs(audio_samples))) if len(audio_samples) else 0.0
    return audio_samples if loudest_value == 0 else (audio_samples / loudest_value * peak_level).astype(np.float32)


def convert_samples_to_wav_bytes(audio_samples, sample_rate):
    """
    Encode float samples as 16-bit WAV bytes for st.audio and downloads.

    Parameters:
        audio_samples (np.ndarray): samples in -1..1.
        sample_rate (int): samples per second.
    Returns:
        bytes: WAV file content.
    """
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sample_rate, (np.clip(audio_samples, -1, 1) * 32767).astype(np.int16))
    return wav_buffer.getvalue()


def assemble_narration(sentence_clips, persona_settings, voice_speed):
    """
    Join the spoken sentences, apply the character effect and the chosen speed, and make a WAV file.

    Parameters:
        sentence_clips (list[tuple]): (samples, sample_rate) per sentence, in story order.
        persona_settings (dict): chosen storyteller voice.
        voice_speed (float): speed from the slider (0.5x to 2x).
    Returns:
        tuple(bytes, float): WAV bytes and audio length in seconds.
    """
    sample_rate = sentence_clips[0][1]
    short_pause = np.zeros(int(SENTENCE_PAUSE_SECONDS * sample_rate), dtype=np.float32)
    joined_samples = np.concatenate([np.concatenate([clip_samples, short_pause]) for clip_samples, _ in sentence_clips])
    voiced_samples = change_pitch_and_tempo(audio_samples=joined_samples, sample_rate=sample_rate,
                                            semitone_shift=persona_settings["semitone_shift"],
                                            tempo_factor=persona_settings["tempo_factor"] * voice_speed)
    if persona_settings["robot_effect"]:
        voiced_samples = add_robot_effect(audio_samples=voiced_samples, sample_rate=sample_rate)
    voiced_samples = normalise_loudness(audio_samples=voiced_samples, peak_level=NARRATION_PEAK_LEVEL)
    return convert_samples_to_wav_bytes(audio_samples=voiced_samples, sample_rate=sample_rate), len(voiced_samples) / sample_rate


def collect_sentence_clips(story_sentences, speech_jobs, queue_sentence_for_speech,
                           local_speech_resources, persona_settings):
    """
    Wait for every sentence of the final story to be spoken (most are already finished in the background).
    If any online sentence fails, the whole story is re-spoken with the local Hugging Face voice.

    Parameters:
        story_sentences (list[str]): sentences of the final story.
        speech_jobs (dict): sentence -> future from start_speech_queue.
        queue_sentence_for_speech (function): starts speech for a sentence not yet queued.
        local_speech_resources: warmed Piper fallback kept alive by Streamlit's cache.
        persona_settings (dict): selected or region-matched voice settings.
    Returns:
        tuple(list, bool): clips in order, True if the fallback voice was used.
    """
    for story_sentence in story_sentences:
        queue_sentence_for_speech(sentence_text=story_sentence)
    try:
        return [speech_jobs[story_sentence].result(timeout=SPEECH_RESULT_TIMEOUT_SECONDS)
                for story_sentence in story_sentences], False
    except Exception as online_voice_error:
        LOGGER.warning("Online narration failed; re-speaking with Piper: %s", online_voice_error)
        fallback_model_file = select_local_model_file(persona_settings=persona_settings)
        return [speak_sentence_with_piper(sentence_text=story_sentence, model_file=fallback_model_file)
                for story_sentence in story_sentences], True


def create_narration_for_story(story_text, persona_settings, voice_speed):
    """
    Speak a finished story from scratch (used when the child changes the voice or speed later).
    All sentences are spoken in parallel.

    Parameters:
        story_text (str): the story.
        persona_settings (dict): chosen storyteller voice.
        voice_speed (float): chosen speed.
    Returns:
        dict: wav_bytes, audio_seconds, voice_seconds, voice_engine.
    """
    start_time = time.perf_counter()
    local_speech_resources = None
    speech_workers = None
    google_voice_available = check_google_voice_available()
    voice_engine = choose_voice_engine(persona_settings=persona_settings,
                                       google_voice_available=google_voice_available)
    try:
        if voice_engine == "piper":
            local_speech_resources = load_piper_voice(
                model_file=select_local_model_file(persona_settings=persona_settings))
        speech_workers, speech_jobs, queue_sentence_for_speech = start_speech_queue(
            voice_engine=voice_engine, persona_settings=persona_settings,
            local_speech_resources=local_speech_resources)
        sentence_clips, used_fallback_voice = collect_sentence_clips(
            story_sentences=split_text_into_sentences(story_text=story_text), speech_jobs=speech_jobs,
            queue_sentence_for_speech=queue_sentence_for_speech,
            local_speech_resources=local_speech_resources, persona_settings=persona_settings)
        speech_workers.shutdown(wait=False)
        speech_workers = None
        wav_bytes, audio_seconds = assemble_narration(sentence_clips=sentence_clips,
                                                      persona_settings=persona_settings, voice_speed=voice_speed)
        return {"wav_bytes": wav_bytes, "audio_seconds": audio_seconds,
                "voice_seconds": time.perf_counter() - start_time,
                "voice_engine": describe_voice_engine(voice_engine=voice_engine, persona_settings=persona_settings,
                                                       used_fallback_voice=used_fallback_voice)}
    except Exception:
        LOGGER.exception("Narration generation failed")
        raise
    finally:
        if speech_workers is not None:
            speech_workers.shutdown(wait=False, cancel_futures=True)
        if local_speech_resources is not None:
            del local_speech_resources
        if should_release_models():
            clear_cached_model(cached_loader=load_piper_voice)


def describe_voice_engine(voice_engine, persona_settings, used_fallback_voice):
    """
    Describe which engine produced the voice (shown in the grown-ups panel).

    Parameters:
        voice_engine (str): "google" or "piper".
        persona_settings (dict): chosen storyteller voice.
        used_fallback_voice (bool): True if the backup voice was needed.
    Returns:
        str: description.
    """
    if used_fallback_voice or (voice_engine == "piper" and persona_settings["engine"] == "google"):
        fallback_model_file = select_local_model_file(persona_settings=persona_settings)
        return f"{LOCAL_SPEECH_MODEL_NAME} ({fallback_model_file}; local fallback because Google TTS was unavailable)"
    if voice_engine == "google":
        return f"Google TTS (gTTS, accent domain '{persona_settings['accent_domain']}')"
    return f"{LOCAL_SPEECH_MODEL_NAME} ({persona_settings['piper_model_file']}; local Hugging Face ONNX model)"


@st.cache_data(show_spinner=False, max_entries=12)
def create_greeting_audio(voice_key, voice_speed):
    """
    Make a short hello from the chosen storyteller, so the audio player is always on screen and
    children can hear each voice before choosing a picture.

    Parameters:
        voice_key (str): key of VOICE_PERSONAS.
        voice_speed (float): chosen speed.
    Returns:
        bytes: WAV bytes (empty bytes if no voice is available).
    """
    persona_settings = VOICE_PERSONAS[voice_key]
    storyteller_name = re.sub(r"\s*\(.*\)", "", persona_settings["label"]).split(" ", 1)[-1]
    greeting_text = f"Hello! I am {storyteller_name}. Drop a picture, and I will tell you a story!"
    used_piper = False
    try:
        if persona_settings["engine"] == "google":
            try:
                greeting_clip = speak_sentence_with_google(sentence_text=greeting_text,
                                                           accent_domain=persona_settings["accent_domain"])
            except Exception:
                LOGGER.warning("Google greeting failed for %s; using Piper fallback", voice_key)
                used_piper = True
                greeting_clip = speak_sentence_with_piper(
                    sentence_text=greeting_text,
                    model_file=select_local_model_file(persona_settings=persona_settings))
        else:
            used_piper = True
            greeting_clip = speak_sentence_with_piper(
                sentence_text=greeting_text,
                model_file=select_local_model_file(persona_settings=persona_settings))
        greeting_wav_bytes, _ = assemble_narration(sentence_clips=[greeting_clip],
                                                    persona_settings=persona_settings,
                                                    voice_speed=voice_speed)
        return greeting_wav_bytes
    except Exception:
        LOGGER.exception("Greeting audio failed for voice: %s", voice_key)
        return b""
    finally:
        if used_piper and should_release_models():
            clear_cached_model(cached_loader=load_piper_voice)


# ---------- 7. Complete picture -> story -> voice run ----------
def create_story_and_voice(picture, theme_key, target_word_count, persona_settings,
                           voice_speed, story_slot, caption_slot, scroll_slot):
    """
    Run all three stages for a new picture: caption, live story writing with background speech, narration.

    Parameters:
        picture (PIL.Image.Image): the uploaded picture.
        theme_key (str): chosen theme.
        target_word_count (int): chosen length.
        persona_settings (dict): chosen storyteller voice.
        voice_speed (float): chosen speed.
        story_slot (st.empty): placeholder where the story appears.
        caption_slot (st.empty): area directly below the picture for the generated caption.
        scroll_slot (st.empty): hidden area used to scroll the page down to the story.
    Returns:
        tuple(dict, dict): story_result and narration_result.
    """
    start_time = time.perf_counter()
    LOGGER.info("Starting picture-to-story run: theme=%s target_words=%d", theme_key, target_word_count)
    show_placeholder(display_slot=story_slot, placeholder_text="🔎 Looking closely at your picture...")
    caption_resources = None
    caption_precision = "unknown"
    try:
        caption_load_start_time = time.perf_counter()
        caption_resources = load_caption_pipeline(model_name=IMAGE_CAPTION_MODEL_NAME)
        caption_load_seconds = time.perf_counter() - caption_load_start_time
        caption_precision = caption_resources["precision"]
        picture_caption, caption_seconds = describe_picture(
            caption_pipeline=caption_resources["pipeline"],
            picture=prepare_image_for_model(picture=picture, maximum_side_pixels=MODEL_INPUT_IMAGE_SIZE))
    except Exception:
        LOGGER.exception("Stage 1 caption generation failed")
        raise
    finally:
        if caption_resources is not None:
            del caption_resources
        if should_release_models():
            clear_cached_model(cached_loader=load_caption_pipeline, loader_arguments=(IMAGE_CAPTION_MODEL_NAME,))
            LOGGER.info("Released Stage 1 model after caption generation")
    show_caption_card(caption_slot=caption_slot, picture_caption=picture_caption)
    LOGGER.info("Stage 1: caption model ready in %.2f s, caption written in %.2f s", caption_load_seconds, caption_seconds)

    generation_resources = None
    speech_workers = None
    try:
        story_load_start_time = time.perf_counter()
        generation_resources = load_story_and_voice_models(persona_settings=persona_settings)
        story_load_seconds = time.perf_counter() - story_load_start_time
        LOGGER.info("Stage 2: story model (and voice) ready in %.2f s", story_load_seconds)
        voice_engine = choose_voice_engine(persona_settings=persona_settings,
                                           google_voice_available=generation_resources["google_voice_available"])
        speech_workers, speech_jobs, queue_sentence_for_speech = start_speech_queue(
            voice_engine=voice_engine, persona_settings=persona_settings,
            local_speech_resources=generation_resources["local_speech"])
        # Scroll down so the child can watch the story being written.
        request_auto_scroll(scroll_slot=scroll_slot, target_selector=".st-key-live_story_box", also_show_audio=False)
        story_writing_start_time = time.perf_counter()
        story_generation = write_story_live(
            story_pipeline=generation_resources["story"]["pipeline"], image_caption=picture_caption,
            theme_key=theme_key, target_word_count=target_word_count, story_slot=story_slot,
            queue_sentence_for_speech=queue_sentence_for_speech)
        story_ready_time = time.perf_counter()
        story_writing_seconds = story_ready_time - story_writing_start_time
        LOGGER.info("Stage 2: story written in %.2f s (%d words)", story_writing_seconds, story_generation["word_count"])
        show_story_card(story_slot=story_slot, story_text=story_generation["story"],
                        word_count=story_generation["word_count"])

        sentence_clips, used_fallback_voice = collect_sentence_clips(
            story_sentences=split_text_into_sentences(story_text=story_generation["story"]), speech_jobs=speech_jobs,
            queue_sentence_for_speech=queue_sentence_for_speech,
            local_speech_resources=generation_resources["local_speech"], persona_settings=persona_settings)
        speech_workers.shutdown(wait=False)
        speech_workers = None
        wav_bytes, audio_seconds = assemble_narration(sentence_clips=sentence_clips,
                                                      persona_settings=persona_settings, voice_speed=voice_speed)
        voice_ready_time = time.perf_counter()
        LOGGER.info("Stage 3: narration ready %.2f s after the story", voice_ready_time - story_ready_time)
        story_precision = generation_resources["story"]["precision"]
    except Exception:
        LOGGER.exception("Stage 2 or Stage 3 generation failed")
        raise
    finally:
        if speech_workers is not None:
            speech_workers.shutdown(wait=False, cancel_futures=True)
        if generation_resources is not None:
            del generation_resources
        if should_release_models():
            release_stage_two_and_three_models()
        log_memory_use(moment_description="after the story run")

    first_words_time = story_generation["first_words_time"] or story_ready_time
    story_result = {"caption": picture_caption, "story": story_generation["story"],
                    "word_count": story_generation["word_count"], "used_backup_story": story_generation["used_backup_story"],
                    "caption_seconds": caption_seconds, "first_words_seconds": first_words_time - start_time,
                    "story_seconds": story_ready_time - start_time, "caption_precision": caption_precision,
                    "story_precision": story_precision, "caption_load_seconds": caption_load_seconds,
                    "story_load_seconds": story_load_seconds, "story_writing_seconds": story_writing_seconds}
    narration_result = {"wav_bytes": wav_bytes, "audio_seconds": audio_seconds,
                        "voice_seconds": voice_ready_time - story_ready_time, "total_seconds": voice_ready_time - start_time,
                        "voice_engine": describe_voice_engine(voice_engine=voice_engine, persona_settings=persona_settings,
                                                              used_fallback_voice=used_fallback_voice)}
    LOGGER.info("TIMING SUMMARY | caption model %.2f s | caption %.2f s | story model %.2f s | first words at %.2f s | "
                "story writing %.2f s | voice after story %.2f s | TOTAL %.2f s",
                caption_load_seconds, caption_seconds, story_load_seconds, story_result["first_words_seconds"],
                story_writing_seconds, narration_result["voice_seconds"], narration_result["total_seconds"])
    return story_result, narration_result


# ---------- 8. Screen (display) functions ----------
def render_theme_picker():
    """
    Left panel: list of story worlds (themes) and the animated theme mascot.

    Returns:
        str: the selected theme key.
    """
    with st.container(border=True):
        st.markdown('<div class="premium-panel-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-card"><p class="panel-title">🎨 Pick a story world</p></div>', unsafe_allow_html=True)
        selected_theme_key = st.radio("Story world", options=list(STORY_THEMES.keys()),
                                      format_func=lambda theme_key: STORY_THEMES[theme_key]["selector_label"],
                                      key="theme_choice", label_visibility="collapsed")
        chosen_theme = STORY_THEMES[selected_theme_key]
        st.markdown(f'<div class="panel-card"><div class="mascot">{chosen_theme["mascot"]}</div>'
                    f'<div class="world-name">Portal opened! Welcome to<br><b>{chosen_theme["world_name"]}</b>!</div></div>',
                    unsafe_allow_html=True)
    return selected_theme_key


def render_story_controls():
    """
    Right panel: story length slider, voice speed slider and storyteller voice dropdown.

    Returns:
        tuple(int, float, str): target word count, voice speed, voice key.
    """
    with st.container(border=True):
        st.markdown('<div class="premium-panel-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-card"><p class="panel-title">📏 Story size</p></div>', unsafe_allow_html=True)
        target_word_count = st.slider("Story size (words)", min_value=MINIMUM_STORY_WORDS, max_value=MAXIMUM_STORY_WORDS,
                                      value=DEFAULT_STORY_WORDS, step=STORY_WORD_STEP, format="%d words",
                                      key="story_size_choice", label_visibility="collapsed")
        if target_word_count <= 60:
            length_mood = "Quick Tale ⚡"
        elif target_word_count >= 90:
            length_mood = "Big Adventure 🌟"
        else:
            length_mood = "Magic Middle ✨"
        st.markdown(f'<div class="control-impact">{target_word_count} words · {length_mood}</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-card"><p class="panel-title">🐢 Sleepy Turtle → Zooming Rocket 🚀</p></div>',
                    unsafe_allow_html=True)
        voice_speed = st.select_slider("Voice speed", options=VOICE_SPEED_CHOICES, value=DEFAULT_VOICE_SPEED,
                                       format_func=lambda speed_value: f"{speed_value:.2f}x",
                                       key="voice_speed_choice", label_visibility="collapsed")
        if voice_speed < 0.9:
            speed_mood = "Sleepy Turtle 🐢"
        elif voice_speed > 1.25:
            speed_mood = "Zooming Rocket 🚀"
        else:
            speed_mood = "Just Right ⭐"
        st.markdown(f'<div class="control-impact">{voice_speed:.2f}x · {speed_mood}</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-card"><p class="panel-title">🗣️ Storyteller voice</p></div>', unsafe_allow_html=True)
        selected_voice_key = st.selectbox("Storyteller voice", options=list(VOICE_PERSONAS.keys()),
                                          index=list(VOICE_PERSONAS.keys()).index(DEFAULT_VOICE_KEY),
                                          format_func=lambda voice_key: VOICE_PERSONAS[voice_key]["label"],
                                          key="voice_choice", label_visibility="collapsed")
        chosen_persona = VOICE_PERSONAS[selected_voice_key]
        st.markdown(f'<div class="character-card"><div class="mascot">{chosen_persona["avatar"]}</div>'
                    f'<div class="world-name"><b>Your storyteller is ready!</b><br>'
                    f'{html.escape(chosen_persona["about"])}</div></div>', unsafe_allow_html=True)
    return target_word_count, voice_speed, selected_voice_key


def show_placeholder(display_slot, placeholder_text):
    """
    Show a friendly dashed placeholder card (keeps picture/story areas always visible).

    Parameters:
        display_slot (st.empty): where to show it.
        placeholder_text (str): the message.
    """
    display_slot.markdown(f'<div class="placeholder-card">{placeholder_text}</div>', unsafe_allow_html=True)


def show_caption_card(caption_slot, picture_caption):
    """
    Show the Stage 1 caption (what TaleTwinkle sees) directly below the uploaded picture.

    Parameters:
        caption_slot (st.empty): the area under the picture.
        picture_caption (str): the cleaned caption.
    """
    safe_caption = html.escape(picture_caption)
    caption_slot.markdown(f'<div class="caption-card">👀 <b>I can see:</b> {safe_caption}</div>',
                          unsafe_allow_html=True)


def show_story_card(story_slot, story_text, word_count):
    """
    Show the finished story in a big, easy-to-read card.

    Parameters:
        story_slot (st.empty): where to show it.
        story_text (str): the story.
        word_count (int): number of words.
    """
    safe_story_html = html.escape(story_text).replace("\n", "<br>")
    story_slot.markdown(f'<div class="story-card"><span class="word-chip">📖 {word_count} words</span><br>'
                        f'{safe_story_html}</div>',
                        unsafe_allow_html=True)


def build_auto_scroll_script(target_selector, also_show_audio, request_number):
    """
    Build a tiny script that smoothly scrolls the page to the story (and, if asked, far enough
    to show the audio player too). It retries for a few seconds because Streamlit draws elements
    slightly after the Python code sends them.

    Parameters:
        target_selector (str): CSS selector of the element to scroll to.
        also_show_audio (bool): True = make sure the audio player is also visible on screen.
        request_number (int): unique number so the browser runs the script again for every new request.
    Returns:
        str: HTML with the scroll script.
    """
    audio_check = "true" if also_show_audio else "false"
    return f"""
    <script>
    // TaleTwinkle auto-scroll request {request_number}
    const parentDocument = window.parent.document;
    let scrollAttempts = 0;
    function scrollToStory() {{
        const targetElements = parentDocument.querySelectorAll("{target_selector}");
        const targetElement = targetElements[targetElements.length - 1];
        if (!targetElement) {{
            if (scrollAttempts++ < 40) {{ setTimeout(scrollToStory, 150); }}
            return;
        }}
        targetElement.scrollIntoView({{behavior: "smooth", block: "start"}});
        if ({audio_check}) {{
            setTimeout(function () {{
                const audioPlayers = parentDocument.querySelectorAll("audio");
                const audioPlayer = audioPlayers[audioPlayers.length - 1];
                if (audioPlayer && audioPlayer.getBoundingClientRect().bottom > window.parent.innerHeight) {{
                    audioPlayer.scrollIntoView({{behavior: "smooth", block: "end"}});
                }}
            }}, 800);
        }}
    }}
    scrollToStory();
    </script>
    """


def request_auto_scroll(scroll_slot, target_selector, also_show_audio):
    """
    Scroll the page down to the story (used when the story starts being written and when the
    narration is ready), so young children do not need to scroll by themselves.

    Parameters:
        scroll_slot (st.empty): hidden area at the bottom of the page that holds the scroll script.
        target_selector (str): CSS selector of the element to scroll to.
        also_show_audio (bool): True = also bring the audio player into view.
    """
    st.session_state["scroll_request_count"] += 1
    scroll_script = build_auto_scroll_script(target_selector=target_selector, also_show_audio=also_show_audio,
                                             request_number=st.session_state["scroll_request_count"])
    with scroll_slot:
        components.html(scroll_script, height=0)


def increase_story_version():
    """
    Button callback: ask for a brand-new story for the same picture.
    """
    st.session_state["story_version"] += 1


def show_grown_up_details(story_result, narration_result, theme_key, voice_key):
    """
    Collapsible panel for parents / teachers / graders: timings (2 decimals), caption, models and downloads.

    Parameters:
        story_result (dict): caption, story and timings.
        narration_result (dict): audio and timings.
        theme_key (str): chosen theme.
        voice_key (str): chosen voice.
    """
    with st.expander("🧑‍🏫 For grown-ups: how TaleTwinkle made this story"):
        timing_columns = st.columns(4)
        timing_columns[0].metric("Picture → words", f"{story_result['caption_seconds']:.2f} s")
        timing_columns[1].metric("First story words", f"{story_result['first_words_seconds']:.2f} s")
        timing_columns[2].metric("Whole story", f"{story_result['story_seconds']:.2f} s")
        timing_columns[3].metric("Voice ready", f"{narration_result['total_seconds']:.2f} s")
        if "story_writing_seconds" in story_result:
            st.caption(f"Stage details: caption model ready in {story_result['caption_load_seconds']:.2f} s · "
                       f"caption {story_result['caption_seconds']:.2f} s · story model ready in "
                       f"{story_result['story_load_seconds']:.2f} s · story writing {story_result['story_writing_seconds']:.2f} s.")
        st.caption(f"Voice finished {narration_result['voice_seconds']:.2f} s after the story, because sentences were "
                   f"spoken in the background while the story was being written. "
                   f"Audio length: {narration_result['audio_seconds']:.2f} s.")
        st.markdown(f"**What the picture shows (Stage 1):** {story_result['caption']}")
        st.markdown(f"**Theme:** {STORY_THEMES[theme_key]['label']} &nbsp;|&nbsp; "
                    f"**Voice:** {VOICE_PERSONAS[voice_key]['label']} "
                    f"&nbsp;|&nbsp; **Words:** {story_result['word_count']}")
        st.markdown(f"**Models:** image-to-text `{IMAGE_CAPTION_MODEL_NAME}` ({story_result['caption_precision']}) → "
                    f"text-generation `{STORY_GENERATION_MODEL_NAME}` ({story_result['story_precision']}) → "
                    f"speech: {narration_result['voice_engine']}")
        if story_result["used_backup_story"]:
            st.info("The story model had a problem, so a safe backup story was used this time.")
        download_columns = st.columns(2)
        download_columns[0].download_button("📄 Save story text", data=story_result["story"],
                                            file_name="taletwinkle_story.txt", mime="text/plain")
        download_columns[1].download_button("🎧 Save story audio", data=narration_result["wav_bytes"],
                                            file_name="taletwinkle_story.wav", mime="audio/wav")


# ---------- 9. Input / process / output helpers used by main() ----------
def collect_child_choices():
    """
    INPUT: draw the three-column screen (story worlds | picture | controls) plus the full-width
    story and audio area, and read every choice the child makes.

    Returns:
        tuple(dict, dict): child_choices (theme, length, speed, voice, picture file) and
                           screen_slots (the places on screen where results are shown).
    """
    left_column, centre_column, right_column = st.columns([1.05, 2.45, 1.20], gap="large",
                                                          vertical_alignment="top")
    with left_column:
        selected_theme_key = render_theme_picker()
    with right_column:
        target_word_count, voice_speed, selected_voice_key = render_story_controls()
    with centre_column:
        uploaded_picture_file = st.file_uploader("📸 Drop a picture here, or tap to choose one!",
                                                 type=ACCEPTED_PICTURE_TYPES, key="picture_upload")
        image_slot = st.empty()
        caption_slot = st.empty()

    # Full-width output keeps the complete story and audio easy to read without squeezing either panel.
    story_slot = st.empty()
    sound_label_slot = st.empty()
    sound_slot = st.empty()
    extras_area = st.container()
    scroll_slot = st.empty()

    child_choices = {"theme_key": selected_theme_key, "target_word_count": target_word_count,
                     "voice_speed": voice_speed, "voice_key": selected_voice_key,
                     "persona_settings": VOICE_PERSONAS[selected_voice_key],
                     "uploaded_picture_file": uploaded_picture_file}
    screen_slots = {"centre_column": centre_column, "image_slot": image_slot, "caption_slot": caption_slot,
                    "story_slot": story_slot, "sound_label_slot": sound_label_slot, "sound_slot": sound_slot,
                    "extras_area": extras_area, "scroll_slot": scroll_slot}
    return child_choices, screen_slots


def open_picture_safely(uploaded_picture_file, centre_column):
    """
    Open the uploaded picture, or show a friendly message if the file is not a picture.

    Parameters:
        uploaded_picture_file (UploadedFile or None): file from the uploader.
        centre_column (st.delta_generator.DeltaGenerator): where to show the message.
    Returns:
        PIL.Image.Image or None: the picture, or None if there is no usable picture.
    """
    if uploaded_picture_file is None:
        return None
    try:
        return open_uploaded_picture(uploaded_picture_file=uploaded_picture_file)
    except Exception:
        LOGGER.exception("Uploaded picture could not be opened")
        with centre_column:
            st.error("😕 Oops! That file is not a picture I can open. Please try a JPG or PNG picture.")
        return None


def build_request_keys(uploaded_picture_file, child_choices):
    """
    Build the keys that decide what must be redone: a new story (new picture, theme, length or
    "another story") or only a new voice (new storyteller or speed).

    Parameters:
        uploaded_picture_file (UploadedFile): file from the uploader.
        child_choices (dict): result of collect_child_choices().
    Returns:
        tuple(tuple, tuple): story_request_key and narration_request_key.
    """
    story_request_key = (calculate_picture_fingerprint(picture_bytes=uploaded_picture_file.getvalue()),
                         child_choices["theme_key"], child_choices["target_word_count"],
                         st.session_state["story_version"])
    narration_request_key = (story_request_key, child_choices["voice_key"], child_choices["voice_speed"])
    return story_request_key, narration_request_key


def run_new_story_request(picture, child_choices, screen_slots, story_request_key, narration_request_key):
    """
    Run all three stages (caption, story, voice) for a new request and remember the results.

    Parameters:
        picture (PIL.Image.Image): the uploaded picture.
        child_choices (dict): result of collect_child_choices().
        screen_slots (dict): places on screen from collect_child_choices().
        story_request_key (tuple): key of this story request.
        narration_request_key (tuple): key of this narration request.
    """
    screen_slots["sound_label_slot"].markdown('<p class="sound-label">🎙️ Your storyteller is getting ready...</p>',
                                              unsafe_allow_html=True)
    try:
        with st.spinner("🪄 Building your story one magical step at a time..."):
            new_story_result, new_narration_result = create_story_and_voice(
                picture=picture, theme_key=child_choices["theme_key"],
                target_word_count=child_choices["target_word_count"],
                persona_settings=child_choices["persona_settings"], voice_speed=child_choices["voice_speed"],
                story_slot=screen_slots["story_slot"], caption_slot=screen_slots["caption_slot"],
                scroll_slot=screen_slots["scroll_slot"])
        st.session_state.update({"story_request_key": story_request_key, "story_result": new_story_result,
                                 "narration_request_key": narration_request_key,
                                 "narration_result": new_narration_result, "celebrate_new_story": True,
                                 "scroll_to_new_audio": True})
    except Exception as stage_error:
        LOGGER.exception("Complete picture-to-story request failed")
        st.session_state["story_request_key"] = None
        show_placeholder(display_slot=screen_slots["story_slot"],
                         placeholder_text=f"😕 Oops! I could not make a story from this picture. Please try another one. "
                                          f"<small>({type(stage_error).__name__})</small>")


def run_voice_change_request(child_choices, screen_slots, narration_request_key):
    """
    Same story, new storyteller or speed: re-record only the voice (the story is not rewritten).

    Parameters:
        child_choices (dict): result of collect_child_choices().
        screen_slots (dict): places on screen from collect_child_choices().
        narration_request_key (tuple): key of this narration request.
    """
    screen_slots["sound_label_slot"].markdown('<p class="sound-label">🎙️ Changing the storyteller voice...</p>',
                                              unsafe_allow_html=True)
    try:
        changed_narration = create_narration_for_story(story_text=st.session_state["story_result"]["story"],
                                                       persona_settings=child_choices["persona_settings"],
                                                       voice_speed=child_choices["voice_speed"])
        changed_narration.update({"total_seconds": changed_narration["voice_seconds"]})
        st.session_state.update({"narration_request_key": narration_request_key, "narration_result": changed_narration,
                                 "scroll_to_new_audio": True})
    except Exception:
        LOGGER.exception("Voice-only regeneration failed")
        st.session_state["narration_request_key"] = None
        with screen_slots["centre_column"]:
            st.warning("🔇 The storyteller lost their voice. Please try another voice.")


def process_picture_request(picture, child_choices, screen_slots):
    """
    PROCESS: show the picture, then decide whether to write a new story, only re-record the voice,
    or reuse the results already made for exactly these choices.

    Parameters:
        picture (PIL.Image.Image or None): the uploaded picture.
        child_choices (dict): result of collect_child_choices().
        screen_slots (dict): places on screen from collect_child_choices().
    Returns:
        tuple(dict or None, dict or None): story_result and narration_result for the current choices.
    """
    if picture is None:
        return None, None
    # Show the picture first (scaled to the preferred size), so the child sees it straight away.
    screen_slots["image_slot"].image(resize_image_for_display(picture=picture,
                                                              longest_side_pixels=PREFERRED_IMAGE_DISPLAY_SIZE))
    show_placeholder(display_slot=screen_slots["caption_slot"], placeholder_text="👀 Looking closely at your picture...")
    story_request_key, narration_request_key = build_request_keys(
        uploaded_picture_file=child_choices["uploaded_picture_file"], child_choices=child_choices)

    if st.session_state["story_request_key"] != story_request_key:
        run_new_story_request(picture=picture, child_choices=child_choices, screen_slots=screen_slots,
                              story_request_key=story_request_key, narration_request_key=narration_request_key)
    elif st.session_state["narration_request_key"] != narration_request_key:
        run_voice_change_request(child_choices=child_choices, screen_slots=screen_slots,
                                 narration_request_key=narration_request_key)

    story_result = None
    narration_result = None
    if st.session_state["story_request_key"] == story_request_key:
        story_result = st.session_state["story_result"]
        show_caption_card(caption_slot=screen_slots["caption_slot"], picture_caption=story_result["caption"])
        if st.session_state["narration_request_key"] == narration_request_key:
            narration_result = st.session_state["narration_result"]
    return story_result, narration_result


def show_story_outputs(picture, story_result, narration_result, child_choices, screen_slots):
    """
    OUTPUT: show placeholders or the finished story, play the narration (autoplay), add the
    "another story" button and grown-ups panel, celebrate, and scroll down to the story and audio.
    Before any story exists, the storyteller's greeting is offered instead.

    Parameters:
        picture (PIL.Image.Image or None): the uploaded picture.
        story_result (dict or None): current story.
        narration_result (dict or None): current narration.
        child_choices (dict): result of collect_child_choices().
        screen_slots (dict): places on screen from collect_child_choices().
    """
    persona_label = html.escape(child_choices["persona_settings"]["label"])
    if picture is None:
        show_placeholder(display_slot=screen_slots["image_slot"], placeholder_text="🖼️ Your picture will pop up here!")
        show_placeholder(display_slot=screen_slots["caption_slot"], placeholder_text="👀 I will describe your picture here.")
        show_placeholder(display_slot=screen_slots["story_slot"], placeholder_text="📖 Your story will appear here...")
    elif story_result is not None:
        show_story_card(story_slot=screen_slots["story_slot"], story_text=story_result["story"],
                        word_count=story_result["word_count"])

    if narration_result is not None:
        screen_slots["sound_label_slot"].markdown(
            f'<p class="sound-label">🎧 {persona_label} is reading your story '
            f'&nbsp;·&nbsp; ⚡ ready in {narration_result["total_seconds"]:.2f} s</p>', unsafe_allow_html=True)
        screen_slots["sound_slot"].audio(narration_result["wav_bytes"], format="audio/wav", autoplay=True)
        with screen_slots["extras_area"]:
            st.button("🔄 Tell me another story!", on_click=increase_story_version, use_container_width=True)
            show_grown_up_details(story_result=story_result, narration_result=narration_result,
                                  theme_key=child_choices["theme_key"], voice_key=child_choices["voice_key"])
        if st.session_state["celebrate_new_story"]:
            st.session_state["celebrate_new_story"] = False
            st.balloons()
        if st.session_state["scroll_to_new_audio"]:
            # New narration: scroll so the story and the playing audio player are both on screen.
            st.session_state["scroll_to_new_audio"] = False
            request_auto_scroll(scroll_slot=screen_slots["scroll_slot"], target_selector=".story-card",
                                also_show_audio=True)
    else:
        greeting_wav_bytes = create_greeting_audio(voice_key=child_choices["voice_key"],
                                                   voice_speed=child_choices["voice_speed"])
        screen_slots["sound_label_slot"].markdown(f'<p class="sound-label">🎧 Press ▶ to hear {persona_label} say hello</p>',
                                                  unsafe_allow_html=True)
        if greeting_wav_bytes:
            screen_slots["sound_slot"].audio(greeting_wav_bytes, format="audio/wav")


# ==============================
# MAIN PART
# ==============================
def main():
    """
    Start of the program flow: set up the page, collect the child's choices (input),
    run the three pipelines (process) and show picture, story and audio (output).
    """
    configure_page()
    initialise_session_state()
    configure_cpu_threads()
    current_theme_key = st.session_state.get("theme_choice", DEFAULT_THEME_KEY)
    apply_page_style(theme_settings=STORY_THEMES[current_theme_key])
    render_header()

    # ----- INPUT PART -----
    child_choices, screen_slots = collect_child_choices()

    # Load and warm up the models once when the app starts (kept for every later story).
    with st.spinner("🪄 Waking up the story machine... (only the first time)"):
        preload_story_models()

    # ----- PROCESS PART -----
    picture = open_picture_safely(uploaded_picture_file=child_choices["uploaded_picture_file"],
                                  centre_column=screen_slots["centre_column"])
    story_result, narration_result = process_picture_request(picture=picture, child_choices=child_choices,
                                                             screen_slots=screen_slots)

    # ----- OUTPUT PART -----
    show_story_outputs(picture=picture, story_result=story_result, narration_result=narration_result,
                       child_choices=child_choices, screen_slots=screen_slots)


if __name__ == "__main__":
    main()
