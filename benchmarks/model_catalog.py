"""Candidate model catalogue used by the TaleTwinkle benchmarks."""


IMAGE_CAPTION_MODELS = [
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
    "microsoft/git-base",
    "microsoft/git-base-coco",
    "microsoft/git-base-textcaps",
    "microsoft/git-large",
    "microsoft/git-large-coco",
    "microsoft/git-large-textcaps",
    "nlpconnect/vit-gpt2-image-captioning",
    "ydshieh/vit-gpt2-coco-en",
]


STORY_MODELS = [
    "roneneldan/TinyStories-Instruct-1M",
    "roneneldan/TinyStories-Instruct-3M",
    "roneneldan/TinyStories-Instruct-8M",
    "roneneldan/TinyStories-Instruct-28M",
    "roneneldan/TinyStories-Instruct-33M",
    "roneneldan/TinyStories-1M",
    "roneneldan/TinyStories-33M",
    "google/flan-t5-small",
    "google/flan-t5-base",
    "distilbert/distilgpt2",
]


TTS_CANDIDATES = [
    "facebook/mms-tts-eng",
    "microsoft/speecht5_tts",
    "suno/bark-small",
    "coqui/XTTS-v2",
    "espnet/kan-bayashi_ljspeech_vits",
    "espnet/kan-bayashi_vctk_vits",
    "hexgrad/Kokoro-82M",
    "facebook/fastspeech2-en-ljspeech",
    "gTTS-2.5.4",
    "pyttsx3-2.99",
]
