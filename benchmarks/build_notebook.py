"""
Builds TaleTwinkle_Model_Benchmark.ipynb (a Google Colab notebook).

Run:  python build_notebook.py
The notebook compares 10 candidate models for each TaleTwinkle pipeline stage
(image captioning, story generation, text-to-speech) on 10 test images.
"""

import json

notebook_cells = []


def add_markdown_cell(markdown_text):
    """Append a markdown cell to the notebook."""
    notebook_cells.append({"cell_type": "markdown", "metadata": {},
                           "source": markdown_text.strip("\n").splitlines(keepends=True)})


def add_code_cell(code_text):
    """Append a code cell to the notebook."""
    notebook_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": code_text.strip("\n").splitlines(keepends=True)})


# ---------------------------------------------------------------------------
add_markdown_cell(r"""
# TaleTwinkle: Model Benchmark (ISOM5240 Individual Assignment)

This notebook compares **10 candidate Hugging Face models for each pipeline stage** of the TaleTwinkle
storytelling app, using the **10 test images in `test_images/`** (6 illustrated story worlds + 4 photos) and their
expected-object lists in `test_images/test_image_ground_truth.csv`:

| Stage | Pipeline task | What is measured |
|---|---|---|
| 1. Image captioning | `image-to-text` | **Speed** (seconds per image) and **accuracy** (% of expected objects named in the caption + CLIP image-text score) |
| 2. Story generation | `text-generation` | **Image relevance**, **theme match**, **kid suitability** (reading grade + safety), **length compliance**, speed |
| 3. Text-to-speech | `text-to-speech` / gTTS / Piper / Kokoro | **Audio quality**: intelligibility (Whisper word error rate), speed, real-time factor + your listening score |

### How to run
1. **Runtime → Change runtime type → CPU** (keep the default CPU). Streamlit Cloud runs on CPU, so CPU timings are the honest ones.
2. The notebook needs the `test_images/` folder: it clones your GitHub repo (set `GITHUB_REPOSITORY_URL` in cell 1),
   or asks you to upload `test_images.zip` if that is empty.
3. (Optional) Add your Hugging Face token: key icon on the left → *Secrets* → name `HF_TOKEN`, turn on *Notebook access*.
   Two story candidates (Llama 3.2 1B, Gemma 3 270M) are *gated*: accept their licence on their Hugging Face pages first, or they are skipped automatically.
4. **Runtime → Run all.** It takes roughly **45–75 minutes** on CPU. Every model runs inside `try/except`, so one failure does not stop the run.
5. If Colab asks you to **restart the session** after the install cell, click *Restart*, then *Run all* again.
6. At the end, a zip file `taletwinkle_benchmark_results.zip` downloads automatically. It contains:
   * `benchmark_results.xlsx` (all tables)
   * `stories/` (every generated story, to read and rate)
   * `tts_samples/` (every voice sample, to listen to and rate)

Send the zip (or just the Excel file) back so the final models can be locked in `app.py` and the results written into the README.
""")

# ---------------------------------------------------------------------------
add_markdown_cell("## 0. Install packages")
add_code_cell(r"""
# Install the same library versions the Streamlit app uses (transformers < 5.0, as in the course).
%pip -q install "transformers>=4.57.0,<5.0" "torch>=2.6.0,<2.10.0" "sentence-transformers>=3.0.0,<6.0" "datasets>=2.20.0,<5.0" \
    "textstat>=0.7.3,<1.0" "jiwer>=3.0.0,<5.0" "gTTS>=2.5.0,<3.0" "soundfile>=0.12.1,<1.0" \
    "kokoro>=0.9.4,<1.0" "piper-tts>=1.8.0,<2.0" "onnxruntime>=1.20.0,<2.0" \
    "pyttsx3>=2.90,<3.0" "openpyxl>=3.1.0,<4.0" "psutil>=5.9.0,<8.0"
# espeak-ng is needed by Kokoro (phonemizer fallback) and pyttsx3 on Linux.
!apt-get -qq install -y espeak-ng > /dev/null 2>&1
""")

# ---------------------------------------------------------------------------
add_markdown_cell("## 1. Configuration: candidate models, themes, settings")
add_code_cell(r"""
# ==============================
# IMPORT PART
# ==============================
import gc
import io
import os
import re
import time
import shutil
import zipfile
import traceback

import numpy as np
import pandas as pd
import psutil
import soundfile as sf
import torch
from PIL import Image
from transformers import pipeline, set_seed

# ==============================
# CONFIGURATION PART
# ==============================
# DEVICE = -1 means CPU (honest timings for Streamlit Cloud). Use 0 only if you want a quick GPU dry run.
DEVICE = -1
NUMBER_OF_TEST_IMAGES = 10
TTS_TEST_STORY_COUNT = 3          # voice tests use 3 stories (each engine speaks the same 3 stories)
RANDOM_SEED = 42
GITHUB_REPOSITORY_URL = ""   # e.g. "https://github.com/your-name/taletwinkle" (public repo with test_images/)
OUTPUT_FOLDER = "taletwinkle_benchmark"
TEST_IMAGE_FOLDER = os.path.join(OUTPUT_FOLDER, "test_images")
STORY_FOLDER = os.path.join(OUTPUT_FOLDER, "stories")
TTS_SAMPLE_FOLDER = os.path.join(OUTPUT_FOLDER, "tts_samples")
for folder_path in [OUTPUT_FOLDER, TEST_IMAGE_FOLDER, STORY_FOLDER, TTS_SAMPLE_FOLDER]:
    os.makedirs(folder_path, exist_ok=True)

# Hugging Face token from Colab Secrets (optional; needed only for gated models).
HF_TOKEN = None
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")
print(f"Hugging Face token found: {HF_TOKEN is not None}")

# ---- Stage 1: 10 image-captioning candidates (pipeline task: image-to-text) ----
CAPTION_MODEL_CANDIDATES = [
    "Salesforce/blip-image-captioning-base",        # course example
    "Salesforce/blip-image-captioning-large",
    "microsoft/git-base-coco",
    "microsoft/git-large-coco",
    "microsoft/git-base",
    "microsoft/git-base-textcaps",
    "microsoft/git-large-textcaps",
    "microsoft/git-large",
    "nlpconnect/vit-gpt2-image-captioning",
    "ydshieh/vit-gpt2-coco-en",
]

# ---- Stage 2: 10 story-generation candidates (pipeline task: text-generation) ----
# prompt_style: "chat" = instruction model with chat template, "tinystories" / "genre" / "plain" = raw text models
STORY_MODEL_CANDIDATES = [
    {"model_name": "Qwen/Qwen3-0.6B",                       "prompt_style": "qwen3_chat"},
    {"model_name": "Qwen/Qwen2.5-0.5B-Instruct",            "prompt_style": "chat"},
    {"model_name": "Qwen/Qwen2.5-1.5B-Instruct",            "prompt_style": "chat"},
    {"model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",   "prompt_style": "chat"},
    {"model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",   "prompt_style": "chat"},
    {"model_name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",   "prompt_style": "chat"},
    {"model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",    "prompt_style": "chat"},
    {"model_name": "google/flan-t5-base",                   "prompt_style": "t5"},
    {"model_name": "google/flan-t5-small",                  "prompt_style": "t5"},
    {"model_name": "roneneldan/TinyStories-Instruct-33M",   "prompt_style": "tinystories"},
]
# The app shrinks models to int8 (PyTorch dynamic quantization) for speed on Streamlit Cloud: test the same way.
SHRINK_MODELS_WITH_INT8 = True

# ---- Themes (identical wording to app.py); keys match the "theme" column of test_image_ground_truth.csv ----
THEME_STORY_INSTRUCTIONS = {
    "Fairy Tale": "Make it a magical fairy tale with kind fairies, castles, wishes or talking animals.",
    "Space Quest": "Make it a friendly space adventure with rockets, twinkly stars, planets or cute friendly aliens.",
    "Gentle Mystery": "Make it a gentle, cosy mystery where the friends find clues and solve a small puzzle. Nothing scary.",
    "Jungle Adventure": "Make it an exciting jungle adventure with exploring, brave friends, animals and a treasure map.",
    "Silly Poem": "Write it as a short, silly rhyming poem with short lines that rhyme and a giggly surprise.",
    "Ocean Magic": "Make it a magical underwater adventure with friendly fish, shiny shells and sparkly coral.",
}
THEME_ZERO_SHOT_LABELS = {
    "Fairy Tale": "magical fairy tale",
    "Space Quest": "space adventure",
    "Gentle Mystery": "mystery with clues",
    "Jungle Adventure": "jungle adventure",
    "Silly Poem": "rhyming poem",
    "Ocean Magic": "underwater ocean story",
}
THEME_NAMES = list(THEME_STORY_INSTRUCTIONS.keys())
STORY_TARGET_WORDS_CYCLE = [75, 50, 100]   # images cycle through these slider values
MINIMUM_STORY_WORDS = 50
MAXIMUM_STORY_WORDS = 100

set_seed(RANDOM_SEED)
print(f"Torch threads: {torch.get_num_threads()}, device: {'CPU' if DEVICE == -1 else 'GPU'}")
""")

# ---------------------------------------------------------------------------
add_markdown_cell("## 2. Shared helper functions (same logic as `app.py`)")
add_code_cell(r"""
# ==============================
# FUNCTION PART (shared with app.py)
# ==============================
def clean_caption(raw_caption):
    '''
    Remove common captioning artefacts (e.g. "arafed", "there is", "illustration of")
    so that the story generator receives a clean description.

    Parameters:
        raw_caption (str): caption produced by the image-to-text model.
    Returns:
        str: cleaned caption in lower case.
    '''
    cleaned_caption = raw_caption.lower().strip()
    unwanted_phrases = ["arafed", "araffe", "arafe", "there is ", "there are ", "this is ",
                        "an image of ", "a picture of ", "a photo of ", "an illustration of ", "illustration of ",
                        " illustration", "a cartoon of ", "cartoon of ", "a painting of ", "painting of ",
                        " background", "a drawing of "]
    for unwanted_phrase in unwanted_phrases:
        cleaned_caption = cleaned_caption.replace(unwanted_phrase, " ")
    # Remove immediately repeated words (some models stutter, e.g. "a dog dog").
    cleaned_caption = re.sub(r"\b(\w+)( \1\b)+", r"\1", cleaned_caption)
    cleaned_caption = re.sub(r"\s+", " ", cleaned_caption).strip(" ,.")
    return cleaned_caption


def build_story_messages(image_caption, theme_name, target_word_count):
    '''
    Build the chat messages (system + user) that ask an instruction model
    for a short, kid-friendly story in the selected theme.
    '''
    system_message = ("You are a kind storyteller for children aged 3 to 10. "
                      "Use short sentences and simple, happy words. "
                      "Never include anything scary, violent, sad or unsafe.")
    user_message = (f"Write a story for young children, about {target_word_count} words long, "
                    f"based on this picture: \"{image_caption}\". "
                    f"Include the things you can see in the picture. "
                    f"{THEME_STORY_INSTRUCTIONS[theme_name]} "
                    f"Start in a fun, surprising way and give it a happy ending. Write only the story, with no title.")
    return [{"role": "system", "content": system_message},
            {"role": "user", "content": user_message}]


def build_plain_story_prompt(image_caption, theme_name, prompt_style):
    '''Build a raw-text prompt for non-chat story models (TinyStories, genre GPT-2).'''
    if prompt_style == "tinystories":
        return f"Summary: {image_caption}. It is a {THEME_ZERO_SHOT_LABELS[theme_name]} for little kids.\nStory: "
    if prompt_style == "t5":
        return (f"Write a happy children's story of about 75 words about {image_caption}. "
                f"{THEME_STORY_INSTRUCTIONS[theme_name]}")
    return f"Once upon a time, there was {image_caption}."


def trim_story_to_word_limit(story_text, target_word_count, is_poem):
    '''
    Trim a generated story to complete sentences (or poem lines) close to the
    target word count, always staying inside 50-100 words when possible.

    Returns:
        tuple(str, int): the trimmed story and its word count.
    '''
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

    upper_word_limit = min(MAXIMUM_STORY_WORDS, target_word_count + 10)
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
    final_word_count = len(trimmed_story.split())
    return trimmed_story, final_word_count


def concept_recall(text_value, expected_concepts):
    '''
    Share of expected object groups (e.g. "forest|woods") that are named in the text (0-1).
    '''
    lower_text = str(text_value).lower()
    concept_groups = [group for group in str(expected_concepts).split(";") if group.strip()]
    found_groups = [group for group in concept_groups
                    if any(re.search(rf"\b{re.escape(synonym.strip().lower())}", lower_text) for synonym in group.split("|"))]
    return len(found_groups) / max(len(concept_groups), 1)


def shrink_model_to_int8(loaded_pipeline):
    '''Same int8 dynamic quantization the app uses (returns the precision used).'''
    if not SHRINK_MODELS_WITH_INT8:
        return "float32"
    try:
        torch.ao.quantization.quantize_dynamic(loaded_pipeline.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        return "int8"
    except Exception:
        return "float32"


def get_process_memory_mb():
    '''Return the current process memory (resident set size) in MB.'''
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def count_model_parameters_millions(loaded_pipeline):
    '''Return the number of model parameters in millions.'''
    return sum(parameter.numel() for parameter in loaded_pipeline.model.parameters()) / 1e6


def release_memory():
    '''Free memory between models.'''
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("Helper functions ready.")
""")

# ---------------------------------------------------------------------------
add_markdown_cell(r"""
## 3. Load the 10 test images and their expected objects
The same 10 images are shipped with the app in `test_images/` (6 illustrated story worlds + 4 photos).
`test_image_ground_truth.csv` lists the objects a good caption should mention (synonyms separated by `|`).
""")
add_code_cell(r"""
import glob

if not os.path.isdir("test_images") and GITHUB_REPOSITORY_URL:
    !git clone -q {GITHUB_REPOSITORY_URL} taletwinkle_repository
    if os.path.isdir("taletwinkle_repository/test_images"):
        shutil.copytree("taletwinkle_repository/test_images", "test_images")
if not os.path.isdir("test_images"):
    from google.colab import files
    print("Please upload test_images.zip (the test_images folder from the project, zipped).")
    uploaded_files = files.upload()
    with zipfile.ZipFile(io.BytesIO(list(uploaded_files.values())[0])) as uploaded_zip:
        uploaded_zip.extractall(".")
    if not os.path.isdir("test_images"):
        os.makedirs("test_images", exist_ok=True)
        for extracted_path in glob.glob("*.png") + glob.glob("*.jpg") + glob.glob("*.csv"):
            shutil.move(extracted_path, "test_images")

ground_truth_table = pd.read_csv("test_images/test_image_ground_truth.csv")
test_items = []
for ground_truth_row in ground_truth_table.itertuples():
    test_items.append({"file_name": ground_truth_row.image_name, "theme": ground_truth_row.theme,
                       "expected_concepts": ground_truth_row.expected_concepts,
                       "image": Image.open(os.path.join("test_images", ground_truth_row.image_name)).convert("RGB")})
shutil.copytree("test_images", TEST_IMAGE_FOLDER, dirs_exist_ok=True)
reference_table = ground_truth_table
print(f"Loaded {len(test_items)} test images.")
display(ground_truth_table)
""")

# ---------------------------------------------------------------------------
add_markdown_cell(r"""
## 4. Stage 1: image captioning, 10 models × 10 images
Accuracy = % of the expected objects (ground truth) named in the caption. A reference-free **CLIP score**
(`openai/clip-vit-base-patch32`, image-caption match) is also reported. Captions are repetition-controlled exactly as in the app.
Timing: model loaded once (load time reported separately), then one warm-up call, then each image timed.
""")
add_code_cell(r"""
caption_detail_rows = []
caption_model_rows = []

for caption_model_name in CAPTION_MODEL_CANDIDATES:
    print(f"\n=== Caption model: {caption_model_name}")
    memory_before_mb = get_process_memory_mb()
    try:
        load_start_time = time.perf_counter()
        caption_pipeline = pipeline("image-to-text", model=caption_model_name, device=DEVICE, token=HF_TOKEN)
        parameter_millions = count_model_parameters_millions(loaded_pipeline=caption_pipeline)
        model_precision = shrink_model_to_int8(loaded_pipeline=caption_pipeline)
        load_seconds = time.perf_counter() - load_start_time
        caption_settings = {"max_new_tokens": 30, "repetition_penalty": 1.3, "no_repeat_ngram_size": 3}
        caption_pipeline(test_items[0]["image"], generate_kwargs=caption_settings)   # warm-up
        for test_item in test_items:
            start_time = time.perf_counter()
            caption_output = caption_pipeline(test_item["image"], generate_kwargs=caption_settings)
            elapsed_seconds = time.perf_counter() - start_time
            raw_caption = caption_output[0]["generated_text"]
            caption_detail_rows.append({"model": caption_model_name, "file_name": test_item["file_name"],
                                        "raw_caption": raw_caption, "clean_caption": clean_caption(raw_caption=raw_caption),
                                        "seconds": round(elapsed_seconds, 2)})
            print(f"  {test_item['file_name']}: {raw_caption}  ({elapsed_seconds:.2f} s)")
        caption_model_rows.append({"model": caption_model_name, "status": "ok", "precision": model_precision, "load_seconds": round(load_seconds, 2),
                                   "parameters_millions": round(parameter_millions, 1),
                                   "memory_increase_mb": round(get_process_memory_mb() - memory_before_mb, 1)})
        del caption_pipeline
    except Exception as model_error:
        print(f"  FAILED: {str(model_error)[:300]}")
        caption_model_rows.append({"model": caption_model_name, "status": f"failed: {str(model_error)[:150]}"})
    release_memory()

caption_detail_table = pd.DataFrame(caption_detail_rows)
""")
add_code_cell(r"""
# ---- Score the captions: expected-object accuracy + CLIP score ----
from sentence_transformers import SentenceTransformer, util
from transformers import CLIPModel, CLIPProcessor

similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
concepts_by_file = {test_item["file_name"]: test_item["expected_concepts"] for test_item in test_items}
images_by_file = {test_item["file_name"]: test_item["image"] for test_item in test_items}


def reference_similarity(candidate_text, reference_texts):
    '''Best cosine similarity (0-1) between a text and any reference text.'''
    candidate_embedding = similarity_model.encode(candidate_text, convert_to_tensor=True)
    reference_embeddings = similarity_model.encode(reference_texts, convert_to_tensor=True)
    return float(util.cos_sim(candidate_embedding, reference_embeddings).max())


def clip_image_text_score(pil_image, caption_text):
    '''CLIP cosine similarity x 100 between an image and a caption.'''
    clip_inputs = clip_processor(text=[caption_text], images=pil_image, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        clip_outputs = clip_model(**clip_inputs)
    image_embedding = clip_outputs.image_embeds / clip_outputs.image_embeds.norm(dim=-1, keepdim=True)
    text_embedding = clip_outputs.text_embeds / clip_outputs.text_embeds.norm(dim=-1, keepdim=True)
    return float((image_embedding * text_embedding).sum()) * 100


if not caption_detail_table.empty:
    caption_detail_table["accuracy_similarity"] = [
        round(concept_recall(text_value=row.clean_caption, expected_concepts=concepts_by_file[row.file_name]), 4)
        for row in caption_detail_table.itertuples()]
    caption_detail_table["clip_score"] = [
        round(clip_image_text_score(pil_image=images_by_file[row.file_name], caption_text=row.clean_caption), 2)
        for row in caption_detail_table.itertuples()]

caption_summary_table = pd.DataFrame(caption_model_rows)
if not caption_detail_table.empty:
    caption_means = caption_detail_table.groupby("model").agg(
        accuracy_percent=("accuracy_similarity", lambda values: round(values.mean() * 100, 2)),
        clip_score=("clip_score", lambda values: round(values.mean(), 2)),
        mean_seconds_per_image=("seconds", lambda values: round(values.mean(), 2))).reset_index()
    caption_summary_table = caption_summary_table.merge(caption_means, on="model", how="left")
    successful_rows = caption_summary_table["status"] == "ok"
    best_accuracy = caption_summary_table.loc[successful_rows, "accuracy_percent"].max()
    fastest_seconds = caption_summary_table.loc[successful_rows, "mean_seconds_per_image"].min()
    # Overall score: 70% accuracy (relative to best) + 30% speed (relative to fastest).
    caption_summary_table["overall_score"] = (
        0.7 * caption_summary_table["accuracy_percent"] / best_accuracy
        + 0.3 * fastest_seconds / caption_summary_table["mean_seconds_per_image"]).round(4) * 100
    caption_summary_table = caption_summary_table.sort_values("overall_score", ascending=False).reset_index(drop=True)
    caption_summary_table.insert(0, "rank", range(1, len(caption_summary_table) + 1))

display(caption_summary_table)
BEST_CAPTION_MODEL = caption_summary_table.iloc[0]["model"]
best_caption_by_file = caption_detail_table[caption_detail_table["model"] == BEST_CAPTION_MODEL].set_index("file_name")["clean_caption"].to_dict()
print(f"Best caption model: {BEST_CAPTION_MODEL}")
""")

# ---------------------------------------------------------------------------
add_markdown_cell(r"""
## 5. Stage 2: story generation, 10 models × 10 images
Each image uses its theme from the ground-truth file and a target length (75 / 50 / 100 words, like the slider).
All models receive the **same captions** (from the best caption model) and the **same prompt as the app**, then the app's
trimming rule is applied.

Metrics (each 0–1):
* **image relevance**: 50% expected objects named in the story + 50% similarity of the story to the caption (all-MiniLM-L6-v2)
* **theme match**: zero-shot probability of the requested theme (`facebook/bart-large-mnli`)
* **kid readability**: 1.0 at reading grade ≤ 2, falling to 0 at grade 10 (Flesch-Kincaid, `textstat`)
* **safety**: 1 − toxicity (`unitary/toxic-bert`)
* **length OK**: 1 if 50–100 words after trimming

`kids_story_score = 0.35 relevance + 0.25 theme + 0.20 readability + 0.10 safety + 0.10 length`,
`overall_score = 80% kids_story_score (relative to best) + 20% speed (relative to fastest)`.
""")
add_code_cell(r"""
story_detail_rows = []
story_model_rows = []

for story_candidate in STORY_MODEL_CANDIDATES:
    story_model_name = story_candidate["model_name"]
    prompt_style = story_candidate["prompt_style"]
    print(f"\n=== Story model: {story_model_name}")
    memory_before_mb = get_process_memory_mb()
    try:
        load_start_time = time.perf_counter()
        pipeline_task = "text2text-generation" if prompt_style == "t5" else "text-generation"
        story_pipeline = pipeline(pipeline_task, model=story_model_name, device=DEVICE, token=HF_TOKEN)
        parameter_millions = count_model_parameters_millions(loaded_pipeline=story_pipeline)
        model_precision = shrink_model_to_int8(loaded_pipeline=story_pipeline)
        load_seconds = time.perf_counter() - load_start_time
        end_of_text_id = story_pipeline.tokenizer.eos_token_id
        for image_index, test_item in enumerate(test_items):
            theme_name = test_item["theme"]
            target_word_count = STORY_TARGET_WORDS_CYCLE[image_index % len(STORY_TARGET_WORDS_CYCLE)]
            image_caption = best_caption_by_file[test_item["file_name"]]
            generation_settings = {"max_new_tokens": int(min(100, target_word_count + 10) * 1.45) + 8, "do_sample": True,
                                   "temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.1}
            if prompt_style != "t5":
                generation_settings.update({"pad_token_id": end_of_text_id, "return_full_text": False})
            set_seed(RANDOM_SEED)
            start_time = time.perf_counter()
            if prompt_style in ("chat", "qwen3_chat"):
                chat_template_settings = ({"tokenizer_encode_kwargs": {"enable_thinking": False}}
                                          if prompt_style == "qwen3_chat" else {})
                story_output = story_pipeline(build_story_messages(image_caption=image_caption, theme_name=theme_name,
                                                                   target_word_count=target_word_count),
                                              **chat_template_settings, **generation_settings)
                generated_value = story_output[0]["generated_text"]
                raw_story = generated_value[-1]["content"] if isinstance(generated_value, list) else generated_value
            else:
                plain_prompt = build_plain_story_prompt(image_caption=image_caption, theme_name=theme_name, prompt_style=prompt_style)
                story_output = story_pipeline(plain_prompt, **generation_settings)
                opening_text = plain_prompt if prompt_style == "plain" else ""
                raw_story = f"{opening_text} {story_output[0]['generated_text']}".strip()
            elapsed_seconds = time.perf_counter() - start_time
            final_story, final_word_count = trim_story_to_word_limit(story_text=raw_story, target_word_count=target_word_count,
                                                                     is_poem=(theme_name == "Poem"))
            story_detail_rows.append({"model": story_model_name, "file_name": test_item["file_name"], "theme": theme_name,
                                      "target_words": target_word_count, "caption_used": image_caption,
                                      "story": final_story, "word_count": final_word_count, "seconds": round(elapsed_seconds, 2)})
            print(f"  {test_item['file_name']} [{theme_name}, {target_word_count}w] {final_word_count} words, {elapsed_seconds:.2f} s")
        story_model_rows.append({"model": story_model_name, "status": "ok", "precision": model_precision, "load_seconds": round(load_seconds, 2),
                                 "parameters_millions": round(parameter_millions, 1),
                                 "memory_increase_mb": round(get_process_memory_mb() - memory_before_mb, 1)})
        del story_pipeline
    except Exception as model_error:
        print(f"  FAILED: {str(model_error)[:300]}")
        story_model_rows.append({"model": story_model_name, "status": f"failed: {str(model_error)[:150]}"})
    release_memory()

story_detail_table = pd.DataFrame(story_detail_rows)
""")
add_code_cell(r"""
# ---- Score the stories ----
import textstat

zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=DEVICE)
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", top_k=None, device=DEVICE)
theme_label_list = list(THEME_ZERO_SHOT_LABELS.values())


def theme_match_probability(story_text, theme_name):
    '''Zero-shot probability that the story matches the requested theme.'''
    zero_shot_result = zero_shot_classifier(story_text, candidate_labels=theme_label_list,
                                            hypothesis_template="This children's story is a {}.")
    label_scores = dict(zip(zero_shot_result["labels"], zero_shot_result["scores"]))
    return float(label_scores[THEME_ZERO_SHOT_LABELS[theme_name]])


def kid_readability_score(story_text):
    '''1.0 at reading grade <= 2, falling linearly to 0 at grade 10.'''
    reading_grade = textstat.flesch_kincaid_grade(story_text)
    return float(np.clip(1 - max(reading_grade - 2, 0) / 8, 0, 1)), reading_grade


def safety_score(story_text):
    '''1 minus the toxic-bert "toxic" probability.'''
    toxicity_output = toxicity_classifier(story_text[:2000])
    toxicity_labels = toxicity_output[0] if isinstance(toxicity_output[0], list) else toxicity_output
    toxic_probability = max([label["score"] for label in toxicity_labels if label["label"].lower() == "toxic"] or [0.0])
    return 1 - float(toxic_probability)


if not story_detail_table.empty:
    relevance_values, theme_values, readability_values, grade_values, safety_values = [], [], [], [], []
    for row in story_detail_table.itertuples():
        story_text = row.story if row.story.strip() else "(empty)"
        relevance_values.append(0.5 * concept_recall(text_value=story_text, expected_concepts=concepts_by_file[row.file_name])
                                + 0.5 * reference_similarity(candidate_text=story_text, reference_texts=[row.caption_used]))
        theme_values.append(theme_match_probability(story_text=story_text, theme_name=row.theme))
        readability_value, reading_grade = kid_readability_score(story_text=story_text)
        readability_values.append(readability_value)
        grade_values.append(reading_grade)
        safety_values.append(safety_score(story_text=story_text))
    story_detail_table["image_relevance"] = np.round(relevance_values, 4)
    story_detail_table["theme_match"] = np.round(theme_values, 4)
    story_detail_table["reading_grade"] = np.round(grade_values, 2)
    story_detail_table["kid_readability"] = np.round(readability_values, 4)
    story_detail_table["safety"] = np.round(safety_values, 4)
    story_detail_table["fixed_opening_repeated"] = story_detail_table["story"].str.contains("picture came alive", case=False).astype(int)
    story_detail_table["length_ok"] = story_detail_table["word_count"].between(MINIMUM_STORY_WORDS, MAXIMUM_STORY_WORDS).astype(int)
    story_detail_table["kids_story_score"] = (0.35 * story_detail_table["image_relevance"] + 0.25 * story_detail_table["theme_match"]
                                              + 0.20 * story_detail_table["kid_readability"] + 0.10 * story_detail_table["safety"]
                                              + 0.10 * story_detail_table["length_ok"]).round(4)
    story_detail_table["your_rating_1_to_5"] = ""      # optional: read the stories and rate them yourself

story_summary_table = pd.DataFrame(story_model_rows)
if not story_detail_table.empty:
    story_means = story_detail_table.groupby("model").agg(
        kids_story_score=("kids_story_score", lambda values: round(values.mean() * 100, 2)),
        image_relevance=("image_relevance", lambda values: round(values.mean() * 100, 2)),
        theme_match=("theme_match", lambda values: round(values.mean() * 100, 2)),
        reading_grade=("reading_grade", lambda values: round(values.mean(), 2)),
        safety=("safety", lambda values: round(values.mean() * 100, 2)),
        length_ok_percent=("length_ok", lambda values: round(values.mean() * 100, 2)),
        mean_words=("word_count", lambda values: round(values.mean(), 2)),
        mean_seconds_per_story=("seconds", lambda values: round(values.mean(), 2))).reset_index()
    story_summary_table = story_summary_table.merge(story_means, on="model", how="left")
    successful_rows = story_summary_table["status"] == "ok"
    best_kids_score = story_summary_table.loc[successful_rows, "kids_story_score"].max()
    fastest_seconds = story_summary_table.loc[successful_rows, "mean_seconds_per_story"].min()
    story_summary_table["overall_score"] = (
        0.8 * story_summary_table["kids_story_score"] / best_kids_score
        + 0.2 * fastest_seconds / story_summary_table["mean_seconds_per_story"]).round(4) * 100
    story_summary_table = story_summary_table.sort_values("overall_score", ascending=False).reset_index(drop=True)
    story_summary_table.insert(0, "rank", range(1, len(story_summary_table) + 1))
    story_detail_table.to_csv(os.path.join(STORY_FOLDER, "all_generated_stories.csv"), index=False)

display(story_summary_table)
BEST_STORY_MODEL = story_summary_table.iloc[0]["model"]
print(f"Best story model: {BEST_STORY_MODEL}")
del zero_shot_classifier, toxicity_classifier
release_memory()
""")

# ---------------------------------------------------------------------------
add_markdown_cell(r"""
## 6. Stage 3: text-to-speech, 10+ voice options
Each engine speaks the same stories (written by the best story model). Measured:
* **synthesis seconds** and **real-time factor** (seconds needed per second of audio; lower is better)
* **intelligibility** = 1 − word error rate when `openai/whisper-base.en` transcribes the audio back to text
* **your listening score**: open `tts_samples/` and rate each voice 1–5 in the Excel sheet (clarity, warmth, fit for kids)

gTTS needs internet (Streamlit Cloud has it). Piper and Kokoro use Hugging Face-hosted voice checkpoints.
The production app selects Piper because its quantized ONNX voices are much faster on CPU while still offering
distinct male/female checkpoints. Child-style choices use a clearly labelled pitch effect rather than claiming
that an adult training voice is a real child.
""")
add_code_cell(r"""
import jiwer

tts_test_stories = []
if not story_detail_table.empty:
    best_story_rows = story_detail_table[(story_detail_table["model"] == BEST_STORY_MODEL) & (story_detail_table["length_ok"] == 1)]
    tts_test_stories = best_story_rows["story"].head(TTS_TEST_STORY_COUNT).tolist()
if len(tts_test_stories) < TTS_TEST_STORY_COUNT:
    tts_test_stories.append("Once upon a time, a little puppy found a shiny red ball in the park. "
                            "He rolled it to his friend, a fluffy white cat. They laughed and played all afternoon. "
                            "When the sun went down, they curled up together under a big green tree. "
                            "The puppy whispered, thank you for playing with me. The cat purred happily, "
                            "and they both fell fast asleep, dreaming of more fun tomorrow.")
tts_test_stories = tts_test_stories[:TTS_TEST_STORY_COUNT]


def audio_from_gtts(story_text, top_level_domain):
    '''Synthesize speech with Google Text-to-Speech (accent set by the domain).'''
    from gtts import gTTS
    mp3_buffer = io.BytesIO()
    gTTS(text=story_text, lang="en", tld=top_level_domain).write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)
    audio_samples, sample_rate = sf.read(mp3_buffer, dtype="float32")
    return audio_samples, sample_rate


KOKORO_PIPELINES = {}
def audio_from_kokoro(story_text, voice_name):
    '''Synthesize speech with hexgrad/Kokoro-82M (voice prefix a=American, b=British).'''
    from kokoro import KPipeline
    language_code = voice_name[0]
    if language_code not in KOKORO_PIPELINES:
        KOKORO_PIPELINES[language_code] = KPipeline(lang_code=language_code)
    audio_chunks = [np.asarray(chunk_audio) for _, _, chunk_audio in KOKORO_PIPELINES[language_code](story_text, voice=voice_name, speed=1.0)]
    return np.concatenate(audio_chunks), 24000


PIPER_VOICES = {}
def audio_from_piper(story_text, model_file):
    '''Synthesize speech locally with a Piper checkpoint hosted on Hugging Face.'''
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice
    if model_file not in PIPER_VOICES:
        model_path = hf_hub_download(repo_id="rhasspy/piper-voices", filename=model_file)
        config_path = hf_hub_download(repo_id="rhasspy/piper-voices", filename=f"{model_file}.json")
        PIPER_VOICES[model_file] = PiperVoice.load(model_path=model_path, config_path=config_path, use_cuda=False)
    chunks = list(PIPER_VOICES[model_file].synthesize(story_text))
    return np.concatenate([chunk.audio_float_array for chunk in chunks]), chunks[0].sample_rate


HF_TTS_PIPELINES = {}
def audio_from_hf_pipeline(story_text, model_name, forward_parameters=None):
    '''Synthesize speech with a transformers text-to-speech pipeline.'''
    if model_name not in HF_TTS_PIPELINES:
        HF_TTS_PIPELINES[model_name] = pipeline("text-to-speech", model=model_name, device=DEVICE, token=HF_TOKEN)
    speech_output = HF_TTS_PIPELINES[model_name](story_text, forward_params=forward_parameters or {})
    return np.squeeze(speech_output["audio"]), speech_output["sampling_rate"]


SPEECHT5_EMBEDDING = {}
def audio_from_speecht5(story_text):
    '''Synthesize speech with microsoft/speecht5_tts using a CMU Arctic female x-vector.'''
    if "embedding" not in SPEECHT5_EMBEDDING:
        from huggingface_hub import hf_hub_download
        zip_path = hf_hub_download(repo_id="Matthijs/cmu-arctic-xvectors", filename="spkrec-xvect.zip", repo_type="dataset")
        with zipfile.ZipFile(zip_path) as vector_zip:
            female_vector_name = [name for name in vector_zip.namelist() if "slt" in name and name.endswith(".npy")][0]
            SPEECHT5_EMBEDDING["embedding"] = torch.tensor(np.load(io.BytesIO(vector_zip.read(female_vector_name)))).unsqueeze(0)
    return audio_from_hf_pipeline(story_text=story_text, model_name="microsoft/speecht5_tts",
                                  forward_parameters={"speaker_embeddings": SPEECHT5_EMBEDDING["embedding"]})


def audio_from_pyttsx3(story_text):
    '''Synthesize speech offline with pyttsx3 (espeak on Linux).'''
    import pyttsx3
    speech_engine = pyttsx3.init()
    temporary_path = os.path.join(OUTPUT_FOLDER, "pyttsx3_temp.wav")
    speech_engine.save_to_file(story_text, temporary_path)
    speech_engine.runAndWait()
    audio_samples, sample_rate = sf.read(temporary_path, dtype="float32")
    return audio_samples, sample_rate


TTS_CANDIDATES = [
    {"engine": "gTTS UK English (co.uk)",           "synthesize": lambda text: audio_from_gtts(story_text=text, top_level_domain="co.uk")},
    {"engine": "gTTS US English (com)",             "synthesize": lambda text: audio_from_gtts(story_text=text, top_level_domain="com")},
    {"engine": "gTTS Australian English (com.au)",  "synthesize": lambda text: audio_from_gtts(story_text=text, top_level_domain="com.au")},
    {"engine": "gTTS Indian English (co.in)",       "synthesize": lambda text: audio_from_gtts(story_text=text, top_level_domain="co.in")},
    {"engine": "rhasspy/piper-voices US male Ryan", "synthesize": lambda text: audio_from_piper(story_text=text, model_file="en/en_US/ryan/medium/en_US-ryan-medium.onnx")},
    {"engine": "rhasspy/piper-voices UK female Alba", "synthesize": lambda text: audio_from_piper(story_text=text, model_file="en/en_GB/alba/medium/en_GB-alba-medium.onnx")},
    {"engine": "hexgrad/Kokoro-82M US female (af_heart)", "synthesize": lambda text: audio_from_kokoro(story_text=text, voice_name="af_heart")},
    {"engine": "hexgrad/Kokoro-82M UK female (bf_emma)",  "synthesize": lambda text: audio_from_kokoro(story_text=text, voice_name="bf_emma")},
    {"engine": "hexgrad/Kokoro-82M UK male (bm_george)",  "synthesize": lambda text: audio_from_kokoro(story_text=text, voice_name="bm_george")},
    {"engine": "facebook/mms-tts-eng",              "synthesize": lambda text: audio_from_hf_pipeline(story_text=text, model_name="facebook/mms-tts-eng")},
    {"engine": "kakao-enterprise/vits-ljs",         "synthesize": lambda text: audio_from_hf_pipeline(story_text=text, model_name="kakao-enterprise/vits-ljs")},
    {"engine": "microsoft/speecht5_tts",            "synthesize": lambda text: audio_from_speecht5(story_text=text)},
    {"engine": "suno/bark-small",                   "synthesize": lambda text: audio_from_hf_pipeline(story_text=text, model_name="suno/bark-small")},
    {"engine": "pyttsx3 (offline espeak)",          "synthesize": lambda text: audio_from_pyttsx3(story_text=text)},
]

tts_detail_rows = []
for tts_candidate in TTS_CANDIDATES:
    engine_name = tts_candidate["engine"]
    print(f"\n=== Voice engine: {engine_name}")
    stories_for_engine = tts_test_stories[:1] if "bark" in engine_name else tts_test_stories   # Bark is very slow on CPU
    try:
        tts_candidate["synthesize"]("Hello there.")          # warm-up (loads the model once)
        for story_index, story_text in enumerate(stories_for_engine, start=1):
            start_time = time.perf_counter()
            audio_samples, sample_rate = tts_candidate["synthesize"](story_text)
            elapsed_seconds = time.perf_counter() - start_time
            audio_samples = np.asarray(audio_samples, dtype=np.float32)
            if audio_samples.ndim > 1:
                audio_samples = audio_samples.mean(axis=1)
            audio_duration_seconds = len(audio_samples) / sample_rate
            safe_engine_name = re.sub(r"[^A-Za-z0-9]+", "_", engine_name).strip("_")
            sample_path = os.path.join(TTS_SAMPLE_FOLDER, f"{safe_engine_name}_story{story_index}.wav")
            sf.write(sample_path, audio_samples, sample_rate)
            tts_detail_rows.append({"engine": engine_name, "story_index": story_index, "story_text": story_text,
                                    "seconds": round(elapsed_seconds, 2), "audio_seconds": round(audio_duration_seconds, 2),
                                    "real_time_factor": round(elapsed_seconds / max(audio_duration_seconds, 0.01), 3),
                                    "sample_file": os.path.basename(sample_path)})
            print(f"  story {story_index}: {elapsed_seconds:.2f} s for {audio_duration_seconds:.2f} s of audio")
    except Exception as engine_error:
        print(f"  FAILED: {str(engine_error)[:300]}")
        tts_detail_rows.append({"engine": engine_name, "story_index": 0, "status": f"failed: {str(engine_error)[:150]}"})
    release_memory()

tts_detail_table = pd.DataFrame(tts_detail_rows)
""")
add_code_cell(r"""
# ---- Intelligibility: transcribe every sample back with Whisper and compute word error rate ----
speech_recognizer = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=DEVICE)


def normalize_for_wer(text_value):
    '''Lower-case and strip punctuation so WER only counts real word mistakes.'''
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", str(text_value).lower())).strip()


word_error_rates = []
for row in tts_detail_table.itertuples():
    if getattr(row, "sample_file", None) is None or pd.isna(row.sample_file):
        word_error_rates.append(np.nan)
        continue
    transcription = speech_recognizer(os.path.join(TTS_SAMPLE_FOLDER, row.sample_file),
                                      chunk_length_s=30, return_timestamps=False)["text"]
    word_error_rates.append(jiwer.wer(normalize_for_wer(row.story_text), normalize_for_wer(transcription)))
tts_detail_table["word_error_rate"] = np.round(word_error_rates, 4)
tts_detail_table["intelligibility_percent"] = ((1 - tts_detail_table["word_error_rate"].clip(0, 1)) * 100).round(2)

tts_summary_table = tts_detail_table.dropna(subset=["word_error_rate"]).groupby("engine").agg(
    intelligibility_percent=("intelligibility_percent", lambda values: round(values.mean(), 2)),
    mean_seconds=("seconds", lambda values: round(values.mean(), 2)),
    real_time_factor=("real_time_factor", lambda values: round(values.mean(), 3))).reset_index()
fastest_seconds = tts_summary_table["mean_seconds"].min()
tts_summary_table["overall_score"] = (0.6 * tts_summary_table["intelligibility_percent"] / tts_summary_table["intelligibility_percent"].max()
                                      + 0.4 * fastest_seconds / tts_summary_table["mean_seconds"]).round(4) * 100
tts_summary_table["your_listening_score_1_to_5"] = ""
failed_engines = (tts_detail_table[tts_detail_table["status"].notna()][["engine", "status"]]
                  if "status" in tts_detail_table.columns else pd.DataFrame())
tts_summary_table = tts_summary_table.sort_values("overall_score", ascending=False).reset_index(drop=True)
tts_summary_table.insert(0, "rank", range(1, len(tts_summary_table) + 1))
display(tts_summary_table)
if not failed_engines.empty:
    display(failed_engines)
""")

# ---------------------------------------------------------------------------
add_markdown_cell("## 7. Export everything (Excel + zip download)")
add_code_cell(r"""
import platform
import transformers

environment_table = pd.DataFrame([
    {"item": "python", "value": platform.python_version()},
    {"item": "transformers", "value": transformers.__version__},
    {"item": "torch", "value": torch.__version__},
    {"item": "device", "value": "CPU" if DEVICE == -1 else "GPU"},
    {"item": "cpu_threads", "value": torch.get_num_threads()},
    {"item": "processor", "value": platform.processor() or platform.machine()},
    {"item": "best_caption_model", "value": BEST_CAPTION_MODEL},
    {"item": "best_story_model", "value": BEST_STORY_MODEL},
    {"item": "best_tts_engine_by_score", "value": tts_summary_table.iloc[0]["engine"] if not tts_summary_table.empty else "n/a"},
])

excel_path = os.path.join(OUTPUT_FOLDER, "benchmark_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as excel_writer:
    environment_table.to_excel(excel_writer, sheet_name="Environment", index=False)
    caption_summary_table.to_excel(excel_writer, sheet_name="Caption_Summary", index=False)
    caption_detail_table.to_excel(excel_writer, sheet_name="Caption_Details", index=False)
    story_summary_table.to_excel(excel_writer, sheet_name="Story_Summary", index=False)
    story_detail_table.to_excel(excel_writer, sheet_name="Story_Details", index=False)
    tts_summary_table.to_excel(excel_writer, sheet_name="TTS_Summary", index=False)
    tts_detail_table.to_excel(excel_writer, sheet_name="TTS_Details", index=False)
    reference_table.to_excel(excel_writer, sheet_name="Test_Images", index=False)

archive_path = shutil.make_archive("taletwinkle_benchmark_results", "zip", OUTPUT_FOLDER)
print(f"Saved {excel_path} and {archive_path}")
print(f"Best caption model : {BEST_CAPTION_MODEL}")
print(f"Best story model   : {BEST_STORY_MODEL}")
try:
    from google.colab import files
    files.download(archive_path)
except Exception:
    print("Download the zip from the Files panel on the left.")
""")

# ---------------------------------------------------------------------------
notebook_document = {
    "cells": notebook_cells,
    "metadata": {"colab": {"provenance": []},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 0,
}
with open("TaleTwinkle_Model_Benchmark.ipynb", "w", encoding="utf-8") as notebook_file:
    json.dump(notebook_document, notebook_file, indent=1, ensure_ascii=False)
print(f"Wrote notebook with {len(notebook_cells)} cells.")
