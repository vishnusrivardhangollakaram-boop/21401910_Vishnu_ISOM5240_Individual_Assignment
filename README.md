# ✨ TaleTwinkle Picture Stories for Kids

**ISOM5240 Deep Learning Business Applications with Python: Individual Assignment**
Storytelling application using Hugging Face Transformers pipelines, deployed on Streamlit Cloud.

> 🔗 **Live app:** `https://<your-app-name>.streamlit.app` ← *replace with your Streamlit Cloud URL*
> 📦 **Files:** `app.py` · `requirements.txt` · `packages.txt` · `.streamlit/config.toml` · `assets/backgrounds/` · `test_images/` · `benchmark/` · `tests/`

A child (aged 3–10) drops a picture onto the screen. TaleTwinkle shows the picture, describes it, writes a short story
about it **word by word on screen** in the chosen story world, and reads it aloud in the chosen storyteller voice.
No button needs to be pressed.

---

## 1. Business and user requirements

| Requirement (brief + class) | How TaleTwinkle meets it |
|---|---|
| Python app using Hugging Face **Transformers pipelines** | `pipeline("image-to-text")` → `pipeline("text-generation")` → speech: gTTS (named in the brief) + Hugging Face `pipeline("text-to-speech")` |
| **Image input** uploaded by the user | Big drag-and-drop / tap-to-choose box in the centre (JPG, JPEG, PNG, WEBP) |
| **Story of 50–100 words** from details in the image | An instruction-following model gets the caption, theme and length. Output is trimmed to whole sentences within 50–100 words; the **slider** sets the length (default 75) |
| **Text-to-speech** | Narration plays automatically: 5 real voices + 4 cartoon characters, speed 0.5×–2× |
| **Deployed on Streamlit Cloud** | Public app, `app.py` + `requirements.txt` in a GitHub repo |
| **Users are 3–10 years old** | No button needed; big fonts; illustrated worlds; simple words; kid-safety filter; the picture, story and audio player are always on screen |
| **Short attention span → fast** | Models load and warm up once at start-up; int8 models; the story streams word by word; generation stops once the story is long enough; each sentence is voiced **while the rest is still being written** |
| Picture must be shown | The picture appears first, scaled to **460 px** on the longest side (small pictures scaled up, big pictures scaled down) |
| Accent for Hong Kong children | Default voice: **Story Lady Lily**, a real British female voice |

## 2. Features

| Area | What the child sees |
|---|---|
| **Centre** | 📸 Drop-a-picture box → the picture → 📖 the story, appearing word by word → 🎧 narration that plays by itself (the player shows **pause** while playing and **play** after it ends) → 🔄 *Tell me another story!* |
| **Left: story worlds** | 🏰 Fairy Tale · 🚀 Space Quest · 🔎 Gentle Mystery · 🦜 Jungle Adventure · 🎵 Silly Poem · 🐠 Ocean Magic. Each changes the **illustrated background, floating emojis, mascot and story style** |
| **Right: controls** | 📏 Story size (50–100 words, default 75) · 🐢 Voice speed 🐇 (0.5×–2×, default 1×) · 🗣️ Storyteller voice with an animated avatar |
| **Voices** | 👩‍🏫 Story Lady Lily (British, default) · 🦜 Polly the Parrot · 🐱 Whiskers the Kitten · 👩 Aunt Amy (American) · 👩 Aunty Chloe (Australian) · 👩‍🏫 Teacher Priya (Indian English) · 👨 Uncle Max (man's voice) · 🐻 Bruno the Bear · 🤖 Robo Beep |
| **Before a picture** | A friendly greeting from the chosen storyteller (press ▶ to hear each voice) |
| **Celebration** | 🎈 balloons when a story is ready |
| **For grown-ups** | Expandable panel: timings for every stage (2 decimals), the caption, the models used and their precision, download buttons for the story text and audio |
| **Never breaks** | Friendly message for bad files. Safe backup story if the story model fails or writes something unsuitable. Local Hugging Face voice if Google TTS is unreachable |

## 3. How it works

```
picture ─► [1] image-to-text ─► caption ─► [2] text-generation (streamed) ─► 50-100 word story ─► [3] voice ─► autoplay
           blip-image-captioning-base       Qwen2.5-0.5B-Instruct                                   gTTS / mms-tts-eng
           int8, repetition-controlled      int8, theme + length prompt,        each finished sentence is sent to
                                            early stop, trim + safety check     the voice engine in the background
```

**Why the speech is almost instant:** as soon as a sentence is finished on screen, it is sent to the voice engine in a
background thread. Google voices for several sentences are requested in parallel. By the time the last sentence is written,
most of the narration already exists. The sentences are then joined, the character effect and speed are applied, and the audio plays.

**Voices:** Lily, Amy, Chloe and Priya are **real, unaltered** Google voices (UK, US, AU and Indian English).
Uncle Max is the **real** male voice of `facebook/mms-tts-eng`. Only the cartoon characters (parrot, kitten, bear, robot)
use pitch/tempo effects, and they are meant to sound like cartoons. The speed slider uses WSOLA time-stretching,
so **speed changes do not change the pitch**.

## 4. Model selection: 10 candidates per stage

Every candidate has a Hugging Face **model card** and a **"Use this model → Transformers"** snippet, and fits the task.
All tests use the **same 10 test images** in `test_images/` (6 illustrated worlds + 4 photos, with expected objects in
`test_image_ground_truth.csv`), on **CPU**, as on Streamlit Cloud.

### 4.1 Image captioning: speed and accuracy (`benchmark/run1_initial_screening/`)
Accuracy = % of expected objects named in the caption. Overall = 70 % accuracy (relative to best) + 30 % speed (relative to fastest).

| Rank | Model | Accuracy % | Seconds / image | Overall |
|---|---|---|---|---|
| 1 | `Salesforce/blip-image-captioning-base` ✅ | 29.17 | 0.55 | 76.08 |
| 2 | `microsoft/git-base-textcaps` | 35.83 | 2.78 | 73.80 |
| 3 | `Salesforce/blip-image-captioning-large` | 34.17 | 1.74 | 72.82 |
| 4 | `microsoft/git-base-coco` | 32.50 | 1.79 | 69.39 |
| 5 | `ydshieh/vit-gpt2-coco-en` | 18.33 | 0.35 | 65.81 |
| 6 | `microsoft/git-large-textcaps` | 30.83 | 9.47 | 61.35 |
| 7 | `nlpconnect/vit-gpt2-image-captioning` | 18.33 | 0.54 | 55.47 |
| 8 | `microsoft/git-base` | 22.50 | 2.50 | 48.18 |
| 9 | `microsoft/git-large-coco` | 21.67 | 7.02 | 43.83 |
| 10 | `microsoft/git-large` | 12.50 | 6.51 | 26.04 |

**Choice:** BLIP-base, the best accuracy for its speed (3–5× faster than the more accurate models). Its known weaknesses
(word loops such as "jungle jungle jungle", and words like "painting of" or "illustration") are fixed with
`repetition_penalty` / `no_repeat_ngram_size` and a caption-cleaning function.

### 4.2 Story generation: image relevance and kid suitability by theme
**Run 1 (`benchmark/run1_initial_screening/`)** tested small story models that do **not** follow instructions:

| Rank | Model | Overall % | Image relevance % | Theme match % | Mean words | Seconds |
|---|---|---|---|---|---|---|
| 1 | `google/flan-t5-base` | 48.42 | 21.67 | 45.00 | 85.90 | 4.49 |
| 2 | `google/flan-t5-small` | 42.19 | 18.33 | 30.00 | 88.00 | 1.60 |
| 3 | `roneneldan/TinyStories-Instruct-28M` | 37.56 | 6.67 | 10.00 | 74.60 | 0.72 |
| 5 | `roneneldan/TinyStories-33M` | 32.44 | 3.33 | 0.00 | 21.80 | 0.33 |
| … | 6 more TinyStories / distilgpt2 variants | ≤ 31.66 | ≤ 5.83 | ≤ 10.00 | | |

**Finding:** these models ignore the picture (image relevance ≤ 22 %) and the theme. FLAN-T5 repeats sentences. Using them
needs a hard-coded opening such as *"Once upon a time, the picture came alive: …"* plus ready-made filler sentences,
so every story sounds the same. **Decision:** switch to **instruction-tuned chat models**, which follow the prompt
(picture + theme + length + kid-safe style).

**Run 2 (`benchmark/TaleTwinkle_Model_Benchmark.ipynb`)** compares 10 candidates with the app's exact prompt and int8 setting:
Qwen2.5-0.5B/1.5B-Instruct, SmolLM2-135M/360M/1.7B-Instruct, TinyLlama-1.1B-Chat, FLAN-T5 base/small, TinyStories-33M and
TinyStories-Instruct-33M. Kids story score = 35 % image relevance + 25 % theme match + 20 % reading level + 10 % safety + 10 % length.

| Rank | Model | Kids score | Relevance % | Theme % | Reading grade | Length OK % | Seconds / story | Overall |
|---|---|---|---|---|---|---|---|---|
| ⏳ | *Run 2 results are added here from `benchmark_results.xlsx`* | | | | | | | |

**Choice:** `Qwen/Qwen2.5-0.5B-Instruct`, the smallest model that reliably follows the picture, theme and length instructions
and fits Streamlit Cloud's memory in int8 (about 0.6 GB).

### 4.3 Text-to-speech: audio quality
| Engine | Voices | Quality | Speed | Decision |
|---|---|---|---|---|
| **gTTS** (`co.uk`, `us`, `com.au`, `co.in`) | Real female UK, US, AU, Indian English | Natural | ~1 s (parallel sentences) | ✅ Lily (default), Amy, Chloe, Priya, cartoon voices |
| **`facebook/mms-tts-eng`** | One real male voice | Clear | Fast, local | ✅ Uncle Max, Bruno; offline backup |
| `microsoft/speecht5_tts` | x-vector speakers | Good | Slower, needs speaker embeddings | ✗ |
| `suno/bark-small` | Expressive presets | Very good | Far too slow on CPU | ✗ |
| `hexgrad/Kokoro-82M` | US/UK male + female | Very good | Moderate | ✗ not a Transformers pipeline; extra system packages |
| `kakao-enterprise/vits-ljs` | One US female | Good | Fast | ✗ one voice only |
| `espnet/kan-bayashi_ljspeech_vits`, `facebook/fastspeech2-en-ljspeech` | One voice | Good | Medium | ✗ heavy frameworks |
| `pyttsx3` | System voices | Robotic | Fast | ✗ |
| *Pitch-shifted MMS personas (earlier prototype)* | 21 "voices" from one male voice | Poor, unnatural | | ✗ replaced by real voices |

Intelligibility (Whisper word error rate) and speed for these engines are measured in stage 3 of the notebook.

## 5. Performance on Streamlit Cloud
The label under the story shows *"ready in X s"*, and *For grown-ups* shows every stage. Measured on the deployed app:

| Test picture | Theme / words | Picture → words (s) | First story words (s) | Whole story (s) | Voice ready (s) |
|---|---|---|---|---|---|
| ⏳ | | | | | |

## 6. Testing
### 6.1 Automated offline tests: `python tests/run_offline_tests.py` (43 / 43 passed)
Stand-ins for Streamlit, Transformers, torch and gTTS (`tests/stubs/`) run every function and the whole screen flow without
downloading models. Full table: `tests/offline_test_results.md`. Highlights:

| Area | Checks (all ✅) |
|---|---|
| Pictures | Small pictures scaled up (120×80 → 460×307), large scaled down (4000×3000 → 460×345), transparent PNG on white |
| Captions | "jungle jungle jungle…" → "jungle"; "underwater background with corals" → "underwater with corals"; "illustration" / "painting of" removed |
| Story | Trimmed to 53 / 80 / 93 words for slider 50 / 75 / 100, ending on a full sentence; poem line breaks kept; unsafe words blocked; backup story is safe and has no fixed opening |
| Speed features | Sentences queued for speech while writing (first sentence voiced 0.94 s before the story ended); **early stop: 87 words generated instead of 360**; models shrunk to int8 |
| Voices | 0.5× / 2× speed keep pitch at 220.00 Hz; parrot 311.00 Hz (expected 311.13); bear 179.50 Hz (expected 179.73); default is the real British lady; order Lily → Parrot → Kitten |
| Screen flow | Placeholders + greeting player before upload; story card shown **before** the audio; narration autoplays; voice change re-records audio only; theme change writes a new story; no music anywhere |
| Never breaks | Bad file → friendly error; story crash → backup story; unsafe story → backup; Google TTS offline → Hugging Face voice |

### 6.2 Manual functional test on the deployed app
| # | Check | Result |
|---|---|---|
| 1 | App opens; placeholders, 6 worlds, sliders, voice list and greeting player visible | ⏳ |
| 2 | Drop a picture → picture at preferred size, story appears word by word, narration starts by itself | ⏳ |
| 3 | Story is 50–100 words, simple, about the picture, and in the chosen world's style | ⏳ |
| 4 | Story size 50 / 75 / 100 → matching lengths | ⏳ |
| 5 | Each world changes the background, emojis, mascot and story style | ⏳ |
| 6 | Voice speed 0.5× and 2× → slower / faster, same pitch | ⏳ |
| 7 | All 9 voices work; Lily is a female British voice | ⏳ |
| 8 | Player shows pause while playing and play after it ends | ⏳ |
| 9 | "Tell me another story!" → new story for the same picture | ⏳ |
| 10 | Non-picture file → friendly message, no crash | ⏳ |
| 11 | Works on phone / tablet | ⏳ |

## 7. Code structure (course conventions)
* **IMPORT PART** (imports, settings, model names, themes, voices) · **FUNCTION PART** (52 functions, each with a docstring,
  grouped into page, models, picture, Stage 1, Stage 2, Stage 3, full run and screen) · **MAIN PART**
  (`def main()` with **INPUT / PROCESS / OUTPUT** sub-sections, started by `if __name__ == "__main__": main()`).
* No classes. Every function is called with **keyword arguments** (`parameter=argument`). **f-strings** everywhere, with times shown to **`.2f`**.
* Descriptive variable names (`uploaded_picture_file`, `target_word_count`, `queue_sentence_for_speech`, ...).

```
taletwinkle/
├── app.py                       # the Streamlit app (main file)
├── requirements.txt             # Python packages with min/max versions
├── packages.txt                 # system packages for Streamlit Cloud (ffmpeg, libsndfile)
├── .streamlit/config.toml       # light theme, 15 MB upload limit
├── assets/backgrounds/*.jpg     # 6 illustrated story-world backgrounds
├── test_images/                 # 10 test images + test_image_ground_truth.csv
├── benchmark/                   # Colab notebook (10 models x 3 stages) + run 1 screening results
└── tests/                       # offline automated tests + stand-in modules
```

## 8. Deploy on Streamlit Cloud
1. Create a **public** GitHub repository (README on, licence GPL v3) and upload every file and folder above.
   The hidden `.streamlit` folder matters: on the GitHub website, use *Add file → Create new file* and type `.streamlit/config.toml`.
2. <https://share.streamlit.io> → **Create app** → your repo, branch `main`, main file **`app.py`**.
   Under **Advanced settings**, choose **Python 3.11**. Click **Deploy**.
3. The first start downloads and warms up the models (a few minutes). After that they stay loaded.
4. If the app ever gets stuck: *Manage app → Reboot*.

## 9. Honest limitations
* gTTS needs internet (Streamlit Cloud has it) and sends the story text to Google's TTS service. The man's voice and the offline backup run locally.
* There is no free engine with genuine child or grandparent voices, so TaleTwinkle offers cartoon characters instead of fake-sounding "kids" or "grandparents".
* Streamlit Community Cloud has limited CPU and about 2.7 GB of memory. The story is therefore written by a small model (in int8), and the full story
  takes a few seconds; the first words appear almost immediately.
* Browsers only allow autoplay after the page has been tapped. Uploading a picture counts as a tap.

## 10. Credits and acknowledgements
* Models: the Hugging Face model authors (see the model cards). Speech: Google Translate TTS via `gTTS`; `facebook/mms-tts-eng`.
* Story-world backgrounds and the 6 illustrated test images: created for this project. Photos of people in `test_images/`: ISOM5240 course sample images.
* Font: Fredoka (Google Fonts). Emojis: Unicode.
* **Use of generative AI:** Claude (Anthropic) and ChatGPT (OpenAI) were used as coding assistants, as the course syllabus permits. All code was reviewed and tested.
