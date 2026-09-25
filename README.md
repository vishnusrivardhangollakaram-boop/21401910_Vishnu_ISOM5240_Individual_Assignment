# ✨ TaleTwinkle — Picture Stories for Kids

**ISOM5240 Deep Learning Business Applications with Python — Individual Assignment**

TaleTwinkle is a Streamlit application for children aged 3–10. A child drops in a picture; the app immediately shows it, describes it, writes a themed 50–100 word story on screen, and reads the story aloud. No generate button is required.

## Selected production models

| Stage | Selected model | Code in this submission | Why selected |
|---|---|---|---|
| Image → caption | `Salesforce/blip-image-captioning-base` | `pipeline("image-to-text")` | Best measured accuracy/speed balance on the 10 project images |
| Caption → story | `Qwen/Qwen3-0.6B` | `pipeline("text-generation")` with chat messages and non-thinking mode | Ranked first in the focused five-image quality/speed benchmark |
| Story → local speech | `rhasspy/piper-voices` | Piper ONNX checkpoints downloaded from Hugging Face | Distinct voices, local fallback and very fast CPU synthesis |
| Story → regional speech | gTTS | UK, US, Australian and Indian English | Familiar regional voices; Piper automatically takes over if the service is unavailable |

Model cards and inference documentation:

- [BLIP image captioning model card](https://huggingface.co/Salesforce/blip-image-captioning-base) — used through the Transformers `image-to-text` pipeline.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) — used through the Transformers `text-generation` pipeline with `enable_thinking=False`.
- [Piper voices repository](https://huggingface.co/rhasspy/piper-voices) — ONNX voice checkpoints loaded by `piper-tts`.

`roneneldan/TinyStories-33M` was considered and benchmarked, but it was not selected. It is fast, yet it frequently ignores the image/theme instructions and encouraged repetitive template text. Qwen3 does not add the repeated “Once upon a time, the picture came alive” prefix.

## Business and user requirements

| Requirement | Implementation |
|---|---|
| Engaging app for children aged 3–10 | Bright theme worlds, large controls, themed backgrounds, animated emoji decorations and child-friendly language |
| Low user knowledge | One central drag-and-drop uploader; processing starts automatically |
| 50–100 word story | Slider from 50 to 100, default 75; output is trimmed at a complete sentence and safety checked |
| Relevant to the uploaded image | BLIP caption is included in Qwen’s prompt and displayed directly below the image |
| Theme choice | Fairy Tale, Space Quest, Gentle Mystery, Jungle Adventure, Silly Poem and Ocean Magic |
| Attractive, easy UI | Premium themed side panels sit level with the image; the live and final story use wide white reading cards |
| Image, story and audio always visible | Full-width output area below the picture/caption keeps the story and audio player on screen |
| Fast, memory-safe response | BLIP runs and is released first; Qwen and the required Piper voice then load together, story words stream, and Stage 2/3 resources are released after each run |
| Audio control | 0.5×–2× speed slider; pitch-preserving WSOLA processing; automatic playback after generation |
| Reliable deployment | Explicit minimum/maximum dependency ranges, system packages, local speech fallback and automated offline tests |
| No unwanted music | There is no background music code, package or audio asset |

## Interface and voices

- Left premium panel: six story worlds. Each changes the page background, colour, mascot, decorations and writing instruction.
- Centre: uploader, resized image and generated caption.
- Auto-scroll: the page glides down to the story while it is being written, and again when the narration is ready, so the story and the playing audio player are on screen without the child scrolling.
- Right premium panel: story length, voice speed and storyteller.
- Below: a full-width live white story box, final story card, audio player and “Tell me another story” control.

The final order is **👩‍🏫 Story Lady Lily**, **🦜 Polly**, **🤖 Robo Beep**, **👧 Giggle Grace**, **🧭 Captain Finn**, **👩 Aunty Chloe**, **🎩 Sir Alan**, **🐱 Whiskers**, **👩 Aunt Amy** and **👩‍🏫 Teacher Priya**. Giggle Grace is transparently labelled as a pitch-adjusted child-style effect, not a recording of a child. Uncle Max, Bruno the Bear, Jolly Joe, Hero Bryce and Buddy Ben have been removed.

## Processing flow

1. The uploaded image is opened safely, phone rotation is corrected and transparency is placed on white.
2. A display copy is resized to a 460 px longest side. Small images scale up; large images scale down.
3. BLIP produces a caption from a separate model copy no larger than 512 px, then its in-memory resource is released.
4. Caption-cleaning removes common benchmark artefacts such as repeated words, “illustration” and “painting of”.
5. Only after captioning, Qwen and the required local Piper voice load concurrently. Google voices do not load Piper unless fallback is needed.
6. Qwen receives the caption, chosen world, child-safe writing rules and target word count.
7. Story tokens stream into an opaque white reading card. Each completed sentence is queued for speech in the background.
8. The final output is checked for length, complete punctuation and unsafe words. A safe backup story is available if generation fails.
9. Speech clips are joined, voice speed/effects are applied, loudness is normalised and a WAV player autoplays.
10. Stage 2 and Stage 3 model resources are explicitly released after success or failure. Python logging records stages, timings and exceptions without logging private image/audio data.

## Model comparison

The reproducible notebook is [benchmarks/TaleTwinkle_Model_Benchmark.ipynb](benchmarks/TaleTwinkle_Model_Benchmark.ipynb). It uses the same ten images and saves per-model details, summaries, audio samples and an Excel workbook.

### Image-caption candidates (10)

| Rank | Candidate | Accuracy % | Seconds/image | Overall |
|---:|---|---:|---:|---:|
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

Caption accuracy is the percentage of expected objects named. Overall score weights relative accuracy at 70% and relative speed at 30%.

### Story candidates (10)

The current notebook configuration compares:

1. `Qwen/Qwen3-0.6B` ✅
2. `Qwen/Qwen2.5-0.5B-Instruct`
3. `Qwen/Qwen2.5-1.5B-Instruct`
4. `HuggingFaceTB/SmolLM2-135M-Instruct`
5. `HuggingFaceTB/SmolLM2-360M-Instruct`
6. `HuggingFaceTB/SmolLM2-1.7B-Instruct`
7. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
8. `google/flan-t5-base`
9. `google/flan-t5-small`
10. `roneneldan/TinyStories-Instruct-33M`

The kids’ story score weights image relevance (35%), theme match (25%), reading level (20%), safety (10%) and valid length (10%).

The checked-in `run1_initial_screening/story_summary.csv` is retained as the earlier screening evidence. The later `run2_qwen3_focused/` test reused five saved BLIP captions and compared only Qwen3, the current Qwen2.5 model and FLAN-T5-base, as requested; it did not fabricate or rerun results for the remaining models.

### Focused Qwen3 challenger test (5 images)

| Rank | Model | Story quality | Mean time | Median time | Overall |
|---:|---|---:|---:|---:|---:|
| 1 | `Qwen/Qwen3-0.6B` ✅ | 62.71% | 11.27 s | 10.14 s | **88.60** |
| 2 | `Qwen/Qwen2.5-0.5B-Instruct` | 57.79% | 13.68 s | 15.37 s | **80.81** |
| 3 | `google/flan-t5-base` | 27.98% | 4.84 s | 5.05 s | **55.70** |

Qwen3 is selected because it ranked first on the same 80% relative story-quality / 20% relative-speed rule. It is called as `pipeline(messages, tokenizer_encode_kwargs={"enable_thinking": False})`; Transformers adds the generation prompt automatically for a chat ending in a user message. The measured CPU completion time is still above the desired 3–4 seconds, so the app streams words immediately, narrates completed sentences concurrently and releases Stage 2 after each run. The full method, raw stories and precision experiment are documented in [`benchmarks/run2_qwen3_focused/`](benchmarks/run2_qwen3_focused/README.md).

Model-card research also identified `Qwen/Qwen3.5-0.8B`, `google/gemma-3-270m-it`, `meta-llama/Llama-3.2-1B-Instruct` and `tiiuae/Falcon3-1B-Instruct` as possible future challengers. They are not labelled as benchmark winners because they were not run in this focused test and the larger models increase Streamlit Cloud memory risk.

### Speech candidates (10+)

The notebook compares four gTTS regions, Piper US/UK voices, Kokoro voices, MMS, VITS, SpeechT5, Bark and pyttsx3. Evaluation uses synthesis time, real-time factor, Whisper word error rate and a 1–5 human listening score.

Production uses gTTS plus Piper. In a real local CPU timing check, Piper Ryan loaded in **1.29 s** and generated a **73-word** narration in **0.97 s**. A tested quantized Kokoro alternative still needed 6–10 s for only six words on the same CPU, so it was rejected for this attention-sensitive use case.

## Testing

Run:

```bash
python tests/run_offline_tests.py
```

Current result: **56/56 passed**. The suite does not download models; small stand-ins exercise the app’s functions and complete Streamlit flow.

Coverage includes:

- image scaling in both directions, transparent images and all six background files;
- caption cleaning, 50/75/100-word limits, poem formatting and safety filtering;
- no fixed story prefix and safe fallback generation;
- background sentence queuing and early generation stop;
- auto-scroll to the live story and to the story + audio (once per new story or voice, never on plain reruns);
- cache limits plus BLIP → cleanup → Qwen/Piper lifecycle ordering;
- pitch-preserving 0.5×/2× speed and character effects;
- final ten-voice order, with Captain Finn and Sir Alan as the two local male choices and Giggle Grace as the child-style choice;
- caption below image, white live story panel and story displayed before audio;
- autoplay, cached reruns, voice-only rerender, theme regeneration and broken-file handling;
- Google outage → local Piper narration.

Detailed evidence is saved in [tests/offline_test_results.md](tests/offline_test_results.md).

## Code quality and structure

`app.py` follows the requested course style:

- separate **IMPORT PART**, **FUNCTION PART** and **MAIN PART** comments;
- small modular functions with docstrings and descriptive variable names;
- values passed explicitly as `parameter=argument`;
- f-strings and `.2f` timing output;
- no application classes;
- `def main()` with INPUT / PROCESS / OUTPUT sections;
- `if __name__ == "__main__": main()` entry point.

## Project files

```text
TaleTwinkle/
├── app.py
├── requirements.txt
├── packages.txt
├── .streamlit/config.toml
├── assets/backgrounds/                 # six PNG story-world backgrounds
├── benchmarks/
│   ├── TaleTwinkle_Model_Benchmark.ipynb
│   ├── build_notebook.py
│   └── run1_initial_screening/
├── test_images/                        # ten images + ground truth CSV
└── tests/                              # offline suite, result report and test doubles
```

## Run locally

Use Python 3.11 or 3.12:

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the selected Hugging Face files. Later runs reuse Hugging Face's disk cache, while the app deliberately reloads and releases Stage 1/2/3 model objects to remain within Streamlit Cloud RAM limits.

## Deploy to Streamlit Community Cloud

1. Put the complete folder in a GitHub repository. Keep `.streamlit/config.toml`.
2. In Streamlit Community Cloud, create an app from the repository and set the entry point to `app.py`.
3. Choose Python 3.11 where the deployment settings permit it.
4. Deploy and allow the first model download/warm-up to complete.
5. Record the final public URL and manual cloud timing results in the assessment submission.

## Honest limitations

- gTTS needs an internet connection and sends story text to its service. Piper is the local fallback.
- Browser autoplay may require a user interaction; uploading a picture normally supplies that interaction.
- Child-style voices are transparent audio effects, not real children’s recordings.
- The first cloud run is slower because model files must download. Later runs avoid downloading again, but model objects are deliberately reloaded from disk to keep RAM bounded.
- The final public Streamlit URL and manual device/browser checks must be completed after repository deployment.

## Credits

- Hugging Face model authors and model cards for BLIP, Qwen, Piper and all benchmark candidates.
- Google Translate TTS through the `gTTS` Python package.
- Fredoka via Google Fonts and Unicode emoji.
- Generative AI coding assistance was used; all code and tests were reviewed for this submission.
