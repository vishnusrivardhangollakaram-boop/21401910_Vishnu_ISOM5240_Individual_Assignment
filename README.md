# ✨ TaleTwinkle — Picture Stories for Kids

**ISOM5240 Deep Learning Business Applications with Python — Individual Assignment**

TaleTwinkle is a Streamlit application for children aged 3–10. A child drops in a picture; the app immediately shows it, describes it, writes a themed 50–100 word story on screen, and reads the story aloud. No generate button is required.

## Selected production models

| Stage | Selected model | Code in this submission | Why selected |
|---|---|---|---|
| Image → caption | `Salesforce/blip-image-captioning-base` | `pipeline("image-to-text")` | Best measured accuracy/speed balance on the 10 project images |
| Caption → story | `Qwen/Qwen2.5-0.5B-Instruct` | `pipeline("text-generation")` | Follows image, theme, safety and word-count instructions much better than TinyStories |
| Story → local speech | `rhasspy/piper-voices` | Piper ONNX checkpoints downloaded from Hugging Face | Distinct voices, local fallback and very fast CPU synthesis |
| Story → regional speech | gTTS | UK, US, Australian and Indian English | Familiar regional voices; Piper automatically takes over if the service is unavailable |

Model cards and inference documentation:

- [BLIP image captioning model card](https://huggingface.co/Salesforce/blip-image-captioning-base) — used through the Transformers `image-to-text` pipeline.
- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — used through the Transformers `text-generation` pipeline.
- [Piper voices repository](https://huggingface.co/rhasspy/piper-voices) — ONNX voice checkpoints loaded by `piper-tts`.

`roneneldan/TinyStories-33M` was considered and benchmarked, but it was not selected. It is fast, yet it frequently ignores the image/theme instructions and encouraged repetitive template text. The current code therefore keeps Qwen and does not add the repeated “Once upon a time, the picture came alive” prefix.

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
| Fast response | Cached/warmed models, dynamic int8 quantisation, streamed story words, early stopping and background sentence narration |
| Audio control | 0.5×–2× speed slider; pitch-preserving WSOLA processing; automatic playback after generation |
| Reliable deployment | Explicit minimum/maximum dependency ranges, system packages, local speech fallback and automated offline tests |
| No unwanted music | There is no background music code, package or audio asset |

## Interface and voices

- Left premium panel: six story worlds. Each changes the page background, colour, mascot, decorations and writing instruction.
- Centre: uploader, resized image and generated caption.
- Right premium panel: story length, voice speed and storyteller.
- Below: a full-width live white story box, final story card, audio player and “Tell me another story” control.

The default is **👩‍🏫 Story Lady Lily (British)**. The voice list also includes:

- Cartoon companions: 🦜 Polly, 🐱 Whiskers and 🤖 Robo Beep.
- Regional women: 👩 Aunt Amy, 👩 Aunty Chloe and 👩‍🏫 Teacher Priya.
- Four distinct local male checkpoints: 🧭 Captain Finn (Ryan), 🎙️ Jolly Joe, 🦸 Hero Bryce and 🎩 Sir Alan.
- Two clearly labelled child-style effects: 🧒 Buddy Ben and 👧 Giggle Grace. These are pitch-adjusted voices, not falsely represented as recordings of children.

The disliked **Uncle Max** and **Bruno the Bear** choices have been removed.

## Processing flow

1. The uploaded image is opened safely, phone rotation is corrected and transparency is placed on white.
2. A display copy is resized to a 460 px longest side. Small images scale up; large images scale down.
3. BLIP produces a caption from a separate model copy no larger than 512 px.
4. Caption-cleaning removes common benchmark artefacts such as repeated words, “illustration” and “painting of”.
5. Qwen receives the caption, chosen world, child-safe writing rules and target word count.
6. Story tokens stream into a white reading card. Each completed sentence is queued for speech in the background.
7. The final output is checked for length, complete punctuation and unsafe words. A safe backup story is available if generation fails.
8. Speech clips are joined, voice speed/effects are applied, loudness is normalised and a WAV player autoplays.

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

The notebook compares:

1. `Qwen/Qwen2.5-0.5B-Instruct` ✅
2. `Qwen/Qwen2.5-1.5B-Instruct`
3. `HuggingFaceTB/SmolLM2-135M-Instruct`
4. `HuggingFaceTB/SmolLM2-360M-Instruct`
5. `HuggingFaceTB/SmolLM2-1.7B-Instruct`
6. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
7. `google/flan-t5-base`
8. `google/flan-t5-small`
9. `roneneldan/TinyStories-33M`
10. `roneneldan/TinyStories-Instruct-33M`

The kids’ story score weights image relevance (35%), theme match (25%), reading level (20%), safety (10%) and valid length (10%). Qwen 0.5B is selected because it is the smallest practical instruction model that obeys all four prompt constraints while fitting Streamlit Cloud memory after int8 quantisation.

### Speech candidates (10+)

The notebook compares four gTTS regions, Piper US/UK voices, Kokoro voices, MMS, VITS, SpeechT5, Bark and pyttsx3. Evaluation uses synthesis time, real-time factor, Whisper word error rate and a 1–5 human listening score.

Production uses gTTS plus Piper. In a real local CPU timing check, Piper Ryan loaded in **1.29 s** and generated a **73-word** narration in **0.97 s**. A tested quantized Kokoro alternative still needed 6–10 s for only six words on the same CPU, so it was rejected for this attention-sensitive use case.

## Testing

Run:

```bash
python tests/run_offline_tests.py
```

Current result: **49/49 passed**. The suite does not download models; small stand-ins exercise the app’s functions and complete Streamlit flow.

Coverage includes:

- image scaling in both directions, transparent images and all six background files;
- caption cleaning, 50/75/100-word limits, poem formatting and safety filtering;
- no fixed story prefix and safe fallback generation;
- background sentence queuing and early generation stop;
- pitch-preserving 0.5×/2× speed and character effects;
- four male and two child-style voices, with Max/Bear absent;
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

The first run downloads the selected Hugging Face models and voice checkpoint. Later runs use the cache.

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
- The first uncached cloud start is slower because model files must download. In-session runs are the relevant performance measure.
- The final public Streamlit URL and manual device/browser checks must be completed after repository deployment.

## Credits

- Hugging Face model authors and model cards for BLIP, Qwen, Piper and all benchmark candidates.
- Google Translate TTS through the `gTTS` Python package.
- Fredoka via Google Fonts and Unicode emoji.
- Generative AI coding assistance was used; all code and tests were reviewed for this submission.
