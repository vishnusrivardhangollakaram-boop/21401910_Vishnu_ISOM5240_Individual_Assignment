Offline test run: 61/61 passed

| # | Test | Result | Details |
|---|---|---|---|
| 1 | Small picture scaled UP to preferred size | ✅ Pass | 120x80 -> 460x307 |
| 2 | Large picture scaled DOWN to preferred size | ✅ Pass | 4000x3000 -> 460x345 |
| 3 | Model input never enlarged | ✅ Pass | (120, 80) |
| 4 | Transparent PNG opened as RGB on white | ✅ Pass | mode=RGB, pixel=(255, 255, 255) |
| 5 | All six dynamic theme backgrounds exist | ✅ Pass | 6/6 files found |
| 6 | Caption cleaned: 'arafed there is a cat cat si...' | ✅ Pass | -> 'a cat sitting on a sofa' |
| 7 | Caption cleaned: 'jungle jungle jungle jungle ...' | ✅ Pass | -> 'jungle' |
| 8 | Caption cleaned: 'underwater background with c...' | ✅ Pass | -> 'underwater with corals and fishes' |
| 9 | Caption cleaned: 'children playing in the park...' | ✅ Pass | -> 'children playing in the park' |
| 10 | Caption cleaned: 'a painting of a river surrou...' | ✅ Pass | -> 'a river surrounded by trees' |
| 11 | Story trimmed for slider = 50 | ✅ Pass | 53 words, ends with ' the clouds.' |
| 12 | Story trimmed for slider = 75 | ✅ Pass | 80 words, ends with 'morrow. Pop!' |
| 13 | Story trimmed for slider = 100 | ✅ Pass | 93 words, ends with ' sunny park.' |
| 14 | Poem keeps its line breaks | ✅ Pass | 63 words, 7 lines |
| 15 | Safety filter blocks unsafe words | ✅ Pass | 'gun' blocked |
| 16 | Safety filter allows normal words | ✅ Pass | 'warm' allowed (not 'war') |
| 17 | Backup story is 50-100 words, safe, no fixed 'picture came alive' opening | ✅ Pass | 82 words |
| 18 | Prompt contains caption, theme, length and 'no title' | ✅ Pass | chat messages built |
| 19 | Chat settings: Qwen3 thinking disabled; Qwen2.5 and SmolLM2 need none; bfloat16 loading | ✅ Pass | production story model: Qwen/Qwen2.5-0.5B-Instruct |
| 20 | Finished sentences queued one by one during streaming | ✅ Pass | ['The dog ran fast.', 'The cat jumped!', 'Then they slept.'] |
| 21 | Resource caches have explicit size limits | ✅ Pass | {'backgrounds': 6, 'Piper voices': 2, 'greetings': 12} |
| 22 | BLIP is released before Qwen and Piper load | ✅ Pass | Stage 1 load → caption/release → Stage 2/3 load |
| 23 | Explicit Stage 2 and Stage 3 cleanup clears both model caches | ✅ Pass | Qwen and Piper cache clear calls recorded |
| 24 | Early stop: long model output cut at the word limit | ✅ Pass | 87 words with early stop vs 360 without (target 75) |
| 25 | Voice speed 0.50x keeps pitch | ✅ Pass | 59.98 s (expected 60.00), pitch 220.00 Hz, took 0.70 s |
| 26 | Voice speed 2.00x keeps pitch | ✅ Pass | 15.04 s (expected 15.00), pitch 220.00 Hz, took 0.16 s |
| 27 | Cartoon pitch: 🦜 Polly the Parrot | ✅ Pass | 311.00 Hz (expected 311.13) |
| 28 | Cartoon pitch: 👧 Giggle Grace (Kid-Style Girl) | ✅ Pass | 261.50 Hz (expected 261.63) |
| 29 | Default voice is the real British lady (no pitch effect) | ✅ Pass | 👩‍🏫 Story Lady Lily (British); 5 real unaltered voices |
| 30 | Voice order matches the final kid-friendly character list | ✅ Pass | ['lily_british', 'polly_parrot', 'robo_beep', 'giggle_grace', 'captain_finn', 'chloe_australian', 'sir_alan', 'whiskers_kitten', 'amy_american', 'priya_indian'] |
| 31 | Only Captain Finn and Sir Alan remain as local male voices | ✅ Pass | male choices=['captain_finn', 'sir_alan'] |
| 32 | Giggle Grace remains as the child-style voice | ✅ Pass | one clearly labelled child-style voice |
| 33 | Robot effect changes the sound | ✅ Pass | 60 Hz buzz added |
| 34 | MP3 decoding (ffmpeg fallback) | ✅ Pass | 25344 samples at 24000 Hz |
| 35 | Narration assembled (3 sentences, robot, 1.5x) | ✅ Pass | 2.43 s of audio |
| 36 | First visit: picture, caption and story placeholders plus greeting audio are shown | ✅ Pass | 3 placeholders, greeting player (no autoplay) |
| 37 | No background music anywhere | ✅ Pass | music code and files removed |
| 38 | Start-up model loading follows KEEP_MODELS_LOADED | ✅ Pass | KEEP_MODELS_LOADED=True, models preloaded=True |
| 39 | PyTorch limited to the usable CPU cores | ✅ Pass | 2 threads |
| 40 | Caption and story models are used for the first story | ✅ Pass | BLIP then Qwen |
| 41 | Stage timings recorded separately (model loading vs writing) | ✅ Pass | story writing 0.64 s |
| 42 | Upload: picture shown, story streamed, narration autoplays (no button click) | ✅ Pass | caption='a dog playing with a red ball in the park', 53 words |
| 43 | Caption is displayed directly below the uploaded image | ✅ Pass | a dog playing with a red ball in the park |
| 44 | Streaming story is enclosed in the white live-story panel | ✅ Pass | live story panel marker rendered |
| 45 | Story card is shown BEFORE the audio player | ✅ Pass | story card call #30, audio call #32 |
| 46 | Speech started in the background while the story was still being written | ✅ Pass | first sentence sent 0.78 s before the story finished |
| 47 | Timings recorded (2 decimals shown) | ✅ Pass | first words 0.14 s, story 0.78 s, voice ready 0.89 s (voice 0.12 s after story) |
| 48 | Auto-scroll: to the live story while writing, then to the story + audio when ready | ✅ Pass | 2 scroll requests during the first story |
| 49 | Rerun with same choices does not rewrite the story | ✅ Pass | story reused from session state |
| 50 | Auto-scroll does not repeat on a plain rerun | ✅ Pass | no scroll request |
| 51 | Voice / speed change re-records audio only | ✅ Pass | new narration, same story |
| 52 | Auto-scroll to the new audio after a voice change | ✅ Pass | one scroll request |
| 53 | Theme change writes a new story | ✅ Pass | new Ocean Magic story |
| 54 | Broken file shows a friendly error (no crash) | ✅ Pass | friendly error message shown |
| 55 | Story model crash -> safe backup story | ✅ Pass | 56 words |
| 56 | Unsafe story -> safe backup story | ✅ Pass | blocked |
| 57 | Google TTS offline -> local Piper voice used, story still narrated | ✅ Pass | rhasspy/piper-voices (en/en_US/amy/medium/en_US-amy-medium.onnx; local fallback because Google TTS was unavailable) |
| 58 | Male voice uses a local Hugging Face Piper model | ✅ Pass | rhasspy/piper-voices (en/en_US/ryan/medium/en_US-ryan-medium.onnx; local Hugging Face ONNX model) |
| 59 | Normal memory: models stay loaded for the next story | ✅ Pass | no model released at 1500 MB |
| 60 | High memory: models are released automatically after use | ✅ Pass | released above 2600 MB |
| 61 | KEEP_MODELS_LOADED = False restores release-after-every-story | ✅ Pass | setting respected |
