Offline test run: 49/49 passed

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
| 19 | Finished sentences queued one by one during streaming | ✅ Pass | ['The dog ran fast.', 'The cat jumped!', 'Then they slept.'] |
| 20 | Early stop: long model output cut at the word limit | ✅ Pass | 87 words with early stop vs 360 without (target 75) |
| 21 | Voice speed 0.50x keeps pitch | ✅ Pass | 59.98 s (expected 60.00), pitch 220.00 Hz, took 0.45 s |
| 22 | Voice speed 2.00x keeps pitch | ✅ Pass | 15.04 s (expected 15.00), pitch 220.00 Hz, took 0.10 s |
| 23 | Cartoon pitch: 🦜 Polly the Parrot | ✅ Pass | 311.00 Hz (expected 311.13) |
| 24 | Cartoon pitch: 🧒 Buddy Ben (Kid-Style Boy) | ✅ Pass | 261.50 Hz (expected 261.63) |
| 25 | Cartoon pitch: 👧 Giggle Grace (Kid-Style Girl) | ✅ Pass | 261.50 Hz (expected 261.63) |
| 26 | Default voice is the real British lady (no pitch effect) | ✅ Pass | 👩‍🏫 Story Lady Lily (British); 5 real unaltered voices |
| 27 | Voice order: Lily, then Parrot, then Kitten | ✅ Pass | ['lily_british', 'polly_parrot', 'whiskers_kitten'] |
| 28 | Four distinct male voices replace Uncle Max and Bruno | ✅ Pass | male choices=['captain_finn', 'joe_storyteller', 'hero_bryce', 'sir_alan'] |
| 29 | One boy-style and one girl-style voice are available | ✅ Pass | kid choices=['buddy_ben', 'giggle_grace'] |
| 30 | Robot effect changes the sound | ✅ Pass | 60 Hz buzz added |
| 31 | MP3 decoding (ffmpeg fallback) | ✅ Pass | 25344 samples at 24000 Hz |
| 32 | Narration assembled (3 sentences, robot, 1.5x) | ✅ Pass | 2.43 s of audio |
| 33 | First visit: picture, caption and story placeholders plus greeting audio are shown | ✅ Pass | 3 placeholders, greeting player (no autoplay) |
| 34 | No background music anywhere | ✅ Pass | music code and files removed |
| 35 | Models shrunk to int8 at start-up | ✅ Pass | 4 models quantized |
| 36 | Upload: picture shown, story streamed, narration autoplays (no button click) | ✅ Pass | caption='a dog playing with a red ball in the park', 79 words |
| 37 | Caption is displayed directly below the uploaded image | ✅ Pass | a dog playing with a red ball in the park |
| 38 | Streaming story is enclosed in the white live-story panel | ✅ Pass | live story panel marker rendered |
| 39 | Story card is shown BEFORE the audio player | ✅ Pass | story card call #27, audio call #29 |
| 40 | Speech started in the background while the story was still being written | ✅ Pass | first sentence sent 0.96 s before the story finished |
| 41 | Timings recorded (2 decimals shown) | ✅ Pass | first words 0.01 s, story 0.84 s, voice ready 0.98 s (voice 0.14 s after story) |
| 42 | Rerun with same choices does not rewrite the story | ✅ Pass | story reused from session state |
| 43 | Voice / speed change re-records audio only | ✅ Pass | new narration, same story |
| 44 | Theme change writes a new story | ✅ Pass | new Ocean Magic story |
| 45 | Broken file shows a friendly error (no crash) | ✅ Pass | friendly error message shown |
| 46 | Story model crash -> safe backup story | ✅ Pass | 85 words |
| 47 | Unsafe story -> safe backup story | ✅ Pass | blocked |
| 48 | Google TTS offline -> local Piper voice used, story still narrated | ✅ Pass | rhasspy/piper-voices (en/en_US/amy/medium/en_US-amy-medium.onnx; local fallback because Google TTS was unavailable) |
| 49 | Male voice uses a local Hugging Face Piper model | ✅ Pass | rhasspy/piper-voices (en/en_US/ryan/medium/en_US-ryan-medium.onnx; local Hugging Face ONNX model) |
