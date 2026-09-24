Offline test run: 43/43 passed

| # | Test | Result | Details |
|---|---|---|---|
| 1 | Small picture scaled UP to preferred size | ✅ Pass | 120x80 -> 460x307 |
| 2 | Large picture scaled DOWN to preferred size | ✅ Pass | 4000x3000 -> 460x345 |
| 3 | Model input never enlarged | ✅ Pass | (120, 80) |
| 4 | Transparent PNG opened as RGB on white | ✅ Pass | mode=RGB, pixel=(255, 255, 255) |
| 5 | Caption cleaned: 'arafed there is a cat cat si...' | ✅ Pass | -> 'a cat sitting on a sofa' |
| 6 | Caption cleaned: 'jungle jungle jungle jungle ...' | ✅ Pass | -> 'jungle' |
| 7 | Caption cleaned: 'underwater background with c...' | ✅ Pass | -> 'underwater with corals and fishes' |
| 8 | Caption cleaned: 'children playing in the park...' | ✅ Pass | -> 'children playing in the park' |
| 9 | Caption cleaned: 'a painting of a river surrou...' | ✅ Pass | -> 'a river surrounded by trees' |
| 10 | Story trimmed for slider = 50 | ✅ Pass | 53 words, ends with ' the clouds.' |
| 11 | Story trimmed for slider = 75 | ✅ Pass | 80 words, ends with 'morrow. Pop!' |
| 12 | Story trimmed for slider = 100 | ✅ Pass | 93 words, ends with ' sunny park.' |
| 13 | Poem keeps its line breaks | ✅ Pass | 63 words, 7 lines |
| 14 | Safety filter blocks unsafe words | ✅ Pass | 'gun' blocked |
| 15 | Safety filter allows normal words | ✅ Pass | 'warm' allowed (not 'war') |
| 16 | Backup story is 50-100 words, safe, no fixed 'picture came alive' opening | ✅ Pass | 82 words |
| 17 | Prompt contains caption, theme, length and 'no title' | ✅ Pass | chat messages built |
| 18 | Finished sentences queued one by one during streaming | ✅ Pass | ['The dog ran fast.', 'The cat jumped!', 'Then they slept.'] |
| 19 | Early stop: long model output cut at the word limit | ✅ Pass | 87 words with early stop vs 360 without (target 75) |
| 20 | Voice speed 0.50x keeps pitch | ✅ Pass | 59.98 s (expected 60.00), pitch 220.00 Hz, took 0.50 s |
| 21 | Voice speed 2.00x keeps pitch | ✅ Pass | 15.04 s (expected 15.00), pitch 220.00 Hz, took 0.12 s |
| 22 | Cartoon pitch: 🦜 Polly the Parrot | ✅ Pass | 311.00 Hz (expected 311.13) |
| 23 | Cartoon pitch: 🐻 Bruno the Bear | ✅ Pass | 179.50 Hz (expected 179.73) |
| 24 | Default voice is the real British lady (no pitch effect) | ✅ Pass | 👩‍🏫 Story Lady Lily (British); 5 real unaltered voices |
| 25 | Voice order: Lily, then Parrot, then Kitten | ✅ Pass | ['lily_british', 'polly_parrot', 'whiskers_kitten'] |
| 26 | Robot effect changes the sound | ✅ Pass | 60 Hz buzz added |
| 27 | MP3 decoding (ffmpeg fallback) | ✅ Pass | 25344 samples at 24000 Hz |
| 28 | Narration assembled (3 sentences, robot, 1.5x) | ✅ Pass | 2.43 s of audio |
| 29 | First visit: picture + story placeholders and a greeting audio player are shown | ✅ Pass | 2 placeholders, greeting player (no autoplay) |
| 30 | No background music anywhere | ✅ Pass | music code and files removed |
| 31 | Models shrunk to int8 at start-up | ✅ Pass | 4 models quantized |
| 32 | Upload: picture shown, story streamed, narration autoplays (no button click) | ✅ Pass | caption='a dog playing with a red ball in the park', 79 words |
| 33 | Story card is shown BEFORE the audio player | ✅ Pass | story card call #21, audio call #23 |
| 34 | Speech started in the background while the story was still being written | ✅ Pass | first sentence sent 0.95 s before the story finished |
| 35 | Timings recorded (2 decimals shown) | ✅ Pass | first words 0.01 s, story 0.84 s, voice ready 1.01 s (voice 0.16 s after story) |
| 36 | Rerun with same choices does not rewrite the story | ✅ Pass | story reused from session state |
| 37 | Voice / speed change re-records audio only | ✅ Pass | new narration, same story |
| 38 | Theme change writes a new story | ✅ Pass | new Ocean Magic story |
| 39 | Broken file shows a friendly error (no crash) | ✅ Pass | friendly error message shown |
| 40 | Story model crash -> safe backup story | ✅ Pass | 85 words |
| 41 | Unsafe story -> safe backup story | ✅ Pass | blocked |
| 42 | Google TTS offline -> Hugging Face voice used, story still narrated | ✅ Pass | facebook/mms-tts-eng (backup: Google TTS unreachable) |
| 43 | Man's voice uses the local Hugging Face pipeline | ✅ Pass | facebook/mms-tts-eng (Hugging Face pipeline) |
