# Model Selection Report

This report records the evidence behind TaleTwinkle’s three production models. It supplements the raw CSV files and avoids hiding failed or unsuitable candidates.

## Evaluation design

- Ten fixed images were used for image captioning.
- The same BLIP Base captions were supplied to all ten raw story candidates.
- Image scoring measures expected concept-group recall and runtime.
- Story scoring combines image concept recall (40%), child-suitability proxy (30%), theme keywords (20%), and length (10%).
- Automated scores were followed by manual output review because repeated or incoherent text can receive an inflated keyword score.
- TTS selection used a ten-engine compatibility screen, followed by executable waveform testing of ten child-facing personas from the selected model.
- The final production pipeline was tested separately on all ten images with exact word count, safety, audio creation, and stage timing.

## Decisions

### Image captioning

Selected `Salesforce/blip-image-captioning-base`. GIT Base TextCaps had the highest concept recall, but BLIP Base was approximately five times faster, lighter, and less prone to reading nonexistent text into illustrated scenes. The selection optimizes the assignment’s combined accuracy-and-speed requirement rather than one metric alone.

### Story generation

Selected `roneneldan/TinyStories-33M`. FLAN-T5 ranked highest numerically but repeatedly emitted the same sentence. The selected model is trained for simple stories and is small enough for CPU deployment. Production safeguards provide image grounding, genre hooks, blocked-term fallback, repeated-sentence removal, exact 50–100-word control, and a positive ending.

### Text-to-speech

Selected `facebook/mms-tts-eng`. SpeechT5 and Bark offer richer native variation but add speaker-embedding/vocoder complexity or excessive CPU latency. gTTS was prototyped successfully but rejected because it would disclose the generated story to an external service. MMS keeps inference local and supports fast, deterministic persona effects.

## Reproducibility files

- `image_caption_detailed_results.csv`
- `image_caption_summary.csv`
- `story_detailed_results.csv`
- `story_summary.csv`
- `tts_model_screening.csv`
- `tts_voice_persona_results.csv`
- `production_acceptance_results.csv`
- `production_acceptance_summary.csv`

## Interpretation warning

Concept recall is useful for comparing the same fixed examples, but it does not capture narrative coherence, warmth, prosody, or delight. “Accuracy” in this report means the documented test metric, not a universal probability that an output is correct. Human review remains part of model selection and final grading.
