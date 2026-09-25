# Focused Qwen3 Stage 2 benchmark

## Purpose

This focused run tests whether `Qwen/Qwen3-0.6B` should replace the current Stage 2 story model without rerunning the complete ten-model benchmark. It compares Qwen3 with the current production model and the strongest quality model from the earlier saved screening.

## Controlled setup

- Five fixed project images: Fairy Tale, Space Quest, Gentle Mystery, Jungle Adventure and Silly Poem.
- The previously saved `Salesforce/blip-image-captioning-base` captions were reused; image captioning was not rerun.
- Identical child-safety prompt, theme instruction, target lengths and random seed (`42`).
- CPU only, two PyTorch threads, Python 3.12.14, PyTorch 2.9.1 and Transformers 4.57.6.
- Qwen3 received chat messages directly through the Transformers `text-generation` pipeline with `tokenizer_encode_kwargs={"enable_thinking": False}`. Because the final message is from the user, the pipeline automatically applies the generation prompt.
- Generation time excludes model download/load time. Load values remain in the CSV for diagnosis but are not used in ranking because model cache state differed.

## Scoring

The focused test uses transparent offline checks so it can run without downloading separate evaluator models:

- 35% expected picture-concept recall.
- 25% requested-theme keyword coverage.
- 20% Flesch-Kincaid readability for young children.
- 10% safety vocabulary check.
- 10% compliance with the required 50–100 words.

The final score follows the main notebook's selection rule: 80% relative story quality and 20% relative generation speed.

## Results

| Rank | Model | Story quality | Image relevance | Theme match | Child suitability | Length compliance | Mean time | Median time | Overall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `Qwen/Qwen3-0.6B` | 62.71% | 36.67% | 86.67% | 73.89% | 60.00% | 11.27 s | 10.14 s | **88.60** |
| 2 | `Qwen/Qwen2.5-0.5B-Instruct` | 57.79% | 33.33% | 66.67% | 64.68% | 100.00% | 13.68 s | 15.37 s | **80.81** |
| 3 | `google/flan-t5-base` | 27.98% | 3.33% | 6.67% | 56.95% | 80.00% | 4.84 s | 5.05 s | **55.70** |

Qwen3 ranked first. Its stories were more strongly themed and slightly more image-relevant than Qwen2.5, while Qwen2.5 followed the word range more consistently. FLAN-T5 was faster but produced repetitive or irrelevant text, so its speed did not compensate for the quality failure.

## Precision decision

An additional Qwen3-only optimization test forced float32 loading followed by dynamic int8 conversion. It improved mean generation time from 11.27 seconds to 7.40 seconds and achieved 100% length compliance in that run, but process memory increased by about 2.95 GB during loading/conversion. That variant was rejected because the project has already encountered Streamlit resource-limit errors. Production therefore uses Qwen3 bfloat16, direct chat messages, non-thinking mode, live token streaming and explicit model cleanup after each story.

The CPU completion time remains above the desired 3–4 second target. Live streaming and sentence-level background narration reduce perceived waiting time, but this result must not be reported as a sub-four-second full-generation benchmark.

Raw evidence is in `stage2_qwen3_focused_detailed.csv` and `stage2_qwen3_focused_summary.csv`.
