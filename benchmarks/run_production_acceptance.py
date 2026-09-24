"""Run the production TaleTwinkle pipeline against all ten test images."""

# Import Part

import hashlib
import sys
import time
from pathlib import Path

import pandas as pd


# Allow this benchmark to import the production app from the project root.
PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIRECTORY))

import app  # noqa: E402  (Path setup must happen before this local import.)


# Function Part

BENCHMARK_DIRECTORY = Path(__file__).resolve().parent
GROUND_TRUTH_PATH = BENCHMARK_DIRECTORY / "test_image_ground_truth.csv"
TEST_IMAGE_DIRECTORY = PROJECT_DIRECTORY / "test_images"
OUTPUT_PATH = BENCHMARK_DIRECTORY / "production_acceptance_results.csv"
SUMMARY_PATH = BENCHMARK_DIRECTORY / "production_acceptance_summary.csv"

THEME_UI_NAMES = {
    "Fairy Tale": "Fairy Tale 🏰",
    "Space Quest": "Space Quest 🚀",
    "Gentle Mystery": "Gentle Mystery 🔎",
    "Jungle Adventure": "Jungle Adventure 🦜",
    "Silly Poem": "Silly Poem 🎵",
    "Ocean Magic": "Ocean Magic 🐠",
}


def calculate_concept_recall(story_text: str, expected_concepts: str) -> float:
    """Calculate expected image-concept recall for one final story."""
    normalized_story = story_text.lower()
    concept_groups = [concept_group.split("|") for concept_group in expected_concepts.split(";")]
    matched_group_count = sum(
        any(synonym.strip() in normalized_story for synonym in concept_group)
        for concept_group in concept_groups
    )
    return matched_group_count / max(len(concept_groups), 1)


def run_one_acceptance_case(test_case: dict[str, object]) -> dict[str, object]:
    """Process one image and capture quality, safety, audio, and timing evidence."""
    image_path = TEST_IMAGE_DIRECTORY / str(test_case["image_name"])
    image_bytes = image_path.read_bytes()
    theme_name = THEME_UI_NAMES[str(test_case["theme"])]
    random_seed = int(hashlib.sha256(image_bytes).hexdigest()[:8], 16)

    case_start_time = time.perf_counter()
    caption_start_time = time.perf_counter()
    image_caption = app.generate_image_caption(image_bytes)
    caption_seconds = time.perf_counter() - caption_start_time

    story_start_time = time.perf_counter()
    generated_story = app.generate_children_story(
        image_caption,
        theme_name,
        app.DEFAULT_STORY_WORDS,
        random_seed,
    )
    story_seconds = time.perf_counter() - story_start_time

    audio_start_time = time.perf_counter()
    narrated_audio = app.create_narrated_story_audio(
        generated_story,
        "🌙 Luna — Story Lady (Default)",
        1.00,
        theme_name,
    )
    audio_seconds = time.perf_counter() - audio_start_time
    total_seconds = time.perf_counter() - case_start_time

    word_count = app.count_story_words(generated_story)
    return {
        "image_name": test_case["image_name"],
        "theme": test_case["theme"],
        "caption": image_caption,
        "story": generated_story,
        "word_count": word_count,
        "exact_length_pass": word_count == app.DEFAULT_STORY_WORDS,
        "safety_pass": not app.contains_unsuitable_language(generated_story),
        "concept_recall": calculate_concept_recall(
            generated_story,
            str(test_case["expected_concepts"]),
        ),
        "audio_created": len(narrated_audio) > 44,
        "audio_size_bytes": len(narrated_audio),
        "caption_seconds": caption_seconds,
        "story_seconds": story_seconds,
        "audio_seconds": audio_seconds,
        "total_seconds": total_seconds,
    }


def save_acceptance_summary(detailed_results: pd.DataFrame) -> None:
    """Save one concise summary separating first-run and warm-run performance."""
    warm_results = detailed_results.iloc[1:] if len(detailed_results) > 1 else detailed_results
    summary = pd.DataFrame(
        [
            {
                "test_images": len(detailed_results),
                "functionality_pass_percent": 100.0
                * float(
                    (
                        detailed_results["exact_length_pass"]
                        & detailed_results["safety_pass"]
                        & detailed_results["audio_created"]
                    ).mean()
                ),
                "mean_concept_recall_percent": 100.0
                * float(detailed_results["concept_recall"].mean()),
                "cold_first_run_seconds": float(detailed_results.iloc[0]["total_seconds"]),
                "warm_mean_caption_seconds": float(warm_results["caption_seconds"].mean()),
                "warm_mean_story_seconds": float(warm_results["story_seconds"].mean()),
                "warm_mean_audio_seconds": float(warm_results["audio_seconds"].mean()),
                "warm_mean_total_seconds": float(warm_results["total_seconds"].mean()),
                "warm_max_total_seconds": float(warm_results["total_seconds"].max()),
            }
        ]
    )
    summary.to_csv(SUMMARY_PATH, index=False)


def main() -> None:
    """Execute and save all ten production acceptance cases."""
    test_cases = pd.read_csv(GROUND_TRUTH_PATH).fillna("").to_dict("records")
    all_results = []
    for test_case in test_cases:
        result = run_one_acceptance_case(test_case)
        all_results.append(result)
        print(
            f"{result['image_name']} | words={result['word_count']} | "
            f"recall={result['concept_recall']:.2f} | total={result['total_seconds']:.2f}s",
            flush=True,
        )

    detailed_results = pd.DataFrame(all_results)
    detailed_results.to_csv(OUTPUT_PATH, index=False)
    save_acceptance_summary(detailed_results)
    print(f"Saved {len(detailed_results)} production acceptance results.", flush=True)


if __name__ == "__main__":
    main()
