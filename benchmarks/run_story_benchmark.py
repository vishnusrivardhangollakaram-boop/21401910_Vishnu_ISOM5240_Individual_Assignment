"""Benchmark ten Hugging Face story models using the selected image captions."""

# Import Part

import argparse
import gc
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

from model_catalog import STORY_MODELS


# Function Part

BENCHMARK_DIRECTORY = Path(__file__).resolve().parent
CAPTION_RESULTS_PATH = BENCHMARK_DIRECTORY / "image_caption_detailed_results.csv"
GROUND_TRUTH_PATH = BENCHMARK_DIRECTORY / "test_image_ground_truth.csv"
TARGET_WORD_COUNT = 75

THEME_KEYWORDS = {
    "Fairy Tale": ["magic", "castle", "fairy", "wish", "kingdom"],
    "Space Quest": ["space", "star", "planet", "rocket", "moon", "alien"],
    "Gentle Mystery": ["clue", "mystery", "secret", "discover", "wonder"],
    "Jungle Adventure": ["adventure", "jungle", "explore", "journey", "map"],
    "Silly Poem": ["rhyme", "silly", "song", "dance", "giggle"],
    "Ocean Magic": ["ocean", "sea", "fish", "coral", "underwater", "bubble"],
}

UNSUITABLE_WORDS = {
    "blood",
    "kill",
    "killed",
    "death",
    "dead",
    "gun",
    "weapon",
    "horror",
    "nightmare",
    "hate",
    "stupid",
}


def select_caption_inputs() -> pd.DataFrame:
    """Use BLIP Base captions so every story model receives identical inputs."""
    caption_results = pd.read_csv(CAPTION_RESULTS_PATH).fillna("")
    selected_captions = caption_results[
        caption_results["model_name"] == "Salesforce/blip-image-captioning-base"
    ][["image_name", "caption"]]
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH)
    return ground_truth.merge(selected_captions, on="image_name", how="left")


def build_story_prompt(caption: str, theme: str) -> str:
    """Build one explicit child-safe prompt from an image caption and theme."""
    story_form = "rhyming poem" if theme == "Silly Poem" else "story"
    return (
        f"Write a {TARGET_WORD_COUNT}-word {theme.lower()} {story_form} for children aged 3 to 10. "
        f"Base it closely on this picture: {caption}. Use simple, warm words, a clear beginning, "
        "one gentle problem, and a happy ending. Avoid violence, frightening events, brands, "
        "and mature topics. Give only the story."
    )


def create_story_pipeline(model_name: str):
    """Create the correct Transformers pipeline for the candidate architecture."""
    task_name = "text2text-generation" if "flan-t5" in model_name.lower() else "text-generation"
    return pipeline(
        task_name,
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
    )


def extract_generated_story(model_name: str, prompt: str, model_output: object) -> str:
    """Extract generated text and remove the echoed prompt when necessary."""
    generated_text = str(model_output[0].get("generated_text", "")).strip()
    if "flan-t5" not in model_name.lower() and generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    return generated_text


def generate_story(story_pipeline, model_name: str, prompt: str) -> str:
    """Generate a deterministic story suitable for comparable testing."""
    generation_arguments = {
        "max_new_tokens": 120,
        "do_sample": False,
        "num_return_sequences": 1,
    }
    model_output = story_pipeline(prompt, **generation_arguments)
    return extract_generated_story(model_name, prompt, model_output)


def calculate_concept_recall(story: str, expected_concepts: str) -> float:
    """Measure how many expected image concepts appear in the story."""
    normalized_story = story.lower()
    concept_groups = [group.split("|") for group in expected_concepts.split(";")]
    matched_groups = sum(
        any(synonym.strip() in normalized_story for synonym in group)
        for group in concept_groups
    )
    return matched_groups / max(len(concept_groups), 1)


def calculate_child_suitability(story: str) -> float:
    """Score safety, vocabulary simplicity, sentence length and repetition."""
    words = re.findall(r"[A-Za-z']+", story.lower())
    if not words:
        return 0.0

    safety_score = 1.0 if not (set(words) & UNSUITABLE_WORDS) else 0.0
    average_word_length = sum(len(word) for word in words) / len(words)
    vocabulary_score = max(0.0, min(1.0, (7.0 - average_word_length) / 2.5))
    sentences = [sentence for sentence in re.split(r"[.!?]+", story) if sentence.strip()]
    average_sentence_words = len(words) / max(len(sentences), 1)
    sentence_score = max(0.0, min(1.0, (24.0 - average_sentence_words) / 12.0))
    unique_word_ratio = len(set(words)) / len(words)
    repetition_score = min(1.0, unique_word_ratio / 0.55)
    return (
        (0.40 * safety_score)
        + (0.20 * vocabulary_score)
        + (0.20 * sentence_score)
        + (0.20 * repetition_score)
    )


def calculate_theme_match(story: str, theme: str) -> float:
    """Measure whether the generated content reflects the selected genre."""
    normalized_story = story.lower()
    theme_keywords = THEME_KEYWORDS[theme]
    matched_keywords = sum(keyword in normalized_story for keyword in theme_keywords)
    return min(1.0, matched_keywords / 2.0)


def calculate_length_score(word_count: int) -> float:
    """Score closeness to the requested 75-word target and assignment range."""
    if not 50 <= word_count <= 100:
        return 0.0
    return max(0.0, 1.0 - abs(TARGET_WORD_COUNT - word_count) / TARGET_WORD_COUNT)


def score_story(story: str, expected_concepts: str, theme: str) -> dict[str, float]:
    """Calculate all documented story-quality scores."""
    word_count = len(re.findall(r"[A-Za-z']+", story))
    image_relevance = calculate_concept_recall(story, expected_concepts)
    child_suitability = calculate_child_suitability(story)
    theme_match = calculate_theme_match(story, theme)
    length_score = calculate_length_score(word_count)
    overall_score = (
        (0.40 * image_relevance)
        + (0.30 * child_suitability)
        + (0.20 * theme_match)
        + (0.10 * length_score)
    )
    return {
        "word_count": float(word_count),
        "image_relevance": image_relevance,
        "child_suitability": child_suitability,
        "theme_match": theme_match,
        "length_score": length_score,
        "overall_score": overall_score,
    }


def release_model_resources() -> None:
    """Release model memory before the next candidate is loaded."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_results(all_results: list[dict[str, object]]) -> None:
    """Checkpoint detailed results and one summary row per model."""
    detailed_results = pd.DataFrame(all_results)
    detailed_results.to_csv(BENCHMARK_DIRECTORY / "story_detailed_results.csv", index=False)
    successful_results = detailed_results[detailed_results["status"] == "success"]
    summary = (
        successful_results.groupby("model_name", as_index=False)
        .agg(
            overall_score=("overall_score", "mean"),
            image_relevance=("image_relevance", "mean"),
            child_suitability=("child_suitability", "mean"),
            theme_match=("theme_match", "mean"),
            length_score=("length_score", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_word_count=("word_count", "mean"),
            successful_stories=("image_name", "count"),
        )
        .sort_values(["overall_score", "mean_runtime_seconds"], ascending=[False, True])
    )
    summary.to_csv(BENCHMARK_DIRECTORY / "story_summary.csv", index=False)


def parse_arguments() -> argparse.Namespace:
    """Parse optional model filters for resumable execution."""
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--models", nargs="*", default=STORY_MODELS)
    return argument_parser.parse_args()


def main() -> None:
    """Run and checkpoint the ten-model story benchmark."""
    arguments = parse_arguments()
    test_inputs = select_caption_inputs()
    detailed_result_path = BENCHMARK_DIRECTORY / "story_detailed_results.csv"
    if detailed_result_path.exists():
        all_results = pd.read_csv(detailed_result_path).fillna("").to_dict("records")
    else:
        all_results = []

    completed_models = {
        str(result["model_name"])
        for result in all_results
        if result.get("status") == "success"
    }

    for model_name in arguments.models:
        if model_name in completed_models:
            print(f"SKIP | {model_name} | already completed", flush=True)
            continue

        try:
            story_pipeline = create_story_pipeline(model_name)
            for test_input in test_inputs.to_dict("records"):
                prompt = build_story_prompt(test_input["caption"], test_input["theme"])
                start_time = time.perf_counter()
                story = generate_story(story_pipeline, model_name, prompt)
                runtime_seconds = time.perf_counter() - start_time
                scores = score_story(
                    story,
                    test_input["expected_concepts"],
                    test_input["theme"],
                )
                result = {
                    "model_name": model_name,
                    "image_name": test_input["image_name"],
                    "theme": test_input["theme"],
                    "caption": test_input["caption"],
                    "story": story,
                    "runtime_seconds": runtime_seconds,
                    "status": "success",
                    "error": "",
                    **scores,
                }
                all_results.append(result)
                print(
                    f"{model_name} | {test_input['image_name']} | "
                    f"score={scores['overall_score']:.2f} | "
                    f"words={int(scores['word_count'])} | time={runtime_seconds:.2f}s",
                    flush=True,
                )
            del story_pipeline
            release_model_resources()
        except Exception as error:  # Preserve other model results if one candidate fails.
            print(f"ERROR | {model_name} | {type(error).__name__}: {error}", flush=True)
            all_results.append(
                {
                    "model_name": model_name,
                    "image_name": "",
                    "theme": "",
                    "caption": "",
                    "story": "",
                    "runtime_seconds": 0.0,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "word_count": 0.0,
                    "image_relevance": 0.0,
                    "child_suitability": 0.0,
                    "theme_match": 0.0,
                    "length_score": 0.0,
                    "overall_score": 0.0,
                }
            )
            release_model_resources()
        save_results(all_results)

    save_results(all_results)
    print("Saved story benchmark results.", flush=True)


if __name__ == "__main__":
    main()
