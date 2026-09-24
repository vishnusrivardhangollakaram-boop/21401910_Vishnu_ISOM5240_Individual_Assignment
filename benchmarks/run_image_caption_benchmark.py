"""Benchmark ten Hugging Face image-caption models on ten local images."""

# Import Part

import argparse
import csv
import gc
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import pipeline

from model_catalog import IMAGE_CAPTION_MODELS


# Function Part

BENCHMARK_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = BENCHMARK_DIRECTORY.parent
TEST_IMAGE_DIRECTORY = PROJECT_DIRECTORY / "test_images"
GROUND_TRUTH_PATH = BENCHMARK_DIRECTORY / "test_image_ground_truth.csv"


def load_ground_truth() -> list[dict[str, str]]:
    """Load the expected concepts for each benchmark image."""
    with GROUND_TRUTH_PATH.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def extract_caption(model_output: object) -> str:
    """Extract caption text from common image-to-text pipeline outputs."""
    if isinstance(model_output, list) and model_output:
        first_result = model_output[0]
        if isinstance(first_result, dict):
            return str(
                first_result.get("generated_text")
                or first_result.get("caption")
                or first_result.get("text")
                or ""
            ).strip()
    return str(model_output).strip()


def calculate_concept_recall(caption: str, expected_concepts: str) -> float:
    """Calculate transparent keyword-concept recall for one caption."""
    normalized_caption = caption.lower()
    concept_groups = [group.split("|") for group in expected_concepts.split(";")]
    matched_groups = sum(
        any(synonym.strip() in normalized_caption for synonym in group)
        for group in concept_groups
    )
    return matched_groups / max(len(concept_groups), 1)


def release_model_resources() -> None:
    """Release model memory before loading the next candidate."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_model(model_name: str, test_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Run one caption model on every test image and return detailed results."""
    model_results = []
    caption_pipeline = pipeline(
        "image-to-text",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
    )

    for test_row in test_rows:
        image_path = TEST_IMAGE_DIRECTORY / test_row["image_name"]
        resized_image = Image.open(image_path).convert("RGB")
        resized_image.thumbnail((768, 768), Image.Resampling.LANCZOS)

        start_time = time.perf_counter()
        model_output = caption_pipeline(resized_image, max_new_tokens=40)
        runtime_seconds = time.perf_counter() - start_time
        caption = extract_caption(model_output)
        concept_recall = calculate_concept_recall(caption, test_row["expected_concepts"])

        model_results.append(
            {
                "model_name": model_name,
                "image_name": test_row["image_name"],
                "caption": caption,
                "concept_recall": concept_recall,
                "runtime_seconds": runtime_seconds,
                "status": "success",
                "error": "",
            }
        )
        print(
            f"{model_name} | {test_row['image_name']} | "
            f"recall={concept_recall:.2f} | time={runtime_seconds:.2f}s | {caption}"
        , flush=True)

    del caption_pipeline
    release_model_resources()
    return model_results


def summarize_results(detailed_results: pd.DataFrame) -> pd.DataFrame:
    """Create one comparable summary row per model."""
    successful_results = detailed_results[detailed_results["status"] == "success"]
    summary = (
        successful_results.groupby("model_name", as_index=False)
        .agg(
            mean_concept_recall=("concept_recall", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            successful_images=("image_name", "count"),
        )
        .sort_values(
            ["mean_concept_recall", "mean_runtime_seconds"],
            ascending=[False, True],
        )
    )
    summary["accuracy_percent"] = summary["mean_concept_recall"] * 100.0
    return summary


def save_results(all_results: list[dict[str, object]]) -> None:
    """Checkpoint detailed and summary results after every model."""
    detailed_results = pd.DataFrame(all_results)
    detailed_results.to_csv(BENCHMARK_DIRECTORY / "image_caption_detailed_results.csv", index=False)
    summarize_results(detailed_results).to_csv(
        BENCHMARK_DIRECTORY / "image_caption_summary.csv", index=False
    )


def parse_arguments() -> argparse.Namespace:
    """Parse optional model filters for resumable benchmarking."""
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--models",
        nargs="*",
        default=IMAGE_CAPTION_MODELS,
        help="Model IDs to benchmark; defaults to all ten candidates.",
    )
    return argument_parser.parse_args()


def main() -> None:
    """Run the image-caption benchmark and save detailed and summary CSV files."""
    arguments = parse_arguments()
    test_rows = load_ground_truth()
    detailed_result_path = BENCHMARK_DIRECTORY / "image_caption_detailed_results.csv"
    if detailed_result_path.exists():
        previous_results = pd.read_csv(detailed_result_path).fillna("")
        all_results = previous_results.to_dict("records")
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
            all_results.extend(benchmark_model(model_name, test_rows))
        except Exception as error:  # Continue so one incompatible model does not end the study.
            print(f"ERROR | {model_name} | {type(error).__name__}: {error}")
            all_results.append(
                {
                    "model_name": model_name,
                    "image_name": "",
                    "caption": "",
                    "concept_recall": 0.0,
                    "runtime_seconds": 0.0,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            release_model_resources()
        save_results(all_results)

    save_results(all_results)
    print("Saved image-caption benchmark results.", flush=True)


if __name__ == "__main__":
    main()
