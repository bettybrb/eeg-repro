import argparse
from pathlib import Path

import pandas as pd

from pipeline.config import CONFIG


def summarise(raw_results, participant_output, method_output):
    raw_results = Path(raw_results)

    if not raw_results.exists():
        raise FileNotFoundError(f"Raw result table not found: {raw_results}")

    results = pd.read_csv(raw_results)
    successful = results[results["status"] == "success"].copy()

    if successful.empty:
        raise RuntimeError("The raw table contains no successful runs")

    successful["test_misclass"] = pd.to_numeric(
        successful["test_misclass"], errors="raise"
    )
    successful["test_accuracy"] = 1.0 - successful["test_misclass"]
    successful["generator_seed"] = pd.to_numeric(
        successful["generator_seed"], errors="coerce"
    )

    subject_generator = (
        successful.groupby(
            ["protocol_id", "method", "subject_id", "generator_seed"],
            dropna=False,
            as_index=False,
        )
        .agg(
            classifier_seed_mean_test_accuracy=("test_accuracy", "mean"),
            n_classifier_runs=("classifier_seed", "count"),
        )
    )

    participant_summary = (
        subject_generator.groupby(
            ["protocol_id", "method", "subject_id"],
            as_index=False,
        )
        .agg(
            participant_test_accuracy=("classifier_seed_mean_test_accuracy", "mean"),
            n_generator_conditions=(
                "classifier_seed_mean_test_accuracy",
                "size",
            ),
            n_classifier_runs=("n_classifier_runs", "sum"),
        )
    )

    method_summary = (
        participant_summary.groupby(
            ["protocol_id", "method"],
            as_index=False,
        )
        .agg(
            macro_mean_test_accuracy=("participant_test_accuracy", "mean"),
            participant_std_test_accuracy=("participant_test_accuracy", "std"),
            n_subjects=("subject_id", "nunique"),
            n_classifier_runs=("n_classifier_runs", "sum"),
        )
    )

    participant_output = Path(participant_output)
    method_output = Path(method_output)

    participant_output.parent.mkdir(parents=True, exist_ok=True)
    method_output.parent.mkdir(parents=True, exist_ok=True)

    participant_summary.to_csv(participant_output, index=False)
    method_summary.to_csv(method_output, index=False)

    print(f"Participant summary: {participant_output}", flush=True)
    print(f"Method summary: {method_output}", flush=True)
    print(method_summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-results", default=str(CONFIG.raw_results_csv))
    parser.add_argument(
        "--participant-output",
        default=str(CONFIG.participant_summary_csv),
    )
    parser.add_argument(
        "--method-output",
        default=str(CONFIG.method_summary_csv),
    )
    args = parser.parse_args()

    summarise(
        raw_results=args.raw_results,
        participant_output=args.participant_output,
        method_output=args.method_output,
    )


if __name__ == "__main__":
    main()
