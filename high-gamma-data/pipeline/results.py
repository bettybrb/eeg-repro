import csv
from pathlib import Path

import pandas as pd


RESULT_COLUMNS = [
    "status",
    "protocol_id",
    "method",
    "subject_id",
    "generator_seed",
    "classifier_seed",
    "selected_epoch",
    "validation_misclass",
    "test_misclass",
    "n_real_train_trials",
    "n_synthetic_train_trials",
    "n_valid_trials",
    "n_test_trials",
    "n_channels",
    "n_times",
    "train_data_type",
    "split_file",
    "train_data_file",
    "runtime_seconds",
    "notes",
    "error",
]


def _validate_existing_schema(results_csv):
    results_csv = Path(results_csv)

    if not results_csv.exists() or results_csv.stat().st_size == 0:
        return

    existing_columns = list(pd.read_csv(results_csv, nrows=0).columns)

    if existing_columns != RESULT_COLUMNS:
        raise RuntimeError(
            f"Incompatible schema in {results_csv}\n"
            f"Existing: {existing_columns}\n"
            f"Expected: {RESULT_COLUMNS}\n"
            "Archive or remove the old file before continuing."
        )


def append_row(results_csv, row):
    results_csv = Path(results_csv)
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    _validate_existing_schema(results_csv)

    file_exists = results_csv.exists() and results_csv.stat().st_size > 0

    with results_csv.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {column: row.get(column, "") for column in RESULT_COLUMNS}
        )


def reset_results(config):
    config.raw_results_csv.unlink(missing_ok=True)


def save_success(config, row):
    append_row(config.raw_results_csv, row)


def save_failure(
    config,
    method,
    subject_id,
    classifier_seed,
    generator_seed,
    error,
    split_file="",
    notes="",
):
    row = {
        "status": "failed",
        "protocol_id": config.protocol_id,
        "method": method,
        "subject_id": subject_id,
        "generator_seed": generator_seed,
        "classifier_seed": classifier_seed,
        "split_file": str(split_file),
        "notes": notes,
        "error": repr(error),
    }
    append_row(config.raw_results_csv, row)


def _normalise_seed(value):
    if pd.isna(value) or value == "":
        return None
    return int(value)


def load_completed(config, method):
    path = Path(config.raw_results_csv)

    if not path.exists() or path.stat().st_size == 0:
        return set()

    _validate_existing_schema(path)

    results = pd.read_csv(path)

    successful = results[
        (results["status"] == "success")
        & (results["method"] == method)
        & (results["protocol_id"] == config.protocol_id)
    ]

    return {
        (
            int(row["subject_id"]),
            int(row["classifier_seed"]),
            _normalise_seed(row["generator_seed"]),
        )
        for _, row in successful.iterrows()
    }
