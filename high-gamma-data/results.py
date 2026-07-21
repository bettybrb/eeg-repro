import csv
from pathlib import Path

import pandas as pd


RESULT_COLUMNS = [
    "status",
    "experiment_type",
    "subject_id",
    "seed",
    "best_epoch",
    "best_valid_misclass",
    "best_valid_accuracy",
    "best_test_misclass",
    "best_test_accuracy",
    "last_test_misclass",
    "last_test_accuracy",
    "n_train_trials",
    "n_valid_trials",
    "n_test_trials",
    "n_channels",
    "n_times",
    "train_data",
    "valid_data",
    "test_data",
    "split_mode",
    "notes",
    "error",
]


def append_row(results_csv, row):
    Path(results_csv).parent.mkdir(parents=True, exist_ok=True)
    results_csv = Path(results_csv)
    file_exists = results_csv.exists()

    with results_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)

        if not file_exists or results_csv.stat().st_size == 0:
            writer.writeheader()

        writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})


def save_success(config, row):
    append_row(config.results_csv, row)


def save_failure(config, experiment_type, subject_id, seed, error):
    row = {
        "status": "failed",
        "experiment_type": experiment_type,
        "subject_id": subject_id,
        "seed": seed,
        "error": repr(error),
    }

    append_row(config.results_csv, row)


def load_completed(config, experiment_type):
    path = Path(config.results_csv)

    if not path.exists():
        return set()

    df = pd.read_csv(path)

    if len(df) == 0:
        return set()

    successful = df[
        (df["status"] == "success")
        & (df["experiment_type"] == experiment_type)
    ]

    return set(
        (int(row["subject_id"]), int(row["seed"]))
        for _, row in successful.iterrows()
    )
