from __future__ import annotations

import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.classifier import run_classifier
from pipeline.config import CONFIG
from pipeline.data import make_signal_and_target
from pipeline.generators import PreparedTrainingData
from pipeline.results import (
    load_completed,
    save_failure,
    save_success,
)
from pipeline.splits import load_real_split


# Only the genuinely new intermediate mixtures are trained here.
# The existing real baseline supplies 0% synthetic, while the existing
# VAE-reconstruction method supplies 100% synthetic.
RATIO_METHODS = {
    "gaussian_cct_replace_25pct_real": 0.25,
    "gaussian_cct_replace_50pct_real": 0.50,
    "gaussian_cct_replace_75pct_real": 0.75,
}

METHOD_ORDER = {
    "gaussian_cct_replace_0pct_real": 0,
    "gaussian_cct_replace_25pct_real": 25,
    "gaussian_cct_replace_50pct_real": 50,
    "gaussian_cct_replace_75pct_real": 75,
    "gaussian_cct_replace_100pct_real": 100,
}

RATIO_PROTOCOL_ID = (
    "gaussian_class_channel_time_fixed_size_ratios_90_10_seed42"
)

RATIO_DIRECTORY = (
    CONFIG.output_directory
    / "results"
    / "ratios"
)

RATIO_RAW_RESULTS = (
    RATIO_DIRECTORY
    / "gaussian_replacement_ratio_runs.csv"
)

RATIO_PARTICIPANT_SUMMARY = (
    RATIO_DIRECTORY
    / "gaussian_replacement_ratio_participant_summary.csv"
)

RATIO_METHOD_SUMMARY = (
    RATIO_DIRECTORY
    / "gaussian_replacement_ratio_method_summary.csv"
)

COMPLETION_MARKER = (
    CONFIG.manifest_directory
    / "gaussian_replacement_ratios_completed.txt"
)


def reconstruction_file(
    subject_id: int,
    generator_seed: int,
) -> Path:
    return (
        (CONFIG.gaussian_data_directory / "gaussian_class_channel_time")
        / (
            f"S{subject_id:02d}_"
            f"generator-seed{generator_seed}.npz"
        )
    )


def load_reconstruction(
    subject_id: int,
    generator_seed: int,
    expected_X: np.ndarray,
    expected_y: np.ndarray,
) -> tuple[np.ndarray, Path]:
    path = reconstruction_file(
        subject_id,
        generator_seed,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing final Gaussian class-channel-time generation: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        X_reconstructed = np.asarray(
            data["X"],
            dtype=np.float32,
        )
        y_reconstructed = np.asarray(
            data["y"],
            dtype=np.int64,
        )

    if X_reconstructed.shape != expected_X.shape:
        raise RuntimeError(
            f"{path} has shape "
            f"{X_reconstructed.shape}; "
            f"expected {expected_X.shape}"
        )

    if not np.array_equal(
        y_reconstructed,
        expected_y,
    ):
        raise RuntimeError(
            f"{path} labels do not match "
            "the frozen training labels"
        )

    if not np.isfinite(
        X_reconstructed
    ).all():
        raise RuntimeError(
            f"{path} contains NaN or infinity"
        )

    return X_reconstructed, path


def indices_to_replace(
    labels: np.ndarray,
    subject_id: int,
    replacement_ratio: float,
) -> np.ndarray:
    """
    Select a fixed, class-balanced, nested subset.

    For each participant and class, the same fixed ordering is used
    for every ratio and generator seed. Therefore the 25% subset is
    contained in the 50% subset, which is contained in the 75% subset.
    """

    selected_indices: list[int] = []

    for class_id in CONFIG.class_ids:
        class_indices = np.flatnonzero(
            labels == class_id
        )

        if len(class_indices) == 0:
            raise RuntimeError(
                f"No training trials for class {class_id}"
            )

        selection_seed = (
            20260806
            + subject_id * 100
            + class_id
        )

        rng = np.random.RandomState(
            selection_seed
        )

        ordered_indices = rng.permutation(
            class_indices
        )

        number_to_replace = int(
            round(
                replacement_ratio
                * len(class_indices)
            )
        )

        selected_indices.extend(
            ordered_indices[
                :number_to_replace
            ].tolist()
        )

    return np.asarray(
        sorted(selected_indices),
        dtype=np.int64,
    )


def build_fixed_size_mixture(
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_reconstructed: np.ndarray,
    subject_id: int,
    replacement_ratio: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Replace selected real trials by their corresponding reconstructions.

    Total size, labels, class counts and trial order remain unchanged.
    """

    replacement_indices = (
        indices_to_replace(
            labels=y_real,
            subject_id=subject_id,
            replacement_ratio=(
                replacement_ratio
            ),
        )
    )

    X_mixed = np.asarray(
        X_real,
        dtype=np.float32,
    ).copy()

    y_mixed = np.asarray(
        y_real,
        dtype=np.int64,
    ).copy()

    X_mixed[
        replacement_indices
    ] = X_reconstructed[
        replacement_indices
    ]

    number_synthetic = len(
        replacement_indices
    )
    number_real = (
        len(X_mixed)
        - number_synthetic
    )

    if X_mixed.shape != X_real.shape:
        raise RuntimeError(
            "Fixed-size replacement changed "
            "the EEG shape"
        )

    if not np.array_equal(
        y_mixed,
        y_real,
    ):
        raise RuntimeError(
            "Fixed-size replacement changed labels"
        )

    if not np.isfinite(
        X_mixed
    ).all():
        raise RuntimeError(
            "Mixed training EEG contains "
            "NaN or infinity"
        )

    return (
        X_mixed,
        y_mixed,
        number_real,
        number_synthetic,
    )


def verify_main_results() -> pd.DataFrame:
    if not CONFIG.raw_results_csv.exists():
        raise FileNotFoundError(
            "Missing main result table: "
            f"{CONFIG.raw_results_csv}"
        )

    results = pd.read_csv(
        CONFIG.raw_results_csv
    )

    successful = results[
        results["status"] == "success"
    ].copy()

    # The final summary printed by the completed run contains:
    # 27 baseline rows + 12 methods × 81 rows = 999.
    if len(successful) != 999:
        raise RuntimeError(
            "Expected 999 successful main-experiment "
            f"rows, found {len(successful)}"
        )

    baseline_count = int(
        (
            successful["method"]
            == "baseline"
        ).sum()
    )

    reconstruction_count = int(
        (
            successful["method"]
            == "gaussian_class_channel_time"
        ).sum()
    )

    if baseline_count != 27:
        raise RuntimeError(
            f"Expected 27 baseline rows, "
            f"found {baseline_count}"
        )

    if reconstruction_count != 81:
        raise RuntimeError(
            "Expected 81 VAE-reconstruction rows, "
            f"found {reconstruction_count}"
        )

    print(
        "Main result table passed: "
        "999 successful rows.",
        flush=True,
    )

    return successful


def preflight() -> None:
    verify_main_results()

    validated = 0

    for subject_id in CONFIG.subject_numbers:
        split = load_real_split(
            subject_id,
            CONFIG,
        )

        for generator_seed in (
            CONFIG.generator_seeds
        ):
            load_reconstruction(
                subject_id=subject_id,
                generator_seed=(
                    generator_seed
                ),
                expected_X=split.X_train,
                expected_y=split.y_train,
            )

            validated += 1

    if validated != 27:
        raise RuntimeError(
            f"Expected 27 reconstruction files, "
            f"validated {validated}"
        )

    print(
        "All 27 final reconstruction "
        "datasets passed.",
        flush=True,
    )


def run_ratio_experiments() -> None:
    ratio_config = replace(
        CONFIG,
        protocol_id=RATIO_PROTOCOL_ID,
        raw_results_csv=RATIO_RAW_RESULTS,
        participant_summary_csv=(
            RATIO_PARTICIPANT_SUMMARY
        ),
        method_summary_csv=(
            RATIO_METHOD_SUMMARY
        ),
    )

    RATIO_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for method, ratio in (
        RATIO_METHODS.items()
    ):
        completed = load_completed(
            ratio_config,
            method,
        )

        print(
            "\n"
            + "=" * 80,
            flush=True,
        )
        print(
            f"RATIO METHOD: {method}",
            flush=True,
        )
        print(
            f"Replacement ratio: "
            f"{ratio * 100:.0f}% synthetic",
            flush=True,
        )
        print(
            "=" * 80,
            flush=True,
        )

        for subject_id in (
            CONFIG.subject_numbers
        ):
            split = load_real_split(
                subject_id,
                CONFIG,
            )

            valid_set = (
                make_signal_and_target(
                    split.X_valid,
                    split.y_valid,
                )
            )

            test_set = (
                make_signal_and_target(
                    split.X_test,
                    split.y_test,
                )
            )

            for generator_seed in (
                CONFIG.generator_seeds
            ):
                (
                    X_reconstructed,
                    reconstruction_path,
                ) = load_reconstruction(
                    subject_id=subject_id,
                    generator_seed=(
                        generator_seed
                    ),
                    expected_X=split.X_train,
                    expected_y=split.y_train,
                )

                (
                    X_mixed,
                    y_mixed,
                    number_real,
                    number_synthetic,
                ) = build_fixed_size_mixture(
                    X_real=split.X_train,
                    y_real=split.y_train,
                    X_reconstructed=(
                        X_reconstructed
                    ),
                    subject_id=subject_id,
                    replacement_ratio=ratio,
                )

                train_set = (
                    make_signal_and_target(
                        X_mixed,
                        y_mixed,
                    )
                )

                notes = (
                    f"Fixed-size paired replacement; "
                    f"requested synthetic ratio="
                    f"{ratio * 100:.0f}%; "
                    f"real trials={number_real}; "
                    f"synthetic trials="
                    f"{number_synthetic}; "
                    f"total trials={len(X_mixed)}; "
                    "class counts and frozen label "
                    "order preserved; "
                    "replacement subsets nested "
                    "across ratios; "
                    "validation and test are real EEG"
                )

                prepared_training = (
                    PreparedTrainingData(
                        dataset=train_set,
                        train_data_type=(
                            "real_vae_fixed_size_"
                            "replacement"
                        ),
                        train_data_file=str(
                            reconstruction_path
                        ),
                        n_real_train_trials=(
                            number_real
                        ),
                        n_synthetic_train_trials=(
                            number_synthetic
                        ),
                        notes=notes,
                    )
                )

                for classifier_seed in (
                    CONFIG.classifier_seeds
                ):
                    run_key = (
                        subject_id,
                        classifier_seed,
                        generator_seed,
                    )

                    if run_key in completed:
                        print(
                            "SKIP "
                            f"method={method}, "
                            f"subject={subject_id}, "
                            f"generator_seed="
                            f"{generator_seed}, "
                            f"classifier_seed="
                            f"{classifier_seed}",
                            flush=True,
                        )
                        continue

                    print(
                        "START "
                        f"method={method}, "
                        f"subject={subject_id}, "
                        f"generator_seed="
                        f"{generator_seed}, "
                        f"classifier_seed="
                        f"{classifier_seed}, "
                        f"real={number_real}, "
                        f"synthetic="
                        f"{number_synthetic}",
                        flush=True,
                    )

                    try:
                        row = run_classifier(
                            train_set=train_set,
                            valid_set=valid_set,
                            test_set=test_set,
                            method=method,
                            subject_id=subject_id,
                            classifier_seed=(
                                classifier_seed
                            ),
                            generator_seed=(
                                generator_seed
                            ),
                            prepared_training=(
                                prepared_training
                            ),
                            split_file=(
                                split.split_file
                            ),
                            config=ratio_config,
                        )

                        save_success(
                            ratio_config,
                            row,
                        )

                        print(
                            "DONE "
                            f"method={method}, "
                            f"subject={subject_id}, "
                            f"generator_seed="
                            f"{generator_seed}, "
                            f"classifier_seed="
                            f"{classifier_seed}, "
                            f"test_accuracy="
                            f"{1.0 - row['test_misclass']:.4f}",
                            flush=True,
                        )

                    except Exception as error:
                        print(
                            "\nFULL TRACEBACK:",
                            flush=True,
                        )
                        traceback.print_exc()

                        save_failure(
                            config=ratio_config,
                            method=method,
                            subject_id=subject_id,
                            classifier_seed=(
                                classifier_seed
                            ),
                            generator_seed=(
                                generator_seed
                            ),
                            error=error,
                            split_file=(
                                split.split_file
                            ),
                            notes=notes,
                        )


def create_combined_summary() -> None:
    main_successful = verify_main_results()

    if not RATIO_RAW_RESULTS.exists():
        raise FileNotFoundError(
            f"Missing ratio result table: "
            f"{RATIO_RAW_RESULTS}"
        )

    ratio_results = pd.read_csv(
        RATIO_RAW_RESULTS
    )

    ratio_successful = ratio_results[
        ratio_results["status"]
        == "success"
    ].copy()

    expected_new_rows = (
        len(RATIO_METHODS)
        * len(CONFIG.subject_numbers)
        * len(CONFIG.generator_seeds)
        * len(CONFIG.classifier_seeds)
    )

    if len(ratio_successful) != (
        expected_new_rows
    ):
        raise RuntimeError(
            "Expected "
            f"{expected_new_rows} successful "
            "new ratio rows, found "
            f"{len(ratio_successful)}"
        )

    baseline = main_successful[
        main_successful["method"]
        == "baseline"
    ].copy()

    baseline["ratio_method"] = (
        "gaussian_cct_replace_0pct_real"
    )

    reconstruction = main_successful[
        main_successful["method"]
        == "gaussian_class_channel_time"
    ].copy()

    reconstruction["ratio_method"] = (
        "gaussian_cct_replace_100pct_real"
    )

    intermediate = ratio_successful.copy()

    intermediate["ratio_method"] = (
        intermediate["method"]
    )

    combined = pd.concat(
        [
            baseline,
            intermediate,
            reconstruction,
        ],
        ignore_index=True,
        sort=False,
    )

    combined["test_misclass"] = (
        pd.to_numeric(
            combined["test_misclass"],
            errors="raise",
        )
    )

    combined["test_accuracy"] = (
        1.0
        - combined["test_misclass"]
    )

    combined["generator_seed_numeric"] = (
        pd.to_numeric(
            combined["generator_seed"],
            errors="coerce",
        )
    )

    duplicate_columns = [
        "ratio_method",
        "subject_id",
        "generator_seed_numeric",
        "classifier_seed",
    ]

    if combined.duplicated(
        subset=duplicate_columns,
        keep=False,
    ).any():
        raise RuntimeError(
            "Duplicate successful ratio keys "
            "were detected"
        )

    subject_generator = (
        combined.groupby(
            [
                "ratio_method",
                "subject_id",
                "generator_seed_numeric",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            classifier_seed_mean_test_accuracy=(
                "test_accuracy",
                "mean",
            ),
            n_classifier_runs=(
                "classifier_seed",
                "count",
            ),
        )
    )

    participant_summary = (
        subject_generator.groupby(
            [
                "ratio_method",
                "subject_id",
            ],
            as_index=False,
        )
        .agg(
            participant_test_accuracy=(
                "classifier_seed_mean_test_accuracy",
                "mean",
            ),
            n_generator_conditions=(
                "classifier_seed_mean_test_accuracy",
                "size",
            ),
            n_classifier_runs=(
                "n_classifier_runs",
                "sum",
            ),
        )
    )

    participant_summary[
        "synthetic_percentage"
    ] = participant_summary[
        "ratio_method"
    ].map(
        METHOD_ORDER
    )

    method_summary = (
        participant_summary.groupby(
            [
                "ratio_method",
                "synthetic_percentage",
            ],
            as_index=False,
        )
        .agg(
            macro_mean_test_accuracy=(
                "participant_test_accuracy",
                "mean",
            ),
            participant_std_test_accuracy=(
                "participant_test_accuracy",
                "std",
            ),
            n_subjects=(
                "subject_id",
                "nunique",
            ),
            n_classifier_runs=(
                "n_classifier_runs",
                "sum",
            ),
        )
        .sort_values(
            "synthetic_percentage"
        )
    )

    participant_summary = (
        participant_summary.sort_values(
            [
                "synthetic_percentage",
                "subject_id",
            ]
        )
    )

    participant_summary.to_csv(
        RATIO_PARTICIPANT_SUMMARY,
        index=False,
    )

    method_summary.to_csv(
        RATIO_METHOD_SUMMARY,
        index=False,
    )

    expected_combined_rows = (
        27
        + expected_new_rows
        + 81
    )

    if len(combined) != (
        expected_combined_rows
    ):
        raise RuntimeError(
            "Expected "
            f"{expected_combined_rows} combined "
            f"ratio rows, found {len(combined)}"
        )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )
    print(
        "FIXED-SIZE GAUSSIAN CLASS-CHANNEL-TIME REPLACEMENT SUMMARY",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )
    print(
        method_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        ),
        flush=True,
    )

    print(
        "\nRaw new ratio results:",
        RATIO_RAW_RESULTS,
        flush=True,
    )
    print(
        "Participant summary:",
        RATIO_PARTICIPANT_SUMMARY,
        flush=True,
    )
    print(
        "Method summary:",
        RATIO_METHOD_SUMMARY,
        flush=True,
    )

    COMPLETION_MARKER.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    COMPLETION_MARKER.write_text(
        "VAE fixed-size replacement "
        "ratio experiment completed.\n",
        encoding="utf-8",
    )


def main() -> None:
    print(
        "=" * 80,
        flush=True,
    )
    print(
        "VAE RECONSTRUCTION FIXED-SIZE "
        "REPLACEMENT EXPERIMENT",
        flush=True,
    )
    print(
        "New conditions: 25%, 50%, "
        "75% synthetic",
        flush=True,
    )
    print(
        "Existing endpoints reused: "
        "0% and 100% synthetic",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    preflight()
    run_ratio_experiments()
    create_combined_summary()

    print(
        "\nRATIO EXPERIMENT COMPLETED",
        flush=True,
    )


if __name__ == "__main__":
    main()
