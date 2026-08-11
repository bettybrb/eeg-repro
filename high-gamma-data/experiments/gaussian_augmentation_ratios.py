#!/usr/bin/env python3

"""
Additive Gaussian augmentation experiment.

Unlike the fixed-size replacement experiment, all 259 real
training trials are retained. Increasing amounts of the existing
Gaussian class-channel-time synthetic data are added on top.

Conditions:
    0%   -> 259 real +   0 synthetic   [existing baseline reused]
   +25%  -> 259 real + ~65 synthetic
   +50%  -> 259 real + ~129 synthetic
   +75%  -> 259 real + ~194 synthetic
  +100%  -> 259 real + 259 synthetic

Validation and official test data remain unchanged and entirely real.
"""

import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.braindecode_setup import (
    apply_compatibility_patches,
    setup_logging,
)

apply_compatibility_patches()

from pipeline.classifier import run_classifier
from pipeline.config import CONFIG
from pipeline.data import load_train_valid_test
from pipeline.generators import (
    PreparedTrainingData,
    copy_dataset,
    prepare_training_data,
)
from pipeline.results import (
    load_completed,
    save_failure,
    save_success,
)


BASE_SYNTHETIC_METHOD = (
    "gaussian_class_channel_time"
)

AUGMENTATION_RATIOS = {
    "gaussian_cct_add_25pct_synth": 0.25,
    "gaussian_cct_add_50pct_synth": 0.50,
    "gaussian_cct_add_75pct_synth": 0.75,
    "gaussian_cct_add_100pct_synth": 1.00,
}

SUBJECTS = tuple(CONFIG.subject_numbers)
GENERATOR_SEEDS = tuple(CONFIG.generator_seeds)
CLASSIFIER_SEEDS = tuple(CONFIG.classifier_seeds)

# Fixed only for deciding which synthetic trials enter the
# 25/50/75% subsets. The subsets are nested and do not change
# across classifier seeds.
SELECTION_SEED = 20260807

RESULT_DIRECTORY = (
    CONFIG.project_root
    / "outputs"
    / "results"
    / "ratios"
)

RAW_RESULTS = (
    RESULT_DIRECTORY
    / "gaussian_augmentation_ratio_runs.csv"
)

PARTICIPANT_SUMMARY = (
    RESULT_DIRECTORY
    / "gaussian_augmentation_ratio_participant_summary.csv"
)

METHOD_SUMMARY = (
    RESULT_DIRECTORY
    / "gaussian_augmentation_ratio_method_summary.csv"
)

# Existing final results are used only to recover the already
# completed real-only baseline.
MAIN_RESULTS = Path(
    CONFIG.raw_results_csv
)


def select_nested_synthetic_subset(
    X_synthetic,
    y_synthetic,
    ratio,
    subject_id,
    generator_seed,
):
    """
    Select a fixed class-stratified subset of synthetic EEG.

    A single fixed ordering is created for every
    subject × generator seed × class.

    Therefore:
        25% subset ⊂ 50% subset ⊂ 75% subset ⊂ 100% set.
    """

    selected_indices = []

    for class_id in CONFIG.class_ids:
        class_indices = np.flatnonzero(
            y_synthetic == class_id
        )

        if len(class_indices) == 0:
            raise RuntimeError(
                f"No synthetic trials for class {class_id}"
            )

        selection_rng = np.random.RandomState(
            SELECTION_SEED
            + subject_id * 10000
            + generator_seed * 100
            + int(class_id)
        )

        ordered_indices = selection_rng.permutation(
            class_indices
        )

        number_to_add = int(
            round(
                ratio
                * len(class_indices)
            )
        )

        selected_indices.extend(
            ordered_indices[
                :number_to_add
            ].tolist()
        )

    selected_indices = np.asarray(
        selected_indices,
        dtype=np.int64,
    )

    return (
        np.asarray(
            X_synthetic[selected_indices],
            dtype=np.float32,
        ),
        np.asarray(
            y_synthetic[selected_indices],
            dtype=np.int64,
        ),
    )


def build_augmented_training_set(
    real_train_set,
    full_synthetic_training,
    ratio,
    subject_id,
    generator_seed,
    classifier_seed,
):
    """
    Keep every real training trial and add a class-stratified
    subset of the Gaussian CCT synthetic data.
    """

    X_synthetic, y_synthetic = (
        select_nested_synthetic_subset(
            X_synthetic=(
                full_synthetic_training
                .dataset
                .X
            ),
            y_synthetic=(
                full_synthetic_training
                .dataset
                .y
            ),
            ratio=ratio,
            subject_id=subject_id,
            generator_seed=generator_seed,
        )
    )

    number_real = int(
        len(real_train_set.X)
    )
    number_synthetic = int(
        len(X_synthetic)
    )

    combined_X = np.concatenate(
        [
            np.asarray(
                real_train_set.X,
                dtype=np.float32,
            ),
            X_synthetic,
        ],
        axis=0,
    )

    combined_y = np.concatenate(
        [
            np.asarray(
                real_train_set.y,
                dtype=np.int64,
            ),
            y_synthetic,
        ],
        axis=0,
    )

    # Same sample set for all classifier seeds.
    # Only the initial presentation order changes with the
    # classifier seed, as part of classifier randomness.
    ordering_rng = np.random.RandomState(
        920000
        + subject_id * 10000
        + generator_seed * 1000
        + classifier_seed * 10
        + int(ratio * 100)
    )

    order = ordering_rng.permutation(
        len(combined_X)
    )

    augmented_set = copy_dataset(
        real_train_set
    )

    augmented_set.X = combined_X[
        order
    ]

    augmented_set.y = combined_y[
        order
    ]

    total = (
        number_real
        + number_synthetic
    )

    final_synthetic_percentage = (
        100.0
        * number_synthetic
        / total
    )

    notes = (
        "Additive augmentation with Gaussian "
        "class-channel-time synthetic EEG; "
        f"all {number_real} real training trials retained; "
        f"{number_synthetic} synthetic trials added; "
        f"synthetic amount={ratio * 100:.0f}% "
        "relative to original real training size; "
        f"final training size={total}; "
        f"final synthetic composition="
        f"{final_synthetic_percentage:.2f}%; "
        "synthetic subset class-stratified and nested; "
        "validation and official test sets remain entirely real."
    )

    return PreparedTrainingData(
        dataset=augmented_set,
        train_data_type=(
            "real_plus_synthetic_gaussian_cct"
        ),
        train_data_file=(
            full_synthetic_training
            .train_data_file
        ),
        n_real_train_trials=(
            number_real
        ),
        n_synthetic_train_trials=(
            number_synthetic
        ),
        notes=notes,
    )


def summarise_results():
    """
    Aggregate in the same hierarchy as the final experiment:

    classifier seeds
        -> generator seed
        -> participant
        -> macro mean across participants.
    """

    if not RAW_RESULTS.exists():
        raise RuntimeError(
            f"Missing results file: {RAW_RESULTS}"
        )

    raw = pd.read_csv(
        RAW_RESULTS
    )

    successful = raw[
        raw["status"] == "success"
    ].copy()

    successful["test_accuracy"] = (
        1.0
        - pd.to_numeric(
            successful[
                "test_misclass"
            ],
            errors="raise",
        )
    )

    participant_rows = []
    method_rows = []


    # --------------------------------------------------------
    # Existing 0% real-only baseline
    # --------------------------------------------------------

    final_results = pd.read_csv(
        MAIN_RESULTS
    )

    baseline = final_results[
        (
            final_results["status"]
            == "success"
        )
        & (
            final_results["method"]
            == "baseline"
        )
        & (
            final_results["protocol_id"]
            == CONFIG.protocol_id
        )
    ].copy()

    baseline[
        "test_accuracy"
    ] = (
        1.0
        - pd.to_numeric(
            baseline[
                "test_misclass"
            ],
            errors="raise",
        )
    )

    baseline_participant = (
        baseline
        .groupby(
            "subject_id"
        )[
            "test_accuracy"
        ]
        .mean()
    )

    if len(
        baseline_participant
    ) != 9:
        raise RuntimeError(
            "Expected completed baseline for "
            f"9 participants, found "
            f"{len(baseline_participant)}"
        )

    for (
        subject_id,
        accuracy,
    ) in baseline_participant.items():

        participant_rows.append(
            {
                "ratio_method": (
                    "baseline_real_only"
                ),
                "synthetic_added_percentage_of_real": 0,
                "subject_id": int(
                    subject_id
                ),
                "participant_mean_test_accuracy": float(
                    accuracy
                ),
            }
        )

    method_rows.append(
        {
            "ratio_method": (
                "baseline_real_only"
            ),
            "synthetic_added_percentage_of_real": 0,
            "mean_final_synthetic_composition_percentage": 0.0,
            "macro_mean_test_accuracy": float(
                baseline_participant.mean()
            ),
            "participant_std_test_accuracy": float(
                baseline_participant.std()
            ),
            "mean_real_train_trials": 259.0,
            "mean_synthetic_train_trials": 0.0,
            "mean_total_train_trials": 259.0,
            "n_subjects": int(
                len(
                    baseline_participant
                )
            ),
            "n_classifier_runs": int(
                len(
                    baseline
                )
            ),
        }
    )


    # --------------------------------------------------------
    # Additive synthetic conditions
    # --------------------------------------------------------

    for (
        method,
        ratio,
    ) in AUGMENTATION_RATIOS.items():

        method_data = successful[
            successful["method"]
            == method
        ].copy()

        if len(method_data) == 0:
            continue

        # Average classifier seeds first.
        subject_generator = (
            method_data
            .groupby(
                [
                    "subject_id",
                    "generator_seed",
                ],
                as_index=False,
            )[
                "test_accuracy"
            ]
            .mean()
        )

        # Then average generator repetitions within participant.
        participant = (
            subject_generator
            .groupby(
                "subject_id"
            )[
                "test_accuracy"
            ]
            .mean()
        )

        for (
            subject_id,
            accuracy,
        ) in participant.items():

            participant_rows.append(
                {
                    "ratio_method": method,
                    "synthetic_added_percentage_of_real": int(
                        ratio * 100
                    ),
                    "subject_id": int(
                        subject_id
                    ),
                    "participant_mean_test_accuracy": float(
                        accuracy
                    ),
                }
            )

        real_counts = pd.to_numeric(
            method_data[
                "n_real_train_trials"
            ],
            errors="raise",
        )

        synthetic_counts = pd.to_numeric(
            method_data[
                "n_synthetic_train_trials"
            ],
            errors="raise",
        )

        total_counts = (
            real_counts
            + synthetic_counts
        )

        final_synthetic_composition = (
            100.0
            * synthetic_counts
            / total_counts
        )

        method_rows.append(
            {
                "ratio_method": method,
                "synthetic_added_percentage_of_real": int(
                    ratio * 100
                ),
                "mean_final_synthetic_composition_percentage": float(
                    final_synthetic_composition.mean()
                ),
                "macro_mean_test_accuracy": float(
                    participant.mean()
                ),
                "participant_std_test_accuracy": float(
                    participant.std()
                ),
                "mean_real_train_trials": float(
                    real_counts.mean()
                ),
                "mean_synthetic_train_trials": float(
                    synthetic_counts.mean()
                ),
                "mean_total_train_trials": float(
                    total_counts.mean()
                ),
                "n_subjects": int(
                    len(
                        participant
                    )
                ),
                "n_classifier_runs": int(
                    len(
                        method_data
                    )
                ),
            }
        )


    participant_summary = pd.DataFrame(
        participant_rows
    )

    method_summary = pd.DataFrame(
        method_rows
    ).sort_values(
        "synthetic_added_percentage_of_real"
    )

    PARTICIPANT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    participant_summary.to_csv(
        PARTICIPANT_SUMMARY,
        index=False,
    )

    method_summary.to_csv(
        METHOD_SUMMARY,
        index=False,
    )

    print()
    print("=" * 80)
    print(
        "REAL + GAUSSIAN CLASS-CHANNEL-TIME "
        "ADDITIVE AUGMENTATION SUMMARY"
    )
    print("=" * 80)

    print(
        method_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Raw augmentation results:",
        RAW_RESULTS,
    )

    print(
        "Participant summary:",
        PARTICIPANT_SUMMARY,
    )

    print(
        "Method summary:",
        METHOD_SUMMARY,
    )


def main():
    setup_logging()

    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Same frozen protocol and classifier configuration,
    # but write this extension to its own result table.
    config = replace(
        CONFIG,
        raw_results_csv=RAW_RESULTS,
    )

    completed = {
        method: load_completed(
            config,
            method,
        )
        for method
        in AUGMENTATION_RATIOS
    }

    print("=" * 80)
    print(
        "GAUSSIAN CCT ADDITIVE AUGMENTATION EXPERIMENT"
    )
    print("=" * 80)

    print(
        "Real trials retained per participant:",
        CONFIG.expected_train_trials,
    )

    print(
        "Synthetic amounts:",
        "25%, 50%, 75%, 100% of original real training size",
    )

    print(
        "Subjects:",
        SUBJECTS,
    )

    print(
        "Generator seeds:",
        GENERATOR_SEEDS,
    )

    print(
        "Classifier seeds:",
        CLASSIFIER_SEEDS,
    )

    print(
        "New classifier runs:",
        (
            len(AUGMENTATION_RATIOS)
            * len(SUBJECTS)
            * len(GENERATOR_SEEDS)
            * len(CLASSIFIER_SEEDS)
        ),
    )

    print(
        "0% condition: existing baseline reused; not rerun."
    )

    print(
        "Results:",
        RAW_RESULTS,
    )


    for subject_id in SUBJECTS:

        try:
            (
                real_train_set,
                valid_set,
                test_set,
                split_file,
            ) = load_train_valid_test(
                subject_id=subject_id,
                config=config,
            )

        except Exception as error:
            print(
                "\nFAILED TO LOAD REAL SPLIT "
                f"FOR SUBJECT {subject_id}"
            )
            traceback.print_exc()
            continue


        for generator_seed in GENERATOR_SEEDS:

            # Load the exact full Gaussian CCT synthetic
            # dataset already produced by the final pipeline.
            try:
                full_synthetic_training = (
                    prepare_training_data(
                        real_train_set=(
                            real_train_set
                        ),
                        method=(
                            BASE_SYNTHETIC_METHOD
                        ),
                        subject_id=(
                            subject_id
                        ),
                        generator_seed=(
                            generator_seed
                        ),
                        split_file=(
                            split_file
                        ),
                        config=config,
                        overwrite_gaussian=False,
                    )
                )

            except Exception as error:
                print(
                    "\nFAILED TO LOAD GAUSSIAN DATA | "
                    f"subject={subject_id} | "
                    f"generator_seed={generator_seed}"
                )
                traceback.print_exc()

                for (
                    method,
                    ratio,
                ) in AUGMENTATION_RATIOS.items():

                    for classifier_seed in CLASSIFIER_SEEDS:
                        key = (
                            subject_id,
                            classifier_seed,
                            generator_seed,
                        )

                        if key in completed[
                            method
                        ]:
                            continue

                        save_failure(
                            config=config,
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
                                split_file
                            ),
                        )

                continue


            for (
                method,
                ratio,
            ) in AUGMENTATION_RATIOS.items():

                for classifier_seed in CLASSIFIER_SEEDS:

                    run_key = (
                        subject_id,
                        classifier_seed,
                        generator_seed,
                    )

                    if run_key in completed[
                        method
                    ]:
                        print(
                            f"SKIP method={method}, "
                            f"subject={subject_id}, "
                            f"generator_seed="
                            f"{generator_seed}, "
                            f"classifier_seed="
                            f"{classifier_seed}",
                            flush=True,
                        )
                        continue


                    try:
                        prepared_training = (
                            build_augmented_training_set(
                                real_train_set=(
                                    real_train_set
                                ),
                                full_synthetic_training=(
                                    full_synthetic_training
                                ),
                                ratio=ratio,
                                subject_id=(
                                    subject_id
                                ),
                                generator_seed=(
                                    generator_seed
                                ),
                                classifier_seed=(
                                    classifier_seed
                                ),
                            )
                        )

                        print(
                            f"START method={method}, "
                            f"subject={subject_id}, "
                            f"generator_seed="
                            f"{generator_seed}, "
                            f"classifier_seed="
                            f"{classifier_seed} | "
                            f"real="
                            f"{prepared_training.n_real_train_trials} | "
                            f"synthetic="
                            f"{prepared_training.n_synthetic_train_trials} | "
                            f"total="
                            f"{len(prepared_training.dataset.X)}",
                            flush=True,
                        )

                        row = run_classifier(
                            train_set=(
                                prepared_training
                                .dataset
                            ),
                            valid_set=valid_set,
                            test_set=test_set,
                            method=method,
                            subject_id=(
                                subject_id
                            ),
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
                                split_file
                            ),
                            config=config,
                        )

                        save_success(
                            config,
                            row,
                        )

                        completed[
                            method
                        ].add(
                            run_key
                        )

                        print(
                            f"DONE method={method}, "
                            f"subject={subject_id}, "
                            f"generator_seed="
                            f"{generator_seed}, "
                            f"classifier_seed="
                            f"{classifier_seed} | "
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
                            config=config,
                            method=method,
                            subject_id=(
                                subject_id
                            ),
                            classifier_seed=(
                                classifier_seed
                            ),
                            generator_seed=(
                                generator_seed
                            ),
                            error=error,
                            split_file=(
                                split_file
                            ),
                            notes=(
                                prepared_training.notes
                                if "prepared_training"
                                in locals()
                                else ""
                            ),
                        )

                        print(
                            f"FAILED method={method}, "
                            f"subject={subject_id}, "
                            f"generator_seed="
                            f"{generator_seed}, "
                            f"classifier_seed="
                            f"{classifier_seed} | "
                            f"{error!r}",
                            flush=True,
                        )


    summarise_results()


if __name__ == "__main__":
    main()
