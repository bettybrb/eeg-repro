#!/usr/bin/env python3

import argparse
import copy
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from braindecode_setup import apply_compatibility_patches

apply_compatibility_patches()

from braindecode.datautil.signal_target import SignalAndTarget

import run
from config import CONFIG


# ============================================================
# PREVIOUS EXPERIMENTAL DESIGN: ADDITIVE AUGMENTATION
#
# All real training trials were retained and reconstructed
# trials were added on top. This increased the total size of
# the training set.
#
# Kept here to document and reproduce the earlier experiment.
# ============================================================

# RATIO_EXPERIMENTS = {
#     "vae_real_plus_25pct_synthetic": 0.25,
#     "vae_real_plus_50pct_synthetic": 0.50,
#     "vae_real_plus_100pct_synthetic": 1.00,
#     "vae_real_plus_200pct_synthetic": 2.00,
# }


# ============================================================
# CURRENT EXPERIMENTAL DESIGN: FIXED-SIZE REPLACEMENT
#
# A proportion of the real training trials is removed and
# replaced by the same number of reconstructed trials.
#
# Total training-set size and class counts remain unchanged.
# ============================================================

RATIO_EXPERIMENTS = {
    "vae_replace_0pct_real": 0.00, #baseline test
    "vae_replace_25pct_real": 0.25,
    "vae_replace_50pct_real": 0.50,
    "vae_replace_75pct_real": 0.75,
    "vae_replace_100pct_real": 1.00,
}


def parse_integer_list(value):
    return tuple(
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    )


def make_dataset(X, y):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    if X.ndim != 3:
        raise RuntimeError(
            "Expected EEG shape "
            f"(trials, channels, time), got {X.shape}"
        )

    if len(X) != len(y):
        raise RuntimeError(
            f"EEG/label length mismatch: {len(X)} versus {len(y)}"
        )

    return SignalAndTarget(X, y)


def copy_dataset(dataset):
    return make_dataset(
        np.array(dataset.X, dtype=np.float32, copy=True),
        np.array(dataset.y, dtype=np.int64, copy=True),
    )


def find_real_split_directory(subjects):
    candidates = [
        Path("saved_vae_run1/classifier_real_splits"),
        Path("saved_vae_run2/classifier_real_splits"),
        Path("saved_vae/classifier_real_splits"),
    ]

    for candidate in candidates:
        if all(
            (
                candidate
                / f"S{subject_id:02d}_real_splits.npz"
            ).exists()
            for subject_id in subjects
        ):
            return candidate.resolve()

    searched = "\n".join(
        f"  {candidate}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        "Could not find a complete real-split directory.\n"
        f"Searched:\n{searched}"
    )


@lru_cache(maxsize=None)
def load_real_arrays(real_split_directory, subject_id):
    split_file = (
        Path(real_split_directory)
        / f"S{subject_id:02d}_real_splits.npz"
    )

    with np.load(split_file) as data:
        arrays = {
            "X_train": np.asarray(
                data["X_train"],
                dtype=np.float32,
            ),
            "y_train": np.asarray(
                data["y_train"],
                dtype=np.int64,
            ),
            "X_valid": np.asarray(
                data["X_valid"],
                dtype=np.float32,
            ),
            "y_valid": np.asarray(
                data["y_valid"],
                dtype=np.int64,
            ),
            "X_test": np.asarray(
                data["X_test"],
                dtype=np.float32,
            ),
            "y_test": np.asarray(
                data["y_test"],
                dtype=np.int64,
            ),
        }

    return arrays


@lru_cache(maxsize=None)
def load_reconstruction(vae_directory, subject_id):
    reconstruction_file = (
        Path(vae_directory)
        / f"S{subject_id:02d}_seed0_vae_recon.npz"
    )

    if not reconstruction_file.exists():
        raise FileNotFoundError(
            f"Missing reconstruction: {reconstruction_file}"
        )

    with np.load(reconstruction_file) as data:
        X = np.asarray(
            data["X_recon"],
            dtype=np.float32,
        )
        y = np.asarray(
            data["y"],
            dtype=np.int64,
        )

    if X.ndim != 3:
        raise RuntimeError(
            f"Unexpected reconstruction shape in "
            f"{reconstruction_file}: {X.shape}"
        )

    if len(X) != len(y):
        raise RuntimeError(
            f"Reconstruction/label mismatch in "
            f"{reconstruction_file}"
        )

    return X, y, reconstruction_file


def fixed_class_order(
    labels,
    class_id,
    subject_id,
    source_number,
):
    """
    Produce one fixed ordering per subject, class and VAE run.

    The 25% subset is therefore contained within the 50% subset,
    and the 50% subset is contained within the 100% subset.
    The selected trials do not change across classifier seeds.
    """
    indices = np.flatnonzero(labels == class_id)

    selection_seed = (
        20260715
        + subject_id * 1000
        + int(class_id) * 10
        + source_number
    )

    rng = np.random.RandomState(selection_seed)

    return rng.permutation(indices)


def select_from_source(
    X,
    y,
    real_labels,
    amount_relative_to_real,
    subject_id,
    source_number,
):
    selected_indices = []

    for class_id in np.unique(real_labels):
        number_real_in_class = int(
            np.sum(real_labels == class_id)
        )

        number_required = int(
            round(
                amount_relative_to_real
                * number_real_in_class
            )
        )

        ordered_indices = fixed_class_order(
            labels=y,
            class_id=class_id,
            subject_id=subject_id,
            source_number=source_number,
        )

        if number_required > len(ordered_indices):
            raise RuntimeError(
                f"Subject {subject_id}, class {class_id}: "
                f"requested {number_required} synthetic trials "
                f"but source {source_number} contains only "
                f"{len(ordered_indices)}."
            )

        selected_indices.extend(
            ordered_indices[:number_required].tolist()
        )

    selected_indices = np.asarray(
        selected_indices,
        dtype=np.int64,
    )

    return X[selected_indices], y[selected_indices]


def build_synthetic_subset(
    subject_id,
    ratio,
    real_labels,
    run1_directory,
    run2_directory,
):
    X_run1, y_run1, run1_file = load_reconstruction(
        str(run1_directory),
        subject_id,
    )

    if ratio <= 1.0:
        X_synthetic, y_synthetic = select_from_source(
            X=X_run1,
            y=y_run1,
            real_labels=real_labels,
            amount_relative_to_real=ratio,
            subject_id=subject_id,
            source_number=1,
        )

        return (
            X_synthetic,
            y_synthetic,
            [str(run1_file)],
            False,
        )

    if ratio != 2.0:
        raise ValueError(
            f"Unsupported synthetic ratio: {ratio}"
        )

    X_run2, y_run2, run2_file = load_reconstruction(
        str(run2_directory),
        subject_id,
    )

    if X_run1.shape != X_run2.shape:
        raise RuntimeError(
            f"Subject {subject_id}: run-1/run-2 "
            f"shape mismatch: {X_run1.shape} versus "
            f"{X_run2.shape}"
        )

    if not np.array_equal(y_run1, y_run2):
        raise RuntimeError(
            f"Subject {subject_id}: run-1/run-2 "
            "labels are not identical."
        )

    # For 200%, select 100% from run 1 and 100% from run 2.
    X_first, y_first = select_from_source(
        X=X_run1,
        y=y_run1,
        real_labels=real_labels,
        amount_relative_to_real=1.0,
        subject_id=subject_id,
        source_number=1,
    )

    X_second, y_second = select_from_source(
        X=X_run2,
        y=y_run2,
        real_labels=real_labels,
        amount_relative_to_real=1.0,
        subject_id=subject_id,
        source_number=2,
    )

    sources_identical = np.array_equal(
        X_run1,
        X_run2,
    )

    return (
        np.concatenate(
            [X_first, X_second],
            axis=0,
        ).astype(np.float32),
        np.concatenate(
            [y_first, y_second],
            axis=0,
        ).astype(np.int64),
        [
            str(run1_file),
            str(run2_file),
        ],
        sources_identical,
    )

def build_fixed_size_replacement(
    X_real,
    y_real,
    X_synthetic,
    y_synthetic,
    replacement_ratio,
    subject_id,
):
    """
    Replace a fixed proportion of the real training trials with
    reconstructed trials while preserving:

        1. the original total training-set size;
        2. the original number of trials in each class;
        3. the same selected trials across classifier seeds.

    For example, replacement_ratio=0.25 means that approximately
    25% of the trials in each class are replaced by reconstructed
    trials from the same class.
    """
    kept_real_indices = []

    number_removed_real = 0

    for class_id in np.unique(y_real):
        real_class_indices = fixed_class_order(
            labels=y_real,
            class_id=class_id,
            subject_id=subject_id,
            source_number=0,
        )

        number_real_in_class = len(real_class_indices)

        number_to_replace = int(
            round(
                replacement_ratio
                * number_real_in_class
            )
        )

        # The first trials in the fixed ordering are removed.
        # This makes the 25% removed subset contained within the
        # 50%, 75% and 100% removed subsets.
        class_indices_to_keep = real_class_indices[
            number_to_replace:
        ]

        kept_real_indices.extend(
            class_indices_to_keep.tolist()
        )

        number_removed_real += number_to_replace

        number_synthetic_in_class = int(
            np.sum(y_synthetic == class_id)
        )

        if number_synthetic_in_class != number_to_replace:
            raise RuntimeError(
                f"Subject {subject_id}, class {class_id}: "
                f"removed {number_to_replace} real trials but "
                f"selected {number_synthetic_in_class} "
                "synthetic trials."
            )

    kept_real_indices = np.asarray(
        kept_real_indices,
        dtype=np.int64,
    )

    X_real_remaining = X_real[
        kept_real_indices
    ].astype(np.float32)

    y_real_remaining = y_real[
        kept_real_indices
    ].astype(np.int64)

    replacement_X = np.concatenate(
        [
            X_real_remaining,
            X_synthetic,
        ],
        axis=0,
    ).astype(np.float32)

    replacement_y = np.concatenate(
        [
            y_real_remaining,
            y_synthetic,
        ],
        axis=0,
    ).astype(np.int64)

    if len(replacement_X) != len(X_real):
        raise RuntimeError(
            f"Fixed-size replacement failed for subject "
            f"{subject_id}: original total={len(X_real)}, "
            f"replacement total={len(replacement_X)}."
        )

    original_classes, original_counts = np.unique(
        y_real,
        return_counts=True,
    )

    replacement_classes, replacement_counts = np.unique(
        replacement_y,
        return_counts=True,
    )

    if not np.array_equal(
        original_classes,
        replacement_classes,
    ):
        raise RuntimeError(
            f"Subject {subject_id}: class labels changed "
            "during replacement."
        )

    if not np.array_equal(
        original_counts,
        replacement_counts,
    ):
        raise RuntimeError(
            f"Subject {subject_id}: per-class trial counts "
            "changed during replacement."
        )

    return (
        replacement_X,
        replacement_y,
        len(X_real_remaining),
        number_removed_real,
    )
    
def preflight_checks(
    subjects,
    real_split_directory,
    run1_directory,
    run2_directory,
):
    print("=" * 78, flush=True)
    print("PREFLIGHT CHECKS", flush=True)
    print("=" * 78, flush=True)
    print(
        f"Real splits: {real_split_directory}",
        flush=True,
    )
    print(
        f"VAE run 1: {run1_directory}",
        flush=True,
    )
    print(
        f"VAE run 2: {run2_directory}",
        flush=True,
    )

    for subject_id in subjects:
        real = load_real_arrays(
            str(real_split_directory),
            subject_id,
        )

        X_run1, y_run1, _ = load_reconstruction(
            str(run1_directory),
            subject_id,
        )

        X_run2, y_run2, _ = load_reconstruction(
            str(run2_directory),
            subject_id,
        )

        real_shape = real["X_train"].shape[1:]

        if X_run1.shape[1:] != real_shape:
            raise RuntimeError(
                f"Subject {subject_id}: run-1 shape "
                f"{X_run1.shape} does not match real EEG "
                f"{real['X_train'].shape}."
            )

        if X_run2.shape[1:] != real_shape:
            raise RuntimeError(
                f"Subject {subject_id}: run-2 shape "
                f"{X_run2.shape} does not match real EEG "
                f"{real['X_train'].shape}."
            )

        if not np.array_equal(y_run1, y_run2):
            raise RuntimeError(
                f"Subject {subject_id}: run-1 and run-2 "
                "labels differ."
            )

        identical = np.array_equal(
            X_run1,
            X_run2,
        )

        print(
            f"Subject {subject_id}: "
            f"real_train={len(real['X_train'])}, "
            f"run1={len(X_run1)}, "
            f"run2={len(X_run2)}, "
            f"run1_run2_identical={identical}",
            flush=True,
        )

        if identical:
            print(
                f"WARNING: Subject {subject_id}'s two VAE "
                "reconstruction sets are identical. Its 200% "
                "condition doubles the synthetic quantity but "
                "does not add independent synthetic diversity.",
                flush=True,
            )

    print("Preflight checks passed.", flush=True)


def print_final_summary(results_file):
    results_file = Path(results_file)

    if not results_file.exists():
        print(
            f"No results file found at {results_file}",
            flush=True,
        )
        return

    results = pd.read_csv(results_file)

    if "status" in results.columns:
        results = results[
            results["status"]
            .astype(str)
            .str.lower()
            == "success"
        ].copy()

    if "last_test_accuracy" in results.columns:
        results["final_test_accuracy"] = pd.to_numeric(
            results["last_test_accuracy"],
            errors="coerce",
        )
    elif "last_test_misclass" in results.columns:
        results["final_test_accuracy"] = (
            1.0
            - pd.to_numeric(
                results["last_test_misclass"],
                errors="coerce",
            )
        )
    else:
        print(
            "Could not find last-test accuracy or "
            "misclassification columns.",
            flush=True,
        )
        return

    experiment_column = next(
        (
            column
            for column in [
                "experiment_type",
                "experiment",
                "experiment_name",
            ]
            if column in results.columns
        ),
        None,
    )

    if experiment_column is None:
        print(
            "Could not identify the experiment column. "
            f"Columns: {list(results.columns)}",
            flush=True,
        )
        return

    summary = (
        results.groupby(experiment_column)[
            "final_test_accuracy"
        ]
        .agg(["mean", "std", "count"])
        .reindex(RATIO_EXPERIMENTS.keys())
    )

    summary["mean"] *= 100.0
    summary["std"] *= 100.0

    print("\n" + "=" * 78, flush=True)
    print(
        "FINAL TEST ACCURACY SUMMARY (%)",
        flush=True,
    )
    print("=" * 78, flush=True)
    print(
        summary.to_string(
            float_format=lambda value: f"{value:.2f}"
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run1-dir",
        default="saved_vae_run1",
    )

    parser.add_argument(
        "--run2-dir",
        default="saved_vae_run2",
    )

    parser.add_argument(
        "--subjects",
        default="1,2,3,4,5,6,7,8,9",
    )

    parser.add_argument(
        "--classifier-seeds",
        default="0,1,2",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--results",
        default=(
            "results/"
            "vae_fixed_size_replacement_results.csv"
        ),
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    args = parser.parse_args()

    subjects = parse_integer_list(args.subjects)
    classifier_seeds = parse_integer_list(
        args.classifier_seeds
    )

    run1_directory = Path(
        args.run1_dir
    ).resolve()

    run2_directory = Path(
        args.run2_dir
    ).resolve()

    if not run1_directory.is_dir():
        raise FileNotFoundError(
            f"Missing VAE run-1 directory: "
            f"{run1_directory}"
        )

    if not run2_directory.is_dir():
        raise FileNotFoundError(
            f"Missing VAE run-2 directory: "
            f"{run2_directory}"
        )

    real_split_directory = (
        find_real_split_directory(subjects)
    )

    preflight_checks(
        subjects=subjects,
        real_split_directory=real_split_directory,
        run1_directory=run1_directory,
        run2_directory=run2_directory,
    )

    def load_matching_real_splits(
        subject_id,
        config,
    ):
        arrays = load_real_arrays(
            str(real_split_directory),
            subject_id,
        )

        return (
            make_dataset(
                np.array(
                    arrays["X_train"],
                    copy=True,
                ),
                np.array(
                    arrays["y_train"],
                    copy=True,
                ),
            ),
            make_dataset(
                np.array(
                    arrays["X_valid"],
                    copy=True,
                ),
                np.array(
                    arrays["y_valid"],
                    copy=True,
                ),
            ),
            make_dataset(
                np.array(
                    arrays["X_test"],
                    copy=True,
                ),
                np.array(
                    arrays["y_test"],
                    copy=True,
                ),
            ),
        )

    def apply_ratio_transformation(
        train_set,
        valid_set,
        test_set,
        experiment_type,
        subject_id,
        seed,
    ):
        train_set = copy_dataset(train_set)
        valid_set = copy_dataset(valid_set)
        test_set = copy_dataset(test_set)

        if experiment_type not in RATIO_EXPERIMENTS:
            raise ValueError(
                f"Unknown ratio experiment: "
                f"{experiment_type}"
            )

        ratio = RATIO_EXPERIMENTS[
            experiment_type
        ]

        X_synthetic, y_synthetic, sources, identical = (
            build_synthetic_subset(
                subject_id=subject_id,
                ratio=ratio,
                real_labels=train_set.y,
                run1_directory=run1_directory,
                run2_directory=run2_directory,
            )
        )

        original_number_real = len(train_set.X)
        number_synthetic = len(X_synthetic)

        # ====================================================
        # PREVIOUS EXPERIMENTAL DESIGN: ADDITIVE AUGMENTATION
        #
        # All real trials were kept and reconstructed trials
        # were added. Therefore, the total training-set size
        # increased as the synthetic ratio increased.
        #
        # Kept commented out for reproducibility.
        # ====================================================

        # combined_X = np.concatenate(
        #     [train_set.X, X_synthetic],
        #     axis=0,
        # ).astype(np.float32)
        #
        # combined_y = np.concatenate(
        #     [train_set.y, y_synthetic],
        #     axis=0,
        # ).astype(np.int64)


        # ====================================================
        # CURRENT EXPERIMENTAL DESIGN: FIXED-SIZE REPLACEMENT
        #
        # Remove real trials and add the same number of
        # reconstructed trials. Total size and class counts
        # remain unchanged.
        # ====================================================

        (
            combined_X,
            combined_y,
            number_real_remaining,
            number_real_removed,
        ) = build_fixed_size_replacement(
            X_real=train_set.X,
            y_real=train_set.y,
            X_synthetic=X_synthetic,
            y_synthetic=y_synthetic,
            replacement_ratio=ratio,
            subject_id=subject_id,
        )

        # Keep the sample set fixed across classifier seeds, but
        # deterministically change training order with classifier seed.
        ordering_rng = np.random.RandomState(
            900000
            + subject_id * 1000
            + seed * 10
            + int(ratio * 100)
        )

        order = ordering_rng.permutation(
            len(combined_X)
        )

        train_set.X = combined_X[order]
        train_set.y = combined_y[order]

        final_total = len(train_set.X)

        real_percentage = (
            100.0
            * number_real_remaining
            / final_total
        )

        synthetic_percentage = (
            100.0
            * number_synthetic
            / final_total
        )

        notes = (
            f"Original real training trials="
            f"{original_number_real}; "
            f"real trials removed="
            f"{number_real_removed}; "
            f"real trials remaining="
            f"{number_real_remaining}; "
            f"synthetic trials added="
            f"{number_synthetic}; "
            f"replacement ratio="
            f"{ratio * 100:.0f}%; "
            f"final total={final_total}; "
            f"final composition="
            f"{real_percentage:.2f}% real/"
            f"{synthetic_percentage:.2f}% synthetic; "
            f"sources={sources}; "
            "total training-set size fixed; "
            "per-class counts preserved; "
            "validation/test are real EEG"
        )

        if identical:
            notes += (
                "; run-1 and run-2 reconstructions "
                "were identical for this subject"
            )

        print(
            f"REPLACEMENT subject={subject_id}, "
            f"seed={seed}, "
            f"experiment={experiment_type} | "
            f"original_real={original_number_real}, "
            f"real_remaining={number_real_remaining}, "
            f"real_removed={number_real_removed}, "
            f"synthetic_added={number_synthetic}, "
            f"total={final_total}",
            flush=True,
        )

        return (
            train_set,
            valid_set,
            test_set,
            "real_vae_fixed_size_replacement",
            (
                f"replace_"
                f"{int(ratio * 100)}pct_real"
            ),
            notes,
        )

    config = copy.deepcopy(CONFIG)

    config.subject_numbers = subjects
    config.random_seeds = classifier_seeds

    config.experiment_names = tuple(
        RATIO_EXPERIMENTS.keys()
    )

    config.max_epochs = args.epochs
    config.max_increase_epochs = 30
    config.use_cuda = not args.cpu
    config.results_csv = Path(args.results)

    config.results_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Replace only the data source and fixed-size replacement
    # transformation. The existing classifier and result-saving
    # pipeline are retained.

    run.load_train_valid_test = (
        load_matching_real_splits
    )

    run.apply_experiment_transformation = (
        apply_ratio_transformation
    )

    print(
        "STARTING FIXED-SIZE VAE REPLACEMENT EXPERIMENTS",
        flush=True,
    )
    print("=" * 78, flush=True)

    print(
        f"Experiments: "
        f"{tuple(RATIO_EXPERIMENTS.keys())}",
        flush=True,
    )
    print(f"Subjects: {subjects}", flush=True)
    print(
        f"Classifier seeds: {classifier_seeds}",
        flush=True,
    )
    print(f"Maximum epochs: {args.epochs}", flush=True)
    print(
        f"Results file: {config.results_csv}",
        flush=True,
    )
    print(
        f"Classifier device: "
        f"{'CPU' if args.cpu else 'CUDA'}",
        flush=True,
    )

    run.run_all_experiments(config)

    print_final_summary(config.results_csv)


if __name__ == "__main__":
    main()
