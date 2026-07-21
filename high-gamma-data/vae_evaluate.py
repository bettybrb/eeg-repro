#!/usr/bin/env python3

import argparse
import copy
from pathlib import Path

import numpy as np

from braindecode_setup import apply_compatibility_patches

apply_compatibility_patches()

from braindecode.datautil.signal_target import SignalAndTarget

import run
from config import CONFIG


RATIO_EXPERIMENTS = {
    "vae_real_plus_25pct_recon": 0.25,
    "vae_real_plus_50pct_recon": 0.50,
    "vae_real_plus_100pct_recon": 1.00,
}


def parse_int_list(value):
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
        np.array(dataset.X, copy=True),
        np.array(dataset.y, copy=True),
    )


def find_real_split_directory(vae_dir, subjects):
    candidates = [
        vae_dir / "classifier_real_splits",
        Path("saved_vae_run1/classifier_real_splits"),
        Path("saved_vae_run2/classifier_real_splits"),
        Path("saved_vae/classifier_real_splits"),
    ]

    for candidate in candidates:
        complete = all(
            (
                candidate
                / f"S{subject_id:02d}_real_splits.npz"
            ).exists()
            for subject_id in subjects
        )

        if complete:
            return candidate.resolve()

    searched = "\n".join(str(path) for path in candidates)

    raise FileNotFoundError(
        "Could not find a complete classifier_real_splits directory.\n"
        f"Searched:\n{searched}"
    )


def load_reconstruction(vae_dir, subject_id):
    path = (
        vae_dir
        / f"S{subject_id:02d}_seed0_vae_recon.npz"
    )

    if not path.exists():
        raise FileNotFoundError(str(path))

    with np.load(path) as data:
        X_recon = np.asarray(
            data["X_recon"],
            dtype=np.float32,
        )
        y_recon = np.asarray(
            data["y"],
            dtype=np.int64,
        )

    return X_recon, y_recon, path


def select_stratified_reconstruction(
    X_recon,
    y_recon,
    ratio,
    subject_id,
    classifier_seed,
):
    """
    Select a class-balanced proportion of reconstructed trials.

    ratio=0.25 means:
        all real training trials
        plus reconstructed trials equal to approximately 25%
        of the real training-set size
    """
    rng = np.random.RandomState(
        100000 * subject_id + classifier_seed
    )

    selected_indices = []

    for class_id in np.unique(y_recon):
        class_indices = np.flatnonzero(
            y_recon == class_id
        )

        number_to_select = int(
            round(ratio * len(class_indices))
        )

        number_to_select = min(
            number_to_select,
            len(class_indices),
        )

        if number_to_select > 0:
            chosen = rng.choice(
                class_indices,
                size=number_to_select,
                replace=False,
            )

            selected_indices.extend(
                chosen.tolist()
            )

    selected_indices = np.asarray(
        selected_indices,
        dtype=np.int64,
    )

    rng.shuffle(selected_indices)

    return (
        X_recon[selected_indices],
        y_recon[selected_indices],
    )


def compare_reconstruction_runs(
    run1_dir,
    run2_dir,
    subjects,
    output_file,
):
    rows = []

    for subject_id in subjects:
        X1, y1, path1 = load_reconstruction(
            run1_dir,
            subject_id,
        )

        X2, y2, path2 = load_reconstruction(
            run2_dir,
            subject_id,
        )

        if X1.shape != X2.shape:
            raise RuntimeError(
                f"Subject {subject_id} shape mismatch: "
                f"{X1.shape} versus {X2.shape}"
            )

        difference = (
            X1.astype(np.float64)
            - X2.astype(np.float64)
        )

        row = {
            "subject_id": subject_id,
            "run1_file": str(path1),
            "run2_file": str(path2),
            "shape": str(X1.shape),
            "labels_identical": bool(
                np.array_equal(y1, y2)
            ),
            "exactly_identical": bool(
                np.array_equal(X1, X2)
            ),
            "allclose": bool(
                np.allclose(X1, X2)
            ),
            "mean_absolute_difference": float(
                np.mean(np.abs(difference))
            ),
            "rmse": float(
                np.sqrt(np.mean(difference ** 2))
            ),
            "maximum_absolute_difference": float(
                np.max(np.abs(difference))
            ),
        }

        rows.append(row)

        print(
            f"Subject {subject_id}: "
            f"labels_identical={row['labels_identical']} | "
            f"exactly_identical={row['exactly_identical']} | "
            f"MAE={row['mean_absolute_difference']:.6f} | "
            f"RMSE={row['rmse']:.6f}",
            flush=True,
        )

    import pandas as pd

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_file, index=False)

    print(
        f"Saved reconstruction comparison to: {output_file}",
        flush=True,
    )


def run_classification(args):
    vae_dir = Path(args.vae_dir).resolve()
    subjects = parse_int_list(args.subjects)
    classifier_seeds = parse_int_list(
        args.classifier_seeds
    )

    real_split_dir = find_real_split_directory(
        vae_dir,
        subjects,
    )

    print("=" * 72, flush=True)
    print(f"Task: {args.task}", flush=True)
    print(f"VAE directory: {vae_dir}", flush=True)
    print(
        f"Real split directory: {real_split_dir}",
        flush=True,
    )
    print(f"Subjects: {subjects}", flush=True)
    print(
        f"Classifier seeds: {classifier_seeds}",
        flush=True,
    )
    print(f"Maximum epochs: {args.epochs}", flush=True)
    print("=" * 72, flush=True)

    def load_matching_real_splits(
        subject_id,
        config,
    ):
        split_file = (
            real_split_dir
            / f"S{subject_id:02d}_real_splits.npz"
        )

        with np.load(split_file) as data:
            train_set = make_dataset(
                data["X_train"],
                data["y_train"],
            )

            valid_set = make_dataset(
                data["X_valid"],
                data["y_valid"],
            )

            test_set = make_dataset(
                data["X_test"],
                data["y_test"],
            )

        return train_set, valid_set, test_set

    def apply_selected_experiment(
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

        if experiment_type == "baseline":
            return (
                train_set,
                valid_set,
                test_set,
                "real",
                "real_train_valid_test",
                (
                    "Real training EEG; "
                    "real validation and test EEG"
                ),
            )

        X_recon, y_recon, recon_path = (
            load_reconstruction(
                vae_dir,
                subject_id,
            )
        )

        if X_recon.shape[1:] != train_set.X.shape[1:]:
            raise RuntimeError(
                f"Subject {subject_id} VAE shape mismatch: "
                f"reconstruction={X_recon.shape}, "
                f"real={train_set.X.shape}"
            )

        if experiment_type == "vae_recon_train_only":
            train_set.X = X_recon
            train_set.y = y_recon

            return (
                train_set,
                valid_set,
                test_set,
                "synthetic_vae_reconstruction",
                "train_synthetic_only",
                (
                    f"Reconstructed-only training from {recon_path}; "
                    "VAE seed=0; validation/test real"
                ),
            )

        if experiment_type in RATIO_EXPERIMENTS:
            ratio = RATIO_EXPERIMENTS[
                experiment_type
            ]

            X_selected, y_selected = (
                select_stratified_reconstruction(
                    X_recon=X_recon,
                    y_recon=y_recon,
                    ratio=ratio,
                    subject_id=subject_id,
                    classifier_seed=seed,
                )
            )

            number_real = len(train_set.X)
            number_synthetic = len(X_selected)

            train_set.X = np.concatenate(
                [train_set.X, X_selected],
                axis=0,
            ).astype(np.float32)

            train_set.y = np.concatenate(
                [train_set.y, y_selected],
                axis=0,
            ).astype(np.int64)

            final_total = (
                number_real + number_synthetic
            )

            real_percentage = (
                100.0 * number_real / final_total
            )

            synthetic_percentage = (
                100.0
                * number_synthetic
                / final_total
            )

            return (
                train_set,
                valid_set,
                test_set,
                "real_plus_vae_reconstruction",
                "real_plus_reconstructed_augmentation",
                (
                    f"Real={number_real}, "
                    f"reconstructed={number_synthetic}; "
                    f"final composition approximately "
                    f"{real_percentage:.1f}% real and "
                    f"{synthetic_percentage:.1f}% reconstructed; "
                    f"source={recon_path}; "
                    "validation/test real"
                ),
            )

        raise ValueError(
            f"Unknown experiment: {experiment_type}"
        )

    config = copy.deepcopy(CONFIG)

    config.subject_numbers = subjects
    config.random_seeds = classifier_seeds
    config.max_epochs = args.epochs
    config.max_increase_epochs = 30
    config.use_cuda = not args.cpu
    config.results_csv = Path(args.results)

    if args.task == "classify":
        config.experiment_names = (
            "vae_recon_train_only",
        )

    elif args.task == "ratios":
        config.experiment_names = (
            "baseline",
            "vae_real_plus_25pct_recon",
            "vae_real_plus_50pct_recon",
            "vae_real_plus_100pct_recon",
            "vae_recon_train_only",
        )

    else:
        raise ValueError(args.task)

    config.results_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run.load_train_valid_test = (
        load_matching_real_splits
    )

    run.apply_experiment_transformation = (
        apply_selected_experiment
    )

    run.run_all_experiments(config)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices=[
            "compare",
            "classify",
            "ratios",
        ],
        required=True,
    )

    parser.add_argument("--vae-dir")
    parser.add_argument("--other-vae-dir")
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
    parser.add_argument("--results")
    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    args = parser.parse_args()

    subjects = parse_int_list(args.subjects)

    if args.task == "compare":
        if not args.vae_dir:
            raise ValueError(
                "--vae-dir is required"
            )

        if not args.other_vae_dir:
            raise ValueError(
                "--other-vae-dir is required"
            )

        output_file = Path(
            args.results
            or "results/vae_reconstruction_run_comparison.csv"
        )

        compare_reconstruction_runs(
            run1_dir=Path(args.vae_dir).resolve(),
            run2_dir=Path(
                args.other_vae_dir
            ).resolve(),
            subjects=subjects,
            output_file=output_file,
        )

        return

    if not args.vae_dir:
        raise ValueError(
            "--vae-dir is required"
        )

    if not args.results:
        raise ValueError(
            "--results is required"
        )

    run_classification(args)


if __name__ == "__main__":
    main()
