from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RealSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    split_file: Path


def split_file_for_subject(subject_id, config):
    return config.real_split_directory / f"S{subject_id:02d}_real_splits.npz"


def _validate_eeg(X, y, name, expected_trials, config):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    expected_shape = (
        expected_trials,
        config.expected_channels,
        config.expected_times,
    )

    if X.shape != expected_shape:
        raise RuntimeError(
            f"{name} EEG shape is {X.shape}; expected {expected_shape}"
        )

    if y.shape != (expected_trials,):
        raise RuntimeError(
            f"{name} labels have shape {y.shape}; expected {(expected_trials,)}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError(f"{name} EEG contains NaN or infinity")

    if not np.isin(y, config.class_ids).all():
        raise RuntimeError(
            f"{name} contains labels outside {config.class_ids}"
        )

    return X, y


def load_real_split(subject_id, config):
    split_file = split_file_for_subject(subject_id, config)

    if not split_file.exists():
        raise FileNotFoundError(
            f"Missing split file: {split_file}\n"
            "Run `python -m experiments.export_real_splits` first."
        )

    required_keys = {
        "X_train",
        "y_train",
        "X_valid",
        "y_valid",
        "X_test",
        "y_test",
    }

    with np.load(split_file, allow_pickle=False) as data:
        missing = required_keys.difference(data.files)
        if missing:
            raise RuntimeError(
                f"{split_file} is missing keys: {sorted(missing)}"
            )

        X_train, y_train = _validate_eeg(
            data["X_train"],
            data["y_train"],
            "training",
            config.expected_train_trials,
            config,
        )
        X_valid, y_valid = _validate_eeg(
            data["X_valid"],
            data["y_valid"],
            "validation",
            config.expected_valid_trials,
            config,
        )
        X_test, y_test = _validate_eeg(
            data["X_test"],
            data["y_test"],
            "test",
            config.expected_test_trials,
            config,
        )

    return RealSplit(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        split_file=split_file,
    )
