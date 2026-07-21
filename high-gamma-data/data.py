from pathlib import Path
import numpy as np

from braindecode_setup import apply_compatibility_patches

apply_compatibility_patches()

from braindecode.datautil.signal_target import SignalAndTarget


def _make_signal_and_target(X, y):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    if X.ndim != 3:
        raise RuntimeError(f"Expected EEG shape (trials, channels, time), got {X.shape}")

    return SignalAndTarget(X, y)


def load_train_valid_test(subject_id, config):
    split_file = Path("saved_vae/classifier_real_splits") / f"S{subject_id:02d}_real_splits.npz"

    if not split_file.exists():
        raise FileNotFoundError(
            f"Missing {split_file}. Run export_vae_classifier_real_splits.py first."
        )

    data = np.load(split_file)

    train_set = _make_signal_and_target(data["X_train"], data["y_train"])
    valid_set = _make_signal_and_target(data["X_valid"], data["y_valid"])
    test_set = _make_signal_and_target(data["X_test"], data["y_test"])

    return train_set, valid_set, test_set
