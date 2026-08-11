import numpy as np

from pipeline.braindecode_setup import apply_compatibility_patches
from pipeline.splits import load_real_split

apply_compatibility_patches()

from braindecode.datautil.signal_target import SignalAndTarget


def make_signal_and_target(X, y):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    if X.ndim != 3:
        raise RuntimeError(
            f"Expected EEG shape (trials, channels, time), got {X.shape}"
        )

    if y.ndim != 1 or len(X) != len(y):
        raise RuntimeError(
            f"Incompatible EEG and label shapes: {X.shape}, {y.shape}"
        )

    return SignalAndTarget(X, y)


def load_train_valid_test(subject_id, config):
    split = load_real_split(subject_id, config)

    train_set = make_signal_and_target(split.X_train, split.y_train)
    valid_set = make_signal_and_target(split.X_valid, split.y_valid)
    test_set = make_signal_and_target(split.X_test, split.y_test)

    return train_set, valid_set, test_set, split.split_file
