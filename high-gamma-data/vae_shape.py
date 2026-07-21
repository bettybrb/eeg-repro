from pathlib import Path

import numpy as np


def labels_to_zero_based(y):
    y = np.asarray(y).astype(np.int64)
    unique = set(np.unique(y).tolist())

    if unique == {1, 2, 3, 4}:
        return y - 1

    return y


def fix_vae_shape(X, expected_channels):
    """
    Convert VAE output into:
        trials x channels x time

    Handles:
        trials x 1 x channels x time
        trials x channels x time
        trials x time x channels
    """

    X = np.asarray(X)

    if X.ndim == 4:
        X = np.squeeze(X)

    if X.ndim != 3:
        raise ValueError(f"Expected 3D EEG after squeeze, got shape {X.shape}")

    if X.shape[1] == expected_channels:
        return X.astype(np.float32)

    if X.shape[2] == expected_channels:
        X = np.transpose(X, (0, 2, 1))
        return X.astype(np.float32)

    raise ValueError(
        f"Could not find channel axis in VAE output shape {X.shape}. "
        f"Expected {expected_channels} channels."
    )


def load_vae_recon_for_classifier(
    subject_id,
    seed,
    expected_channels,
    min_n_times=1000,
    folder="saved_vae",
):
    path = Path(folder) / f"S{subject_id:02d}_seed{seed}_vae_recon.npz"

    if not path.exists():
        raise FileNotFoundError(f"Missing VAE reconstruction file: {path}")

    data = np.load(path, allow_pickle=True)

    if "X_recon" not in data:
        raise KeyError(f"{path} does not contain X_recon")

    if "y" not in data:
        raise KeyError(f"{path} does not contain y")

    X = fix_vae_shape(data["X_recon"], expected_channels=expected_channels)
    y = labels_to_zero_based(data["y"])

    if X.shape[2] < min_n_times:
        raise ValueError(
            f"VAE output has only {X.shape[2]} time samples, "
            f"but classifier needs at least {min_n_times}."
        )

    if len(X) != len(y):
        raise ValueError(
            f"X and y length mismatch: X has {len(X)} trials, y has {len(y)} labels"
        )

    return X.astype(np.float32), y.astype(np.int64)
