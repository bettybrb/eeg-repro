import copy

import numpy as np


GAUSSIAN_EXPERIMENT_TYPES = [
    "gaussian_channel",
    "gaussian_class_channel_train_only",
    "gaussian_global",
    "gaussian_class",
    "gaussian_time",
    "gaussian_channel_time",
    "gaussian_class_time",
    "gaussian_class_channel_time",
]

VAE_EXPERIMENT_TYPES = [
    "vae_recon_train_only",
]

VAE_SEED_BY_SUBJECT = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 1,
}


def copy_dataset(dataset):
    copied = copy.copy(dataset)
    copied.X = np.array(dataset.X, copy=True)
    copied.y = np.array(dataset.y, copy=True)
    return copied


def safe_std(values):
    std = np.std(values)
    if std == 0 or not np.isfinite(std):
        std = 1e-6
    return std


def get_gaussian_experiment_settings(experiment_type):
    settings = {
        "gaussian_channel": {
            "subject": True,
            "class": False,
            "channel": True,
            "time": False,
            "split_mode": "train_synthetic_only",
            "description": (
                "subject-specific channel Gaussian; one mean/std per channel; "
                "train synthetic only; valid/test real"
            ),
        },
        "gaussian_class_channel_train_only": {
            "subject": True,
            "class": True,
            "channel": True,
            "time": False,
            "split_mode": "train_synthetic_only",
            "description": (
                "subject-specific class + channel Gaussian; one mean/std per class and channel; "
                "train synthetic only; valid/test real"
            ),
        },
        "gaussian_global": {
            "subject": True,
            "class": False,
            "channel": False,
            "time": False,
            "split_mode": "train_synthetic_only",
            "description": (
                "subject-specific global Gaussian; one mean/std for all EEG values; "
                "train synthetic only; valid/test real"
            ),
        },
        "gaussian_class": {
            "subject": True,
            "class": True,
            "channel": False,
            "time": False,
            "split_mode": "train_synthetic_only",
            "description": (
                "subject-specific class Gaussian; one mean/std per class; "
                "train synthetic only; valid/test real"
            ),
        },
        "gaussian_time": {
            "subject": True,
            "class": False,
            "channel": False,
            "time": True,
            "split_mode": "train_synthetic_only",
            "description": (
                "subject-specific time Gaussian; one mean/std per time point; "
                "train synthetic only; valid/test real"
            ),
        },
        "gaussian_channel_time": {
            "subject": True,
            "class": False,
            "channel": True,
            "time": True,
            "split_mode": "train_synthetic_only",
            "description": (
                "subject-specific channel + time Gaussian; one mean/std per channel and time point; "
                "train synthetic only; valid/test real"
            ),
        },
        "gaussian_class_time": {
            "subject": True,
            "class": True,
            "channel": False,
            "time": True,
            "split_mode": "train_synthetic_only",
            "description": (
                "subject-specific class + time Gaussian; one mean/std per class and time point; "
                "train synthetic only; valid/test real"
            ),
        },
        "gaussian_class_channel_time": {
            "subject": True,
            "class": True,
            "channel": True,
            "time": True,
            "split_mode": "train_synthetic_only",
            "description": (
                "subject-specific class + channel + time Gaussian; one mean/std per class, channel, and time point; "
                "train synthetic only; valid/test real"
            ),
        },
    }

    if experiment_type not in settings:
        raise ValueError(f"Unknown Gaussian experiment type: {experiment_type}")

    return settings[experiment_type]


def generate_subject_specific_gaussian_eeg(
    train_set,
    target_set,
    condition_on_class,
    condition_on_channel,
    condition_on_time,
    rng,
):
    train_y = np.asarray(train_set.y)
    target_y = np.asarray(target_set.y)

    fake_X = np.zeros_like(target_set.X)

    n_channels = train_set.X.shape[1]
    n_times = train_set.X.shape[2]

    global_train_values = train_set.X
    global_mean = np.mean(global_train_values)
    global_std = safe_std(global_train_values)

    if condition_on_class:
        class_ids = np.unique(np.concatenate([train_y, target_y]))
    else:
        class_ids = [None]

    for class_id in class_ids:
        if class_id is None:
            train_class_mask = np.ones(len(train_y), dtype=bool)
            target_class_mask = np.ones(len(target_y), dtype=bool)
        else:
            train_class_mask = train_y == class_id
            target_class_mask = target_y == class_id

        if not np.any(target_class_mask):
            continue

        if not np.any(train_class_mask):
            fake_X[target_class_mask, :, :] = rng.normal(
                loc=global_mean,
                scale=global_std,
                size=fake_X[target_class_mask, :, :].shape,
            )
            continue

        if condition_on_channel and condition_on_time:
            for channel_i in range(n_channels):
                for time_i in range(n_times):
                    train_values = train_set.X[train_class_mask, channel_i, time_i]
                    mean = np.mean(train_values)
                    std = safe_std(train_values)

                    fake_X[target_class_mask, channel_i, time_i] = rng.normal(
                        loc=mean,
                        scale=std,
                        size=fake_X[target_class_mask, channel_i, time_i].shape,
                    )

        elif condition_on_channel and not condition_on_time:
            for channel_i in range(n_channels):
                train_values = train_set.X[train_class_mask, channel_i, :]
                mean = np.mean(train_values)
                std = safe_std(train_values)

                fake_X[target_class_mask, channel_i, :] = rng.normal(
                    loc=mean,
                    scale=std,
                    size=fake_X[target_class_mask, channel_i, :].shape,
                )

        elif condition_on_time and not condition_on_channel:
            for time_i in range(n_times):
                train_values = train_set.X[train_class_mask, :, time_i]
                mean = np.mean(train_values)
                std = safe_std(train_values)

                fake_X[target_class_mask, :, time_i] = rng.normal(
                    loc=mean,
                    scale=std,
                    size=fake_X[target_class_mask, :, time_i].shape,
                )

        else:
            train_values = train_set.X[train_class_mask, :, :]
            mean = np.mean(train_values)
            std = safe_std(train_values)

            fake_X[target_class_mask, :, :] = rng.normal(
                loc=mean,
                scale=std,
                size=fake_X[target_class_mask, :, :].shape,
            )

    return fake_X



def load_vae_reconstructed_train(subject_id):
    vae_seed = VAE_SEED_BY_SUBJECT.get(subject_id)

    if vae_seed is None:
        raise ValueError(f"No VAE seed configured for subject {subject_id}")

    vae_path = f"saved_vae/S{subject_id:02d}_seed{vae_seed}_vae_recon.npz"
    data = np.load(vae_path)

    X_recon = data["X_recon"].astype(np.float32)
    y_recon = data["y"].astype(np.int64)

    return X_recon, y_recon, vae_seed, vae_path


def apply_experiment_transformation(
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
            "baseline; no perturbation; real train/valid/test EEG",
        )

    if experiment_type in GAUSSIAN_EXPERIMENT_TYPES:
        settings = get_gaussian_experiment_settings(experiment_type)

        if settings["split_mode"] != "train_synthetic_only":
            raise RuntimeError("Only train_synthetic_only is allowed for Gaussian experiments.")

        rng = np.random.RandomState(seed)

        train_fake = generate_subject_specific_gaussian_eeg(
            train_set=train_set,
            target_set=train_set,
            condition_on_class=settings["class"],
            condition_on_channel=settings["channel"],
            condition_on_time=settings["time"],
            rng=rng,
        )

        train_set.X = train_fake

        return (
            train_set,
            valid_set,
            test_set,
            "synthetic_gaussian",
            "train_synthetic_only",
            settings["description"],
        )

    if experiment_type in VAE_EXPERIMENT_TYPES:
        X_recon, y_recon, vae_seed, vae_path = load_vae_reconstructed_train(subject_id)

        if X_recon.shape[1:] != train_set.X.shape[1:]:
            raise RuntimeError(
                f"VAE shape mismatch for subject {subject_id}: "
                f"VAE {X_recon.shape}, real train {train_set.X.shape}"
            )

        if len(y_recon) != len(train_set.y):
            raise RuntimeError(
                f"VAE label length mismatch for subject {subject_id}: "
                f"VAE {len(y_recon)}, real train {len(train_set.y)}"
            )

        train_set.X = X_recon
        train_set.y = y_recon

        return (
            train_set,
            valid_set,
            test_set,
            "synthetic_vae_reconstruction",
            "train_synthetic_only",
            (
                f"VAE reconstructed train-only EEG from {vae_path}; "
                f"VAE model seed={vae_seed}; valid/test are real EEG"
            ),
        )

    if experiment_type == "hveegnet_recon_train_only":
        raise NotImplementedError(
            "Old hvEEGNet placeholder kept but disabled. Use vae_recon_train_only."
        )

    raise ValueError(f"Unknown experiment_type: {experiment_type}")
