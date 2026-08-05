import copy
from dataclasses import dataclass

import numpy as np


GAUSSIAN_METHODS = {
    "gaussian_unconditional": {
        "class": False,
        "channel": False,
        "time": False,
        "description": (
            "one mean and standard deviation for all training EEG values"
        ),
    },
    "gaussian_channel": {
        "class": False,
        "channel": True,
        "time": False,
        "description": (
            "one mean and standard deviation per EEG channel"
        ),
    },
    "gaussian_class": {
        "class": True,
        "channel": False,
        "time": False,
        "description": (
            "one mean and standard deviation per motor-imagery class"
        ),
    },
    "gaussian_time": {
        "class": False,
        "channel": False,
        "time": True,
        "description": (
            "one mean and standard deviation per time sample"
        ),
    },
    "gaussian_channel_time": {
        "class": False,
        "channel": True,
        "time": True,
        "description": (
            "one mean and standard deviation per channel and time sample"
        ),
    },
    "gaussian_class_time": {
        "class": True,
        "channel": False,
        "time": True,
        "description": (
            "one mean and standard deviation per class and time sample"
        ),
    },
    "gaussian_class_channel": {
        "class": True,
        "channel": True,
        "time": False,
        "description": (
            "one mean and standard deviation per class and channel"
        ),
    },
    "gaussian_class_channel_time": {
        "class": True,
        "channel": True,
        "time": True,
        "description": (
            "one mean and standard deviation per class, channel "
            "and time sample"
        ),
    },
}


NEURAL_GENERATION_METHODS = {
    "vae_reconstruction",
    "conditional_vae_generation",
    "class_specific_vae_generation",
    "hierarchical_conditional_vae_generation",
}


@dataclass(frozen=True)
class PreparedTrainingData:
    dataset: object
    train_data_type: str
    train_data_file: str
    n_real_train_trials: int
    n_synthetic_train_trials: int
    notes: str


def copy_dataset(dataset):
    copied = copy.copy(dataset)
    copied.X = np.asarray(
        dataset.X,
        dtype=np.float32,
    ).copy()
    copied.y = np.asarray(
        dataset.y,
        dtype=np.int64,
    ).copy()
    return copied


def _safe_std(values, axis=None):
    standard_deviation = np.std(
        values,
        axis=axis,
    )

    return np.where(
        np.isfinite(standard_deviation)
        & (standard_deviation > 0),
        standard_deviation,
        1e-6,
    )


def _sample_gaussian_group(
    source,
    output_shape,
    condition_on_channel,
    condition_on_time,
    rng,
):
    if condition_on_channel and condition_on_time:
        mean = np.mean(source, axis=0)
        standard_deviation = _safe_std(
            source,
            axis=0,
        )

    elif condition_on_channel:
        mean = np.mean(
            source,
            axis=(0, 2),
        )[None, :, None]
        standard_deviation = _safe_std(
            source,
            axis=(0, 2),
        )[None, :, None]

    elif condition_on_time:
        mean = np.mean(
            source,
            axis=(0, 1),
        )[None, None, :]
        standard_deviation = _safe_std(
            source,
            axis=(0, 1),
        )[None, None, :]

    else:
        mean = float(np.mean(source))
        standard_deviation = float(
            _safe_std(source)
        )

    return rng.normal(
        loc=mean,
        scale=standard_deviation,
        size=output_shape,
    ).astype(np.float32)


def generate_gaussian_training_data(
    X_train,
    y_train,
    condition_on_class,
    condition_on_channel,
    condition_on_time,
    generator_seed,
):
    """
    Fit Gaussian statistics only to the central real training split
    and generate a synthetic training set with matching labels.
    """

    X_train = np.asarray(
        X_train,
        dtype=np.float32,
    )
    y_train = np.asarray(
        y_train,
        dtype=np.int64,
    )

    if X_train.ndim != 3:
        raise RuntimeError(
            "Expected real training EEG shaped "
            f"(trials, channels, time), got {X_train.shape}"
        )

    if y_train.shape != (len(X_train),):
        raise RuntimeError(
            "Incompatible real training arrays: "
            f"EEG={X_train.shape}, labels={y_train.shape}"
        )

    rng = np.random.RandomState(
        generator_seed
    )
    X_generated = np.empty_like(
        X_train
    )

    if condition_on_class:
        class_ids = np.unique(y_train)
    else:
        class_ids = [None]

    for class_id in class_ids:
        if class_id is None:
            source_mask = np.ones(
                len(y_train),
                dtype=bool,
            )
            target_mask = source_mask
        else:
            source_mask = y_train == class_id
            target_mask = source_mask

        if not np.any(source_mask):
            raise RuntimeError(
                f"No real training trials found for class {class_id}"
            )

        X_generated[target_mask] = (
            _sample_gaussian_group(
                source=X_train[source_mask],
                output_shape=(
                    X_generated[target_mask].shape
                ),
                condition_on_channel=(
                    condition_on_channel
                ),
                condition_on_time=(
                    condition_on_time
                ),
                rng=rng,
            )
        )

    if not np.isfinite(
        X_generated
    ).all():
        raise RuntimeError(
            "Generated Gaussian EEG contains NaN or infinity"
        )

    return X_generated


def _generated_file(
    method,
    subject_id,
    generator_seed,
    config,
):
    filename = (
        f"S{subject_id:02d}_"
        f"generator-seed{generator_seed}.npz"
    )

    if method in GAUSSIAN_METHODS:
        return (
            config.gaussian_data_directory
            / method
            / filename
        )

    if method == "vae_reconstruction":
        return (
            config.vae_reconstruction_directory
            / filename
        )

    if method == "conditional_vae_generation":
        return (
            config.conditional_vae_directory
            / filename
        )

    if method == "class_specific_vae_generation":
        return (
            config.class_specific_vae_directory
            / filename
        )

    if method == "hierarchical_conditional_vae_generation":
        return (
            config.hierarchical_conditional_vae_directory
            / filename
        )

    raise ValueError(
        f"No generated-data path configured for {method}"
    )


def _validate_generated_dataset(
    X_generated,
    y_generated,
    real_train_set,
    method,
):
    X_generated = np.asarray(
        X_generated,
        dtype=np.float32,
    )
    y_generated = np.asarray(
        y_generated,
        dtype=np.int64,
    )

    if X_generated.shape != real_train_set.X.shape:
        raise RuntimeError(
            f"{method} EEG shape {X_generated.shape} "
            "does not match central training shape "
            f"{real_train_set.X.shape}"
        )

    if not np.array_equal(
        y_generated,
        real_train_set.y,
    ):
        raise RuntimeError(
            f"{method} labels do not exactly match "
            "the central training labels"
        )

    if not np.isfinite(
        X_generated
    ).all():
        raise RuntimeError(
            f"{method} EEG contains NaN or infinity"
        )

    return X_generated, y_generated


def prepare_training_data(
    real_train_set,
    method,
    subject_id,
    generator_seed,
    split_file,
    config,
    overwrite_gaussian=False,
):
    """
    Create or load classifier training data.

    Validation and test datasets are not passed here and therefore
    cannot accidentally be transformed.
    """

    if method == "baseline":
        return PreparedTrainingData(
            dataset=copy_dataset(
                real_train_set
            ),
            train_data_type="real",
            train_data_file="",
            n_real_train_trials=len(
                real_train_set.X
            ),
            n_synthetic_train_trials=0,
            notes=(
                "Real training EEG from the central split; "
                "validation and official test EEG are real."
            ),
        )

    if generator_seed is None:
        raise ValueError(
            f"A generator seed is required for {method}"
        )

    generated_file = _generated_file(
        method=method,
        subject_id=subject_id,
        generator_seed=generator_seed,
        config=config,
    )

    if method in GAUSSIAN_METHODS:
        settings = GAUSSIAN_METHODS[
            method
        ]

        if (
            generated_file.exists()
            and not overwrite_gaussian
        ):
            with np.load(
                generated_file,
                allow_pickle=False,
            ) as data:
                X_generated = data["X"]
                y_generated = data["y"]

        else:
            X_generated = (
                generate_gaussian_training_data(
                    X_train=real_train_set.X,
                    y_train=real_train_set.y,
                    condition_on_class=(
                        settings["class"]
                    ),
                    condition_on_channel=(
                        settings["channel"]
                    ),
                    condition_on_time=(
                        settings["time"]
                    ),
                    generator_seed=generator_seed,
                )
            )

            y_generated = np.asarray(
                real_train_set.y,
                dtype=np.int64,
            ).copy()

            generated_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.savez_compressed(
                generated_file,
                X=X_generated,
                y=y_generated,
                protocol_id=config.protocol_id,
                method=method,
                subject_id=subject_id,
                generator_seed=generator_seed,
                split_file=str(split_file),
                source=(
                    "Gaussian statistics fitted only "
                    "to central real training EEG"
                ),
            )

        description = (
            f"{settings['description']}; statistics fitted "
            "only to the central real training split; "
            "validation and test remain real."
        )
        train_data_type = (
            "synthetic_gaussian"
        )

    elif method in NEURAL_GENERATION_METHODS:
        if not generated_file.exists():
            if method == "vae_reconstruction":
                generation_command = (
                    "python -m experiments.vae_make"
                )
            elif method == "conditional_vae_generation":
                generation_command = (
                    "python -m experiments.cvae_generate"
                )
            elif method == "class_specific_vae_generation":
                generation_command = (
                    "python -m "
                    "experiments.vae_class_generate"
                )
            else:
                generation_command = (
                    "python -m "
                    "experiments.hierarchical_cvae_generate"
                )

            raise FileNotFoundError(
                f"Missing generated training data: "
                f"{generated_file}\n"
                f"Run `{generation_command}` first."
            )

        with np.load(
            generated_file,
            allow_pickle=False,
        ) as data:
            if (
                "X" not in data.files
                or "y" not in data.files
            ):
                raise RuntimeError(
                    f"{generated_file} must contain "
                    "the standard keys X and y"
                )

            X_generated = data["X"]
            y_generated = data["y"]

        if method == "vae_reconstruction":
            description = (
                "Hierarchical VAE reconstructions of "
                "central real training trials; validation "
                "and test remain real."
            )
            train_data_type = (
                "synthetic_vae_reconstruction"
            )

        elif method == "conditional_vae_generation":
            description = (
                "Shared flat class-conditioned VAE samples "
                "generated from the latent prior; validation "
                "and test remain real."
            )
            train_data_type = (
                "synthetic_conditional_vae"
            )

        elif method == "class_specific_vae_generation":
            description = (
                "Four independent class-specific hierarchical "
                "hvEEGNet VAEs; genuine prior generation; "
                "validation and test remain real."
            )
            train_data_type = (
                "synthetic_class_specific_vae"
            )

        else:
            description = (
                "Shared hierarchical class-conditioned VAE "
                "samples generated from learned class-dependent "
                "two-level priors; validation and test remain real."
            )
            train_data_type = (
                "synthetic_hierarchical_conditional_vae"
            )

    else:
        raise ValueError(
            f"Unknown method: {method}"
        )

    (
        X_generated,
        y_generated,
    ) = _validate_generated_dataset(
        X_generated=X_generated,
        y_generated=y_generated,
        real_train_set=real_train_set,
        method=method,
    )

    generated_set = copy_dataset(
        real_train_set
    )
    generated_set.X = X_generated
    generated_set.y = y_generated

    return PreparedTrainingData(
        dataset=generated_set,
        train_data_type=train_data_type,
        train_data_file=str(
            generated_file
        ),
        n_real_train_trials=0,
        n_synthetic_train_trials=len(
            X_generated
        ),
        notes=description,
    )
