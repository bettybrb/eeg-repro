import argparse
import gc
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.vae_make import (
    add_vae_repository_to_path,
    configure_numba_cuda,
    make_tensor_dataset,
    parse_int_list,
    set_seed,
)
from pipeline.config import CONFIG
from pipeline.splits import load_real_split


METHOD_NAME = "class_specific_vae_generation"


def output_file(
    subject_id,
    generator_seed,
    config,
):
    return (
        config.class_specific_vae_directory
        / (
            f"S{subject_id:02d}_"
            f"generator-seed{generator_seed}.npz"
        )
    )


def to_three_dimensions(values):
    values = (
        values
        .detach()
        .cpu()
        .numpy()
    )

    if (
        values.ndim == 4
        and values.shape[1] == 1
    ):
        values = values[:, 0]

    if values.ndim != 3:
        raise RuntimeError(
            "Unexpected generated EEG shape: "
            f"{values.shape}"
        )

    return values.astype(
        np.float32
    )


def train_class_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
    number_to_generate,
    subject_id,
    class_id,
    generator_seed,
    generation_seed,
    epochs,
    batch_size,
    device,
):
    """
    Train one hvEEGNet VAE using only one motor-imagery class.

    New trials are sampled from the model's standard-normal latent
    prior. No real EEG trial is encoded during generation.
    """

    from library.config import (
        config_model as cm,
    )
    from library.config import (
        config_training as ct,
    )
    from library.training import (
        train_generic,
    )

    # Same class-specific model-seeding rule as the archived method.
    class_seed = (
        generator_seed
        + 1000 * class_id
    )
    set_seed(
        class_seed
    )

    train_dataset = make_tensor_dataset(
        X_train,
        y_train,
    )
    valid_dataset = make_tensor_dataset(
        X_valid,
        y_valid,
    )

    model_config = (
        cm.get_config_hierarchical_vEEGNet(
            X_train.shape[1],
            X_train.shape[2],
            0,
            0,
        )
    )

    model_config["input_size"] = (
        train_dataset[0][0]
        .unsqueeze(0)
        .shape
    )

    train_config = (
        ct.get_config_vEEGNet_training()
    )

    train_config["epochs"] = epochs
    train_config["batch_size"] = (
        batch_size
    )
    train_config["wandb_training"] = (
        False
    )
    train_config["print_var"] = True
    train_config["device"] = device
    train_config[
        "measure_metrics_during_training"
    ] = model_config["use_classifier"]
    train_config["use_classifier"] = (
        model_config["use_classifier"]
    )
    train_config["notes"] = (
        "Class-specific hvEEGNet VAE "
        "trained from the frozen central split"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Only one temporary checkpoint directory exists at a time.
    with tempfile.TemporaryDirectory(
        prefix=(
            f"class_vae_"
            f"S{subject_id:02d}_"
            f"seed{generator_seed}_"
            f"class{class_id}_"
        )
    ) as temporary_directory:
        model_directory = Path(
            temporary_directory
        )

        train_config[
            "path_to_save_model"
        ] = str(
            model_directory
        )
        train_config[
            "model_artifact_name"
        ] = (
            f"class_specific_vae_"
            f"S{subject_id:02d}_"
            f"seed{generator_seed}_"
            f"class{class_id}"
        )

        model = (
            train_generic
            .get_untrained_model(
                "hvEEGNet_shallow",
                model_config,
            )
            .to(device)
        )

        loss_function = (
            train_generic
            .get_loss_function(
                "hvEEGNet_shallow",
                train_config,
            )
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config["lr"],
            weight_decay=(
                train_config[
                    "optimizer_weight_decay"
                ]
            ),
        )

        if train_config["use_scheduler"]:
            scheduler = (
                torch.optim.lr_scheduler
                .ExponentialLR(
                    optimizer,
                    gamma=(
                        train_config[
                            "lr_decay_rate"
                        ]
                    ),
                )
            )
        else:
            scheduler = None

        print(
            f"Subject {subject_id:02d} | "
            f"generator_seed={generator_seed} | "
            f"class={class_id} | "
            f"train={len(train_dataset)} | "
            f"valid={len(valid_dataset)}",
            flush=True,
        )

        train_generic.train(
            model,
            loss_function,
            optimizer,
            [
                train_loader,
                valid_loader,
            ],
            train_config,
            scheduler,
            model_artifact=None,
        )

        best_model_file = (
            model_directory
            / "model_BEST.pth"
        )

        if not best_model_file.exists():
            raise FileNotFoundError(
                "The external VAE trainer did "
                "not save model_BEST.pth"
            )

        model.load_state_dict(
            torch.load(
                best_model_file,
                map_location=device,
            )
        )
        model.eval()

        latent_shape = list(
            model
            .h_vae
            .hidden_space_shape
        )
        latent_shape[0] = (
            number_to_generate
        )

        # Preserve the archived prior-sampling rule.
        set_seed(
            generation_seed
            + subject_id * 100
            + class_id
        )

        with torch.no_grad():
            latent = torch.randn(
                latent_shape,
                device=device,
            )

            generated = (
                to_three_dimensions(
                    model.generate(
                        latent
                    )
                )
            )

        expected_shape = (
            number_to_generate,
            X_train.shape[1],
            X_train.shape[2],
        )

        if (
            generated.shape
            != expected_shape
        ):
            raise RuntimeError(
                f"Class {class_id} produced "
                f"{generated.shape}; expected "
                f"{expected_shape}"
            )

        del model
        del optimizer

    del train_loader
    del valid_loader

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return generated


def train_subject(
    subject_id,
    generator_seed,
    generation_seed,
    epochs,
    batch_size,
    use_cuda,
    overwrite,
    config,
):
    destination = output_file(
        subject_id,
        generator_seed,
        config,
    )

    if (
        destination.exists()
        and not overwrite
    ):
        print(
            "SKIP existing class-specific "
            f"VAE data: {destination}",
            flush=True,
        )
        return

    if overwrite:
        destination.unlink(
            missing_ok=True
        )

    split = load_real_split(
        subject_id,
        config,
    )

    X_train = np.asarray(
        split.X_train,
        dtype=np.float32,
    )
    y_train = np.asarray(
        split.y_train,
        dtype=np.int64,
    )
    X_valid = np.asarray(
        split.X_valid,
        dtype=np.float32,
    )
    y_valid = np.asarray(
        split.y_valid,
        dtype=np.int64,
    )

    classes = sorted(
        int(value)
        for value
        in np.unique(y_train)
    )

    if (
        classes
        != list(config.class_ids)
    ):
        raise RuntimeError(
            f"Expected classes "
            f"{list(config.class_ids)}, "
            f"found {classes}"
        )

    device = (
        "cuda"
        if (
            use_cuda
            and torch.cuda.is_available()
        )
        else "cpu"
    )

    X_generated = np.empty_like(
        X_train,
        dtype=np.float32,
    )

    class_train_counts = []
    class_valid_counts = []

    print(
        "=" * 72,
        flush=True,
    )
    print(
        "START class-specific VAE generation | "
        f"subject={subject_id} | "
        f"generator_seed={generator_seed} | "
        f"device={device}",
        flush=True,
    )
    print(
        f"Central split: "
        f"{split.split_file}",
        flush=True,
    )

    for class_id in classes:
        train_mask = (
            y_train == class_id
        )
        valid_mask = (
            y_valid == class_id
        )

        number_to_generate = int(
            train_mask.sum()
        )

        if number_to_generate == 0:
            raise RuntimeError(
                f"No training trials "
                f"for class {class_id}"
            )

        if not np.any(
            valid_mask
        ):
            raise RuntimeError(
                f"No validation trials "
                f"for class {class_id}"
            )

        generated = train_class_model(
            X_train=(
                X_train[train_mask]
            ),
            y_train=(
                y_train[train_mask]
            ),
            X_valid=(
                X_valid[valid_mask]
            ),
            y_valid=(
                y_valid[valid_mask]
            ),
            number_to_generate=(
                number_to_generate
            ),
            subject_id=subject_id,
            class_id=class_id,
            generator_seed=(
                generator_seed
            ),
            generation_seed=(
                generation_seed
            ),
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )

        # Restore generated trials to the frozen label order.
        X_generated[
            train_mask
        ] = generated

        class_train_counts.append(
            number_to_generate
        )
        class_valid_counts.append(
            int(valid_mask.sum())
        )

    y_generated = (
        y_train.copy()
    )

    if (
        X_generated.shape
        != X_train.shape
    ):
        raise RuntimeError(
            f"Generated shape "
            f"{X_generated.shape} does not "
            f"match training shape "
            f"{X_train.shape}"
        )

    if not np.array_equal(
        y_generated,
        y_train,
    ):
        raise RuntimeError(
            "Generated labels do not match "
            "the frozen training labels"
        )

    if not np.isfinite(
        X_generated
    ).all():
        raise RuntimeError(
            "Generated EEG contains "
            "NaN or infinity"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        destination,
        X=X_generated,
        y=y_generated,
        protocol_id=(
            config.protocol_id
        ),
        method=METHOD_NAME,
        subject_id=subject_id,
        generator_seed=(
            generator_seed
        ),
        generation_seed=(
            generation_seed
        ),
        maximum_epochs=epochs,
        batch_size=batch_size,
        checkpoint_selection=(
            "lowest class-specific "
            "validation loss"
        ),
        split_file=str(
            split.split_file
        ),
        class_train_counts=(
            np.asarray(
                class_train_counts,
                dtype=np.int64,
            )
        ),
        class_valid_counts=(
            np.asarray(
                class_valid_counts,
                dtype=np.int64,
            )
        ),
        source=(
            "four independent hierarchical "
            "hvEEGNet VAEs, one per class; "
            "genuine prior generation"
        ),
    )

    print(
        f"SAVED {destination}",
        flush=True,
    )
    print(
        f"Shape={X_generated.shape}, "
        f"mean={X_generated.mean():.6f}, "
        f"std={X_generated.std():.6f}, "
        f"class_counts="
        f"{class_train_counts}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate EEG using four "
            "independent class-specific "
            "hierarchical hvEEGNet VAEs."
        )
    )

    parser.add_argument(
        "--repo",
        default=str(
            CONFIG
            .external_vae_repository
        ),
    )
    parser.add_argument(
        "--subjects",
        default=",".join(
            str(subject)
            for subject
            in CONFIG.subject_numbers
        ),
    )
    parser.add_argument(
        "--generator-seeds",
        "--seeds",
        dest="generator_seeds",
        default=",".join(
            str(seed)
            for seed
            in CONFIG.generator_seeds
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=(
            CONFIG
            .class_specific_vae_max_epochs
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=(
            CONFIG
            .class_specific_vae_batch_size
        ),
    )
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=(
            CONFIG
            .class_specific_vae_generation_seed
        ),
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    configure_numba_cuda(
        CONFIG
    )
    add_vae_repository_to_path(
        args.repo
    )

    for subject_id in (
        parse_int_list(
            args.subjects
        )
    ):
        for generator_seed in (
            parse_int_list(
                args.generator_seeds
            )
        ):
            train_subject(
                subject_id=subject_id,
                generator_seed=(
                    generator_seed
                ),
                generation_seed=(
                    args.generation_seed
                ),
                epochs=args.epochs,
                batch_size=(
                    args.batch_size
                ),
                use_cuda=args.cuda,
                overwrite=(
                    args.overwrite
                ),
                config=CONFIG,
            )


if __name__ == "__main__":
    main()
