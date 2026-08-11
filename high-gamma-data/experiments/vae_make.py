import argparse
import gc
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pipeline.config import CONFIG
from pipeline.splits import load_real_split


def parse_int_list(value):
    return [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_numba_cuda(config):
    """
    Point Numba to the local minimal CUDA toolkit used by
    the external soft-DTW implementation.
    """

    cuda_home = (
        config.project_root
        / ".cuda-numba"
    ).resolve()

    nvvm_file = (
        cuda_home
        / "nvvm"
        / "lib64"
        / "libnvvm.so"
    )
    libdevice_directory = (
        cuda_home
        / "nvvm"
        / "libdevice"
    )
    libdevice_files = sorted(
        libdevice_directory.glob(
            "libdevice*.bc"
        )
    )

    if not nvvm_file.is_file():
        raise FileNotFoundError(
            f"Missing local NVVM library: {nvvm_file}"
        )

    if not libdevice_files:
        raise FileNotFoundError(
            "No libdevice bitcode file was found in "
            f"{libdevice_directory}"
        )

    # Numba 0.66 discovers both NVVM and libdevice relative
    # to CUDA_HOME.
    os.environ["CUDA_HOME"] = str(
        cuda_home
    )

    library_directories = [
        cuda_home / "lib",
        cuda_home / "lib64",
        cuda_home / "nvvm" / "lib64",
    ]

    existing_library_path = (
        os.environ.get(
            "LD_LIBRARY_PATH",
            "",
        )
    )

    path_entries = [
        str(directory)
        for directory in library_directories
        if directory.exists()
    ]

    if existing_library_path:
        path_entries.append(
            existing_library_path
        )

    os.environ["LD_LIBRARY_PATH"] = ":".join(
        path_entries
    )

    # Remove obsolete values so there is only one explicit
    # CUDA-discovery mechanism.
    os.environ.pop(
        "NUMBAPRO_NVVM",
        None,
    )
    os.environ.pop(
        "NUMBAPRO_LIBDEVICE",
        None,
    )

    from numba.cuda.cuda_paths import get_cuda_paths

    # Avoid retaining a lookup performed before CUDA_HOME was set.
    if hasattr(
        get_cuda_paths,
        "_cached_result",
    ):
        delattr(
            get_cuda_paths,
            "_cached_result",
        )

    discovered = get_cuda_paths()

    discovered_nvvm = (
        discovered["nvvm"].info
    )
    discovered_libdevice = (
        discovered["libdevice"].info
    )

    if discovered_nvvm is None:
        raise RuntimeError(
            "Numba could not discover libnvvm.so "
            f"under CUDA_HOME={cuda_home}"
        )

    if discovered_libdevice is None:
        raise RuntimeError(
            "Numba could not discover a libdevice "
            f"file under CUDA_HOME={cuda_home}"
        )

    print(
        f"CUDA_HOME: {cuda_home}",
        flush=True,
    )
    print(
        f"Numba NVVM: {discovered_nvvm}",
        flush=True,
    )
    print(
        f"Numba libdevice: {discovered_libdevice}",
        flush=True,
    )


def add_vae_repository_to_path(repository):
    repository = Path(repository).resolve()

    if not repository.exists():
        raise FileNotFoundError(
            f"External VAE repository not found: {repository}"
        )

    sys.path.insert(0, str(repository))


def make_tensor_dataset(X, y):
    """
    Convert central EEG shaped (trials, channels, time) into the
    singleton-dimension format expected by the external VAE.
    """

    X_tensor = torch.from_numpy(
        np.asarray(X, dtype=np.float32)
    ).unsqueeze(1)

    y_tensor = torch.from_numpy(
        np.asarray(y, dtype=np.int64)
    )

    return TensorDataset(
        X_tensor,
        y_tensor,
    )


def reconstruct_dataset(
    model,
    dataset,
    batch_size,
    device,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    reconstruction_batches = []
    label_batches = []

    model.eval()

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            X_reconstructed = model.reconstruct(
                X_batch,
                no_grad=True,
            )

            reconstruction_batches.append(
                X_reconstructed
                .detach()
                .cpu()
                .numpy()
            )
            label_batches.append(
                y_batch
                .detach()
                .cpu()
                .numpy()
            )

    X_raw = np.concatenate(
        reconstruction_batches,
        axis=0,
    )
    y = np.concatenate(
        label_batches,
        axis=0,
    )

    if X_raw.ndim == 4 and X_raw.shape[1] == 1:
        X_reconstructed = X_raw[:, 0, :, :]

    elif X_raw.ndim == 3:
        X_reconstructed = X_raw

    else:
        X_reconstructed = np.squeeze(
            X_raw
        )

        if X_reconstructed.ndim != 3:
            raise RuntimeError(
                "Could not convert VAE reconstruction "
                f"shape {X_raw.shape} into three dimensions"
            )

    return (
        X_reconstructed.astype(np.float32),
        y.astype(np.int64),
        X_raw.shape,
    )


def reconstruction_file(
    subject_id,
    generator_seed,
    config,
):
    return (
        config.vae_reconstruction_directory
        / (
            f"S{subject_id:02d}_"
            f"generator-seed{generator_seed}.npz"
        )
    )


def checkpoint_directory(
    subject_id,
    generator_seed,
    config,
):
    return (
        config.checkpoint_directory
        / "vae_reconstruction"
        / (
            f"S{subject_id:02d}_"
            f"generator-seed{generator_seed}"
        )
    )


def train_subject(
    subject_id,
    generator_seed,
    epochs,
    batch_size,
    use_cuda,
    overwrite,
    config,
):
    output_file = reconstruction_file(
        subject_id,
        generator_seed,
        config,
    )

    model_directory = checkpoint_directory(
        subject_id,
        generator_seed,
        config,
    )

    if output_file.exists() and not overwrite:
        print(
            f"SKIP existing reconstruction: {output_file}",
            flush=True,
        )
        return

    if overwrite:
        output_file.unlink(
            missing_ok=True
        )

        if model_directory.exists():
            shutil.rmtree(
                model_directory
            )

    set_seed(
        generator_seed
    )

    from library.config import config_model as cm
    from library.config import config_training as ct
    from library.training import train_generic

    split = load_real_split(
        subject_id,
        config,
    )

    train_dataset = make_tensor_dataset(
        split.X_train,
        split.y_train,
    )
    valid_dataset = make_tensor_dataset(
        split.X_valid,
        split.y_valid,
    )

    device = (
        "cuda"
        if use_cuda and torch.cuda.is_available()
        else "cpu"
    )

    train_config = (
        ct.get_config_vEEGNet_training()
    )

    train_config["epochs"] = epochs
    train_config["batch_size"] = batch_size
    train_config["wandb_training"] = False
    train_config["print_var"] = True
    train_config["device"] = device
    train_config["path_to_save_model"] = str(
        model_directory
    )
    train_config["model_artifact_name"] = (
        f"vae_reconstruction_"
        f"S{subject_id:02d}_"
        f"generator-seed{generator_seed}"
    )
    train_config["notes"] = (
        "Hierarchical VAE trained from the central "
        "90/10 real EEG split"
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_channels = split.X_train.shape[1]
    n_times = split.X_train.shape[2]

    model_config = (
        cm.get_config_hierarchical_vEEGNet(
            n_channels,
            n_times,
            0,
            0,
        )
    )

    train_config[
        "measure_metrics_during_training"
    ] = model_config["use_classifier"]

    train_config["use_classifier"] = (
        model_config["use_classifier"]
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

    model_config["input_size"] = (
        train_dataset[0][0]
        .unsqueeze(0)
        .shape
    )

    model = train_generic.get_untrained_model(
        "hvEEGNet_shallow",
        model_config,
    )

    model.to(device)

    loss_function = (
        train_generic.get_loss_function(
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
            torch.optim.lr_scheduler.ExponentialLR(
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

    print("=" * 72, flush=True)
    print(
        f"START VAE reconstruction | "
        f"subject={subject_id} | "
        f"generator_seed={generator_seed} | "
        f"device={device}",
        flush=True,
    )
    print(
        f"Central split: {split.split_file}",
        flush=True,
    )
    print(
        f"Train={split.X_train.shape}, "
        f"valid={split.X_valid.shape}",
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
            "The external VAE trainer did not save its "
            f"validation-selected model: {best_model_file}"
        )

    state = torch.load(
        best_model_file,
        map_location=device,
    )

    model.load_state_dict(
        state
    )

    print(
        f"Loaded validation-selected model: "
        f"{best_model_file}",
        flush=True,
    )

    (
        X_reconstructed,
        y_reconstructed,
        original_shape,
    ) = reconstruct_dataset(
        model=model,
        dataset=train_dataset,
        batch_size=batch_size,
        device=device,
    )

    if X_reconstructed.shape != split.X_train.shape:
        raise RuntimeError(
            "VAE reconstruction shape "
            f"{X_reconstructed.shape} does not match "
            f"central training shape {split.X_train.shape}"
        )

    if not np.array_equal(
        y_reconstructed,
        split.y_train,
    ):
        raise RuntimeError(
            "VAE reconstruction labels do not exactly "
            "match the central training labels"
        )

    if not np.isfinite(
        X_reconstructed
    ).all():
        raise RuntimeError(
            "VAE reconstruction contains NaN or infinity"
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_file,
        X=X_reconstructed,
        y=y_reconstructed,
        protocol_id=config.protocol_id,
        method="vae_reconstruction",
        subject_id=subject_id,
        generator_seed=generator_seed,
        maximum_epochs=epochs,
        batch_size=batch_size,
        checkpoint_selection="lowest validation loss",
        split_file=str(split.split_file),
        original_reconstruction_shape=np.asarray(
            original_shape
        ),
        source=(
            "hierarchical VAE reconstruction of "
            "central real training EEG"
        ),
    )

    print(
        f"SAVED {output_file}",
        flush=True,
    )
    print(
        f"Shape={X_reconstructed.shape}, "
        f"mean={X_reconstructed.mean():.6f}, "
        f"std={X_reconstructed.std():.6f}",
        flush=True,
    )

    del model
    del optimizer
    del train_loader
    del valid_loader

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the adopted hierarchical VAE from the "
            "central EEG split and reconstruct only the "
            "real training trials."
        )
    )

    parser.add_argument(
        "--repo",
        default=str(
            CONFIG.external_vae_repository
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
        default=",".join(
            str(seed)
            for seed
            in CONFIG.generator_seeds
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CONFIG.hveegnet_max_epochs,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CONFIG.hveegnet_batch_size,
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

    subjects = parse_int_list(
        args.subjects
    )
    generator_seeds = parse_int_list(
        args.generator_seeds
    )

    for subject_id in subjects:
        for generator_seed in generator_seeds:
            train_subject(
                subject_id=subject_id,
                generator_seed=generator_seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                use_cuda=args.cuda,
                overwrite=args.overwrite,
                config=CONFIG,
            )


if __name__ == "__main__":
    main()
