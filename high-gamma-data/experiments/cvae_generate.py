from __future__ import annotations

import argparse
import gc
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
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


class SharedCVAE(nn.Module):
    """
    Class-conditioned VAE for EEG shaped
    (trials, 22 channels, 1000 time samples).

    The class label is supplied to both the encoder and decoder.
    """

    def __init__(
        self,
        num_channels=22,
        num_timesteps=1000,
        num_classes=4,
        latent_dim=32,
        label_dim=8,
    ):
        super().__init__()

        if (
            num_channels != 22
            or num_timesteps != 1000
        ):
            raise ValueError(
                "This architecture expects EEG shaped "
                "(N, 22, 1000), but received "
                f"channels={num_channels}, "
                f"timesteps={num_timesteps}"
            )

        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(
                1,
                16,
                kernel_size=(1, 25),
                stride=(1, 2),
                padding=(0, 12),
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(
                16,
                32,
                kernel_size=(
                    num_channels,
                    1,
                ),
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(
                32,
                64,
                kernel_size=(1, 5),
                stride=(1, 5),
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(
                64,
                64,
                kernel_size=(1, 5),
                stride=(1, 5),
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )

        self.encoded_shape = (
            64,
            1,
            20,
        )
        encoded_size = int(
            np.prod(
                self.encoded_shape
            )
        )

        self.encoder_label = nn.Embedding(
            num_classes,
            label_dim,
        )
        self.decoder_label = nn.Embedding(
            num_classes,
            label_dim,
        )

        self.fc_mu = nn.Linear(
            encoded_size + label_dim,
            latent_dim,
        )
        self.fc_logvar = nn.Linear(
            encoded_size + label_dim,
            latent_dim,
        )
        self.fc_decode = nn.Linear(
            latent_dim + label_dim,
            encoded_size,
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64,
                64,
                kernel_size=(1, 5),
                stride=(1, 5),
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=(1, 5),
                stride=(1, 5),
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=(
                    num_channels,
                    1,
                ),
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.ConvTranspose2d(
                16,
                1,
                kernel_size=(1, 4),
                stride=(1, 2),
                padding=(0, 1),
            ),
        )

    def encode(
        self,
        X,
        labels,
    ):
        features = self.encoder(
            X.unsqueeze(1)
        ).flatten(
            start_dim=1
        )

        condition = self.encoder_label(
            labels
        )

        conditioned_features = torch.cat(
            (
                features,
                condition,
            ),
            dim=1,
        )

        return (
            self.fc_mu(
                conditioned_features
            ),
            self.fc_logvar(
                conditioned_features
            ),
        )

    @staticmethod
    def reparameterize(
        mu,
        log_variance,
    ):
        standard_deviation = torch.exp(
            0.5 * log_variance
        )

        noise = torch.randn_like(
            standard_deviation
        )

        return (
            mu
            + noise
            * standard_deviation
        )

    def decode(
        self,
        latent,
        labels,
    ):
        condition = self.decoder_label(
            labels
        )

        conditioned_latent = torch.cat(
            (
                latent,
                condition,
            ),
            dim=1,
        )

        features = self.fc_decode(
            conditioned_latent
        )

        features = features.view(
            -1,
            *self.encoded_shape,
        )

        return self.decoder(
            features
        ).squeeze(1)

    def forward(
        self,
        X,
        labels,
    ):
        mu, log_variance = self.encode(
            X,
            labels,
        )

        latent = self.reparameterize(
            mu,
            log_variance,
        )

        reconstruction = self.decode(
            latent,
            labels,
        )

        return (
            reconstruction,
            mu,
            log_variance,
        )


def compute_loss(
    reconstruction,
    original,
    mu,
    log_variance,
    beta,
):
    reconstruction_loss = F.mse_loss(
        reconstruction,
        original,
    )

    kl_loss = -0.5 * torch.mean(
        torch.sum(
            (
                1.0
                + log_variance
                - mu.square()
                - log_variance.exp()
            ),
            dim=1,
        )
    )

    total_loss = (
        reconstruction_loss
        + beta * kl_loss
    )

    return (
        total_loss,
        reconstruction_loss,
        kl_loss,
    )


def run_epoch(
    model,
    loader,
    device,
    beta,
    optimizer,
):
    training = optimizer is not None
    model.train(training)

    totals = np.zeros(
        3,
        dtype=np.float64,
    )
    sample_count = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(
            device,
            non_blocking=True,
        )
        y_batch = y_batch.to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):
            (
                reconstruction,
                mu,
                log_variance,
            ) = model(
                X_batch,
                y_batch,
            )

            (
                total_loss,
                reconstruction_loss,
                kl_loss,
            ) = compute_loss(
                reconstruction,
                X_batch,
                mu,
                log_variance,
                beta,
            )

            if training:
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

        batch_size = X_batch.shape[0]

        totals += (
            batch_size
            * np.asarray(
                [
                    total_loss
                    .detach()
                    .item(),
                    reconstruction_loss
                    .detach()
                    .item(),
                    kl_loss
                    .detach()
                    .item(),
                ],
                dtype=np.float64,
            )
        )

        sample_count += batch_size

    return tuple(
        (
            totals
            / sample_count
        ).tolist()
    )


def generate_from_prior(
    model,
    labels,
    latent_dim,
    batch_size,
    device,
):
    """
    Generate new EEG from standard-normal latent samples.

    This is not reconstruction: no real EEG trial is passed
    through the encoder during generation.
    """

    model.eval()
    generated_batches = []

    with torch.no_grad():
        for start in range(
            0,
            len(labels),
            batch_size,
        ):
            label_batch = torch.as_tensor(
                labels[
                    start:
                    start + batch_size
                ],
                dtype=torch.long,
                device=device,
            )

            latent = torch.randn(
                len(label_batch),
                latent_dim,
                device=device,
            )

            generated = model.decode(
                latent,
                label_batch,
            )

            generated_batches.append(
                generated
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    return np.concatenate(
        generated_batches,
        axis=0,
    )


def output_file(
    subject_id,
    generator_seed,
    config,
):
    return (
        config.conditional_vae_directory
        / (
            f"S{subject_id:02d}_"
            f"generator-seed{generator_seed}.npz"
        )
    )


def model_directory(
    subject_id,
    generator_seed,
    config,
):
    return (
        config.checkpoint_directory
        / "conditional_vae"
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
    latent_dim,
    label_dim,
    learning_rate,
    weight_decay,
    beta,
    kl_warmup_epochs,
    minimum_epochs,
    early_stopping_patience,
    use_cuda,
    overwrite,
    config,
):
    generated_file = output_file(
        subject_id,
        generator_seed,
        config,
    )

    checkpoint_directory = model_directory(
        subject_id,
        generator_seed,
        config,
    )

    checkpoint_file = (
        checkpoint_directory
        / "cvae_best.pt"
    )

    if generated_file.exists() and not overwrite:
        print(
            f"SKIP existing generated EEG: "
            f"{generated_file}",
            flush=True,
        )
        return

    if overwrite:
        generated_file.unlink(
            missing_ok=True
        )

        if checkpoint_directory.exists():
            shutil.rmtree(
                checkpoint_directory
            )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(
        generator_seed
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

    if set(np.unique(y_train)) != set(
        config.class_ids
    ):
        raise RuntimeError(
            "CVAE training split does not contain "
            "all four motor-imagery classes"
        )

    device = torch.device(
        "cuda"
        if use_cuda and torch.cuda.is_available()
        else "cpu"
    )

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(
                X_train
            ),
            torch.from_numpy(
                y_train
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(
                X_valid
            ),
            torch.from_numpy(
                y_valid
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = SharedCVAE(
        num_channels=(
            X_train.shape[1]
        ),
        num_timesteps=(
            X_train.shape[2]
        ),
        num_classes=len(
            config.class_ids
        ),
        latent_dim=latent_dim,
        label_dim=label_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print("=" * 72, flush=True)
    print(
        f"START conditional VAE | "
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
        f"Train={X_train.shape}, "
        f"valid={X_valid.shape}",
        flush=True,
    )
    print(
        f"Parameters={parameter_count:,}, "
        f"latent_dim={latent_dim}",
        flush=True,
    )

    best_validation_loss = float(
        "inf"
    )
    best_validation_mse = float(
        "inf"
    )
    best_validation_kl = float(
        "inf"
    )
    best_epoch = -1

    # During KL warm-up the total objective changes between epochs.
    # Model selection therefore begins only once the full beta value
    # is active. Short smoke runs remain supported by beginning at
    # their final epoch when they end before the warm-up completes.
    selection_start_epoch = min(
        max(1, kl_warmup_epochs),
        epochs,
    )
    effective_minimum_epochs = min(
        max(
            minimum_epochs,
            selection_start_epoch,
        ),
        epochs,
    )

    epochs_without_improvement = 0
    epochs_completed = 0
    stopped_early = False
    training_history = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        current_beta = (
            beta
            * min(
                1.0,
                epoch
                / max(
                    1,
                    kl_warmup_epochs,
                ),
            )
        )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            beta=current_beta,
            optimizer=optimizer,
        )

        validation_metrics = run_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            beta=current_beta,
            optimizer=None,
        )

        print(
            f"Subject {subject_id:02d} | "
            f"epoch {epoch:03d}/{epochs} | "
            f"beta={current_beta:.5f} | "
            f"train total={train_metrics[0]:.5f}, "
            f"mse={train_metrics[1]:.5f}, "
            f"kl={train_metrics[2]:.5f} | "
            f"valid total={validation_metrics[0]:.5f}, "
            f"mse={validation_metrics[1]:.5f}, "
            f"kl={validation_metrics[2]:.5f}",
            flush=True,
        )

        epochs_completed = epoch

        training_history.append(
            {
                "epoch": epoch,
                "beta": current_beta,
                "train_total": train_metrics[0],
                "train_mse": train_metrics[1],
                "train_kl": train_metrics[2],
                "validation_total": validation_metrics[0],
                "validation_mse": validation_metrics[1],
                "validation_kl": validation_metrics[2],
            }
        )

        eligible_for_selection = (
            epoch >= selection_start_epoch
        )

        improved = (
            eligible_for_selection
            and validation_metrics[0]
            < best_validation_loss
        )

        if improved:
            best_validation_loss = (
                validation_metrics[0]
            )
            best_validation_mse = (
                validation_metrics[1]
            )
            best_validation_kl = (
                validation_metrics[2]
            )
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "protocol_id": (
                        config.protocol_id
                    ),
                    "subject_id": (
                        subject_id
                    ),
                    "generator_seed": (
                        generator_seed
                    ),
                    "epoch": epoch,
                    "validation_loss": (
                        best_validation_loss
                    ),
                    "validation_mse": (
                        best_validation_mse
                    ),
                    "validation_kl": (
                        best_validation_kl
                    ),
                    "maximum_epochs": epochs,
                    "minimum_epochs": (
                        effective_minimum_epochs
                    ),
                    "early_stopping_patience": (
                        early_stopping_patience
                    ),
                    "selection_start_epoch": (
                        selection_start_epoch
                    ),
                    "kl_warmup_epochs": (
                        kl_warmup_epochs
                    ),
                    "latent_dim": (
                        latent_dim
                    ),
                    "label_dim": (
                        label_dim
                    ),
                    "num_classes": len(
                        config.class_ids
                    ),
                    "split_file": str(
                        split.split_file
                    ),
                },
                checkpoint_file,
            )

        elif eligible_for_selection:
            epochs_without_improvement += 1

        if (
            epoch >= effective_minimum_epochs
            and epochs_without_improvement
            >= early_stopping_patience
        ):
            stopped_early = True

            print(
                "EARLY STOP | "
                f"epoch={epoch} | "
                f"best_epoch={best_epoch} | "
                f"best_validation_loss="
                f"{best_validation_loss:.6f}",
                flush=True,
            )
            break

    history_file = (
        checkpoint_directory
        / "training_history.npz"
    )

    np.savez_compressed(
        history_file,
        epoch=np.asarray(
            [
                row["epoch"]
                for row in training_history
            ],
            dtype=np.int64,
        ),
        beta=np.asarray(
            [
                row["beta"]
                for row in training_history
            ],
            dtype=np.float64,
        ),
        train_total=np.asarray(
            [
                row["train_total"]
                for row in training_history
            ],
            dtype=np.float64,
        ),
        train_mse=np.asarray(
            [
                row["train_mse"]
                for row in training_history
            ],
            dtype=np.float64,
        ),
        train_kl=np.asarray(
            [
                row["train_kl"]
                for row in training_history
            ],
            dtype=np.float64,
        ),
        validation_total=np.asarray(
            [
                row["validation_total"]
                for row in training_history
            ],
            dtype=np.float64,
        ),
        validation_mse=np.asarray(
            [
                row["validation_mse"]
                for row in training_history
            ],
            dtype=np.float64,
        ),
        validation_kl=np.asarray(
            [
                row["validation_kl"]
                for row in training_history
            ],
            dtype=np.float64,
        ),
    )

    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"CVAE checkpoint was not created: "
            f"{checkpoint_file}"
        )

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # Retain the exact central training-label order so the
    # classifier receives a directly comparable training set.
    generation_labels = (
        y_train.copy()
    )

    prior_seed = (
        generator_seed + 1
    )

    set_seed(
        prior_seed
    )

    X_generated = generate_from_prior(
        model=model,
        labels=generation_labels,
        latent_dim=latent_dim,
        batch_size=batch_size,
        device=device,
    )

    if X_generated.shape != X_train.shape:
        raise RuntimeError(
            f"Generated CVAE shape {X_generated.shape} "
            f"does not match training shape {X_train.shape}"
        )

    if not np.isfinite(
        X_generated
    ).all():
        raise RuntimeError(
            "Generated CVAE EEG contains NaN or infinity"
        )

    np.savez_compressed(
        generated_file,
        X=X_generated,
        y=generation_labels,
        protocol_id=config.protocol_id,
        method=(
            "conditional_vae_generation"
        ),
        subject_id=subject_id,
        generator_seed=generator_seed,
        prior_seed=prior_seed,
        split_file=str(
            split.split_file
        ),
        best_epoch=best_epoch,
        best_validation_loss=(
            best_validation_loss
        ),
        best_validation_mse=(
            best_validation_mse
        ),
        best_validation_kl=(
            best_validation_kl
        ),
        maximum_epochs=epochs,
        minimum_epochs=(
            effective_minimum_epochs
        ),
        epochs_completed=epochs_completed,
        stopped_early=stopped_early,
        early_stopping_patience=(
            early_stopping_patience
        ),
        selection_start_epoch=(
            selection_start_epoch
        ),
        kl_warmup_epochs=(
            kl_warmup_epochs
        ),
        training_history_file=str(
            history_file
        ),
        checkpoint_selection=(
            "lowest validation total loss after KL warm-up"
        ),
        conditioning=(
            "class label supplied to encoder and decoder"
        ),
        source=(
            "new EEG generated from standard-normal "
            "latent prior; not reconstruction"
        ),
    )

    print(
        f"SAVED {generated_file}",
        flush=True,
    )
    print(
        f"Best epoch={best_epoch}, "
        f"best validation loss="
        f"{best_validation_loss:.6f}",
        flush=True,
    )
    print(
        f"Generated shape={X_generated.shape}, "
        f"mean={X_generated.mean():.6f}, "
        f"std={X_generated.std():.6f}",
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
            "Train a subject-specific class-conditioned VAE "
            "from the central split and generate new labelled "
            "EEG from its latent prior."
        )
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
        default=CONFIG.cvae_max_epochs,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CONFIG.cvae_batch_size,
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--label-dim",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=CONFIG.cvae_kl_warmup_epochs,
    )
    parser.add_argument(
        "--minimum-epochs",
        type=int,
        default=CONFIG.cvae_minimum_epochs,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=CONFIG.cvae_early_stopping_patience,
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

    if args.epochs < 1:
        raise ValueError(
            "--epochs must be at least 1"
        )

    if args.minimum_epochs < 1:
        raise ValueError(
            "--minimum-epochs must be at least 1"
        )

    if args.early_stopping_patience < 1:
        raise ValueError(
            "--early-stopping-patience must be at least 1"
        )

    if args.kl_warmup_epochs < 1:
        raise ValueError(
            "--kl-warmup-epochs must be at least 1"
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
                latent_dim=args.latent_dim,
                label_dim=args.label_dim,
                learning_rate=(
                    args.learning_rate
                ),
                weight_decay=(
                    args.weight_decay
                ),
                beta=args.beta,
                kl_warmup_epochs=(
                    args.kl_warmup_epochs
                ),
                minimum_epochs=(
                    args.minimum_epochs
                ),
                early_stopping_patience=(
                    args.early_stopping_patience
                ),
                use_cuda=args.cuda,
                overwrite=args.overwrite,
                config=CONFIG,
            )


if __name__ == "__main__":
    main()
