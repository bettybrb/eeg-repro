from __future__ import annotations

import argparse
import gc
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.cvae_generate import (
    SharedCVAE,
)
from pipeline.config import CONFIG
from pipeline.splits import load_real_split


METHOD_NAME = (
    "hierarchical_conditional_vae_generation"
)


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
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = (
        True
    )
    torch.backends.cudnn.benchmark = (
        False
    )


def split_parameters(values):
    mean, log_variance = values.chunk(
        2,
        dim=1,
    )

    return (
        mean,
        torch.clamp(
            log_variance,
            min=-8.0,
            max=8.0,
        ),
    )


def reparameterize(
    mean,
    log_variance,
):
    standard_deviation = torch.exp(
        0.5 * log_variance
    )

    return (
        mean
        + torch.randn_like(
            standard_deviation
        )
        * standard_deviation
    )


def gaussian_kl(
    posterior_mean,
    posterior_log_variance,
    prior_mean,
    prior_log_variance,
):
    posterior_variance = (
        posterior_log_variance.exp()
    )
    prior_variance = (
        prior_log_variance.exp()
    )

    values = 0.5 * torch.sum(
        (
            prior_log_variance
            - posterior_log_variance
            + (
                posterior_variance
                + (
                    posterior_mean
                    - prior_mean
                ).square()
            )
            / prior_variance
            - 1.0
        ),
        dim=1,
    )

    return torch.mean(
        values
    )


class HierarchicalSharedCVAE(
    nn.Module
):
    """
    Shared two-level class-conditioned VAE.

    One model is trained jointly across all four classes. Both latent
    levels have learned class-dependent priors. Generation samples only
    from these priors and does not encode any real EEG trial.
    """

    def __init__(
        self,
        num_channels=22,
        num_timesteps=1000,
        num_classes=4,
        high_latent_dim=32,
        low_latent_dim=64,
        label_dim=16,
    ):
        super().__init__()

        # Reuse the tested convolutional encoder/decoder dimensions
        # from the existing shared CVAE without changing that model.
        base_model = SharedCVAE(
            num_channels=num_channels,
            num_timesteps=num_timesteps,
            num_classes=num_classes,
            latent_dim=32,
            label_dim=8,
        )

        self.encoder = (
            base_model.encoder
        )
        self.decoder = (
            base_model.decoder
        )

        self.encoded_shape = (
            base_model.encoded_shape
        )
        self.high_latent_dim = (
            high_latent_dim
        )
        self.low_latent_dim = (
            low_latent_dim
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

        self.high_posterior = (
            nn.Sequential(
                nn.Linear(
                    encoded_size
                    + label_dim,
                    256,
                ),
                nn.ELU(),
                nn.Linear(
                    256,
                    2 * high_latent_dim,
                ),
            )
        )

        self.low_posterior = (
            nn.Sequential(
                nn.Linear(
                    encoded_size
                    + label_dim
                    + high_latent_dim,
                    256,
                ),
                nn.ELU(),
                nn.Linear(
                    256,
                    2 * low_latent_dim,
                ),
            )
        )

        # One learned high-level Gaussian prior per class.
        self.high_prior = nn.Embedding(
            num_classes,
            2 * high_latent_dim,
        )

        # The low-level prior depends on class and z_high.
        self.low_prior = nn.Sequential(
            nn.Linear(
                high_latent_dim
                + label_dim,
                128,
            ),
            nn.ELU(),
            nn.Linear(
                128,
                2 * low_latent_dim,
            ),
        )

        self.fc_decode = nn.Sequential(
            nn.Linear(
                high_latent_dim
                + low_latent_dim
                + label_dim,
                encoded_size,
            ),
            nn.ELU(),
        )

        # Begin with approximately standard-normal priors.
        nn.init.zeros_(
            self.high_prior.weight
        )
        nn.init.zeros_(
            self.low_prior[-1].weight
        )
        nn.init.zeros_(
            self.low_prior[-1].bias
        )

    def high_prior_parameters(
        self,
        labels,
    ):
        return split_parameters(
            self.high_prior(
                labels
            )
        )

    def low_prior_parameters(
        self,
        high_latent,
        labels,
    ):
        condition = self.decoder_label(
            labels
        )

        return split_parameters(
            self.low_prior(
                torch.cat(
                    (
                        high_latent,
                        condition,
                    ),
                    dim=1,
                )
            )
        )

    def decode(
        self,
        high_latent,
        low_latent,
        labels,
    ):
        condition = self.decoder_label(
            labels
        )

        features = self.fc_decode(
            torch.cat(
                (
                    high_latent,
                    low_latent,
                    condition,
                ),
                dim=1,
            )
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
        features = self.encoder(
            X.unsqueeze(1)
        ).flatten(
            start_dim=1
        )

        encoder_condition = (
            self.encoder_label(
                labels
            )
        )

        (
            high_mean,
            high_log_variance,
        ) = split_parameters(
            self.high_posterior(
                torch.cat(
                    (
                        features,
                        encoder_condition,
                    ),
                    dim=1,
                )
            )
        )

        (
            high_prior_mean,
            high_prior_log_variance,
        ) = self.high_prior_parameters(
            labels
        )

        high_latent = reparameterize(
            high_mean,
            high_log_variance,
        )

        (
            low_mean,
            low_log_variance,
        ) = split_parameters(
            self.low_posterior(
                torch.cat(
                    (
                        features,
                        encoder_condition,
                        high_latent,
                    ),
                    dim=1,
                )
            )
        )

        (
            low_prior_mean,
            low_prior_log_variance,
        ) = self.low_prior_parameters(
            high_latent,
            labels,
        )

        low_latent = reparameterize(
            low_mean,
            low_log_variance,
        )

        reconstruction = self.decode(
            high_latent,
            low_latent,
            labels,
        )

        return {
            "reconstruction": (
                reconstruction
            ),
            "high_mean": high_mean,
            "high_log_variance": (
                high_log_variance
            ),
            "high_prior_mean": (
                high_prior_mean
            ),
            "high_prior_log_variance": (
                high_prior_log_variance
            ),
            "low_mean": low_mean,
            "low_log_variance": (
                low_log_variance
            ),
            "low_prior_mean": (
                low_prior_mean
            ),
            "low_prior_log_variance": (
                low_prior_log_variance
            ),
        }

    def generate_from_prior(
        self,
        labels,
    ):
        (
            high_mean,
            high_log_variance,
        ) = self.high_prior_parameters(
            labels
        )

        high_latent = reparameterize(
            high_mean,
            high_log_variance,
        )

        (
            low_mean,
            low_log_variance,
        ) = self.low_prior_parameters(
            high_latent,
            labels,
        )

        low_latent = reparameterize(
            low_mean,
            low_log_variance,
        )

        return self.decode(
            high_latent,
            low_latent,
            labels,
        )


def compute_loss(
    outputs,
    original,
    beta,
):
    reconstruction_loss = F.mse_loss(
        outputs["reconstruction"],
        original,
    )

    high_kl = gaussian_kl(
        outputs["high_mean"],
        outputs[
            "high_log_variance"
        ],
        outputs[
            "high_prior_mean"
        ],
        outputs[
            "high_prior_log_variance"
        ],
    )

    low_kl = gaussian_kl(
        outputs["low_mean"],
        outputs[
            "low_log_variance"
        ],
        outputs[
            "low_prior_mean"
        ],
        outputs[
            "low_prior_log_variance"
        ],
    )

    total_kl = (
        high_kl
        + low_kl
    )

    total_loss = (
        reconstruction_loss
        + beta * total_kl
    )

    return (
        total_loss,
        reconstruction_loss,
        total_kl,
        high_kl,
        low_kl,
    )


def run_epoch(
    model,
    loader,
    device,
    beta,
    optimizer,
):
    training = (
        optimizer is not None
    )

    model.train(
        training
    )

    totals = np.zeros(
        5,
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
            outputs = model(
                X_batch,
                y_batch,
            )

            metrics = compute_loss(
                outputs,
                X_batch,
                beta,
            )

            if training:
                metrics[0].backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

        current_batch_size = (
            X_batch.shape[0]
        )

        totals += (
            current_batch_size
            * np.asarray(
                [
                    metric
                    .detach()
                    .item()
                    for metric
                    in metrics
                ],
                dtype=np.float64,
            )
        )

        sample_count += (
            current_batch_size
        )

    return tuple(
        (
            totals
            / sample_count
        ).tolist()
    )


def generate_dataset(
    model,
    labels,
    batch_size,
    device,
):
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

            generated = (
                model.generate_from_prior(
                    label_batch
                )
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
        config
        .hierarchical_conditional_vae_directory
        / (
            f"S{subject_id:02d}_"
            f"generator-seed{generator_seed}.npz"
        )
    )


def train_subject(
    subject_id,
    generator_seed,
    epochs,
    minimum_epochs,
    patience,
    warmup_epochs,
    batch_size,
    high_latent_dim,
    low_latent_dim,
    label_dim,
    beta,
    learning_rate,
    weight_decay,
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
            "SKIP existing hierarchical "
            f"CVAE data: {destination}",
            flush=True,
        )
        return

    if overwrite:
        destination.unlink(
            missing_ok=True
        )

    destination.parent.mkdir(
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

    device = torch.device(
        "cuda"
        if (
            use_cuda
            and torch.cuda.is_available()
        )
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

    model = HierarchicalSharedCVAE(
        num_channels=(
            X_train.shape[1]
        ),
        num_timesteps=(
            X_train.shape[2]
        ),
        num_classes=len(
            config.class_ids
        ),
        high_latent_dim=(
            high_latent_dim
        ),
        low_latent_dim=(
            low_latent_dim
        ),
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

    print(
        "=" * 72,
        flush=True,
    )
    print(
        "START shared hierarchical CVAE | "
        f"subject={subject_id} | "
        f"generator_seed={generator_seed} | "
        f"device={device}",
        flush=True,
    )
    print(
        f"Train={X_train.shape}, "
        f"valid={X_valid.shape}, "
        f"parameters={parameter_count:,}",
        flush=True,
    )

    selection_start = min(
        max(
            1,
            warmup_epochs,
        ),
        epochs,
    )

    effective_minimum = min(
        max(
            minimum_epochs,
            selection_start,
        ),
        epochs,
    )

    best_validation_loss = float(
        "inf"
    )
    best_metrics = None
    best_epoch = -1
    epochs_without_improvement = 0
    epochs_completed = 0
    stopped_early = False
    history = []

    with tempfile.TemporaryDirectory(
        prefix=(
            f"hierarchical_cvae_"
            f"S{subject_id:02d}_"
            f"seed{generator_seed}_"
        )
    ) as temporary_directory:
        checkpoint_file = (
            Path(temporary_directory)
            / "best.pt"
        )

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
                        warmup_epochs,
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

            history.append(
                (
                    epoch,
                    current_beta,
                    *train_metrics,
                    *validation_metrics,
                )
            )

            epochs_completed = epoch

            eligible = (
                epoch >= selection_start
            )

            improved = (
                eligible
                and validation_metrics[0]
                < best_validation_loss
            )

            if improved:
                best_validation_loss = (
                    validation_metrics[0]
                )
                best_metrics = (
                    validation_metrics
                )
                best_epoch = epoch
                epochs_without_improvement = 0

                torch.save(
                    model.state_dict(),
                    checkpoint_file,
                )

            elif eligible:
                epochs_without_improvement += 1

            if (
                epoch >= effective_minimum
                and epochs_without_improvement
                >= patience
            ):
                stopped_early = True

                print(
                    "EARLY STOP | "
                    f"epoch={epoch} | "
                    f"best_epoch={best_epoch}",
                    flush=True,
                )
                break

        if not checkpoint_file.exists():
            raise RuntimeError(
                "Validation-selected checkpoint "
                "was not created"
            )

        model.load_state_dict(
            torch.load(
                checkpoint_file,
                map_location=device,
            )
        )

        generation_labels = (
            y_train.copy()
        )

        prior_seed = (
            generator_seed + 1
        )
        set_seed(
            prior_seed
        )

        X_generated = generate_dataset(
            model=model,
            labels=generation_labels,
            batch_size=batch_size,
            device=device,
        )

    if (
        X_generated.shape
        != X_train.shape
    ):
        raise RuntimeError(
            f"Generated shape "
            f"{X_generated.shape} does "
            f"not match {X_train.shape}"
        )

    if not np.isfinite(
        X_generated
    ).all():
        raise RuntimeError(
            "Generated EEG contains "
            "NaN or infinity"
        )

    np.savez_compressed(
        destination,
        X=X_generated,
        y=generation_labels,
        protocol_id=(
            config.protocol_id
        ),
        method=METHOD_NAME,
        subject_id=subject_id,
        generator_seed=(
            generator_seed
        ),
        prior_seed=prior_seed,
        split_file=str(
            split.split_file
        ),
        best_epoch=best_epoch,
        best_validation_loss=(
            best_validation_loss
        ),
        best_validation_mse=(
            best_metrics[1]
        ),
        best_validation_kl=(
            best_metrics[2]
        ),
        best_validation_high_kl=(
            best_metrics[3]
        ),
        best_validation_low_kl=(
            best_metrics[4]
        ),
        maximum_epochs=epochs,
        minimum_epochs=(
            effective_minimum
        ),
        epochs_completed=(
            epochs_completed
        ),
        stopped_early=(
            stopped_early
        ),
        early_stopping_patience=(
            patience
        ),
        selection_start_epoch=(
            selection_start
        ),
        kl_warmup_epochs=(
            warmup_epochs
        ),
        beta=beta,
        high_latent_dim=(
            high_latent_dim
        ),
        low_latent_dim=(
            low_latent_dim
        ),
        label_dim=label_dim,
        parameter_count=(
            parameter_count
        ),
        history=np.asarray(
            history,
            dtype=np.float64,
        ),
        checkpoint_selection=(
            "lowest validation total "
            "loss after KL warm-up"
        ),
        conditioning=(
            "class conditioning in encoder, "
            "two learned latent priors and decoder"
        ),
        source=(
            "genuine EEG generation from learned "
            "class-dependent hierarchical priors; "
            "not reconstruction"
        ),
    )

    print(
        f"SAVED {destination}",
        flush=True,
    )
    print(
        f"Best epoch={best_epoch}, "
        f"validation loss="
        f"{best_validation_loss:.6f}",
        flush=True,
    )
    print(
        f"Generated shape="
        f"{X_generated.shape}, "
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
            "Train the exploratory shared "
            "hierarchical class-conditioned VAE."
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
        "--cuda",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

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
                epochs=(
                    CONFIG
                    .hierarchical_cvae_max_epochs
                ),
                minimum_epochs=(
                    CONFIG
                    .hierarchical_cvae_minimum_epochs
                ),
                patience=(
                    CONFIG
                    .hierarchical_cvae_early_stopping_patience
                ),
                warmup_epochs=(
                    CONFIG
                    .hierarchical_cvae_kl_warmup_epochs
                ),
                batch_size=(
                    CONFIG
                    .hierarchical_cvae_batch_size
                ),
                high_latent_dim=(
                    CONFIG
                    .hierarchical_cvae_high_latent_dim
                ),
                low_latent_dim=(
                    CONFIG
                    .hierarchical_cvae_low_latent_dim
                ),
                label_dim=(
                    CONFIG
                    .hierarchical_cvae_label_dim
                ),
                beta=(
                    CONFIG
                    .hierarchical_cvae_beta
                ),
                learning_rate=(
                    CONFIG
                    .hierarchical_cvae_learning_rate
                ),
                weight_decay=(
                    CONFIG
                    .hierarchical_cvae_weight_decay
                ),
                use_cuda=args.cuda,
                overwrite=(
                    args.overwrite
                ),
                config=CONFIG,
            )


if __name__ == "__main__":
    main()
