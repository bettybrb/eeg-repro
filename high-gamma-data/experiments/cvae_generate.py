from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SharedCVAE(nn.Module):
    """Compact class-conditioned VAE for 22-channel, 1000-sample EEG."""

    def __init__(
        self,
        num_channels: int = 22,
        num_timesteps: int = 1000,
        num_classes: int = 4,
        latent_dim: int = 32,
        label_dim: int = 8,
    ) -> None:
        super().__init__()

        if num_channels != 22 or num_timesteps != 1000:
            raise ValueError(
                "This compact architecture expects EEG shaped (N, 22, 1000), "
                f"but received channels={num_channels}, timesteps={num_timesteps}."
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
                kernel_size=(num_channels, 1),
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

        self.encoded_shape = (64, 1, 20)
        encoded_size = int(np.prod(self.encoded_shape))

        self.encoder_label = nn.Embedding(num_classes, label_dim)
        self.decoder_label = nn.Embedding(num_classes, label_dim)

        self.fc_mu = nn.Linear(encoded_size + label_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoded_size + label_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim + label_dim, encoded_size)

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
                kernel_size=(num_channels, 1),
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
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x.unsqueeze(1)).flatten(start_dim=1)
        condition = self.encoder_label(labels)
        conditioned = torch.cat((features, condition), dim=1)
        return self.fc_mu(conditioned), self.fc_logvar(conditioned)

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.decoder_label(labels)
        conditioned = torch.cat((z, condition), dim=1)
        features = self.fc_decode(conditioned)
        features = features.view(-1, *self.encoded_shape)
        return self.decoder(features).squeeze(1)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar


def compute_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_loss = F.mse_loss(reconstructed, original)
    kl_loss = -0.5 * torch.mean(
        torch.sum(
            1.0 + logvar - mu.square() - logvar.exp(),
            dim=1,
        )
    )
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss


def run_epoch(
    model: SharedCVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)

    totals = np.zeros(3, dtype=np.float64)
    sample_count = 0

    for x, labels in loader:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            reconstructed, mu, logvar = model(x, labels)
            total_loss, reconstruction_loss, kl_loss = compute_loss(
                reconstructed,
                x,
                mu,
                logvar,
                beta,
            )

            if training:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = x.shape[0]
        totals += batch_size * np.array(
            [
                total_loss.detach().item(),
                reconstruction_loss.detach().item(),
                kl_loss.detach().item(),
            ]
        )
        sample_count += batch_size

    return tuple((totals / sample_count).tolist())


def generate_dataset(
    model: SharedCVAE,
    labels: np.ndarray,
    latent_dim: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    generated_batches: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            label_batch = torch.as_tensor(
                labels[start : start + batch_size],
                dtype=torch.long,
                device=device,
            )
            z = torch.randn(
                len(label_batch),
                latent_dim,
                device=device,
            )
            generated = model.decode(z, label_batch)
            generated_batches.append(
                generated.detach().cpu().numpy().astype(np.float32)
            )

    return np.concatenate(generated_batches, axis=0)


def train_subject(
    args: argparse.Namespace,
    subject: int,
    seed: int,
    device: torch.device,
) -> None:
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    model_directory = output_directory / "models"
    model_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_directory
        / f"S{subject:02d}_seed{seed}_vae_recon.npz"
    )
    checkpoint_path = (
        model_directory
        / f"S{subject:02d}_seed{seed}_cvae_best.pt"
    )

    if output_path.exists() and not args.overwrite:
        print(f"SKIP existing output: {output_path}", flush=True)
        return

    set_seed(seed + subject * 10_000)

    split_path = (
        Path(args.split_dir)
        / f"S{subject:02d}_real_splits.npz"
    )
    if not split_path.exists():
        raise FileNotFoundError(f"Missing real split: {split_path}")

    with np.load(split_path) as split:
        x_train = np.asarray(split["X_train"], dtype=np.float32)
        y_train = np.asarray(split["y_train"], dtype=np.int64)
        x_valid = np.asarray(split["X_valid"], dtype=np.float32)
        y_valid = np.asarray(split["y_valid"], dtype=np.int64)

    if x_train.ndim != 3:
        raise ValueError(
            f"Expected training shape (N, C, T), received {x_train.shape}."
        )
    if x_valid.shape[1:] != x_train.shape[1:]:
        raise ValueError(
            f"Training/validation shape mismatch: "
            f"{x_train.shape} versus {x_valid.shape}."
        )
    if not np.isfinite(x_train).all() or not np.isfinite(x_valid).all():
        raise ValueError("Training or validation EEG contains NaN/Inf values.")

    classes = np.unique(y_train)
    expected_classes = np.arange(args.num_classes)
    if not np.array_equal(classes, expected_classes):
        raise ValueError(
            f"Expected labels {expected_classes.tolist()}, "
            f"received {classes.tolist()}."
        )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_valid),
            torch.from_numpy(y_valid),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = SharedCVAE(
        num_channels=x_train.shape[1],
        num_timesteps=x_train.shape[2],
        num_classes=args.num_classes,
        latent_dim=args.latent_dim,
        label_dim=args.label_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    class_counts = {
        int(class_id): int(np.sum(y_train == class_id))
        for class_id in classes
    }

    print("=" * 72, flush=True)
    print(
        f"START subject={subject}, seed={seed}, device={device}",
        flush=True,
    )
    print(
        f"Train={x_train.shape}, valid={x_valid.shape}, "
        f"class_counts={class_counts}",
        flush=True,
    )
    print(
        f"Parameters={parameter_count:,}, latent_dim={args.latent_dim}",
        flush=True,
    )

    best_validation_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        beta = args.beta * min(
            1.0,
            epoch / max(1, args.kl_warmup_epochs),
        )

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            beta,
            optimizer,
        )
        validation_metrics = run_epoch(
            model,
            valid_loader,
            device,
            beta,
            optimizer=None,
        )

        print(
            f"Subject {subject:02d} | epoch {epoch:03d}/{args.epochs} | "
            f"beta={beta:.5f} | "
            f"train total={train_metrics[0]:.5f}, "
            f"mse={train_metrics[1]:.5f}, kl={train_metrics[2]:.5f} | "
            f"valid total={validation_metrics[0]:.5f}, "
            f"mse={validation_metrics[1]:.5f}, "
            f"kl={validation_metrics[2]:.5f}",
            flush=True,
        )

        if validation_metrics[0] < best_validation_loss:
            best_validation_loss = validation_metrics[0]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "subject": subject,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_loss": best_validation_loss,
                    "latent_dim": args.latent_dim,
                    "label_dim": args.label_dim,
                    "num_classes": args.num_classes,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    generation_labels = np.concatenate(
        [
            np.full(
                np.sum(y_train == class_id),
                class_id,
                dtype=np.int64,
            )
            for class_id in classes
        ]
    )

    set_seed(seed + subject * 10_000 + 1)
    x_generated = generate_dataset(
        model,
        generation_labels,
        args.latent_dim,
        args.batch_size,
        device,
    )

    if x_generated.shape != x_train.shape:
        raise RuntimeError(
            f"Generated shape {x_generated.shape} does not match "
            f"training shape {x_train.shape}."
        )
    if not np.isfinite(x_generated).all():
        raise RuntimeError("Generated EEG contains NaN/Inf values.")

    np.savez_compressed(
        output_path,
        X_recon=x_generated,
        X_generated=x_generated,
        y=generation_labels,
        subject=np.asarray(subject),
        seed=np.asarray(seed),
        best_epoch=np.asarray(best_epoch),
        best_validation_loss=np.asarray(best_validation_loss),
        source=np.asarray("shared_true_cvae_prior_generation"),
        conditioning=np.asarray(
            "class_label_in_encoder_and_decoder"
        ),
    )

    generated_counts = {
        int(class_id): int(np.sum(generation_labels == class_id))
        for class_id in classes
    }

    print(
        f"DONE subject={subject}, seed={seed}, best_epoch={best_epoch}, "
        f"best_valid={best_validation_loss:.5f}",
        flush=True,
    )
    print(f"Saved: {output_path}", flush=True)
    print(
        f"Generated={x_generated.shape}, labels={generated_counts}, "
        f"mean={x_generated.mean():.5f}, "
        f"std={x_generated.std():.5f}, "
        f"min={x_generated.min():.5f}, "
        f"max={x_generated.max():.5f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train one shared class-conditioned VAE per subject and "
            "generate labelled EEG from the standard-normal prior."
        )
    )
    parser.add_argument(
        "--split-dir",
        default="outputs/vae_runs/run1/classifier_real_splits",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/generated/shared_cvae",
    )
    parser.add_argument(
        "--subjects",
        default="1,2,3,4,5,6,7,8,9",
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--label-dim", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-2)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("--cuda was requested, but CUDA is unavailable.")

    device = torch.device(
        "cuda" if args.cuda else "cpu"
    )

    print(
        f"Shared CVAE run | subjects={args.subjects} | "
        f"seeds={args.seeds} | epochs={args.epochs} | device={device}",
        flush=True,
    )

    for subject in parse_int_list(args.subjects):
        for seed in parse_int_list(args.seeds):
            train_subject(args, subject, seed, device)


if __name__ == "__main__":
    main()