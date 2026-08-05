import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.vae_make import add_vae_repo_to_path, parse_int_list, set_seed


def labels_of(dataset):
    return np.array([
        int(y.item() if torch.is_tensor(y) else y)
        for _, y in dataset
    ])


def to_3d(x):
    x = x.detach().cpu().numpy()

    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]

    if x.ndim != 3:
        raise RuntimeError(f"Unexpected generated shape: {x.shape}")

    return x.astype(np.float32)


def run_subject(args, subject, seed):
    set_seed(seed)
    add_vae_repo_to_path(args.repo)

    from library.config import config_dataset as cd
    from library.config import config_model as cm
    from library.config import config_training as ct
    from library.dataset import preprocess as pp
    from library.dataset import support_function as sf
    from library.training import train_generic

    device = (
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "cpu"
    )

    dataset_config = cd.get_moabb_dataset_config([subject])
    dataset_config["percentage_split_train_test"] = -1
    dataset_config["percentage_split_train_validation"] = 0.9
    dataset_config["seed_split"] = 42

    train_data, valid_data, test_data = pp.get_dataset_d2a(
        dataset_config
    )

    train_labels = labels_of(train_data)
    valid_labels = labels_of(valid_data)
    classes = [int(x) for x in sorted(np.unique(train_labels))]

    if classes != [0, 1, 2, 3]:
        raise RuntimeError(f"Expected classes 0–3, found {classes}")

    generated_parts = []
    generated_labels = []

    for class_id in classes:
        class_seed = seed + 1000 * class_id
        set_seed(class_seed)

        train_indices = np.flatnonzero(
            train_labels == class_id
        ).tolist()

        valid_indices = np.flatnonzero(
            valid_labels == class_id
        ).tolist()

        class_train = torch.utils.data.Subset(
            train_data,
            train_indices,
        )

        class_valid = torch.utils.data.Subset(
            valid_data,
            valid_indices,
        )

        model_config = cm.get_config_hierarchical_vEEGNet(
            22,
            1000,
            0,
            0,
        )

        model_config["input_size"] = torch.Size(
            [1, 1, 22, 1000]
        )

        train_config = ct.get_config_vEEGNet_training()
        train_config["epochs"] = args.epochs
        train_config["batch_size"] = args.batch_size
        train_config["wandb_training"] = False
        train_config["print_var"] = True
        train_config["device"] = device
        train_config["measure_metrics_during_training"] = (
            model_config["use_classifier"]
        )
        train_config["use_classifier"] = (
            model_config["use_classifier"]
        )

        model_dir = (
            Path(args.out_dir)
            / "models"
            / f"S{subject:02d}_seed{seed}"
            / f"class_{class_id}"
        )

        model_dir.mkdir(parents=True, exist_ok=True)
        train_config["path_to_save_model"] = str(model_dir)
        train_config["model_artifact_name"] = (
            f"S{subject:02d}_class{class_id}"
        )
        train_config["notes"] = "Class-specific VAE generation"

        train_loader = torch.utils.data.DataLoader(
            class_train,
            batch_size=args.batch_size,
            shuffle=True,
        )

        valid_loader = torch.utils.data.DataLoader(
            class_valid,
            batch_size=args.batch_size,
            shuffle=False,
        )

        model = train_generic.get_untrained_model(
            "hvEEGNet_shallow",
            model_config,
        ).to(device)

        loss = train_generic.get_loss_function(
            "hvEEGNet_shallow",
            train_config,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config["lr"],
            weight_decay=train_config[
                "optimizer_weight_decay"
            ],
        )

        scheduler = None

        if train_config["use_scheduler"]:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=train_config["lr_decay_rate"],
            )

        print(
            f"\nSubject {subject}, class {class_id}: "
            f"{len(class_train)} training trials",
            flush=True,
        )

        train_generic.train(
            model,
            loss,
            optimizer,
            [train_loader, valid_loader],
            train_config,
            scheduler,
            model_artifact=None,
        )

        best_path = model_dir / "model_BEST.pth"
        model.load_state_dict(
            torch.load(best_path, map_location=device)
        )
        model.eval()

        number_to_generate = len(class_train)
        latent_shape = list(model.h_vae.hidden_space_shape)
        latent_shape[0] = number_to_generate

        set_seed(args.generation_seed + subject * 100 + class_id)

        with torch.no_grad():
            latent = torch.randn(
                latent_shape,
                device=device,
            )
            generated = to_3d(model.generate(latent))

        if generated.shape != (
            number_to_generate,
            22,
            1000,
        ):
            raise RuntimeError(
                f"Class {class_id} produced {generated.shape}"
            )

        generated_parts.append(generated)
        generated_labels.append(
            np.full(
                number_to_generate,
                class_id,
                dtype=np.int64,
            )
        )

        del model
        torch.cuda.empty_cache()

    X_generated = np.concatenate(generated_parts)
    y_generated = np.concatenate(generated_labels)

    rng = np.random.RandomState(args.generation_seed)
    order = rng.permutation(len(X_generated))

    X_generated = X_generated[order]
    y_generated = y_generated[order]

    if not np.isfinite(X_generated).all():
        raise RuntimeError("Generated data contain NaN or infinity.")

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir
        / f"S{subject:02d}_seed{seed}_vae_recon.npz"
    )

    np.savez_compressed(
        output_path,
        X_recon=X_generated,
        y=y_generated,
        subject_id=subject,
        seed=seed,
        source="class_specific_VAE_generation",
    )

    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    n_train_valid = len(train_data) + len(valid_data)
    train_idx, valid_idx = sf.get_idx_to_split_data(
        n_train_valid,
        dataset_config["percentage_split_train_validation"],
        dataset_config["seed_split"],
    )

    np.savez(
        split_dir
        / f"S{subject:02d}_seed{seed}_fixed_split.npz",
        train_idx=train_idx,
        validation_idx=valid_idx,
        subject_id=subject,
        seed=seed,
    )

    print("\nGeneration complete")
    print(f"Saved: {output_path}")
    print(f"Shape: {X_generated.shape}")
    print(
        "Labels:",
        np.unique(y_generated, return_counts=True),
    )
    print(
        f"Mean/std: "
        f"{X_generated.mean():.4f} / "
        f"{X_generated.std():.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="external/vae_repo")
    parser.add_argument(
        "--out-dir",
        default="outputs/generated/class_specific_vae",
    )
    parser.add_argument("--subjects", default="1")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=10000,
    )
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    for subject in parse_int_list(args.subjects):
        for seed in parse_int_list(args.seeds):
            run_subject(args, subject, seed)


if __name__ == "__main__":
    main()