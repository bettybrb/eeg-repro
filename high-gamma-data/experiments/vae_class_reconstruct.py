import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.vae_make import parse_int_list, set_seed


def add_repo(path):
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    sys.path.insert(0, str(path))


def get_labels(dataset):
    return np.array([
        int(y.item() if torch.is_tensor(y) else y)
        for _, y in dataset
    ])


def to_3d(x):
    x = x.detach().cpu().numpy()

    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]

    if x.ndim != 3:
        raise RuntimeError(f"Unexpected shape: {x.shape}")

    return x.astype(np.float32)


def reconstruct_subject(args, subject, seed):
    add_repo(args.repo)

    from library.config import config_dataset as cd
    from library.config import config_model as cm
    from library.dataset import preprocess as pp
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

    train_data, _, _ = pp.get_dataset_d2a(dataset_config)
    labels = get_labels(train_data)

    reconstructed = np.empty(
        (len(train_data), 22, 1000),
        dtype=np.float32,
    )

    for class_id in range(4):
        set_seed(seed + 1000 * class_id)

        indices = np.flatnonzero(
            labels == class_id
        ).tolist()

        subset = torch.utils.data.Subset(
            train_data,
            indices,
        )

        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
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

        model = train_generic.get_untrained_model(
            "hvEEGNet_shallow",
            model_config,
        ).to(device)

        model_path = (
            Path(args.model_dir)
            / "models"
            / f"S{subject:02d}_seed{seed}"
            / f"class_{class_id}"
            / "model_BEST.pth"
        )

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        model.eval()

        class_parts = []

        with torch.no_grad():
            for X, _ in loader:
                X = X.to(device)
                class_parts.append(
                    to_3d(model.reconstruct(X))
                )

        class_reconstructed = np.concatenate(
            class_parts
        )

        reconstructed[indices] = class_reconstructed

        print(
            f"Subject {subject}, class {class_id}: "
            f"{class_reconstructed.shape}",
            flush=True,
        )

        del model
        torch.cuda.empty_cache()

    if reconstructed.shape != (259, 22, 1000):
        raise RuntimeError(
            f"Subject {subject}: {reconstructed.shape}"
        )

    if not np.isfinite(reconstructed).all():
        raise RuntimeError(
            f"Subject {subject}: invalid values"
        )

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir
        / f"S{subject:02d}_seed{seed}_vae_recon.npz"
    )

    np.savez_compressed(
        output_path,
        X_recon=reconstructed,
        y=labels,
        subject_id=subject,
        seed=seed,
        source="class_specific_VAE_reconstruction",
    )

    print(
        f"Saved Subject {subject}: {output_path}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        default="external/vae_repo",
    )
    parser.add_argument(
        "--model-dir",
        default="outputs/generated/class_specific_vae",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/reconstructed/class_specific_vae",
    )
    parser.add_argument(
        "--subjects",
        default="1,2,3,4,5,6,7,8,9",
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    for subject in parse_int_list(args.subjects):
        for seed in parse_int_list(args.seeds):
            reconstruct_subject(args, subject, seed)


if __name__ == "__main__":
    main()