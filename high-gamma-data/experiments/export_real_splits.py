import argparse
import os
import sys
from pathlib import Path

import numpy as np

from pipeline.config import CONFIG
from pipeline.splits import load_real_split


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def add_repository_to_path(repository):
    repository = Path(repository).resolve()

    if not repository.exists():
        raise FileNotFoundError(f"External VAE repository not found: {repository}")

    sys.path.insert(0, str(repository))


def dataset_to_numpy(dataset):
    if not hasattr(dataset, "data") or not hasattr(dataset, "labels"):
        raise RuntimeError("Expected external EEG_Dataset with data and labels")

    X = dataset.data.detach().cpu().numpy()
    y = dataset.labels.detach().cpu().numpy()

    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0, :, :]

    if X.ndim != 3:
        raise RuntimeError(f"Unexpected external dataset shape: {X.shape}")

    return X.astype(np.float32), y.astype(np.int64)


def export_subject(subject_id, config, overwrite):
    # Store all MOABB/MNE downloads under the project's organised raw-data folder.
    config.raw_data_directory.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(config.raw_data_directory.resolve())

    split_file = config.real_split_directory / f"S{subject_id:02d}_real_splits.npz"

    if split_file.exists() and not overwrite:
        print(f"SKIP existing split: {split_file}", flush=True)
        return

    from library.config import config_dataset as cd
    from library.dataset import preprocess as pp

    dataset_config = cd.get_moabb_dataset_config([subject_id])
    dataset_config["percentage_split_train_test"] = -1
    dataset_config["percentage_split_train_validation"] = config.training_fraction
    dataset_config["seed_split"] = config.split_seed

    print(f"Creating central split for subject {subject_id}", flush=True)

    train_dataset, valid_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

    if valid_dataset is None:
        raise RuntimeError("External preprocessing returned no validation set")

    X_train, y_train = dataset_to_numpy(train_dataset)
    X_valid, y_valid = dataset_to_numpy(valid_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)

    split_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        split_file,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        protocol_id=config.protocol_id,
        subject_id=subject_id,
        split_seed=config.split_seed,
        training_fraction=config.training_fraction,
        source_repository_commit="010426ea09f4151adc91ee7fcf3e81a3280c51bf",
        source=(
            "external VAE repository Dataset 2a preprocessing; "
            "original test session preserved; original training session divided 90/10"
        ),
    )

    validated = load_real_split(subject_id, config)

    print(f"SAVED {validated.split_file}", flush=True)
    print(f"  train={validated.X_train.shape}", flush=True)
    print(f"  valid={validated.X_valid.shape}", flush=True)
    print(f"  test={validated.X_test.shape}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(CONFIG.external_vae_repository))
    parser.add_argument(
        "--subjects",
        default=",".join(str(subject) for subject in CONFIG.subject_numbers),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    add_repository_to_path(args.repo)

    subjects = parse_int_list(args.subjects)

    for subject_id in subjects:
        export_subject(subject_id=subject_id, config=CONFIG, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
