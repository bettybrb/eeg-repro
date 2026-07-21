import argparse
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_vae_repo_to_path(repo_path):
    repo_path = Path(repo_path).resolve()

    if not repo_path.exists():
        raise FileNotFoundError(f"VAE repo folder not found: {repo_path}")

    sys.path.insert(0, str(repo_path))


def reconstruct_dataset(model, dataset, batch_size, device):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    all_recon = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)

            # hvEEGNet has a reconstruct method.
            x_recon = model.reconstruct(x_batch, no_grad=True)

            all_recon.append(x_recon.detach().cpu().numpy())
            all_labels.append(y_batch.detach().cpu().numpy())

    X_recon_raw = np.concatenate(all_recon, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # Expected native shape is usually:
    # trials x 1 x channels x time
    if X_recon_raw.ndim == 4 and X_recon_raw.shape[1] == 1:
        X_recon = X_recon_raw[:, 0, :, :]
    elif X_recon_raw.ndim == 3:
        X_recon = X_recon_raw
    else:
        X_recon = np.squeeze(X_recon_raw)

        if X_recon.ndim != 3:
            raise ValueError(
                f"Could not convert reconstructed EEG to 3D. "
                f"Original shape was {X_recon_raw.shape}, squeezed shape is {X_recon.shape}"
            )

    return X_recon.astype(np.float32), y.astype(np.int64), X_recon_raw.shape


def train_and_save_one_subject(args, subject_id, seed):
    set_seed(seed)
    add_vae_repo_to_path(args.repo)

    from library.dataset import preprocess as pp
    from library.config import config_dataset as cd
    from library.config import config_training as ct
    from library.config import config_model as cm
    from library.training import train_generic

    print(f"START VAE subject={subject_id} seed={seed}", flush=True)

    dataset_config = cd.get_moabb_dataset_config([subject_id])

    train_config = ct.get_config_vEEGNet_training()
    train_config["epochs"] = args.epochs
    train_config["batch_size"] = args.batch_size
    train_config["wandb_training"] = False
    train_config["print_var"] = True
    train_config["device"] = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    train_config["path_to_save_model"] = str(
        Path(args.out_dir) / "models" / f"S{subject_id:02d}_seed{seed}"
    )
    train_config["model_artifact_name"] = f"vae_S{subject_id:02d}_seed{seed}"
    train_config["notes"] = "local VAE reconstruction for downstream classifier"

    C = 22

    if dataset_config["resample_data"]:
        sampling_freq = dataset_config["resample_freq"]
    else:
        sampling_freq = 250

    T = int((dataset_config["trial_end"] - dataset_config["trial_start"]) * sampling_freq)

    type_decoder = 0
    parameters_map_type = 0

    model_config = cm.get_config_hierarchical_vEEGNet(
        C,
        T,
        type_decoder,
        parameters_map_type,
    )

    train_config["measure_metrics_during_training"] = model_config["use_classifier"]
    train_config["use_classifier"] = model_config["use_classifier"]

    print("Loading Dataset 2a through VAE repo pipeline...", flush=True)

    # Leakage-safe fixed split settings.
    # These make the data split fixed across model seeds.
    # Model seeds may change VAE initialisation/training randomness,
    # but not which trials are train/validation.
    dataset_config['percentage_split_train_test'] = -1
    dataset_config['percentage_split_train_validation'] = 0.9
    dataset_config['seed_split'] = 42

    train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)
    # Save the exact split indices used by the VAE repo.
    # This is needed so the downstream classifier can use:
    #   train = VAE reconstructed training trials
    #   validation = real validation trials
    #   test = real official test/session trials
    from library.dataset import support_function as vae_split_sf
    import os
    import numpy as _np

    n_train_before_validation = len(train_dataset) + len(validation_dataset)
    fixed_train_idx, fixed_validation_idx = vae_split_sf.get_idx_to_split_data(
        n_train_before_validation,
        dataset_config['percentage_split_train_validation'],
        dataset_config['seed_split']
    )

    split_dir = Path(args.out_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"S{subject_id:02d}_seed{seed}_fixed_split.npz"

    _np.savez(
        split_path,
        train_idx=fixed_train_idx,
        validation_idx=fixed_validation_idx,
        subject_id=subject_id,
        seed=seed,
        split_seed=dataset_config['seed_split'],
        percentage_split_train_validation=dataset_config['percentage_split_train_validation'],
        percentage_split_train_test=dataset_config['percentage_split_train_test'],
        n_train_before_validation=n_train_before_validation,
        n_vae_train=len(train_dataset),
        n_vae_validation=len(validation_dataset),
        n_test=len(test_dataset) if test_dataset is not None else -1,
        note="VAE trained only on train_idx; validation_idx saved for downstream real validation; official test remains separate."
    )
    print(f"Saved fixed split indices: {split_path}")


    if validation_dataset is None:
        raise RuntimeError("Validation dataset is None. The VAE training code needs validation data.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
    )

    model_config["input_size"] = train_dataset[0][0].unsqueeze(0).shape

    model = train_generic.get_untrained_model(
        "hvEEGNet_shallow",
        model_config,
    )

    device = train_config["device"]
    model.to(device)

    loss_function = train_generic.get_loss_function(
        "hvEEGNet_shallow",
        train_config,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config["optimizer_weight_decay"],
    )

    if train_config["use_scheduler"]:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=train_config["lr_decay_rate"],
        )
    else:
        lr_scheduler = None

    Path(train_config["path_to_save_model"]).mkdir(parents=True, exist_ok=True)

    print("Training VAE...", flush=True)

    train_generic.train(
        model,
        loss_function,
        optimizer,
        [train_loader, validation_loader],
        train_config,
        lr_scheduler,
        model_artifact=None,
    )

    best_model_path = Path(train_config["path_to_save_model"]) / "model_BEST.pth"

    if best_model_path.exists():
        state = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded best VAE model: {best_model_path}", flush=True)
    else:
        print("WARNING: model_BEST.pth not found. Using final model state.", flush=True)

    print("Reconstructing VAE training EEG...", flush=True)

    X_recon, y, original_recon_shape = reconstruct_dataset(
        model=model,
        dataset=train_dataset,
        batch_size=train_config["batch_size"],
        device=device,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"S{subject_id:02d}_seed{seed}_vae_recon.npz"

    np.savez_compressed(
        out_path,
        X_recon=X_recon,
        y=y,
        subject_id=subject_id,
        seed=seed,
        original_recon_shape=np.asarray(original_recon_shape),
        final_shape=np.asarray(X_recon.shape),
        source="hvEEGNet_paper_native_reconstruction",
    )

    print(f"DONE VAE subject={subject_id} seed={seed}", flush=True)
    print(f"Saved: {out_path}", flush=True)
    print(f"X_recon shape: {X_recon.shape}", flush=True)
    print(f"y shape: {y.shape}", flush=True)
    print(f"labels: {np.unique(y, return_counts=True)}", flush=True)


def compare_run_splits(run1_dir, run2_dir, subjects, seeds):
    """
    Confirm that both reconstruction runs used exactly the same
    train and validation trial indices.
    """
    run1_dir = Path(run1_dir)
    run2_dir = Path(run2_dir)

    checked = 0
    missing = []

    for subject_id in subjects:
        for seed in seeds:
            filename = f"S{subject_id:02d}_seed{seed}_fixed_split.npz"

            run1_path = run1_dir / "splits" / filename
            run2_path = run2_dir / "splits" / filename

            if not run1_path.exists() or not run2_path.exists():
                missing.append(filename)
                continue

            with np.load(run1_path) as run1_split, np.load(run2_path) as run2_split:
                train_matches = np.array_equal(
                    run1_split["train_idx"],
                    run2_split["train_idx"],
                )
                validation_matches = np.array_equal(
                    run1_split["validation_idx"],
                    run2_split["validation_idx"],
                )

            if not train_matches or not validation_matches:
                raise RuntimeError(
                    f"Run-1 and run-2 splits do not match for {filename}"
                )

            checked += 1

    print(
        f"Verified identical run-1/run-2 splits for {checked} "
        f"subject-seed combinations.",
        flush=True,
    )

    if missing:
        print(
            "Split comparison skipped for missing files: "
            + ", ".join(missing),
            flush=True,
        )


def run_in_fresh_process(args, experiment):
    """
    Run each repeated experiment in a fresh Python process so run 1
    and run 2 begin from equivalent process states.
    """
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--experiment",
        experiment,
        "--repo",
        args.repo,
        "--run1-dir",
        args.run1_dir,
        "--run2-dir",
        args.run2_dir,
        "--subjects",
        args.subjects,
        "--seeds",
        args.seeds,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
    ]

    if args.cuda:
        command.append("--cuda")

    print(
        f"Launching {experiment} in a fresh Python process.",
        flush=True,
    )

    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo", default="external/vae_repo")

    parser.add_argument(
        "--experiment",
        choices=["run1", "run2", "both"],
        default="run2",
        help=(
            "Choose which repeated VAE experiment to run. "
            "'both' runs run1 first and then run2."
        ),
    )

    parser.add_argument("--run1-dir", default="saved_vae_run1")
    parser.add_argument("--run2-dir", default="saved_vae_run2")

    parser.add_argument("--subjects", default="1")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    subjects = parse_int_list(args.subjects)
    seeds = parse_int_list(args.seeds)

    if args.experiment == "both":
        run_in_fresh_process(args, "run1")
        run_in_fresh_process(args, "run2")

        compare_run_splits(
            args.run1_dir,
            args.run2_dir,
            subjects,
            seeds,
        )
        return

    if args.experiment == "run1":
        args.out_dir = args.run1_dir
    else:
        args.out_dir = args.run2_dir

    print("=" * 70, flush=True)
    print(f"VAE experiment: {args.experiment}", flush=True)
    print(f"Output directory: {args.out_dir}", flush=True)
    print(f"Subjects: {subjects}", flush=True)
    print(f"Seeds: {seeds}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print("=" * 70, flush=True)

    for subject_id in subjects:
        for seed in seeds:
            train_and_save_one_subject(
                args,
                subject_id,
                seed,
            )

    if args.experiment == "run2":
        compare_run_splits(
            args.run1_dir,
            args.run2_dir,
            subjects,
            seeds,
        )


if __name__ == "__main__":
    main()
