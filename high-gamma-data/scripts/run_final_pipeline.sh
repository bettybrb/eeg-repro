#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="/home/jovyan/eeg-repro/high-gamma-data"
CLASSIFIER_PY="$PROJECT_ROOT/../env/bin/python"
VAE_PY="$PROJECT_ROOT/../vae-env/bin/python"

SUBJECTS="1,2,3,4,5,6,7,8,9"
GENERATOR_SEEDS="0,1,2"
CLASSIFIER_SEEDS="0,1,2"

METHODS="baseline,gaussian_unconditional,gaussian_channel,gaussian_class,gaussian_time,gaussian_channel_time,gaussian_class_time,gaussian_class_channel,gaussian_class_channel_time,vae_reconstruction,conditional_vae_generation"

cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1

# Numba must see the local CUDA toolkit before Python starts.
export CUDA_HOME="$PROJECT_ROOT/.cuda-numba"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:${LD_LIBRARY_PATH:-}"

export MNE_DATA="$PROJECT_ROOT/data/raw_data"
export MNE_DATASETS_BNCI_PATH="$PROJECT_ROOT/data/raw_data"

mkdir -p \
  "$MNE_DATA" \
  outputs/logs \
  outputs/manifests

failure_handler() {
    exit_code=$?

    echo
    echo "============================================================"
    echo "FINAL PIPELINE STOPPED"
    echo "Exit code: $exit_code"
    echo "Time: $(date --iso-8601=seconds)"
    echo "Review the final-pipeline log before restarting."
    echo "============================================================"

    exit "$exit_code"
}

trap failure_handler ERR

echo "============================================================"
echo "FINAL EXPERIMENT PIPELINE"
echo "Started: $(date --iso-8601=seconds)"
echo "Project: $PROJECT_ROOT"
echo "Subjects: $SUBJECTS"
echo "Generator seeds: $GENERATOR_SEEDS"
echo "Classifier seeds: $CLASSIFIER_SEEDS"
echo "============================================================"

echo
echo "[PREFLIGHT] Checking environments and configuration"

test -x "$CLASSIFIER_PY" || {
    echo "Missing classifier Python: $CLASSIFIER_PY"
    exit 1
}

test -x "$VAE_PY" || {
    echo "Missing VAE Python: $VAE_PY"
    exit 1
}

"$VAE_PY" - <<'PY'
import torch

from pipeline.config import CONFIG

assert CONFIG.subject_numbers == tuple(range(1, 10))
assert CONFIG.generator_seeds == (0, 1, 2)
assert CONFIG.classifier_seeds == (0, 1, 2)

assert CONFIG.hveegnet_max_epochs == 80
assert CONFIG.hveegnet_batch_size == 30

assert CONFIG.cvae_max_epochs == 100
assert CONFIG.cvae_minimum_epochs == 20
assert CONFIG.cvae_early_stopping_patience == 15
assert CONFIG.cvae_kl_warmup_epochs == 10

assert CONFIG.max_epochs == 120
assert CONFIG.max_increase_epochs == 30
assert CONFIG.debug is False

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available in the VAE environment."
    )

print("Protocol:", CONFIG.protocol_id)
print("CUDA device:", torch.cuda.get_device_name(0))
print("Configuration preflight passed.")
PY

"$CLASSIFIER_PY" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available in the classifier environment."
    )

print(
    "Classifier CUDA device:",
    torch.cuda.get_device_name(0),
)
PY

echo
echo "[STAGE 1/6] Exporting and validating central real splits"

MNE_DATA="$MNE_DATA" \
MNE_DATASETS_BNCI_PATH="$MNE_DATASETS_BNCI_PATH" \
"$VAE_PY" \
  -m experiments.export_real_splits \
  --repo external/vae_repo \
  --subjects "$SUBJECTS"

"$VAE_PY" - <<'PY'
import numpy as np

from pipeline.config import CONFIG
from pipeline.splits import load_real_split

for subject_id in CONFIG.subject_numbers:
    split = load_real_split(
        subject_id,
        CONFIG,
    )

    development_counts = np.bincount(
        np.concatenate(
            [
                split.y_train,
                split.y_valid,
            ]
        ),
        minlength=4,
    )

    test_counts = np.bincount(
        split.y_test,
        minlength=4,
    )

    assert np.array_equal(
        development_counts,
        np.array([72, 72, 72, 72]),
    )
    assert np.array_equal(
        test_counts,
        np.array([72, 72, 72, 72]),
    )

    print(
        f"PASS split S{subject_id:02d}: "
        f"{split.X_train.shape}, "
        f"{split.X_valid.shape}, "
        f"{split.X_test.shape}"
    )

print("All central splits passed.")
PY

echo
echo "[STAGE 2/6] Training final hvEEGNet reconstruction models"

"$VAE_PY" \
  -m experiments.vae_make \
  --repo external/vae_repo \
  --subjects "$SUBJECTS" \
  --generator-seeds "$GENERATOR_SEEDS" \
  --cuda

echo
echo "[STAGE 3/6] Training final conditional VAE models"

"$VAE_PY" \
  -m experiments.cvae_generate \
  --subjects "$SUBJECTS" \
  --generator-seeds "$GENERATOR_SEEDS" \
  --cuda

echo
echo "[STAGE 4/6] Validating all neural generated datasets"

"$VAE_PY" - <<'PY'
from pathlib import Path

import numpy as np

from pipeline.config import CONFIG
from pipeline.splits import load_real_split

methods = {
    "vae_reconstruction": (
        CONFIG.vae_reconstruction_directory
    ),
    "conditional_vae_generation": (
        CONFIG.conditional_vae_directory
    ),
}

validated_files = 0

for subject_id in CONFIG.subject_numbers:
    split = load_real_split(
        subject_id,
        CONFIG,
    )

    for generator_seed in CONFIG.generator_seeds:
        filename = (
            f"S{subject_id:02d}_"
            f"generator-seed{generator_seed}.npz"
        )

        for method, directory in methods.items():
            path = directory / filename

            if not path.exists():
                raise RuntimeError(
                    f"Missing {method} output: {path}"
                )

            with np.load(
                path,
                allow_pickle=False,
            ) as data:
                X = np.asarray(
                    data["X"],
                    dtype=np.float32,
                )
                y = np.asarray(
                    data["y"],
                    dtype=np.int64,
                )

            assert X.shape == (
                259,
                22,
                1000,
            )
            assert y.shape == (259,)
            assert np.array_equal(
                y,
                split.y_train,
            )
            assert np.isfinite(X).all()

            standard_deviation = float(
                X.std()
            )

            if standard_deviation <= 0:
                raise RuntimeError(
                    f"{path} is constant."
                )

            validated_files += 1

            print(
                f"PASS {method} "
                f"S{subject_id:02d} "
                f"seed={generator_seed}: "
                f"mean={X.mean():.6f}, "
                f"std={standard_deviation:.6f}"
            )

assert validated_files == 54

print(
    "All 54 neural generated datasets passed."
)
PY

echo
echo "[STAGE 5/6] Running all final classifier experiments"

# There is deliberately no --debug.
# There is deliberately no --fresh-results.
# A clean first run creates the result table, while a restarted
# run skips combinations that already completed successfully.
"$CLASSIFIER_PY" \
  -m experiments.run \
  --methods "$METHODS" \
  --subjects "$SUBJECTS" \
  --classifier-seeds "$CLASSIFIER_SEEDS" \
  --generator-seeds "$GENERATOR_SEEDS"

echo
echo "[STAGE 6/6] Creating summaries and checking completeness"

"$CLASSIFIER_PY" \
  -m experiments.summarise_results

"$CLASSIFIER_PY" - <<'PY'
from itertools import product

import pandas as pd

from pipeline.config import CONFIG

results = pd.read_csv(
    CONFIG.raw_results_csv
)

successful = results[
    results["status"] == "success"
].copy()

successful["subject_id"] = (
    successful["subject_id"].astype(int)
)
successful["classifier_seed"] = (
    successful["classifier_seed"].astype(int)
)

def normalise_generator_seed(value):
    if pd.isna(value):
        return None
    return int(value)

successful["normalised_generator_seed"] = (
    successful["generator_seed"].apply(
        normalise_generator_seed
    )
)

actual_keys = {
    (
        row.method,
        int(row.subject_id),
        row.normalised_generator_seed,
        int(row.classifier_seed),
    )
    for row in successful.itertuples()
}

expected_keys = set()

for subject_id, classifier_seed in product(
    CONFIG.subject_numbers,
    CONFIG.classifier_seeds,
):
    expected_keys.add(
        (
            "baseline",
            subject_id,
            None,
            classifier_seed,
        )
    )

synthetic_methods = [
    method
    for method in CONFIG.experiment_names
    if method != "baseline"
]

for (
    method,
    subject_id,
    generator_seed,
    classifier_seed,
) in product(
    synthetic_methods,
    CONFIG.subject_numbers,
    CONFIG.generator_seeds,
    CONFIG.classifier_seeds,
):
    expected_keys.add(
        (
            method,
            subject_id,
            generator_seed,
            classifier_seed,
        )
    )

missing = expected_keys.difference(
    actual_keys
)
unexpected = actual_keys.difference(
    expected_keys
)

if missing:
    raise RuntimeError(
        "Missing successful combinations: "
        + repr(sorted(missing))
    )

if unexpected:
    raise RuntimeError(
        "Unexpected successful combinations: "
        + repr(sorted(unexpected))
    )

if len(successful) != 837:
    raise RuntimeError(
        "Expected 837 successful result rows, "
        f"found {len(successful)}."
    )

if successful.duplicated(
    subset=[
        "protocol_id",
        "method",
        "subject_id",
        "generator_seed",
        "classifier_seed",
    ],
    keep=False,
).any():
    raise RuntimeError(
        "Duplicate successful result rows detected."
    )

if not successful[
    "test_misclass"
].between(0.0, 1.0).all():
    raise RuntimeError(
        "Invalid test misclassification value."
    )

failed_rows = results[
    results["status"] == "failed"
]

print()
print("Successful final rows:", len(successful))
print("Failure-attempt rows:", len(failed_rows))
print("Expected final rows: 837")
print("Final result completeness passed.")
PY

date --iso-8601=seconds \
  > outputs/manifests/final_pipeline_completed.txt

echo
echo "============================================================"
echo "FINAL PIPELINE COMPLETED"
echo "Completed: $(date --iso-8601=seconds)"
echo "============================================================"
