from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentConfiguration:
    protocol_id: str = "vae_repo_90_10_seed42"

    project_root: Path = PROJECT_ROOT

    raw_data_directory: Path = PROJECT_ROOT / "data" / "raw_data"

    processed_data_directory: Path = (
        PROJECT_ROOT / "data" / "processed" / "vae_repo_90_10_seed42"
    )
    real_split_directory: Path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "vae_repo_90_10_seed42"
        / "real_splits"
    )

    generated_data_directory: Path = (
        PROJECT_ROOT / "data" / "generated" / "vae_repo_90_10_seed42"
    )
    gaussian_data_directory: Path = (
        PROJECT_ROOT
        / "data"
        / "generated"
        / "vae_repo_90_10_seed42"
        / "gaussian"
    )
    vae_reconstruction_directory: Path = (
        PROJECT_ROOT
        / "data"
        / "generated"
        / "vae_repo_90_10_seed42"
        / "vae_reconstruction"
    )
    conditional_vae_directory: Path = (
        PROJECT_ROOT
        / "data"
        / "generated"
        / "vae_repo_90_10_seed42"
        / "conditional_vae"
    )

    output_directory: Path = PROJECT_ROOT / "outputs"
    checkpoint_directory: Path = PROJECT_ROOT / "outputs" / "checkpoints"
    log_directory: Path = PROJECT_ROOT / "outputs" / "logs"
    manifest_directory: Path = PROJECT_ROOT / "outputs" / "manifests"
    figure_directory: Path = PROJECT_ROOT / "outputs" / "figures"

    raw_results_csv: Path = (
        PROJECT_ROOT / "outputs" / "results" / "raw" / "runs.csv"
    )
    participant_summary_csv: Path = (
        PROJECT_ROOT
        / "outputs"
        / "results"
        / "summary"
        / "participant_summary.csv"
    )
    method_summary_csv: Path = (
        PROJECT_ROOT
        / "outputs"
        / "results"
        / "summary"
        / "method_summary.csv"
    )

    external_vae_repository: Path = PROJECT_ROOT / "external" / "vae_repo"

    subject_numbers: tuple[int, ...] = tuple(range(1, 10))
    classifier_seeds: tuple[int, ...] = (0, 1, 2)
    generator_seeds: tuple[int, ...] = (0,)

    experiment_names: tuple[str, ...] = (
        "baseline",
        "gaussian_global",
        "gaussian_channel",
        "gaussian_class",
        "gaussian_time",
        "gaussian_channel_time",
        "gaussian_class_time",
        "gaussian_class_channel",
        "gaussian_class_channel_time",
        "vae_reconstruction",
        "conditional_vae_generation",
    )

    split_seed: int = 42
    training_fraction: float = 0.9

    expected_train_trials: int = 259
    expected_valid_trials: int = 29
    expected_test_trials: int = 288
    expected_channels: int = 22
    expected_times: int = 1000
    class_ids: tuple[int, ...] = (0, 1, 2, 3)

    model_name: str = "shallow"
    input_time_length: int = 1000
    batch_size: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 120
    max_increase_epochs: int = 30

    debug: bool = False
    use_cuda: bool = True


CONFIG = ExperimentConfiguration()
