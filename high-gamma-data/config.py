from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfiguration:
    dataset_directory: Path = Path("data/raw_data")

    subject_numbers: tuple = tuple(range(1, 10))

    # Same classifier seeds as your previous baseline/Gaussian full runs.
    random_seeds: tuple = (0, 1, 2)

    # VAE classification only.
    experiment_names: tuple = (
        "vae_recon_train_only",
    )

    model_name: str = "shallow"
    low_cut_hz: float = 4.0

    input_time_length: int = 1000
    batch_size: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    max_epochs: int = 120
    max_increase_epochs: int = 30

    debug: bool = False
    use_cuda: bool = True

    results_csv: Path = Path("results/vae_only_classifier_results.csv")


CONFIG = ExperimentConfiguration()
