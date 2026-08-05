import traceback

from pipeline.braindecode_setup import apply_compatibility_patches, setup_logging
from pipeline.classifier import run_classifier
from pipeline.config import CONFIG
from pipeline.data import load_train_valid_test
from pipeline.generators import apply_experiment_transformation
from pipeline.results import load_completed, save_failure, save_success


def run_all_experiments(config):
    apply_compatibility_patches()
    setup_logging()

    for experiment_type in config.experiment_names:
        completed = load_completed(config, experiment_type)

        print("\n" + "=" * 80, flush=True)
        print(f"Running experiment: {experiment_type}", flush=True)
        print(f"Combined results file: {config.results_csv}", flush=True)
        print("=" * 80, flush=True)

        total_runs = 0
        successful_runs = 0
        skipped_runs = 0
        failed_runs = 0

        for subject_id in config.subject_numbers:
            for seed in config.random_seeds:
                total_runs += 1

                if (subject_id, seed) in completed:
                    skipped_runs += 1
                    print(
                        f"SKIP experiment={experiment_type}, subject={subject_id}, seed={seed}",
                        flush=True,
                    )
                    continue

                print(
                    f"START experiment={experiment_type}, subject={subject_id}, seed={seed}",
                    flush=True,
                )

                try:
                    train_set, valid_set, test_set = load_train_valid_test(
                        subject_id=subject_id,
                        config=config,
                    )

                    (
                        train_set,
                        valid_set,
                        test_set,
                        train_data,
                        split_mode,
                        notes,
                    ) = apply_experiment_transformation(
                        train_set=train_set,
                        valid_set=valid_set,
                        test_set=test_set,
                        experiment_type=experiment_type,
                        subject_id=subject_id,
                        seed=seed,
                    )

                    row = run_classifier(
                        train_set=train_set,
                        valid_set=valid_set,
                        test_set=test_set,
                        experiment_type=experiment_type,
                        subject_id=subject_id,
                        seed=seed,
                        train_data=train_data,
                        split_mode=split_mode,
                        notes=notes,
                        config=config,
                    )

                    save_success(config, row)
                    successful_runs += 1

                    print(
                        f"DONE experiment={experiment_type}, subject={subject_id}, seed={seed} | "
                        f"best_epoch={row['best_epoch']} | "
                        f"best_valid_misclass={row['best_valid_misclass']:.4f} | "
                        f"best_test_misclass={row['best_test_misclass']:.4f} | "
                        f"best_test_accuracy={row['best_test_accuracy']:.4f} | "
                        f"last_test_misclass={row['last_test_misclass']:.4f}",
                        flush=True,
                    )

                except Exception as error:
                    failed_runs += 1
                    print("\nFULL TRACEBACK:", flush=True)
                    traceback.print_exc()

                    save_failure(
                        config=config,
                        experiment_type=experiment_type,
                        subject_id=subject_id,
                        seed=seed,
                        error=error,
                    )

                    print(
                        f"FAILED experiment={experiment_type}, subject={subject_id}, seed={seed} | {repr(error)}",
                        flush=True,
                    )

        print(
            f"SUMMARY experiment={experiment_type}, total={total_runs}, "
            f"done={successful_runs}, skipped={skipped_runs}, failed={failed_runs}",
            flush=True,
        )

    print(f"SUMMARY all results saved to: {config.results_csv}", flush=True)


if __name__ == "__main__":
    run_all_experiments(CONFIG)
