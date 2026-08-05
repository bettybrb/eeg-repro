import argparse
import traceback
from dataclasses import replace

from pipeline.braindecode_setup import (
    apply_compatibility_patches,
    setup_logging,
)
from pipeline.classifier import (
    run_classifier,
)
from pipeline.config import CONFIG
from pipeline.data import (
    load_train_valid_test,
)
from pipeline.generators import (
    prepare_training_data,
)
from pipeline.results import (
    load_completed,
    reset_results,
    save_failure,
    save_success,
)


def parse_int_list(value):
    return [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def parse_string_list(value):
    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def generator_seeds_for_method(
    method,
    generator_seeds,
):
    if method == "baseline":
        return [None]

    return generator_seeds


def run_all_experiments(
    config,
    methods,
    subjects,
    classifier_seeds,
    generator_seeds,
    overwrite_gaussian,
):
    apply_compatibility_patches()
    setup_logging()

    unknown_methods = set(
        methods
    ).difference(
        config.experiment_names
    )

    if unknown_methods:
        raise ValueError(
            f"Unknown methods: "
            f"{sorted(unknown_methods)}\n"
            f"Allowed methods: "
            f"{sorted(config.experiment_names)}"
        )

    for method in methods:
        completed = load_completed(
            config,
            method,
        )

        print(
            "\n" + "=" * 80,
            flush=True,
        )
        print(
            f"METHOD: {method}",
            flush=True,
        )
        print(
            "RAW RESULT TABLE: "
            f"{config.raw_results_csv}",
            flush=True,
        )
        print(
            "=" * 80,
            flush=True,
        )

        successful_runs = 0
        skipped_runs = 0
        failed_runs = 0

        method_generator_seeds = (
            generator_seeds_for_method(
                method,
                generator_seeds,
            )
        )

        for subject_id in subjects:
            try:
                (
                    real_train_set,
                    valid_set,
                    test_set,
                    split_file,
                ) = load_train_valid_test(
                    subject_id=subject_id,
                    config=config,
                )

            except Exception as error:
                print(
                    "\nFULL TRACEBACK:",
                    flush=True,
                )
                traceback.print_exc()

                for generator_seed in (
                    method_generator_seeds
                ):
                    for classifier_seed in (
                        classifier_seeds
                    ):
                        save_failure(
                            config=config,
                            method=method,
                            subject_id=subject_id,
                            classifier_seed=(
                                classifier_seed
                            ),
                            generator_seed=(
                                generator_seed
                            ),
                            error=error,
                        )
                        failed_runs += 1

                continue

            for generator_seed in (
                method_generator_seeds
            ):
                try:
                    prepared_training = (
                        prepare_training_data(
                            real_train_set=(
                                real_train_set
                            ),
                            method=method,
                            subject_id=(
                                subject_id
                            ),
                            generator_seed=(
                                generator_seed
                            ),
                            split_file=(
                                split_file
                            ),
                            config=config,
                            overwrite_gaussian=(
                                overwrite_gaussian
                            ),
                        )
                    )

                except Exception as error:
                    print(
                        "\nFULL TRACEBACK:",
                        flush=True,
                    )
                    traceback.print_exc()

                    for classifier_seed in (
                        classifier_seeds
                    ):
                        save_failure(
                            config=config,
                            method=method,
                            subject_id=subject_id,
                            classifier_seed=(
                                classifier_seed
                            ),
                            generator_seed=(
                                generator_seed
                            ),
                            error=error,
                            split_file=(
                                split_file
                            ),
                        )
                        failed_runs += 1

                    continue

                for classifier_seed in (
                    classifier_seeds
                ):
                    run_key = (
                        subject_id,
                        classifier_seed,
                        generator_seed,
                    )

                    if run_key in completed:
                        skipped_runs += 1

                        print(
                            f"SKIP method={method}, "
                            f"subject={subject_id}, "
                            f"generator_seed="
                            f"{generator_seed}, "
                            f"classifier_seed="
                            f"{classifier_seed}",
                            flush=True,
                        )
                        continue

                    print(
                        f"START method={method}, "
                        f"subject={subject_id}, "
                        f"generator_seed="
                        f"{generator_seed}, "
                        f"classifier_seed="
                        f"{classifier_seed}",
                        flush=True,
                    )

                    try:
                        row = run_classifier(
                            train_set=(
                                prepared_training
                                .dataset
                            ),
                            valid_set=valid_set,
                            test_set=test_set,
                            method=method,
                            subject_id=(
                                subject_id
                            ),
                            classifier_seed=(
                                classifier_seed
                            ),
                            generator_seed=(
                                generator_seed
                            ),
                            prepared_training=(
                                prepared_training
                            ),
                            split_file=(
                                split_file
                            ),
                            config=config,
                        )

                        save_success(
                            config,
                            row,
                        )
                        successful_runs += 1

                        print(
                            f"DONE method={method}, "
                            f"subject={subject_id}, "
                            f"generator_seed="
                            f"{generator_seed}, "
                            f"classifier_seed="
                            f"{classifier_seed} | "
                            f"selected_epoch="
                            f"{row['selected_epoch']} | "
                            f"validation_misclass="
                            f"{row['validation_misclass']:.4f} | "
                            f"test_misclass="
                            f"{row['test_misclass']:.4f}",
                            flush=True,
                        )

                    except Exception as error:
                        print(
                            "\nFULL TRACEBACK:",
                            flush=True,
                        )
                        traceback.print_exc()

                        save_failure(
                            config=config,
                            method=method,
                            subject_id=(
                                subject_id
                            ),
                            classifier_seed=(
                                classifier_seed
                            ),
                            generator_seed=(
                                generator_seed
                            ),
                            error=error,
                            split_file=(
                                split_file
                            ),
                            notes=(
                                prepared_training
                                .notes
                            ),
                        )
                        failed_runs += 1

                        print(
                            f"FAILED method={method}, "
                            f"subject={subject_id}, "
                            f"generator_seed="
                            f"{generator_seed}, "
                            f"classifier_seed="
                            f"{classifier_seed} | "
                            f"{error!r}",
                            flush=True,
                        )

        print(
            f"METHOD SUMMARY {method}: "
            f"successful={successful_runs}, "
            f"skipped={skipped_runs}, "
            f"failed={failed_runs}",
            flush=True,
        )

    print(
        f"All raw results: "
        f"{config.raw_results_csv}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--methods",
        default=",".join(
            CONFIG.experiment_names
        ),
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
        "--classifier-seeds",
        default=",".join(
            str(seed)
            for seed
            in CONFIG.classifier_seeds
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
        "--overwrite-gaussian",
        action="store_true",
    )
    parser.add_argument(
        "--fresh-results",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )

    args = parser.parse_args()

    config = replace(
        CONFIG,
        debug=args.debug,
    )

    if args.fresh_results:
        reset_results(
            config
        )

    run_all_experiments(
        config=config,
        methods=parse_string_list(
            args.methods
        ),
        subjects=parse_int_list(
            args.subjects
        ),
        classifier_seeds=(
            parse_int_list(
                args.classifier_seeds
            )
        ),
        generator_seeds=(
            parse_int_list(
                args.generator_seeds
            )
        ),
        overwrite_gaussian=(
            args.overwrite_gaussian
        ),
    )


if __name__ == "__main__":
    main()
