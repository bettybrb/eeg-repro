import gc
import time

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import optim

from pipeline.braindecode_setup import (
    apply_compatibility_patches,
)

apply_compatibility_patches()

from braindecode.datautil.iterators import (
    CropsFromTrialsIterator,
)
from braindecode.experiments.experiment import (
    Experiment,
)
from braindecode.experiments.monitors import (
    CroppedTrialMisclassMonitor,
    LossMonitor,
    MisclassMonitor,
    RuntimeMonitor,
    compute_trial_labels_from_crop_preds,
)
from braindecode.experiments.stopcriteria import (
    MaxEpochs,
    NoDecrease,
    Or,
)
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import (
    ShallowFBCSPNet,
)
from braindecode.models.util import (
    to_dense_prediction_model,
)
from braindecode.torch_ext.constraints import (
    MaxNormDefaultConstraint,
)
from braindecode.torch_ext.util import (
    np_to_var,
    set_random_seeds,
)


def _evaluate_test_once(
    model,
    test_set,
    iterator,
    input_time_length,
    use_cuda,
):
    """
    Evaluate the restored validation-selected model on the
    official real test set once.
    """

    prediction_batches = []
    model.eval()

    with th.no_grad():
        for inputs, _ in iterator.get_batches(
            test_set,
            shuffle=False,
        ):
            input_variables = np_to_var(
                inputs
            )

            if use_cuda:
                input_variables = (
                    input_variables.cuda()
                )

            predictions = model(
                input_variables
            )

            prediction_batches.append(
                predictions
                .cpu()
                .detach()
                .numpy()
            )

    if not prediction_batches:
        raise RuntimeError(
            "No test predictions were produced"
        )

    all_predictions = np.concatenate(
        prediction_batches,
        axis=0,
    )

    # Legacy Braindecode monitors use one leading dimension
    # containing every crop prediction.
    all_predictions = all_predictions[
        np.newaxis,
        :,
    ]

    predicted_labels = (
        compute_trial_labels_from_crop_preds(
            all_predictions,
            input_time_length,
            test_set.X,
        )
    )

    if predicted_labels.shape != test_set.y.shape:
        raise RuntimeError(
            "Predicted labels have shape "
            f"{predicted_labels.shape}; "
            f"expected {test_set.y.shape}"
        )

    return float(
        1.0
        - np.mean(
            predicted_labels
            == test_set.y
        )
    )


def run_classifier(
    train_set,
    valid_set,
    test_set,
    method,
    subject_id,
    classifier_seed,
    generator_seed,
    prepared_training,
    split_file,
    config,
):
    start_time = time.perf_counter()

    if config.debug:
        max_epochs = 4
    else:
        max_epochs = config.max_epochs

    use_cuda = bool(
        config.use_cuda
        and th.cuda.is_available()
    )

    set_random_seeds(
        classifier_seed,
        cuda=use_cuda,
    )

    observed_classes = set(
        np.unique(train_set.y).tolist()
    )
    expected_classes = set(
        config.class_ids
    )

    if observed_classes != expected_classes:
        raise RuntimeError(
            "Classifier training labels are "
            f"{sorted(observed_classes)}; "
            f"expected {sorted(expected_classes)}"
        )

    n_classes = len(
        config.class_ids
    )
    n_channels = int(
        train_set.X.shape[1]
    )

    if config.model_name == "deep":
        model = Deep4Net(
            n_channels,
            n_classes,
            input_time_length=(
                config.input_time_length
            ),
            final_conv_length=2,
        ).create_network()

    elif config.model_name == "shallow":
        model = ShallowFBCSPNet(
            n_channels,
            n_classes,
            input_time_length=(
                config.input_time_length
            ),
            final_conv_length=30,
        ).create_network()

    else:
        raise ValueError(
            "model_name must be 'deep' or 'shallow'"
        )

    to_dense_prediction_model(
        model
    )

    if use_cuda:
        model.cuda()

    model.eval()

    one_trial = np_to_var(
        train_set.X[
            :1,
            :,
            : config.input_time_length,
            None,
        ]
    )

    if use_cuda:
        one_trial = one_trial.cuda()

    with th.no_grad():
        output = model(
            one_trial
        )

    n_predictions_per_input = int(
        output
        .cpu()
        .detach()
        .numpy()
        .shape[2]
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    iterator = CropsFromTrialsIterator(
        batch_size=config.batch_size,
        input_time_length=(
            config.input_time_length
        ),
        n_preds_per_input=(
            n_predictions_per_input
        ),
        seed=classifier_seed,
    )

    monitors = [
        LossMonitor(),
        MisclassMonitor(
            col_suffix="sample_misclass"
        ),
        CroppedTrialMisclassMonitor(
            input_time_length=(
                config.input_time_length
            )
        ),
        RuntimeMonitor(),
    ]

    def loss_function(
        predictions,
        targets,
    ):
        return F.nll_loss(
            th.mean(
                predictions,
                dim=2,
            ),
            targets,
        )

    stop_criterion = Or(
        [
            MaxEpochs(
                max_epochs
            ),
            NoDecrease(
                "valid_misclass",
                config.max_increase_epochs,
            ),
        ]
    )

    # The test set is deliberately excluded from training
    # monitoring. Braindecode restores the validation-selected
    # model after the stopping criterion is reached.
    experiment = Experiment(
        model=model,
        train_set=train_set,
        valid_set=valid_set,
        test_set=None,
        iterator=iterator,
        loss_function=loss_function,
        optimizer=optimizer,
        model_constraint=(
            MaxNormDefaultConstraint()
        ),
        monitors=monitors,
        stop_criterion=stop_criterion,
        remember_best_column=(
            "valid_misclass"
        ),
        run_after_early_stop=False,
        cuda=use_cuda,
        do_early_stop=True,
    )

    experiment.run()

    if experiment.rememberer is None:
        raise RuntimeError(
            "Validation-selected model state "
            "was not recorded"
        )

    selected_epoch = int(
        experiment.rememberer.best_epoch
    )
    validation_misclass = float(
        experiment.rememberer.lowest_val
    )

    test_misclass = (
        _evaluate_test_once(
            model=model,
            test_set=test_set,
            iterator=iterator,
            input_time_length=(
                config.input_time_length
            ),
            use_cuda=use_cuda,
        )
    )

    runtime_seconds = float(
        time.perf_counter()
        - start_time
    )

    row = {
        "status": "success",
        "protocol_id": config.protocol_id,
        "method": method,
        "subject_id": subject_id,
        "generator_seed": generator_seed,
        "classifier_seed": classifier_seed,
        "selected_epoch": selected_epoch,
        "validation_misclass": (
            validation_misclass
        ),
        "test_misclass": test_misclass,
        "n_real_train_trials": (
            prepared_training
            .n_real_train_trials
        ),
        "n_synthetic_train_trials": (
            prepared_training
            .n_synthetic_train_trials
        ),
        "n_valid_trials": int(
            len(valid_set.X)
        ),
        "n_test_trials": int(
            len(test_set.X)
        ),
        "n_channels": int(
            train_set.X.shape[1]
        ),
        "n_times": int(
            train_set.X.shape[2]
        ),
        "train_data_type": (
            prepared_training
            .train_data_type
        ),
        "split_file": str(
            split_file
        ),
        "train_data_file": (
            prepared_training
            .train_data_file
        ),
        "runtime_seconds": (
            runtime_seconds
        ),
        "notes": (
            prepared_training.notes
        ),
        "error": "",
    }

    del experiment
    del optimizer
    del model

    gc.collect()

    if use_cuda:
        th.cuda.empty_cache()

    return row
