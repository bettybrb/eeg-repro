import numpy as np
import torch as th
import torch.nn.functional as F
from torch import optim

from pipeline.braindecode_setup import apply_compatibility_patches

apply_compatibility_patches()

from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import (
    LossMonitor,
    MisclassMonitor,
    RuntimeMonitor,
    CroppedTrialMisclassMonitor,
)
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import np_to_var, set_random_seeds


def run_classifier(
    train_set,
    valid_set,
    test_set,
    experiment_type,
    subject_id,
    seed,
    train_data,
    split_mode,
    notes,
    config,
):
    max_epochs = config.max_epochs
    if config.debug:
        max_epochs = 4

    use_cuda = bool(config.use_cuda and th.cuda.is_available())

    set_random_seeds(seed, cuda=use_cuda)

    n_classes = int(np.max(train_set.y) + 1)
    n_chans = int(train_set.X.shape[1])

    if config.model_name == "deep":
        model = Deep4Net(
            n_chans,
            n_classes,
            input_time_length=config.input_time_length,
            final_conv_length=2,
        ).create_network()

    elif config.model_name == "shallow":
        model = ShallowFBCSPNet(
            n_chans,
            n_classes,
            input_time_length=config.input_time_length,
            final_conv_length=30,
        ).create_network()

    else:
        raise ValueError("model_name must be 'deep' or 'shallow'")

    to_dense_prediction_model(model)

    if use_cuda:
        model.cuda()

    model.eval()

    one_trial = np_to_var(train_set.X[:1, :, :config.input_time_length, None])
    if use_cuda:
        one_trial = one_trial.cuda()

    out = model(one_trial)
    n_preds_per_input = out.cpu().data.numpy().shape[2]

    optimizer = optim.Adam(
        model.parameters(),
        weight_decay=config.weight_decay,
        lr=config.learning_rate,
    )

    iterator = CropsFromTrialsIterator(
        batch_size=config.batch_size,
        input_time_length=config.input_time_length,
        n_preds_per_input=n_preds_per_input,
        seed=seed,
    )

    monitors = [
        LossMonitor(),
        MisclassMonitor(col_suffix="sample_misclass"),
        CroppedTrialMisclassMonitor(input_time_length=config.input_time_length),
        RuntimeMonitor(),
    ]

    loss_function = lambda preds, targets: F.nll_loss(
        th.mean(preds, dim=2),
        targets,
    )

    stop_criterion = Or([
        MaxEpochs(max_epochs),
        NoDecrease("valid_misclass", config.max_increase_epochs),
    ])

    exp = Experiment(
        model,
        train_set,
        valid_set,
        test_set,
        iterator=iterator,
        loss_function=loss_function,
        optimizer=optimizer,
        model_constraint=MaxNormDefaultConstraint(),
        monitors=monitors,
        stop_criterion=stop_criterion,
        remember_best_column="valid_misclass",
        run_after_early_stop=True,
        cuda=use_cuda,
        do_early_stop=True,
    )

    exp.run()

    best_epoch = exp.epochs_df["valid_misclass"].astype(float).idxmin()
    best_row = exp.epochs_df.iloc[best_epoch]
    last_row = exp.epochs_df.iloc[-1]

    best_valid_misclass = float(best_row["valid_misclass"])
    best_test_misclass = float(best_row["test_misclass"])
    last_test_misclass = float(last_row["test_misclass"])

    return {
        "status": "success",
        "experiment_type": experiment_type,
        "subject_id": subject_id,
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_valid_misclass": best_valid_misclass,
        "best_valid_accuracy": 1.0 - best_valid_misclass,
        "best_test_misclass": best_test_misclass,
        "best_test_accuracy": 1.0 - best_test_misclass,
        "last_test_misclass": last_test_misclass,
        "last_test_accuracy": 1.0 - last_test_misclass,
        "n_train_trials": int(len(train_set.X)),
        "n_valid_trials": int(len(valid_set.X)),
        "n_test_trials": int(len(test_set.X)),
        "n_channels": int(train_set.X.shape[1]),
        "n_times": int(train_set.X.shape[2]),
        "train_data": train_data,
        "valid_data": "real",
        "test_data": "real",
        "split_mode": split_mode,
        "notes": notes,
        "error": "",
    }
