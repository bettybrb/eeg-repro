import logging
import sys
import os.path
from collections import OrderedDict
import numpy as np
# compatibility shim for old mne/braindecode code
if not hasattr(np, "str"):
    np.str = str
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "complex"):
    np.complex = complex
    
import mne 
from braindecode.datautil.signalproc import highpass_cnt
import torch.nn.functional as F
import torch as th
from torch import optim
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.experiments.experiment import Experiment
from braindecode.torch_ext.util import np_to_var
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor

from braindecode.datautil.splitters import split_into_two_sets
from braindecode.datautil.trial_segment import \
    create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.datautil.signal_target import SignalAndTarget 

log = logging.getLogger(__name__)
log.setLevel('DEBUG')


def load_bcic_iv_2a_data(filename, low_cut_hz, debug=False):
    log.info("Loading data...")
    cnt = mne.io.read_raw_gdf(filename, preload=True)
    cnt = cnt.pick_types(eeg=True, eog=False)

    if debug:
        cnt.pick_channels(cnt.ch_names[:3])

    events, event_id = mne.events_from_annotations(cnt)
    mi_event_codes = [event_id['769'], event_id['770'], event_id['771'], event_id['772']]
    mi_map = {
        event_id['769']: 0,  # Left Hand
        event_id['770']: 1,  # Right Hand
        event_id['771']: 2,  # Feet
        event_id['772']: 3,  # Tongue
    }

    # keep only MI cue events
    events = events[np.isin(events[:, 2], mi_event_codes)]

    # cleaning window: 0..4 s after cue
    log.info("Cutting trials...")
    epochs_clean = mne.Epochs(
        cnt, events, tmin=0.0, tmax=4.0, baseline=None,
        preload=True, verbose=False
    )
    X_clean = epochs_clean.get_data() * 1e6  # V -> µV
    clean_trial_mask = np.max(np.abs(X_clean), axis=(1, 2)) < 800

    log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(clean_trial_mask),
        len(X_clean),
        np.mean(clean_trial_mask) * 100))

    log.info("Resampling...")
    old_sfreq = cnt.info['sfreq']
    cnt = resample_cnt(cnt, 250.0)

    # adjust event sample indices if sampling rate changed
    if cnt.info['sfreq'] != old_sfreq:
        events = events.copy()
        events[:, 0] = np.round(events[:, 0] * cnt.info['sfreq'] / old_sfreq).astype(int)

    log.info("Highpassing...")
    cnt = mne_apply(
        lambda a: highpass_cnt(a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)

    log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(
            a.T, factor_new=1e-3, init_block_size=1000, eps=1e-4
        ).T,
        cnt)

    # training window: -0.5..4.0 s
    epochs = mne.Epochs(
        cnt, events, tmin=-0.5, tmax=4.0, baseline=None,
        preload=True, verbose=False
    )

    X = epochs.get_data().astype(np.float32)
    y = np.array([mi_map[e] for e in epochs.events[:, 2]], dtype=np.int64)

    X = X[clean_trial_mask]
    y = y[clean_trial_mask]

    log.info("Trial per class:\n%s", __import__("collections").Counter(y))
    return SignalAndTarget(X, y)
    

def load_train_valid_test(
        train_filename, test_filename, low_cut_hz, debug=False):
    log.info("Loading train...")
    full_train_set = load_bcic_iv_2a_data(
        train_filename, low_cut_hz=low_cut_hz, debug=debug)


    log.info("Loading test...")
    test_set = load_bcic_iv_2a_data(
        test_filename, low_cut_hz=low_cut_hz, debug=debug)
    valid_set_fraction = 0.8
    train_set, valid_set = split_into_two_sets(full_train_set,
                                               valid_set_fraction)

    log.info("Train set with {:4d} trials".format(len(train_set.X)))
    if valid_set is not None:
        log.info("Valid set with {:4d} trials".format(len(valid_set.X)))
    log.info("Test set with  {:4d} trials".format(len(test_set.X)))

    return train_set, valid_set, test_set


def run_exp_on_high_gamma_dataset(train_filename, test_filename,
                  low_cut_hz, model_name,
                  max_epochs, max_increase_epochs,
                  np_th_seed,
                  debug):
    input_time_length = 1000
    batch_size = 60
    lr = 1e-3
    weight_decay = 0
    train_set, valid_set, test_set = load_train_valid_test(
        train_filename=train_filename,
        test_filename=test_filename,
        low_cut_hz=low_cut_hz, debug=debug)
    if debug:
        max_epochs = 4

    set_random_seeds(np_th_seed, cuda=True)
    #torch.backends.cudnn.benchmark = True# sometimes crashes?
    n_classes = int(np.max(train_set.y) + 1)
    n_chans = int(train_set.X.shape[1])
    if model_name == 'deep':
        model = Deep4Net(n_chans, n_classes,
                         input_time_length=input_time_length,
                         final_conv_length=2).create_network()
    elif model_name == 'shallow':
        model = ShallowFBCSPNet(
            n_chans, n_classes, input_time_length=input_time_length,
            final_conv_length=30).create_network()

    to_dense_prediction_model(model)
    model.cuda()
    model.eval()

    out = model(np_to_var(train_set.X[:1, :, :input_time_length, None]).cuda())

    n_preds_per_input = out.cpu().data.numpy().shape[2]
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay,
                           lr=lr)

    iterator = CropsFromTrialsIterator(batch_size=batch_size,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input,
                                       seed=np_th_seed)

    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedTrialMisclassMonitor(
                    input_time_length=input_time_length), RuntimeMonitor()]

    model_constraint = MaxNormDefaultConstraint()

    loss_function = lambda preds, targets: F.nll_loss(th.mean(preds, dim=2),
                                                      targets)

    run_after_early_stop = True
    do_early_stop = True
    remember_best_column = 'valid_misclass'
    stop_criterion = Or([MaxEpochs(max_epochs),
                         NoDecrease('valid_misclass', max_increase_epochs)])

    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                     loss_function=loss_function, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column=remember_best_column,
                     run_after_early_stop=run_after_early_stop, cuda=True,
                     do_early_stop=do_early_stop)
    exp.run()
    return exp


#if __name__ == '__main__':
    #logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     #level=logging.DEBUG, stream=sys.stdout)
    #subject_id = 1
    # have to change the data_folder here to make it run.
    #data_folder = '/home/jovyan/eeg-repro/high-gamma-data/data'
    #train_filename =  os.path.join(
        #data_folder, 'train/{:d}.mat'.format(subject_id))
    #test_filename =  os.path.join(
        #data_folder, 'test/{:d}.mat'.format(subject_id))
    #max_epochs = 800
    #max_increase_epochs = 80
    #model_name = 'deep' # or shallow
    #low_cut_hz = 0 # or 4
    #np_th_seed = 0 # random seed for numpy and pytorch
    #debug = False
    #exp = run_exp_on_high_gamma_dataset(train_filename, test_filename,
                                  #low_cut_hz, model_name,
                                  #max_epochs, max_increase_epochs,
                                  #np_th_seed,
                                  #debug)
    #log.info("Last 10 epochs")
    #log.info("\n" + str(exp.epochs_df.iloc[-10:]))

if __name__ == '__main__':
    import csv

    logging.basicConfig(
        format='%(asctime)s %(levelname)s : %(message)s',
        level=logging.INFO,
        stream=sys.stdout
    )

    data_folder = '/home/jovyan/projects/eegnet-replication/data/raw_data'
    max_epochs = 800
    max_increase_epochs = 80
    model_name = 'shallow'
    low_cut_hz = 4
    debug = False
    results_file = 'shallow_2a_4hz_results.csv'

    completed = set()
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((int(row['subject_id']), int(row['seed'])))

    all_results = []

    for subject_id in range(1, 10):
        for np_th_seed in [0, 1, 2]:
            if (subject_id, np_th_seed) in completed:
                log.info(f"Skipping subject={subject_id}, seed={np_th_seed} (already done)")
                continue

            log.info(f"Starting subject={subject_id}, seed={np_th_seed}")

            train_filename = os.path.join(data_folder, f'A0{subject_id}T.gdf')
            test_filename = os.path.join(data_folder, f'A0{subject_id}E.gdf')

            exp = run_exp_on_high_gamma_dataset(
                train_filename, test_filename,
                low_cut_hz, model_name,
                max_epochs, max_increase_epochs,
                np_th_seed,
                debug
            )

            best_epoch = exp.epochs_df['valid_misclass'].astype(float).idxmin()
            best_row = exp.epochs_df.iloc[best_epoch]
            last_row = exp.epochs_df.iloc[-1]

            with open(results_file, 'a') as f:
                if f.tell() == 0:
                    f.write('subject_id,seed,best_epoch,best_valid_misclass,best_test_misclass,last_test_misclass\n')
                f.write(
                    f"{subject_id},{np_th_seed},{int(best_epoch)},"
                    f"{float(best_row['valid_misclass'])},"
                    f"{float(best_row['test_misclass'])},"
                    f"{float(last_row['test_misclass'])}\n"
                )

            log.info(f"Finished subject={subject_id}, seed={np_th_seed}")
    