Reproducibility

Environments

Both environments use Python 3.11.7.

Classifier environment

Used for Braindecode classification.

python3.11 -m venv env
env/bin/pip install -r classifier-requirements-lock.txt
env/bin/pip install -e ./braindecode-legacy

VAE environment

Used for VAE training, reconstruction, and generation.

python3.11 -m venv vae-env
vae-env/bin/pip install -r vae-requirements-lock.txt

External repositories

Braindecode

Repository: TNTLFreiburg/braindecode

Commit: d9feb5c6cfcd203fa8daa79ccd3217712714f330

Local directory: braindecode-legacy/

The project imports Braindecode directly from this local directory.

Hierarchical VAE

Repository: jesus-333/Variational-Autoencoder-for-EEG-analysis

Branch: hvEEGNet_paper

Commit: 010426ea09f4151adc91ee7fcf3e81a3280c51bf

Local directory: high-gamma-data/external/vae_repo/

The VAE scripts import the repository's library modules directly.

Modified external files

VAE repository

library/dataset/download.pyUpdated the MOABB dataset name from BNCI2014001 to BNCI2014_001.

library/model/hierarchical_VAE.pyCorrected the hierarchy-level activity check and added latent-shape validation.

library/training/soft_dtw_cuda.pyReplaced nested min and max operations for Numba/CUDA compatibility.

Braindecode repository

braindecode/datasets/bcic_iv_2a.pyA local modification was detected. Its exact difference still needs to be documented.

Running the project

Classifier experiments:

source env/bin/activate
cd high-gamma-data
python -m experiments.run

VAE experiments:

source vae-env/bin/activate
cd high-gamma-data
python -m experiments.vae_make --help

Raw data, generated outputs, model checkpoints, and logs are not stored in Git.
