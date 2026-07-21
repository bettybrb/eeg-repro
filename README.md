# Synthetic EEG for Motor-Imagery Classification

Work-in-progress MSc Artificial Intelligence thesis project evaluating whether synthetic EEG can support motor-imagery classification.

## Project overview

The project compares several approaches for producing training EEG:

- statistical Gaussian generation;
- hierarchical VAE reconstruction;
- fixed-size replacement of real trials with reconstructed trials;
- class-conditioned VAE generation, currently in development;
- an autoregressive model is planned as an additional comparison.

A fixed High-Gamma-style Braindecode classifier is used for downstream evaluation. Validation and testing use real EEG data.

## Dataset

The experiments use BCI Competition IV Dataset 2a with:

- 9 subjects;
- 4 motor-imagery classes;
- 22 EEG channels;
- subject-specific training and evaluation.

The original EEG data and generated arrays are not included in this repository because of their size and dataset-distribution requirements.

## Current status

This repository is actively being developed. Final experiment tables, consolidated results and documentation will be added after all methods are evaluated under a unified protocol.

## Main files

- `high-gamma-data/run.py` — experiment runner
- `high-gamma-data/classifier.py` — downstream EEG classifier
- `high-gamma-data/data.py` — EEG loading and preprocessing
- `high-gamma-data/generators.py` — Gaussian generators
- `high-gamma-data/vae_make.py` — VAE reconstruction pipeline
- `high-gamma-data/vae_ratio_classify.py` — fixed-size replacement experiments
- `high-gamma-data/vae_shape.py` — reconstruction shape handling
- `high-gamma-data/vae_evaluate.py` — VAE evaluation utilities

## Reproducibility

Environment setup, exact dependency versions and final commands will be documented after the experimental pipeline is finalised.
