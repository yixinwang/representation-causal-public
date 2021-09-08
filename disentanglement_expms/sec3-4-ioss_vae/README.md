# Empirical Studies of IOSS

This directory contains the implementation for the empirical studies
of IOSS (Section 3.4) of the paper.

To download the disentanglement datasets, install
`disentanglement_lib` and run `dlib_download_data`.

To run the study that measures disentanglement with IOSS, run the
`src/run_disentanglement_measure.sh` file, which executes
`src/disentangle_measure.py`.

To run the study that learns disentangled representation with
VAE+IOSS, run the `src/run_disentanglement_learn.sh` file, which
executes `src/disentangle_learn.py`.

The `res/` folder contains the aggregated output files that reproduce
the figures in the paper.

The code extends `disentanglement_lib`
(https://github.com/google-research/disentanglement_lib) and the
`Disent` module (https://github.com/nmichlo/disent).