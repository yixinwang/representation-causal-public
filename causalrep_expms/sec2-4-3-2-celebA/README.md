# Empirical studies of CAUSAL-REP

This folder contains the implementation for Sections 2.4.3 of the
paper, running the supervised CAUSAL-REP on the CelebA dataset.

To run this study, first download the CelebA dataset by running
`src/run_download_celeba.sh` which executes `download_celeba.py`.

Then run `run_celeba.sh` which executes `celeba_supervised_expm.py`.

The `res/` folder contains the aggregated output and the corresponding
table.