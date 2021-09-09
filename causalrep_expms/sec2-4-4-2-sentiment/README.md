# Empirical studies of CAUSAL-REP

This folder contains the implementation for Sections 2.4.4 of the
paper, running the supervised CAUSAL-REP on the IMDB-L, IMDB-S, and
Kindle.

To run this study, first download the datasets from
https://github.com/tapilab/aaai-2021-counterfactuals/tree/main/data

Then run `src/run_preproc_causaltext.sh` to preprocess the datasets,
which executes `preproc_text.py`. The `dat/` folder contains the
prepocessed files.

Finally, run `run_causal_text.sh` which executes `causaltext.py`. The
study is quite sensitive to initialization. The `res/` folder contains
the the parameter configurations and output files that produces the
result.

The preprocessed text data for Section 2.4.4 is also available at
https://www.dropbox.com/sh/sd7ezkgwl110afn/AACVEXEhj8JXohRi119ZhHb_a?dl=0Reference