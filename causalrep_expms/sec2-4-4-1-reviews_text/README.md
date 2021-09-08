# Empirical studies of CAUSAL-REP

This folder contains the implementation for Sections 2.4.4 of the
paper, running the supervised CAUSAL-REP on the Amazon, Tripadvisor,
Yelp reviews corpura.

To run this study, first download the datasets from
http://times.cs.uiuc.edu/wang296/Data/ and
https://www.yelp.com/dataset/documentation/main

Then run `src/prep_reviews.py`, `src/prep_yelp_raw.py`,
`src/prep_yelp_csv.py` to preprocess the reviews data. The `dat/`
folder contains the preprocessed the reviews data.

Finally, run `src/run_causaltext_reviews.sh` which executes
`causaltext_reviews.py`.

The `res/` folder contains the aggregated output and the corresponding
figures and tables.