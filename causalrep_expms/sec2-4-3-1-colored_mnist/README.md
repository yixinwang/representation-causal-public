# Empirical studies of CAUSAL-REP

This folder contains the implementation for Sections 2.4.3 and 2.4.5
of the paper, running the supervised CAUSAL-REP on the colored MNIST
and the unsupervised CAUSAL-REP on colored and shifted MNIST.

To run this study, execute `src/run_colored_mnist.sh` which executes
`src/colored_mnist_supervised_expm.py` for the supervised study and
`src/colored_mnist_unsupervised_expm.py` for the unsupervised study.

The `res/` folder contains the aggregated output and the corresponding
figures.

This implementation extends
https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/colored_mnist/main.py