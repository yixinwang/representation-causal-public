import numpy as np
from sklearn.neural_network import MLPClassifier
import scipy
import sklearn
import torch
from sklearn.inspection import permutation_importance
from torch import nn

def to_numpy(array) -> np.ndarray:
    """
    Handles converting any array like object to a numpy array.
    specifically with support for a tensor
    """
    # TODO: replace... maybe with kornia
    if torch.is_tensor(array):
        return array.cpu().detach().numpy()
    # recursive conversion
    # not super efficient but allows handling of PIL.Image and other nested data.
    elif isinstance(array, (list, tuple)):
        return np.stack([to_numpy(elem) for elem in array], axis=0)
    else:
        return np.array(array)


def uniformize(train_y, eps=1e-8):
    # make the matrix s.t. all entries between 0 and 1
    train_y_true = train_y.copy()
    train_y_uq, train_y_lq = np.quantile(train_y_true,0.9), np.quantile(train_y_true,0.1)
    train_y_true[train_y_true > train_y_uq] = train_y_uq
    train_y_true[train_y_true < train_y_lq] = train_y_lq
    train_y_true = (train_y_true - train_y_true.min(axis=0)) / (train_y_true.max(axis=0) - train_y_true.min(axis=0) + eps)  
    return train_y_true


### Intervention Robustness Score

def IRS_score(gen_factors, latents, diff_quantile=0.99):
    """Computes IRS scores of a dataset.
    Assumes no noise in X and crossed generative factors (i.e. one sample per
    combination of gen_factors). Assumes each g_i is an equally probable
    realization of g_i and all g_i are independent.
    Args:
        gen_factors: Numpy array of shape (num samples, num generative factors),
            matrix of ground truth generative factors.
        latents: Numpy array of shape (num samples, num latent dimensions), matrix
            of latent variables.
        diff_quantile: Float value between 0 and 1 to decide what quantile of diffs
            to select (use 1.0 for the version in the paper).
    Returns:
        Dictionary with IRS scores.
    """
    num_gen = gen_factors.shape[1]
    num_lat = latents.shape[1]

    # Compute normalizer.
    max_deviations = np.max(np.abs(latents - latents.mean(axis=0)), axis=0)
    cum_deviations = np.zeros([num_lat, num_gen])
    for i in range(num_gen):
        print(i)
        unique_factors = np.unique(gen_factors[:, i], axis=0)
        assert unique_factors.ndim == 1
        num_distinct_factors = unique_factors.shape[0]
        for k in range(num_distinct_factors):
            # Compute E[Z | g_i].
            match = gen_factors[:, i] == unique_factors[k]
            e_loc = np.mean(latents[match, :], axis=0)

            # Difference of each value within that group of constant g_i to its mean.
            diffs = np.abs(latents[match, :] - e_loc)
            max_diffs = np.percentile(diffs, q=diff_quantile*100, axis=0)
            cum_deviations[:, i] += max_diffs
        cum_deviations[:, i] /= num_distinct_factors
    # Normalize value of each latent dimension with its maximal deviation.
    normalized_deviations = cum_deviations / max_deviations[:, np.newaxis]
    irs_matrix = 1.0 - normalized_deviations
    disentanglement_scores = irs_matrix.max(axis=1)
    if np.sum(max_deviations) > 0.0:
        avg_score = np.average(disentanglement_scores, weights=max_deviations)
    else:
        avg_score = np.mean(disentanglement_scores)

    parents = irs_matrix.argmax(axis=1)
    score_dict = {}
    score_dict["disentanglement_scores"] = disentanglement_scores
    score_dict["avg_score"] = avg_score
    score_dict["parents"] = parents
    score_dict["IRS_matrix"] = irs_matrix
    score_dict["max_deviations"] = max_deviations
    return score_dict


### DCI Disentanglement Score


def compute_importance_gbt(x_train, y_train, x_test, y_test):
    """Compute importance based on gradient boosted trees."""
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        print(i)

        # model = ensemble.GradientBoostingClassifier(n_iter_no_change=5, tol=0.1)
        model = MLPClassifier(random_state=1, max_iter=200)

        model.fit(x_train.T, y_train[i, :])

        importance_matrix[:, i] = np.abs(permutation_importance(model, x_train.T, y_train[i, :], n_repeats=10,random_state=0)['importances_mean'])
        # importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def DCI_score(mus_train, ys_train, mus_test, ys_test):
    """Computes score based on both training and testing codes and factors."""
    scores = {}
    importance_matrix, train_err, test_err = compute_importance_gbt(
      mus_train, ys_train, mus_test, ys_test)
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    return scores


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                    base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code*code_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                    base=importance_matrix.shape[0])


def completeness(importance_matrix):
    """"Compute completeness of the representation."""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor*factor_importance)



### unsupervised metrics from https://github.com/nmichlo/disent/blob/main/disent/metrics/_unsupervised.py

def gaussian_total_correlation(cov):
    """Computes the total correlation of a Gaussian with covariance matrix cov.
    We use that the total correlation is the KL divergence between the Gaussian
    and the product of its marginals. By design, the means of these two Gaussians
    are zero and the covariance matrix of the second Gaussian is equal to the
    covariance matrix of the first Gaussian with off-diagonal entries set to zero.
    Args:
      cov: Numpy array with covariance matrix.
    Returns:
      Scalar with total correlation.
    """
    return 0.5 * (np.sum(np.log(np.diag(cov))) - np.linalg.slogdet(cov)[1])


def gaussian_wasserstein_correlation(cov):
    """Wasserstein L2 distance between Gaussian and the product of its marginals.
    Args:
      cov: Numpy array with covariance matrix.
    Returns:
      Scalar with score.
    """
    sqrtm = scipy.linalg.sqrtm(cov * np.expand_dims(np.diag(cov), axis=1))
    return 2 * np.trace(cov) - 2 * np.trace(sqrtm)




def histogram_discretize(target, num_bins=20):
    """
    Discretization based on histograms.
    """
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])
    return discretized


def discrete_mutual_info(mus, ys):
    """
    Compute discrete mutual information.
    """
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
    return m


def discrete_entropy(ys):
    """
    Compute discrete mutual information.
    """
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
    return h

def mutual_info(mus_train):
    # Compute average mutual information between different factors.
    num_codes = mus_train.shape[0]
    mus_discrete = histogram_discretize(mus_train, num_bins=20)
    mutual_info_matrix = discrete_mutual_info(mus_discrete, mus_discrete)
    np.fill_diagonal(mutual_info_matrix, 0)
    mutual_info_score = np.sum(mutual_info_matrix) / (num_codes ** 2 - num_codes)
    return mutual_info_score


def betatc_compute_total_correlation(z_sampled, z_mean, z_logvar):
    """
    Estimate total correlation over a batch.
    Reference implementation is from: https://github.com/amir-abdi/disentanglement-pytorch
    """
    # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
    # tensor of size [batch_size, batch_size, num_latents]. In the following
    # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
    log_qz_prob = _betatc_compute_gaussian_log_density(z_sampled.unsqueeze(dim=1), z_mean.unsqueeze(dim=0), z_logvar.unsqueeze(dim=0))

    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
    # + constant) for each sample in the batch, which is a vector of size
    # [batch_size,].
    log_qz_product = log_qz_prob.exp().sum(dim=1, keepdim=False).log().sum(dim=1, keepdim=False)

    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = log_qz_prob.sum(dim=2, keepdim=False).exp().sum(dim=1, keepdim=False).log()

    return (log_qz - log_qz_product).mean()


def _betatc_compute_gaussian_log_density(samples, mean, log_var):
    """
    Estimate the log density of a Gaussian distribution
    Reference implementation is from: https://github.com/amir-abdi/disentanglement-pytorch
    """
    # TODO: can this be replaced with some variant of Normal.log_prob?
    import math
    pi = torch.tensor(math.pi, requires_grad=False)
    normalization = torch.log(2 * pi)
    inv_sigma = torch.exp(-log_var)
    tmp = samples - mean
    return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)



def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation

    mapper = {nn.LeakyReLU: "leaky_relu", nn.ReLU: "relu", nn.Tanh: "tanh",
              nn.Sigmoid: "sigmoid", nn.Softmax: "sigmoid"}
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k

    raise ValueError("Unkown given activation type : {}".format(activation))


def get_gain(activation):
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1

    activation_name = get_activation_name(activation)

    param = None if activation_name != "leaky_relu" else activation.negative_slope
    gain = nn.init.calculate_gain(activation_name, param)

    return gain

# Factor VAE Discriminator

def linear_init(layer, activation="relu"):
    """Initialize a linear layer.
    Args:
        layer (nn.Linear): parameters to initialize.
        activation (`torch.nn.modules.activation` or str, optional) activation that
            will be used on the `layer`.
    """
    x = layer.weight

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity='leaky_relu')
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity='relu')
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))


def weights_init(module):
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        # TO-DO: check litterature
        linear_init(module)
    elif isinstance(module, nn.Linear):
        linear_init(module)
        
class Discriminator(nn.Module):
    def __init__(self,
                 neg_slope=0.2,
                 latent_dim=10,
                 hidden_units=1000):
        """Discriminator proposed in [1].
        Parameters
        ----------
        neg_slope: float
            Hyperparameter for the Leaky ReLu
        latent_dim : int
            Dimensionality of latent variables.
        hidden_units: int
            Number of hidden units in the MLP
        Model Architecture
        ------------
        - 6 layer multi-layer perceptron, each with 1000 hidden units
        - Leaky ReLu activations
        - Output 2 logits
        References:
            [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
            arXiv preprint arXiv:1802.05983 (2018).
        """
        super(Discriminator, self).__init__()

        # Activation parameters
        self.neg_slope = neg_slope
        self.leaky_relu = nn.LeakyReLU(self.neg_slope, True)

        # Layer parameters
        self.z_dim = latent_dim
        self.hidden_units = hidden_units
        # theoretically 1 with sigmoid but gives bad results => use 2 and softmax
        out_units = 2

        # Fully connected layers
        self.lin1 = nn.Linear(self.z_dim, hidden_units)
        self.lin2 = nn.Linear(hidden_units, hidden_units)
        self.lin3 = nn.Linear(hidden_units, hidden_units)
        self.lin4 = nn.Linear(hidden_units, hidden_units)
        self.lin5 = nn.Linear(hidden_units, hidden_units)
        self.lin6 = nn.Linear(hidden_units, out_units)

        self.reset_parameters()

    def forward(self, z):

        # Fully connected layers with leaky ReLu activations
        z = self.leaky_relu(self.lin1(z))
        z = self.leaky_relu(self.lin2(z))
        z = self.leaky_relu(self.lin3(z))
        z = self.leaky_relu(self.lin4(z))
        z = self.leaky_relu(self.lin5(z))
        z = self.lin6(z)

        return z

    def reset_parameters(self):
        self.apply(weights_init)

def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed

def _permute_dims(latent_sample):
    """
    Implementation of Algorithm 1 in ref [1]. Randomly permutes the sample from
    q(z) (latent_dist) across the batch for each of the latent dimensions (mean
    and log_var).
    Parameters
    ----------
    latent_sample: torch.Tensor
        sample from the latent dimension using the reparameterisation trick
        shape : (batch_size, latent_dim).
    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).
    """
    perm = torch.zeros_like(latent_sample)
    batch_size, dim_z = perm.size()

    for z in range(dim_z):
        pi = torch.randperm(batch_size).to(latent_sample.device)
        perm[:, z] = latent_sample[pi, z]

    return perm