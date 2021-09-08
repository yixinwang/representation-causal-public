import argparse
import numpy as np
import numpy.random as npr
import scipy
import os
import seaborn as sns
import matplotlib.pyplot as plt
import time 
import random
from load_data import DSprites, Cars3D, MPI3D, SmallNORB
import pandas as pd

from utils import uniformize, IRS_score, betatc_compute_total_correlation, DCI_score, gaussian_total_correlation, gaussian_wasserstein_correlation, mutual_info, betatc_compute_total_correlation, Discriminator, linear_annealing, _permute_dims

from disent_dataset import DisentDataset
from disent_unsupervised import metric_unsupervised
from disent_sap import metric_sap
from disent_mig import metric_mig
from disent_factorvae import metric_factor_vae
from disent_dci import metric_dci
from disent_betavae import metric_beta_vae

from sklearn import ensemble
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.inspection import permutation_importance

import torch
from torchvision import datasets, transforms
from torch import nn, optim, autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from scipy.spatial.distance import cdist

parser = argparse.ArgumentParser(description='Measure Disentanglement with IOSS')
parser.add_argument('--dataset', choices=['dsprites', 'cars3d', 'mpi3d', 'smallnorb'], default='smallnorb')
# parser.add_argument('--hidden_dim', type=int, default=512)
# parser.add_argument('--lr', type=float, default=0.001)
# parser.add_argument('--z_dim', type=int, default=10)
# parser.add_argument('--batch_size', type=int, default=100) # preferrably large batch size, but for some large datasets, cannot afford large batch sizes on gpu
# parser.add_argument('--vae_epochs', type=int, default=51)
# parser.add_argument('--ioss_weight', type=float, default=1e4)
# parser.add_argument('--beta_weight', type=float, default=0.2)
# parser.add_argument('--gamma_weight', type=float, default=10.) # factor vae weight
parser.add_argument('--spurious_corr', type=float, default=0.90)
parser.add_argument('--train_sample_size', type=int, default=1000)
flags, unparsed = parser.parse_known_args()


N = flags.train_sample_size
corr = flags.spurious_corr

os.environ['DISENTANGLEMENT_LIB_DATA'] = '/proj/sml/usr/yixinwang/representation-causal/src/disentanglement_expms/data/'

eps = 1e-8 # define a small close to zero number

# randseed = 52744889
randseed = int(time.time()*1e7%1e8)
print("random seed: ", randseed)
random.seed(randseed)
np.random.seed(randseed)
torch.manual_seed(randseed)

flags.randseed = randseed

print('Flags:')
for k,v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

out_dir = './measure_out'


if not os.path.exists('./measure_out'):
    os.makedirs('./measure_out')

if flags.dataset == "dsprites":
    dataset = DSprites()
elif flags.dataset == "cars3d":
    dataset = Cars3D()
elif flags.dataset == "mpi3d":
    dataset = MPI3D()
elif flags.dataset == "smallnorb":
    dataset = SmallNORB()

out_dir = './measure_out'



# write function sample correlated ids

# generate data

unique_ys = np.unique(dataset.sample_factors(flags.train_sample_size, npr.RandomState(randseed)), axis=0)
num_uniqs = np.array([len(np.unique(unique_ys[:,i])) for i in range(unique_ys.shape[1])])
y_cols = np.where(num_uniqs > 1)[0] 


D = dataset.sample_factors(2, npr.RandomState(randseed)).shape[1] # number of features
mean = np.zeros(D)
# generate highly correlated training factors
train_cov = np.ones((D, D)) * corr + np.eye(D) * (1 - corr)
train_y_true = npr.multivariate_normal(mean, train_cov, size=N)
train_y_true = uniformize(train_y_true)
train_ys = train_y_true // (1. / (num_uniqs))
assert (train_ys.max(axis=0) - (num_uniqs-1)).sum() == 0
train_xs = dataset.sample_observations_from_factors(train_ys, npr.RandomState(randseed))

# train_ys is the disentangled ground truth factors

def make_entangled_representations(groundtruth_factors, order=3, eps=1e-6, scale=5.):
    num_factors = groundtruth_factors.shape[1] 
    entangled = np.zeros_like(groundtruth_factors)
    betas = (npr.uniform(size=(num_factors, num_factors*order)) - 0.5) * scale
    for j in range(num_factors): 
        entangled[:,j] = (betas[j] * np.column_stack([groundtruth_factors / (groundtruth_factors.std(axis=0) + eps)**(o+1) for o in range(order)])).sum(axis=1)
        entangled[:,j] += npr.normal(size=entangled[:,j].shape)
    return entangled


def IOSS(mu, metric = "euclidean", n_draws=10000, robust_k_prop = 1e-2):
    # stdmu = (mu - np.min(mu,axis=0))/ (np.max(mu,axis=0)-np.min(mu,axis=0))
    # K = np.int(robust_k_prop * mu.shape[0]) + 1
    # # maxs = [np.max(mu[:,i]) for i in range(mu.shape[1])]
    # # mins = [np.min(mu[:,i]) for i in range(mu.shape[1])]
    # maxs = [stdmu[k,j] for j,k in enumerate(np.argsort(-stdmu, axis=0)[K])]
    # mins = [stdmu[k,j] for j,k in enumerate(np.argsort(stdmu, axis=0)[K])]
    # smps = np.column_stack([npr.uniform(low=mins[i], high=maxs[i], size=n_draws) for i in range(stdmu.shape[1])])
    # dist = cdist(smps, stdmu, metric=metric)
    # # IOSS = dist.min(axis=1).mean()
    # min_dist = np.array([dist[k,j] for j,k in enumerate(np.argsort(dist, axis=0)[np.int(robust_k_prop*n_draws)+1])])
    # score = np.max(min_dist)

    stdmu = (mu-np.min(mu,axis=0)) / (np.max(mu,axis=0) - np.min(mu,axis=0))

    # robust_k_prop = 0.001
#     K = np.int(robust_k_prop * stdmu.shape[0])

    maxs = np.max(stdmu, axis=0)
    mins = np.min(stdmu, axis=0)
    smps = (np.column_stack([npr.rand(n_draws) * (maxs[i]-mins[i]) + mins[i] 
                             for i in range(stdmu.shape[1])]))
    min_dist = np.min(cdist(smps, stdmu, metric=metric), axis=1)
    # ortho = (torch.mean(min_dist,dim=0))
    ortho = np.max(min_dist,axis=0)

    # print(IOSS)
    # ortho = (torch.topk(min_dist, np.int(robust_k_prop*n_draws)+1, dim=0))[0][-1]
    return ortho 


def unsupervised_metrics(mus):
    cov_train_mus = np.cov(mus.T)
    gaussian_total_corr_train =  gaussian_total_correlation(cov_train_mus)
    gaussian_wasserstein_corr_train =  gaussian_wasserstein_correlation(cov_train_mus)
    gaussian_wasserstein_corr_norm_train =  gaussian_wasserstein_corr_train / np.sum(np.diag(cov_train_mus))
    # mi = mutual_info(mus)
    return gaussian_total_corr_train, gaussian_wasserstein_corr_norm_train
    # , mi

groundtruth_factors = train_ys
groundtruth_factors = groundtruth_factors[:,np.where(groundtruth_factors.std(axis=0) > 1e-2)[0]]

entangled_rep = make_entangled_representations(groundtruth_factors)

unsupervised_entangled = unsupervised_metrics(entangled_rep)
unsupervised_groundtruth = unsupervised_metrics(groundtruth_factors)


irs_entangle = IRS_score(groundtruth_factors, entangled_rep)['avg_score']
irs_groundtruth = IRS_score(groundtruth_factors, groundtruth_factors)['avg_score']


# only IRS is larger is better, others smaller is better

res = pd.DataFrame({'disentanglement': ['entangled', 'disentangled'], \
    'avg_corr_coef': [np.mean(np.corrcoef(entangled_rep.T)), np.mean(np.corrcoef(groundtruth_factors.T))], \
    'IOSS':[IOSS(entangled_rep), IOSS(groundtruth_factors)], \
    'IRS': [IRS_score(groundtruth_factors, entangled_rep)['avg_score'], IRS_score(groundtruth_factors, groundtruth_factors)['avg_score']], \
    'gaussian_total_correlation': [unsupervised_entangled[0], unsupervised_groundtruth[0]], \
    'gaussian_wasserstein_dependency': [unsupervised_entangled[1], unsupervised_groundtruth[1]], \
    # 'mutual_info': [unsupervised_entangled[2], unsupervised_groundtruth[2]], \
    'dataset': [flags.dataset, flags.dataset]})
res['IOSS_classify'] = np.repeat(res['IOSS'][0]>res['IOSS'][1],2)
res['IRS_classify'] = np.repeat(res['IRS'][0]<res['IRS'][1],2)
res['gaussian_total_correlation_classify'] = np.repeat(res['gaussian_total_correlation'][0]>res['gaussian_total_correlation'][1],2)
res['gaussian_wasserstein_dependency_classify'] = np.repeat(res['gaussian_wasserstein_dependency'][0]>res['gaussian_wasserstein_dependency'][1],2)
res['avg_corr_coef_classify'] = np.repeat(res['avg_corr_coef'][0]>res['avg_corr_coef'][1],2)

print(res.T)
res.to_csv(out_dir + '/disentangle_measure' + str(int(time.time()*1e6)) + '.csv')
