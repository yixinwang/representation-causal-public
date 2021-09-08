import argparse
import numpy as np
import numpy.random as npr
import scipy
import os
import sys
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


parser = argparse.ArgumentParser(description='Disentangled Representation Learning with IOSS')
parser.add_argument('--dataset', choices=['dsprites', 'cars3d', 'mpi3d', 'smallnorb'], default="mpi3d")
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.001)
# parser.add_argument('--mode', type=str, default="logistic", choices=["linear", "logistic"])
parser.add_argument('--z_dim', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=100) # preferrably large batch size, but for some large datasets, cannot afford large batch sizes on gpu
parser.add_argument('--vae_epochs', type=int, default=51)
parser.add_argument('--ioss_weight', type=float, default=1e4)
parser.add_argument('--beta_weight', type=float, default=0.2)
parser.add_argument('--gamma_weight', type=float, default=10.) # factor vae weight
parser.add_argument('--spurious_corr', type=float, default=0.95)
parser.add_argument('--train_sample_size', type=int, default=20000)
flags, unknown = parser.parse_known_args()


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

if not os.path.exists('./learn_out'):
    os.makedirs('./learn_out')

if flags.dataset == "dsprites":
    dataset = DSprites()
elif flags.dataset == "cars3d":
    dataset = Cars3D()
elif flags.dataset == "mpi3d":
    dataset = MPI3D()
elif flags.dataset == "smallnorb":
    dataset = SmallNORB()


unique_ys = np.unique(dataset.sample_factors(10000, npr.RandomState(randseed)), axis=0)
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


# generate weakly correlated test factors for testing (observational)
testobs_cov = np.ones((D, D)) * corr + np.eye(D) * (1 - corr)
testobs_y_true = npr.multivariate_normal(mean, testobs_cov, size=N)
testobs_y_true = uniformize(testobs_y_true)
testobs_ys = testobs_y_true // (1. / (num_uniqs))
assert (testobs_ys.max(axis=0) - (num_uniqs-1)).sum() == 0
testobs_xs = dataset.sample_observations_from_factors(testobs_ys, npr.RandomState(randseed))


# generate weakly correlated test factors for testing (counterfactual)
testct_cov = np.ones((D, D)) * (1 - corr) + np.eye(D) * corr
testct_cov = np.ones((D, D)) * corr + np.eye(D) * (1 - corr)
testct_y_true = npr.multivariate_normal(mean, testct_cov, size=N)
testct_y_true = uniformize(testct_y_true)
testct_ys = testct_y_true // (1. / (num_uniqs))
assert (testct_ys.max(axis=0) - (num_uniqs-1)).sum() == 0
testct_xs = dataset.sample_observations_from_factors(testct_ys, npr.RandomState(randseed))



# final_train_accs = []
# final_test_accs = []


# train_idx = npr.binomial(1, 0.8, size=train_xs.shape[0])

train_data = torch.Tensor(train_xs.reshape(train_xs.shape[0], -1)).float()
testobs_data = torch.Tensor(testobs_xs.reshape(train_xs.shape[0], -1)).float()
testct_data = torch.Tensor(testct_xs.reshape(train_xs.shape[0], -1)).float()

flags.input_dim = train_data.shape[1]

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=flags.batch_size, shuffle=True)

testobs_loader = torch.utils.data.DataLoader(dataset=testobs_data, batch_size=flags.batch_size, shuffle=True)

testct_loader = torch.utils.data.DataLoader(dataset=testct_data, batch_size=flags.batch_size, shuffle=True)

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
            
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, flags.input_dim))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

def IOSS(mu, n_draws=10000, robust_k_prop = 0.01):
    stdmu = (mu - torch.min(mu,dim=0)[0])/ (torch.max(mu,dim=0)[0]-torch.min(mu,dim=0)[0]).cuda()
    
    K = np.int(robust_k_prop * mu.shape[0]) + 1

    maxs = torch.topk(stdmu, K, dim=0)[0][-1,:]
    mins = -(torch.topk(-stdmu, K, dim=0)[0][-1,:])    

    smps = (torch.stack([torch.rand(n_draws).cuda() * (maxs[i]-mins[i]) + mins[i] for i in range(stdmu.shape[1])], dim=1))
    min_dist = (torch.min(torch.cdist(smps, stdmu.cuda()), dim=1)[0])
    # ortho = (torch.mean(min_dist,dim=0))
    ortho = (torch.topk(min_dist, np.int(robust_k_prop*n_draws)+1, dim=0))[0][-1]
    # ortho = torch.max(min_dist,dim=0)[0]
    return ortho



def TotalCorr_betatc_vae(z_sampled, z_mean, z_logvar):
    tc_loss = betatc_compute_total_correlation(
            z_sampled=z_sampled,
            z_mean=z_mean,
            z_logvar=z_logvar,
        )
    return tc_loss

def classical_vae_loss_function(vae, data, factor_vae_discriminator=None, optimizer_factor_vae_discriminator=None):
    recon_batch, mu, log_var = vae(data)
    BCE = F.binary_cross_entropy(recon_batch, data.view(-1, flags.input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def factor_vae_loss_function(vae, data, factor_vae_discriminator, optimizer_factor_vae_discriminator, gamma=flags.gamma_weight, steps_anneal=0):
    """
    Compute the Factor-VAE loss as per Algorithm 2 of [1]
    Parameters
    ----------
    device : torch.device
    gamma : float, optional
        Weight of the TC loss term. `gamma` in the paper.
    discriminator : disvae.discriminator.Discriminator
    optimizer_d : torch.optim
    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.
    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).
    """

    assert factor_vae_discriminator != None
    assert optimizer_factor_vae_discriminator != None
    # factor-vae split data into two batches. In the paper they sample 2 batches
    batch_size = data.size(dim=0)
    half_batch_size = batch_size // 2
    data = data.split(half_batch_size)
    data1 = data[0]
    data2 = data[1]

    # Factor VAE Loss

    recon_batch1, mu1, log_var1 = vae(data1)
    BCE1 = F.binary_cross_entropy(recon_batch1, data1.view(-1, flags.input_dim), reduction='sum')
    KLD1 = -0.5 * torch.sum(1 + log_var1 - mu1.pow(2) - log_var1.exp())
    z_sampled1 = torch.randn_like(torch.exp(0.5*log_var1)).mul(torch.exp(0.5*log_var1)).add_(mu1) 
    d_z = factor_vae_discriminator(z_sampled1)

    # We want log(p_true/p_false). If not using logisitc regression but softmax
    # then p_true = exp(logit_true) / Z; p_false = exp(logit_false) / Z
    # so log(p_true/p_false) = logit_true - logit_false
    factor_vae_tc_loss1 = (d_z[:, 0] - d_z[:, 1]).mean()
    # with sigmoid (not good results) should be `tc_loss = (2 * d_z.flatten()).mean()`

    anneal_reg = linear_annealing(0, 1, flags.vae_epochs, steps_anneal)
    
    vae_loss1 = BCE1 + KLD1 + anneal_reg * gamma * factor_vae_tc_loss1

    vae_loss1 = Variable(vae_loss1, requires_grad = True)

    # Discriminator Loss
    # Get second sample of latent distribution
    recon_batch2, mu2, log_var2 = vae(data2)
    z_sampled2 = torch.randn_like(torch.exp(0.5*log_var2)).mul(torch.exp(0.5*log_var2)).add_(mu2) 
    z_perm = _permute_dims(z_sampled2).detach()
    d_z_perm = factor_vae_discriminator(z_perm)

    

    # Calculate total correlation loss
    # for cross entropy the target is the index => need to be long and says
    # that it's first output for d_z and second for perm
    ones = torch.ones(half_batch_size, dtype=torch.long).cuda()
    zeros = torch.zeros_like(ones)
    d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_perm, ones))
    # with sigmoid would be :
    # d_tc_loss = 0.5 * (self.bce(d_z.flatten(), ones) + self.bce(d_z_perm.flatten(), 1 - ones))

    # TO-DO: check if should also anneals discriminator if not becomes too good ???
    #d_tc_loss = anneal_reg * d_tc_loss

    d_tc_loss = Variable(d_tc_loss, requires_grad = True)

    # Compute discriminator gradients
    optimizer_factor_vae_discriminator.zero_grad()
    d_tc_loss.backward(retain_graph=True)

    optimizer_factor_vae_discriminator.step()
    return vae_loss1


# return reconstruction error + KL divergence losses
def ioss_vae_loss_function(vae, data, factor_vae_discriminator=None, optimizer_factor_vae_discriminator=None, lmbda=flags.ioss_weight):
    recon_batch, mu, log_var = vae(data)
    BCE = F.binary_cross_entropy(recon_batch, data.view(-1, flags.input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    ortho = IOSS(mu)
    # print(ortho, BCE, KLD)
    return BCE + KLD + lmbda * ortho

def beta_vae_loss_function(vae, data, factor_vae_discriminator=None, optimizer_factor_vae_discriminator=None, beta=flags.beta_weight):
    recon_batch, mu, log_var = vae(data)
    BCE = F.binary_cross_entropy(recon_batch, data.view(-1, flags.input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # ortho = IOSS(mu)
    # print(ortho, BCE, KLD)
    return BCE + beta * KLD

def betatc_vae_loss_function(vae, data, factor_vae_discriminator=None, optimizer_factor_vae_discriminator=None, beta=flags.beta_weight):
    recon_batch, mu, log_var = vae(data)
    BCE = F.binary_cross_entropy(recon_batch, data.view(-1, flags.input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    z_sampled = torch.randn_like(torch.exp(0.5*log_var)).mul(torch.exp(0.5*log_var)).add_(mu) 
    TC = TotalCorr_betatc_vae(z_sampled, mu, log_var)
    # ortho = IOSS(mu)
    # print(ortho, BCE, KLD)
    return BCE + beta * KLD + (1-beta) * TC

def train(vae, optimizer_vae, epoch, loss_function, factor_vae_discriminator=None, optimizer_factor_vae_discriminator=None):
    vae.train()
    train_loss = 0
    train_orthos = 0
    train_elbo = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.cuda()
        optimizer_vae.zero_grad()
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(vae, data, factor_vae_discriminator, optimizer_factor_vae_discriminator)
        loss.backward(retain_graph=True)
        train_loss += loss.item()
        train_elbo += classical_vae_loss_function(vae, data).item()
        train_orthos += IOSS(mu).item()
        optimizer_vae.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item() / len(data)))
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
            print('train IOSS', train_orthos / len(train_loader.dataset))
            print('train ELBO', train_elbo / len(train_loader.dataset))
    return train_elbo / len(train_loader.dataset)

def test(vae, loss_function, factor_vae_discriminator=None, optimizer_factor_vae_discriminator=None):
    vae.eval()
    test_loss = 0
    orthos = 0
    test_elbo = 0
    with torch.no_grad():
        for data in testobs_loader:
                data = data.cuda()
                recon, mu, log_var = vae(data)
                
                # sum up batch loss
                test_loss += loss_function(vae, data, factor_vae_discriminator, optimizer_factor_vae_discriminator).item()
                orthos += IOSS(mu).item()
                test_elbo += classical_vae_loss_function(vae, data).item()

    # idx = (npr.rand(testobs_data.shape[0]) > 0.9)
    # ys_test = testobs_data.latents_classes[(1-train_idx)!=0].T[1:]
    # mus_test = vae.cpu()(testobs_data.cpu()).detach().cpu().numpy().T
    # print(scalable_disentanglement_score(ys_test.T[idx], mus_test.T[idx]))
        
    test_loss /= len(testobs_loader.dataset)
    print('====> Testobs set loss: {:.4f}'.format(test_loss))
    print('testobs IOSS', orthos / len(testobs_loader.dataset))
    print('testobs ELBO', test_elbo / len(testobs_loader.dataset))

    return test_elbo / len(testobs_loader.dataset)


# build model
classical_vae = VAE(x_dim=flags.input_dim, h_dim1=flags.hidden_dim, h_dim2=flags.hidden_dim, z_dim=flags.z_dim)
ioss_vae = VAE(x_dim=flags.input_dim, h_dim1=flags.hidden_dim, h_dim2=flags.hidden_dim, z_dim=flags.z_dim)
beta_vae = VAE(x_dim=flags.input_dim, h_dim1=flags.hidden_dim, h_dim2=flags.hidden_dim, z_dim=flags.z_dim)
betatc_vae = VAE(x_dim=flags.input_dim, h_dim1=flags.hidden_dim, h_dim2=flags.hidden_dim, z_dim=flags.z_dim)
factor_vae = VAE(x_dim=flags.input_dim, h_dim1=flags.hidden_dim, h_dim2=flags.hidden_dim, z_dim=flags.z_dim)
factor_vae_discriminator = Discriminator(latent_dim=flags.z_dim,hidden_units=flags.hidden_dim)

if torch.cuda.is_available():
    classical_vae = classical_vae.cuda()
    ioss_vae = ioss_vae.cuda()
    beta_vae = beta_vae.cuda()
    betatc_vae = betatc_vae.cuda()
    factor_vae = factor_vae.cuda()
    factor_vae_discriminator = factor_vae_discriminator.cuda()

optimizer_classical_vae = optim.Adam(classical_vae.parameters())
optimizer_ioss_vae = optim.Adam(ioss_vae.parameters())
optimizer_beta_vae = optim.Adam(beta_vae.parameters())
optimizer_betatc_vae = optim.Adam(betatc_vae.parameters())
optimizer_factor_vae = optim.Adam(factor_vae.parameters())
optimizer_factor_vae_discriminator = optim.Adam(factor_vae_discriminator.parameters())

for epoch in range(1, flags.vae_epochs):
    print("Epoch", epoch)
    print("classical_vae")
    classical_vae_train_elbo = train(classical_vae, optimizer_classical_vae, epoch, classical_vae_loss_function)
    classical_vae_test_elbo = test(classical_vae, classical_vae_loss_function)
    sys.stdout.flush()

for epoch in range(1, flags.vae_epochs):
    print("Epoch", epoch)
    print("ioss_vae")
    ioss_vae_train_elbo = train(ioss_vae, optimizer_ioss_vae, epoch, ioss_vae_loss_function)
    ioss_vae_test_elbo = test(ioss_vae, ioss_vae_loss_function)
    sys.stdout.flush()

for epoch in range(1, flags.vae_epochs):
    print("Epoch", epoch)    
    print("beta_vae")
    beta_vae_train_elbo = train(beta_vae, optimizer_beta_vae, epoch, beta_vae_loss_function)
    beta_vae_test_elbo = test(beta_vae, beta_vae_loss_function)
    sys.stdout.flush()

for epoch in range(1, flags.vae_epochs):
    print("Epoch", epoch)
    print("betatc_vae")
    betatc_vae_train_elbo = train(betatc_vae, optimizer_betatc_vae, epoch, betatc_vae_loss_function)
    betatc_vae_test_elbo = test(betatc_vae, betatc_vae_loss_function)
    sys.stdout.flush()

for epoch in range(1, flags.vae_epochs):
    print("Epoch", epoch)
    print("factor_vae")
    factor_vae_train_elbo = train(factor_vae, optimizer_factor_vae, epoch, factor_vae_loss_function, factor_vae_discriminator, optimizer_factor_vae_discriminator)
    factor_vae_test_elbo = test(factor_vae, factor_vae_loss_function, factor_vae_discriminator, optimizer_factor_vae_discriminator)
    sys.stdout.flush()

def eval_vae_representation(vae, train_data, testobs_data, testct_data, train_ys, testobs_ys, testct_ys, dataset, vae_name, nsmp = 5000):
    print("######################")
    print("Evaluating " + vae_name)
    print("######################")

    def representation_function(train_data):
        train_vae_recon, train_vae_mu, train_vae_logvar = vae(train_data.reshape(train_data.shape[0], -1).float().cuda())
        return train_vae_mu.float().cuda()

    train_vae_mu = representation_function(train_data)
    testobs_vae_mu = representation_function(testobs_data)
    testct_vae_mu = representation_function(testct_data)


    num_train = train_data.shape[0]
    smpidx = npr.randint(train_data.shape[0], size=nsmp)
    # smpidx = np.arange(num_train)
    ys_train = train_ys.T[1:]
    ys_testobs = testobs_ys.T[1:]
    ys_testct = testct_ys.T[1:]
    mus_train = train_vae_mu.detach().cpu().numpy().T
    mus_testobs = testobs_vae_mu.detach().cpu().numpy().T
    mus_testct = testct_vae_mu.detach().cpu().numpy().T

    def take_middle_quantile(mus_train, q=0.01):
        # consider columns of matrix whose value is within (q, 1-q) qunatile of all values. (ignore outliers)
        uq = np.quantile(mus_train.T, 1-q, axis=0)
        lq = np.quantile(mus_train.T, q, axis=0)
        lower_than_uq = np.prod(mus_train.T < np.repeat(uq[np.newaxis,:], mus_train.T.shape[0],axis=0),axis=1)
        higher_than_lq = np.prod(mus_train.T > np.repeat(lq[np.newaxis,:], mus_train.T.shape[0],axis=0),axis=1)
        return mus_train[:,(lower_than_uq * higher_than_lq)==1]

    savefilename_prefix = './learn_out/' + flags.dataset + '_' + vae_name + '_' + str(randseed) 

    train_sns_plot = sns.pairplot(pd.DataFrame(take_middle_quantile(mus_train).T), diag_kind = 'kde', 
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
             size = 4)
    train_sns_plot.savefig(savefilename_prefix + '_train_mu.png') 
    pd.DataFrame(take_middle_quantile(mus_train).T).to_csv(savefilename_prefix + '_train_mu.csv')

    testobs_sns_plot = sns.pairplot(pd.DataFrame(take_middle_quantile(mus_testobs).T), diag_kind = 'kde', 
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
             size = 4)
    testobs_sns_plot.savefig(savefilename_prefix + '_testobs_mu.png') 
    pd.DataFrame(take_middle_quantile(mus_testobs).T).to_csv(savefilename_prefix + '_testobs_mu.csv')

    testct_sns_plot = sns.pairplot(pd.DataFrame(take_middle_quantile(mus_testct).T), diag_kind = 'kde', 
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
             size = 4)
    testct_sns_plot.savefig(savefilename_prefix + '_testct_mu.png') 
    pd.DataFrame(take_middle_quantile(mus_testct).T).to_csv(savefilename_prefix + '_testct_mu.csv')

    irs_train = IRS_score(ys_train.T, mus_train.T)
    irs_testobs = IRS_score(ys_testobs.T, mus_testobs.T)
    irs_testct = IRS_score(ys_testct.T, mus_testct.T)
    irs_oracle = IRS_score(ys_testct.T, ys_testct.T)
    print("irs train", irs_train)
    print("irs testobs", irs_testobs)
    print("irs testct", irs_testct)
    print("irs oracle", irs_oracle)
    sys.stdout.flush()

    disentanglement_metrics = pd.DataFrame(vars(flags
        ), index=[0])

    disentanglement_metrics['vae_name'] = vae_name

    disentanglement_metrics['classical_vae_train_elbo'] = classical_vae_train_elbo
    disentanglement_metrics['classical_vae_test_elbo'] = classical_vae_test_elbo
    disentanglement_metrics['ioss_vae_train_elbo'] = ioss_vae_train_elbo
    disentanglement_metrics['ioss_vae_test_elbo'] = ioss_vae_test_elbo
    disentanglement_metrics['beta_vae_train_elbo'] = beta_vae_train_elbo
    disentanglement_metrics['beta_vae_test_elbo'] = beta_vae_test_elbo
    disentanglement_metrics['betatc_vae_train_elbo'] = betatc_vae_train_elbo
    disentanglement_metrics['betatc_vae_test_elbo'] = betatc_vae_test_elbo
    disentanglement_metrics['factor_vae_train_elbo'] = factor_vae_train_elbo
    disentanglement_metrics['factor_vae_test_elbo'] = factor_vae_test_elbo


    disentanglement_metrics.to_csv(savefilename_prefix + '_disentanglement_metrics.csv')

    disentanglement_metrics['irs_train'] = irs_train['avg_score']
    disentanglement_metrics['irs_testobs'] = irs_testobs['avg_score']
    disentanglement_metrics['irs_testct'] = irs_testct['avg_score']
    disentanglement_metrics['irs_oracle'] = irs_oracle['avg_score']

    disentanglement_metrics.to_csv(savefilename_prefix + '_disentanglement_metrics.csv')

    for ioss_k_prop in [1e-1, 1e-2, 1e-3, 1e-4]:
        ioss_train = IOSS(torch.Tensor(mus_train.T).cuda(), n_draws=1000,robust_k_prop = ioss_k_prop).detach().cpu().numpy()
        ioss_testobs = IOSS(torch.Tensor(mus_testobs.T).cuda(), n_draws=1000,robust_k_prop = ioss_k_prop).detach().cpu().numpy()
        ioss_testct = IOSS(torch.Tensor(mus_testct.T).cuda(), n_draws=1000,robust_k_prop = ioss_k_prop).detach().cpu().numpy()
        ioss_oracle = IOSS(torch.Tensor(ys_train.T).cuda(), n_draws=1000,robust_k_prop = ioss_k_prop).detach().cpu().numpy()

        disentanglement_metrics['ioss_train_'+str(ioss_k_prop)] = ioss_train
        disentanglement_metrics['ioss_testobs_'+str(ioss_k_prop)] = ioss_testobs
        disentanglement_metrics['ioss_testct_'+str(ioss_k_prop)] = ioss_testct
        disentanglement_metrics['ioss_oracle_'+str(ioss_k_prop)] = ioss_oracle

    disentanglement_metrics.to_csv(savefilename_prefix + '_disentanglement_metrics.csv')

    predacc_train = np.zeros(ys_train.T.shape[1])
    predacc_testobs = np.zeros(ys_testobs.T.shape[1])
    predacc_testct = np.zeros(ys_testct.T.shape[1])

    noise_coef = 1.
    ys_train_noisy = (ys_train.T + noise_coef * ys_train.T.std(axis=0) *  npr.normal(size=ys_train.T.shape)).T
    ys_testobs_noisy = (ys_testobs.T + noise_coef * ys_train.T.std(axis=0) *  npr.normal(size=ys_testobs.T.shape)).T
    ys_testct_noisy = (ys_testct.T + noise_coef * ys_train.T.std(axis=0) *  npr.normal(size=ys_testct.T.shape)).T

    # print("downstream prediction with representations")
    # for i in range(ys_testobs.T.shape[1]):
    #     print("outcome", i)
    #     # downstream_predictor = MLPClassifier(random_state=randseed, max_iter=300).fit(mus_train.T, ys_train.T[:,i])
    #     downstream_predictor = MLPRegressor(random_state=randseed, max_iter=200).fit(mus_train.T, ys_train_noisy.T[:,i])
    #     predacc_train[i] = downstream_predictor.score(mus_train.T, ys_train_noisy.T[:,i])
    #     predacc_testobs[i] = downstream_predictor.score(mus_testobs.T, ys_testobs_noisy.T[:,i])
    #     predacc_testct[i] = downstream_predictor.score(mus_testct.T, ys_testct_noisy.T[:,i])

    # print("predacc_train", predacc_train)
    # print("predacc_testobs", predacc_testobs)
    # print("predacc_testct", predacc_testct)
    # disentanglement_metrics['predacc_train'] = predacc_train.mean()
    # disentanglement_metrics['predacc_testobs'] = predacc_testobs.mean()
    # disentanglement_metrics['predacc_testct'] = predacc_testct.mean()
    # sys.stdout.flush()

    disentanglement_metrics.to_csv(savefilename_prefix + '_disentanglement_metrics.csv')

    # print("downstream prediction with raw data")
    # predacc_raw_train = np.zeros(ys_train.T.shape[1])
    # predacc_raw_testobs = np.zeros(ys_testobs.T.shape[1])
    # predacc_raw_testct = np.zeros(ys_testct.T.shape[1])

    # for i in range(ys_testobs.T.shape[1]):
    #     print("outcome", i)
    #     # downstream_predictor = MLPClassifier(random_state=randseed, max_iter=300).fit(mus_train.T, ys_train.T[:,i])
    #     downstream_predictor = MLPRegressor(random_state=randseed, max_iter=200).fit(train_data.numpy(), ys_train_noisy.T[:,i])
    #     predacc_raw_train[i] = downstream_predictor.score(train_data.numpy(), ys_train_noisy.T[:,i])
    #     predacc_raw_testobs[i] = downstream_predictor.score(testobs_data.numpy(), ys_testobs_noisy.T[:,i])
    #     predacc_raw_testct[i] = downstream_predictor.score(testct_data.numpy(), ys_testct_noisy.T[:,i])

    # print("predacc_raw_train", predacc_raw_train)
    # print("predacc_raw_testobs", predacc_raw_testobs)
    # print("predacc_raw_testct", predacc_raw_testct)

    sys.stdout.flush()

    # disentanglement_metrics['predacc_raw_train'] = predacc_raw_train.mean()
    # disentanglement_metrics['predacc_raw_testobs'] = predacc_raw_testobs.mean()
    # disentanglement_metrics['predacc_raw_testct'] = predacc_raw_testct.mean()

    # dci takes very long to compute

    # dci_train = DCI_score(mus_train, ys_train, mus_train, ys_train)
    # dci_testobs = DCI_score(mus_testobs, ys_testobs, mus_testobs, ys_testobs)
    # dci_testct = DCI_score(mus_testct, ys_testct, mus_testct, ys_testct)
    # dci_oracle = DCI_score(ys_train, ys_train, ys_testobs, ys_testobs)
    # disentanglement_metrics['dci_train'] = dci_train['disentanglement']
    # disentanglement_metrics['dci_testobs'] = dci_testobs['disentanglement']
    # disentanglement_metrics['dci_testct'] = dci_testct['disentanglement']
    # disentanglement_metrics['dci_oracle'] = dci_oracle['disentanglement']

    # print("dci train", dci_train)
    # print("dci testobs", dci_testobs)
    # print("dci testct", dci_testct)
    # print("dci oracle", dci_oracle)
    # sys.stdout.flush()

    # disentanglement_metrics.to_csv(savefilename_prefix + '_disentanglement_metrics.csv')

    cov_train_mus = np.cov(mus_train)
    disentanglement_metrics['gaussian_total_corr_train'] =  gaussian_total_correlation(cov_train_mus)
    disentanglement_metrics['gaussian_wasserstein_corr_train'] =  gaussian_wasserstein_correlation(cov_train_mus)
    disentanglement_metrics['gaussian_wasserstein_corr_norm_train'] =  disentanglement_metrics['gaussian_wasserstein_corr_train'] / np.sum(np.diag(cov_train_mus))
    # disentanglement_metrics['mi_train'] = mutual_info(mus_train.T)
    # mi works with discrete data and takes very long to compute

    cov_testobs_mus = np.cov(mus_testobs)
    disentanglement_metrics['gaussian_total_corr_testobs'] =  gaussian_total_correlation(cov_testobs_mus)
    disentanglement_metrics['gaussian_wasserstein_corr_testobs'] =  gaussian_wasserstein_correlation(cov_testobs_mus)
    disentanglement_metrics['gaussian_wasserstein_corr_norm_testobs'] =  disentanglement_metrics['gaussian_wasserstein_corr_testobs'] / np.sum(np.diag(cov_testobs_mus))
    # disentanglement_metrics['mi_testobs'] = mutual_info(mus_testobs.T)

    cov_testct_mus = np.cov(mus_testct)
    disentanglement_metrics['gaussian_total_corr_testct'] =  gaussian_total_correlation(cov_testct_mus)
    disentanglement_metrics['gaussian_wasserstein_corr_testct'] =  gaussian_wasserstein_correlation(cov_testct_mus)
    disentanglement_metrics['gaussian_wasserstein_corr_norm_testct'] =  disentanglement_metrics['gaussian_wasserstein_corr_testct'] / np.sum(np.diag(cov_testct_mus))
    # disentanglement_metrics['mi_testct'] = mutual_info(mus_testct.T)

    disentanglement_metrics.to_csv(savefilename_prefix + '_disentanglement_metrics.csv')

    # below use generative metrics (as opposed to on static testobs and testct)



    # create disent_dataset for evaluation
    disent_dataset = DisentDataset(dataset)
    unsupervised = pd.DataFrame(metric_unsupervised(disent_dataset, representation_function), index=[0])

    disentanglement_metrics = pd.concat([disentanglement_metrics, unsupervised], axis=1)
    disentanglement_metrics.to_csv(savefilename_prefix + '_disentanglement_metrics.csv')

    betavae = pd.DataFrame(metric_beta_vae(disent_dataset, representation_function), index=[0])

    # disentanglement_metrics = pd.concat([disentanglement_metrics, betavae], axis=1)
    # disentanglement_metrics.to_csv(savefilename_prefix + '_disentanglement_metrics.csv')

    # sap = pd.DataFrame(metric_sap(disent_dataset, representation_function), index=[0])

    # disentanglement_metrics = pd.concat([disentanglement_metrics, sap], axis=1)
    # disentanglement_metrics.to_csv(savefilename_prefix + '_disentanglement_metrics.csv')

    # mig = pd.DataFrame(metric_mig(disent_dataset, representation_function), index=[0])

    # disentanglement_metrics = pd.concat([disentanglement_metrics, mig], axis=1)
    # disentanglement_metrics.to_csv(savefilename_prefix + '_disentanglement_metrics.csv')

    # factorvae = pd.DataFrame(metric_factor_vae(disent_dataset, representation_function), index=[0])

    # disentanglement_metrics = pd.concat([disentanglement_metrics, factorvae], axis=1)
    # disentanglement_metrics.to_csv(savefilename_prefix + '_disentanglement_metrics.csv')

    # dci = pd.DataFrame(metric_dci(disent_dataset, representation_function), index=[0])

    # disentanglement_metrics = pd.concat([disentanglement_metrics, dci], axis=1)
    # disentanglement_metrics.to_csv(savefilename_prefix + '_disentanglement_metrics.csv')

    # disentanglement_metrics = pd.concat([disentanglement_metrics, unsupervised], axis=1)

    print(disentanglement_metrics.T)
    sys.stdout.flush()

    disentanglement_metrics.to_csv(savefilename_prefix + '_disentanglement_metrics.csv')

    return disentanglement_metrics

classical_disentanglement_metrics = eval_vae_representation(classical_vae, train_data, testobs_data, testct_data, train_ys, testobs_ys, testct_ys, dataset, vae_name="classical")

ioss_disentanglement_metrics = eval_vae_representation(ioss_vae, train_data, testobs_data, testct_data, train_ys, testobs_ys, testct_ys, dataset, vae_name="ioss")

beta_disentanglement_metrics = eval_vae_representation(beta_vae, train_data, testobs_data, testct_data, train_ys, testobs_ys, testct_ys, dataset, vae_name="beta")

betatc_disentanglement_metrics = eval_vae_representation(betatc_vae, train_data, testobs_data, testct_data, train_ys, testobs_ys, testct_ys, dataset, vae_name="betatc")

factor_disentanglement_metrics = eval_vae_representation(factor_vae, train_data, testobs_data, testct_data, train_ys, testobs_ys, testct_ys, dataset, vae_name="factor")
