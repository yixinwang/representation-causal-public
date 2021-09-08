import argparse
import numpy as np
import pandas as pd
import time
import random
import os
import sys

import numpy.random as npr

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
import statsmodels.api as sm
from numpy.linalg import cond

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch import nn, optim, autograd
from torch.autograd import Variable

from utils import cov

out_dir = 'out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

randseed = int(time.time()*1e7%1e8)
print("random seed: ", randseed)
random.seed(randseed)
np.random.seed(randseed)
torch.manual_seed(randseed)



parser = argparse.ArgumentParser(description='Causal Representation with Linear Models')
parser.add_argument('--N', type=int, default=20000) # number of data points
parser.add_argument('--D', type=int, default=5) # number of features
parser.add_argument('--num_features', type=int, default=2) # when the number of features are too small to capture the true features, it doesn't work.
parser.add_argument('--z_dim', type=int, default=1) # number of pca dimensionality
parser.add_argument('--spurious_corr', type=float, default=0.95)
parser.add_argument('--y_noise', type=float, default=0.1)
parser.add_argument('--l2_reg', type=float, default=1.) # regularization parameter for ridge regression
parser.add_argument('--lognormal', type=int, default=0)
parser.add_argument('--steps', type=int, default=10001)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--mode', type=str, default="linear", choices=["linear", "logistic"])
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--mode_latent', type=str, default="pcaz", choices=["vaez", "bertz", "bertz_cl", "pcaz"])
parser.add_argument('--mode_train_data', type=str, default="text", choices=["text", "bertz"])
flags, unk = parser.parse_known_args()

flags.input_dim = flags.D
z_dim = flags.z_dim

res = pd.DataFrame(vars(flags), index=[0])
res['randseed'] = randseed

print('Flags:')
for k,v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

sys.stdout.flush()


N, D, K, corr, y_noise, alpha, M, lognormal = flags.N, flags.D, flags.z_dim, flags.spurious_corr, flags.y_noise, 100, flags.num_features, flags.lognormal

mean = np.zeros(D)
# designate the core feature
num_corefea = np.int(D/2)
true_cause = np.arange(num_corefea).astype(int)

"""## generate simulated datasets with core and spurious features

The outcome model is the same in training and testing; the outcome
only depends on the core feature.

In the training set, the covariates have high correlation. In the test
set, the covariates have low correlation.
"""

# simulate strongly correlated features for training
train_cov = np.ones((D, D)) * corr + np.eye(D) * (1 - corr)
train_x_true = npr.multivariate_normal(mean, train_cov, size=N)

# simulate weakly correlated features for testing
test_cov = np.ones((D, D)) * (1 - corr) + np.eye(D) * corr
testobs_x_true = npr.multivariate_normal(mean, train_cov, size=N)
testct_x_true = npr.multivariate_normal(mean, test_cov, size=N)

if lognormal:
    train_x_true = np.exp(npr.multivariate_normal(mean, train_cov, size=N)) # exponential of gaussian; no need to be gaussian
    testobs_x_true = np.exp(npr.multivariate_normal(mean, train_cov, size=N)) # exponential of gaussian; no need to be gaussian
    testct_x_true = np.exp(npr.multivariate_normal(mean, test_cov, size=N))  # exponential of gaussian; no need to be gaussian

# add observation noise to the x
# spurious correlation more often occurs when the signal to noise ratio is lower
x_noise = np.array(list(np.ones(num_corefea)*0.4) + list(np.ones(D-num_corefea)*0.3))

train_x = train_x_true + x_noise * npr.normal(size=[N,D])
testobs_x = testobs_x_true + x_noise * npr.normal(size=[N,D])
testct_x = testct_x_true + x_noise * npr.normal(size=[N,D])

print("\ntrain X correlation\n", np.corrcoef(train_x.T))
print("\ntest X correlation\n",np.corrcoef(testct_x.T))

# generate outcome
# toy model y = x + noise
 
truecoeff = npr.uniform(size=num_corefea) * 10 
train_y = train_x_true[:,true_cause].dot(truecoeff) + y_noise * npr.normal(size=N)
testobs_y = testobs_x_true[:,true_cause].dot(truecoeff) + y_noise * npr.normal(size=N)
testct_y = testct_x_true[:,true_cause].dot(truecoeff) + y_noise * npr.normal(size=N)

"""# baseline naive regression on all features"""

def fitcoef(cov_train, train_y, cov_test=None, test_y=None):
    # linearReg
    print("\nlinearReg")
    reg = LinearRegression()
    reg.fit(cov_train, train_y)
    print("coef", reg.coef_, "intercept", reg.intercept_)
    lintrainacc = reg.score(cov_train, train_y)
    print("train accuracy", lintrainacc)
    if cov_test is not None:
        lintestacc = reg.score(cov_test, test_y)
        print("test accuracy", lintestacc)

    # ridgeReg
    print("\nridgeReg")

    ridgealphas, ridgetrainaccs, ridgetestaccs = [], [], []
    for C in [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]:
        tmp_alpha = 1./C
        print("alpha", tmp_alpha)
        reg = Ridge(alpha=tmp_alpha)
        reg.fit(cov_train, train_y)
        print("coef", reg.coef_, "intercept", reg.intercept_)
        ridgetrainacc = reg.score(cov_train, train_y)
        ridgetrainaccs.append(ridgetrainacc)
        ridgealphas.append(tmp_alpha)
        print("train accuracy", ridgetrainacc)
        if cov_test is not None:
            # print("test accuracy", reg.score(cov_test, test_y))
            ridgetestacc = reg.score(cov_test, test_y)
            print("test accuracy", ridgetestacc)
            ridgetestaccs.append(ridgetestacc)
    return lintrainacc, lintestacc, ridgealphas, ridgetrainaccs, ridgetestaccs

"""all three features have coefficient different from zero

test accuracy degrades much from training accuracy.
"""

print("\n###########################\nall features")

cov_train = np.column_stack([train_x])
cov_testct = np.column_stack([testct_x])
cov_testobs = np.column_stack([testobs_x])

naive_testctaccs = fitcoef(cov_train, train_y, cov_testct, testct_y)
naive_testobsaccs = fitcoef(cov_train, train_y, cov_testobs, testobs_y)

def save_regfit_to_df(testaccs):
    lintrainacc, lintestacc, ridgealphas, ridgetrainaccs, ridgetestaccs = testaccs
    assert len(ridgealphas) == len(ridgetrainaccs)
    assert len(ridgetestaccs) == len(ridgetrainaccs)
    res = {'lintrainacc': lintrainacc, 'lintestacc': lintestacc}
    for item in ['ridgetrainaccs', 'ridgetestaccs']:
        for i, alpha in enumerate(ridgealphas):
            curname = item + '_' + str(alpha)
            if item == 'ridgetrainaccs':
                res[curname] = ridgetrainaccs[i]
            elif item == 'ridgetestaccs':
                res[curname] = ridgetestaccs[i]
    return res 

naive_testct_res = {'naive_testct_' + str(key): val for key, val in save_regfit_to_df(naive_testctaccs).items()}
naive_testobs_res = {'naive_testobs_' + str(key): val for key, val in save_regfit_to_df(naive_testobsaccs).items()}


"""next consider oracle, regression only on the core feature"""

print("\n###########################\nall features (oracle)")

cov_train = np.column_stack([train_x[:,true_cause]])
cov_testct = np.column_stack([testct_x[:,true_cause]])
cov_testobs = np.column_stack([testobs_x[:,true_cause]])

oracle_testctaccs = fitcoef(cov_train, train_y, cov_testct, testct_y)
oracle_testobsaccs = fitcoef(cov_train, train_y, cov_testobs, testobs_y)

oracle_testct_res = {'oracle_testct_' + str(key): val for key, val in save_regfit_to_df(oracle_testctaccs).items()}
oracle_testobs_res = {'oracle_testobs_' + str(key): val for key, val in save_regfit_to_df(oracle_testobsaccs).items()}


naive_oracle = dict(list(naive_testct_res.items()) + list(naive_testobs_res.items()) + list(oracle_testct_res.items()) +  list(oracle_testobs_res.items()))

res = pd.concat([pd.DataFrame(naive_oracle, index=[0]), res], axis=1)

"""## causal-rep 

now try adjust for pca factor, then learn feature coefficient,
construct a prediction function using the learned feature mapping,
predict on the test set
"""


# fit pca to high correlated training dataset

pca = PCA(n_components=K)
pca.fit(train_x)
pca.transform(train_x)
print("\n\npca explained ratio", pca.explained_variance_ratio_)

print(np.cumsum(pca.explained_variance_ratio_))


print("truecoeff", truecoeff)
print("\noracle_ct", oracle_testctaccs, "\nnaive_ct", naive_testctaccs)
print("\noracle_obs", oracle_testobsaccs, "\nnaive_obs", naive_testobsaccs)


# next apply causal rep

X_train = torch.from_numpy(train_x).float().cuda()
X_testct = torch.from_numpy(testct_x).float().cuda()
X_testobs = torch.from_numpy(testobs_x).float().cuda()


train_pca_embedding = torch.from_numpy(pca.transform(train_x)).float().cuda()
testct_pca_embedding = torch.from_numpy(pca.transform(testct_x)).float().cuda()
testobs_pca_embedding = torch.from_numpy(pca.transform(testobs_x)).float().cuda()

train_label = torch.unsqueeze(torch.from_numpy(train_y).float(), 1).cuda()
testobs_label = torch.unsqueeze(torch.from_numpy(testobs_y).float(), 1).cuda()
testct_label = torch.unsqueeze(torch.from_numpy(testct_y).float(), 1).cuda()


def compute_prob(logits, mode="logistic"):
    if mode == "linear":
        probs = logits
    elif mode == "logistic":
        probs = nn.Sigmoid()(logits)
    return probs

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input_dim = flags.input_dim
        self.z_dim = z_dim
        self.num_features = flags.num_features
        lin1 = nn.Linear(self.input_dim, self.num_features)
        lin4 = nn.Linear(self.z_dim, 1)
        for lin in [lin1, lin4]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1)
    def forward(self, inputbow, vaez):
        features = torch.matmul(inputbow, F.softmax(self._main[0].weight,dim=1).T)
        return features


def mean_nll(probs, y, mode="logistic"):
    # return nn.functional.binary_cross_entropy_with_logits(logits, y)
    if mode == "linear":
        mean_nll = nn.MSELoss()(probs, y)
    elif mode == "logistic":
        mean_nll = nn.BCELoss()(probs, y)
    return mean_nll

def mean_accuracy(probs, y):
    preds = (probs > 0.5).float()
    return ((preds - y).abs() < 1e-2).float().mean()

envs = [
    {'text': X_train, 'pcaz': train_pca_embedding, 'labels': train_label}, \
    {'text': X_testct, 'pcaz': testct_pca_embedding, 'labels': testct_label}, \
    {'text': X_testobs, 'pcaz': testobs_pca_embedding, 'labels': testobs_label}]

mlp = MLP().cuda()
# optimizer_causalrep = optim.RMSprop(mlp._main.parameters(), lr=flags.lr, weight_decay=1e-12)
optimizer_causalrep = optim.Adam(mlp._main.parameters(), lr=flags.lr, weight_decay=1e-8)


for step in range(flags.steps):
    for i in range(len(envs)):
        env = envs[i]
        features = mlp(env[flags.mode_train_data], env[flags.mode_latent])
        labels = env['labels']

        y = labels - labels.mean()
        X = torch.cat([features, env[flags.mode_latent]], dim=1)
        X = X - X.mean(dim=0)
        X = torch.cat([torch.ones(X.shape[0],1).cuda(), X], dim=1)

        beta = [torch.matmul(
            torch.matmul(
                torch.inverse(flags.l2_reg*torch.eye(X.shape[1]).cuda()+
                    torch.matmul(
                        torch.transpose(X, 0, 1),
                        X)),
                torch.transpose(X, 0, 1)),
            y[:,j]) for j in range(y.shape[1])]

        env['covs'] = cov(torch.cat([beta[0][1:flags.num_features+1] *features, torch.unsqueeze((beta[0][-flags.z_dim:] * env[flags.mode_latent]).sum(dim=1),1)], dim=1))[-1][:-1] # extract the last row to have cov(Features, C)

        env['causalreploss'] = ((features.std(dim=0) * beta[0][1:flags.num_features+1])**2).sum()

         # + 2 * env['covs']).sum() 

         # One may leave out this covariance term in the CAUSAL-REP
         # loss if we already enforce the positivity of
         # representations given the unobserved common cause.


    train_causalreploss = torch.stack([envs[0]['causalreploss']])

    if step % 1 == 0:
        # l1_penalty = F.softmax(mlp._main[0].weight,dim=1).abs().sum()
        l2_penalty = (F.softmax(mlp._main[0].weight,dim=1)**2).sum()

        train_causalrep_loss = -train_causalreploss.clone() + 1e-1 * l2_penalty

        optimizer_causalrep.zero_grad()
        train_causalrep_loss.backward(retain_graph=True)
        optimizer_causalrep.step()


    if step % 100 == 0:

        # print(mlp._main[0].weight)

        train_features, train_y = mlp(envs[0][flags.mode_train_data], envs[0][flags.mode_latent]).clone().cpu().detach().numpy(), envs[0]['labels'].clone().cpu().detach().numpy()
        testct_features, testct_y = mlp(envs[1][flags.mode_train_data], envs[1][flags.mode_latent]).clone().cpu().detach().numpy(), envs[1]['labels'].clone().cpu().detach().numpy()
        testobs_features, testobs_y = mlp(envs[2][flags.mode_train_data], envs[2][flags.mode_latent]).clone().cpu().detach().numpy(), envs[2]['labels'].clone().cpu().detach().numpy()

        C_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]

        # choose C = 10.

        causalrep_alphas, causalrep_trainaccs, causalrep_testobsaccs, causalrep_testctaccs = [], [], [], []
        for C in C_vals:
            alpha = 1./C
            print('\ncausal-pred-w-features', 'C', C)

            # clf = LinearRegression()
            clf = Ridge(alpha=alpha)
            # clf = LogisticRegression(C=C, class_weight='auto', solver='lbfgs')
            clf.fit(train_features, train_y)
            # resulttrain = classification_report((train_y > 0), (clf.predict(train_features) > 0), output_dict=True)
            # resultct = classification_report((testct_y > 0), (clf.predict(testct_features) > 0), output_dict=True)
            # resultobs = classification_report((testobs_y > 0), (clf.predict(testobs_features)> 0), output_dict=True)
            causalrep_trainacc = clf.score(train_features, train_y)
            causalrep_testobsacc = clf.score(testobs_features, testobs_y)
            causalrep_testctacc = clf.score(testct_features, testct_y)
            causalrep_trainaccs.append(causalrep_trainacc)
            causalrep_testobsaccs.append(causalrep_testobsacc)
            causalrep_testctaccs.append(causalrep_testctacc)
            causalrep_alphas.append(alpha)
            print('train',causalrep_trainacc)
            print('testobs',causalrep_testobsacc)
            print('testct',causalrep_testctacc)
            sys.stdout.flush()

    if step % 10 == 0:
        print("itr", np.int32(step),
        "train_causalreploss", train_causalreploss.detach().cpu().numpy())
        sys.stdout.flush()


causalrep_res = {}

assert len(causalrep_alphas) == len(causalrep_trainaccs)
assert len(causalrep_alphas) == len(causalrep_testobsaccs)
assert len(causalrep_alphas) == len(causalrep_testctaccs)
for item in ['causalrep_trainaccs', 'causalrep_testobsaccs', 'causalrep_testctaccs']:
    for i, alpha in enumerate(causalrep_alphas):
        curname = item + '_' + str(alpha)
        if item == 'causalrep_trainaccs':
            causalrep_res[curname] = causalrep_trainaccs[i]
        elif item == 'causalrep_testobsaccs':
            causalrep_res[curname] = causalrep_testobsaccs[i]
        elif item == 'causalrep_testctaccs':
            causalrep_res[curname] = causalrep_testctaccs[i]

res = pd.concat([pd.DataFrame(causalrep_res, index=[0]), res], axis=1)

res.to_csv(out_dir + '/spurious_linear' + str(int(time.time()*1e6)) + '.csv')

