import importlib
import copy
import io, time
from io import BytesIO
import chardet
import os
import collections
from itertools import combinations, cycle, product
import math
import numpy as np
import pandas as pd
import pickle
import tarfile
import random
import re
import requests
from nltk.corpus import stopwords
from scipy.sparse import hstack, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
import torch.nn.functional as F

from tqdm import tqdm
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', 1000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1) # change None to -1


from collections import Counter, defaultdict
import numpy as np
import re
import sys

import sklearn

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score

import torch
from torchvision import datasets, transforms
from torch import nn, optim, autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from transformers import * # here import bert

import warnings
warnings.filterwarnings("ignore")


# from data_structure import Dataset #, get_IMDB, get_kindle
import argparse

import utils
importlib.reload(utils)
from utils import *
from vae import VAE, vae_loss_function, train_vae, test_vae

# randseed = 52744889
randseed = int(time.time()*1e7%1e8)
print("random seed: ", randseed)
sys.stdout.flush()
random.seed(randseed)
np.random.seed(randseed)
torch.manual_seed(randseed)

parser = argparse.ArgumentParser(description='Text Reviews')
parser.add_argument('-d', '--dataset', type=str, default='amazon',choices=['yelp', 'amazon', 'tripadvisor'])
parser.add_argument('--datsubsample', type=int, default=10000)
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--steps', type=int, default=2001)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--l2_reg', type=float, default=1e-3)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--mode', type=str, default="linear", choices=["linear", "logistic"])
parser.add_argument('--z_dim', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--num_features', type=int, default=5)
parser.add_argument('--input_dim', type=int, default=0)
parser.add_argument('--vae_epochs', type=int, default=101)
parser.add_argument('--spurious_corr', type=float, default=0.9)
parser.add_argument('--alter_freq', type=int, default=50)
parser.add_argument('--mode_latent', type=str, default="pcaz", choices=["vaez", "bertz", "bertz_cl", "pcaz"])
parser.add_argument('--mode_train_data', type=str, default="text", choices=["text", "bertz"])
flags, unk = parser.parse_known_args()

res = pd.DataFrame(vars(flags), index=[0])
res['randseed'] = randseed

print(flags)
sys.stdout.flush()

moniker = flags.dataset

out_dir = moniker + '_out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

dat_file = 'dat/'+ moniker + '/' + moniker + '_meta.csv'


# detect encoding
# rawdata=open(dat_file,'rb').read()
# result = chardet.detect(rawdata)
# charenc = result['encoding']
# print(charenc)

if moniker == 'amazon':
    full_dat = pd.read_csv(dat_file)
elif moniker == 'tripadvisor':
    full_dat = pd.read_csv(dat_file, encoding='Windows-1252')
elif moniker == 'yelp':
    full_dat = pd.read_csv(dat_file, lineterminator='\n')
    full_dat = full_dat.rename(columns={'stars_x':'y', 'text':'review_text'})


data = full_dat[full_dat['y']!=3].sample(n=flags.datsubsample)

texts = list(data['review_text'])
labels = (np.array(data['y']) > 3)

split1, split2 = int(0.6*len(texts)), (int(0.6*len(texts)) + int(0.2*len(texts)))

train_text, train_label = texts[:split1], torch.from_numpy(labels[:split1]).float().cuda()
testobs_text, testobs_label = texts[split1:split2], torch.from_numpy(labels[split1:split2]).float().cuda()
testct_text, testct_label = texts[split2:], torch.from_numpy(labels[split2:]).float().cuda()

stop_words = set(stopwords.words('english'))

# vec = CountVectorizer(min_df=5, binary=True, max_df=0.8, ngram_range=(1,3))

vec = TfidfVectorizer(min_df=10, binary=True, max_df=0.8, ngram_range=(1,3))

X_full = vec.fit_transform(train_text)
X_train_full = vec.transform(train_text)
X_testobs_full = vec.transform(testobs_text)
X_testct_full = vec.transform(testct_text)

feats = np.array(vec.get_feature_names())

top_feature_idx, placebo_feature_idx, coef = get_top_terms(vec.transform(train_text), train_label.cpu().numpy(), coef_thresh=0.0, placebo_thresh=0.1) # use coef_threshold=0.0 to take all features, no thresholding happening here.

# top_feature_idx = np.arange(500)

X_train_np = vec.transform(train_text).toarray()
X_testobs_np = vec.transform(testobs_text).toarray()
X_testct_np = vec.transform(testct_text).toarray()


fea_corrcoef = np.corrcoef(X_train_np[:,top_feature_idx].T) - np.eye(X_train_np[:,top_feature_idx].shape[1])
colinear_fea = np.where(fea_corrcoef>0.96)[0]
feature_idx = np.array(list(set(top_feature_idx) - set(colinear_fea)))
# only consider words in feature_idx

id2term = collections.OrderedDict({i:v for i,v in enumerate(feats[feature_idx])})
term2id = collections.OrderedDict({v:i for i,v in enumerate(feats[feature_idx])})


spurious_words = np.array([term2id['as'], term2id['also'], term2id['am'], term2id['an']])


final_train_accs = []
final_test_accs = []
final_train_baselineaccs = []
final_test_baselineaccs = []
final_train_baselinevaeaccs = []
final_test_baselinevaeaccs = []

for restart in range(flags.n_restarts):
    print("Restart", restart)

    def make_environment(texts, labels, e):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a-b).abs() # Assumes both inputs are either 0 or 1
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (labels == 1).float()
        labels = torch_xor(labels, torch_bernoulli(0.35, len(labels)).cuda())
        # Assign a color based on the label; flip the color with probability e
        spurious_counts = torch.stack([torch_xor(labels, torch_bernoulli(e, len(labels)).cuda()) for i in range(len(spurious_words))], axis=1)
        # Apply the color to the image by zeroing out the other color channel
        texts[:,spurious_words] = spurious_counts.cpu().numpy()

        return {
            'texts': torch.from_numpy(texts).float().cuda(),
            'labels': labels[:, None].cuda(),
            'colors': spurious_counts.cuda()
        }

train_data = make_environment(X_train_np[:,feature_idx], train_label, 1-flags.spurious_corr)
X_train, train_label = train_data['texts'], train_data['labels']

testobs_data = make_environment(X_testobs_np[:,feature_idx], testobs_label, 1-flags.spurious_corr)
X_testobs, testobs_label = testobs_data['texts'], testobs_data['labels'] 

testct_data = make_environment(X_testct_np[:,feature_idx], testct_label, 0.9)
X_testct, testct_label = testct_data['texts'], testct_data['labels']

vocabsize = X_train.shape[1]

flags.input_dim = vocabsize

# calculate pca embedding
pca = PCA(n_components=flags.z_dim)
# pca.fit(np.row_stack([X_train_np, X_testobs_np, X_testct_np]))

pca.fit(np.row_stack([X_train_np[:,feature_idx]]))

train_pca_embedding = torch.from_numpy(pca.transform(X_train_np[:,feature_idx])).float().cuda()
testobs_pca_embedding = torch.from_numpy(pca.transform(X_testobs_np[:,feature_idx])).float().cuda()
testct_pca_embedding = torch.from_numpy(pca.transform(X_testct_np[:,feature_idx])).float().cuda()

print(np.cumsum(pca.explained_variance_ratio_))

print(pca.explained_variance_ratio_ * flags.input_dim)

# take only the top pc dimensions with effective sample size > 100

# flags.z_dim = np.sum(pca.explained_variance_ratio_ * flags.input_dim > 30)

# print(flags.z_dim)

# # calculate pca embedding
# pca = PCA(n_components=flags.z_dim)
# # pca.fit(np.row_stack([X_train_np, X_testobs_np, X_testct_np]))

# pca.fit(np.row_stack([X_train_np[:,feature_idx]]))

# train_pca_embedding = torch.from_numpy(pca.transform(X_train_np[:,feature_idx])).float().cuda()
# testobs_pca_embedding = torch.from_numpy(pca.transform(X_testobs_np[:,feature_idx])).float().cuda()
# testct_pca_embedding = torch.from_numpy(pca.transform(X_testct_np[:,feature_idx])).float().cuda()



# flags.num_features = flags.input_dim - flags.z_dim


subset_nonsing=False

if flags.mode_latent == "vaez":
    z_dim = flags.z_dim
elif flags.mode_latent == "bertz":
    z_dim = train_embedding.shape[1]
elif flags.mode_latent == "bertz_cl":
    z_dim = X_train_cl_embedding.shape[1]
    subset_nonsing=True
elif flags.mode_latent == "pcaz":
    z_dim = flags.z_dim




# z_dim = flags.z_dim

print(vocabsize, z_dim)
sys.stdout.flush()


def compute_prob(logits, mode="logistic"):
    if mode == "linear":
        probs = torch.max(torch.stack([logits,torch.zeros_like(logits)],dim=2),dim=2)[0]
        probs = torch.min(torch.stack([probs,torch.ones_like(probs)],dim=2),dim=2)[0]
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
        self._tvaez = nn.Sequential(lin4) 
        self.finallayer = nn.Linear(self.num_features + 1, 1)
    def forward(self, inputbow, vaez):

        features = torch.matmul(inputbow, F.softmax(self._main[0].weight,dim=1).T)
        logits = self.finallayer(torch.cat([features, self._tvaez(vaez)],dim=1))
        probs = compute_prob(logits, mode=flags.mode)
        features_ctr = features - features.mean(dim=0)
        beta_hat = 0.
        feature_hats = 0.
        logit_hats = logits
        prob_hats = probs
        return features, logits, probs, beta_hat, logit_hats, prob_hats


def mean_nll(probs, y, mode="logistic"):
    if mode == "linear":
        mean_nll = nn.MSELoss()(probs, y)
    elif mode == "logistic":
        mean_nll = nn.BCELoss()(probs, y)
    return mean_nll

def mean_accuracy(probs, y):
    preds = (probs > 0.5).float()
    return ((preds - y).abs() < 1e-2).float().mean()

# the Net component is not used
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(flags.num_features, 1)
    def forward(self, x):
        x = self.fc(x)
        return x

def initNet(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)

envs = [
    {'text': X_train, 'pcaz': train_pca_embedding, 'labels': train_label}, \
    {'text': X_testct, 'pcaz': testct_pca_embedding, 'labels': testct_label}, \
    {'text': X_testobs, 'pcaz': testobs_pca_embedding, 'labels': testobs_label}]


if subset_nonsing == True:
    envs[0]['text'] = envs[0]['text'][nonsing_sents]
    envs[0]['labels'] = envs[0]['labels'][nonsing_sents]

if flags.mode_train_data == 'text':
    flags.input_dim = vocabsize
    train_loader = torch.utils.data.DataLoader(dataset=envs[0]['text'].view(-1, flags.input_dim), batch_size=flags.batch_size, shuffle=False)
    testct_loader = torch.utils.data.DataLoader(dataset=envs[1]['text'].view(-1, flags.input_dim), batch_size=flags.batch_size, shuffle=False)
    testobs_loader = torch.utils.data.DataLoader(dataset=envs[2]['text'].view(-1, flags.input_dim), batch_size=flags.batch_size, shuffle=False)
elif flags.mode_train_data == 'bertz':
    flags.input_dim = train_embedding.shape[1]
    train_loader = torch.utils.data.DataLoader(dataset=envs[0]['bertz'].view(-1, flags.input_dim), batch_size=flags.batch_size, shuffle=False)
    testct_loader = torch.utils.data.DataLoader(dataset=envs[1]['bertz'].view(-1, flags.input_dim), batch_size=flags.batch_size, shuffle=False)
    testobs_loader = torch.utils.data.DataLoader(dataset=envs[2]['bertz'].view(-1, flags.input_dim), batch_size=flags.batch_size, shuffle=False)


if flags.mode_latent == 'vae':

    trainvaez_name = flags.dataset + 'k' + str(flags.z_dim) + 'trainvae.pt'
    testctvaez_name = flags.dataset + 'k' + str(flags.z_dim) + 'testctvae.pt'
    testobsvaez_name = flags.dataset + 'k' + str(flags.z_dim) + 'testobsvae.pt'

    envs[0]['vaeimage'] = torch.load(trainvaez_name)[0].detach()
    envs[1]['vaeimage'] = torch.load(testctvaez_name)[0].detach()
    envs[2]['vaeimage'] = torch.load(testobsvaez_name)[0].detach()

    envs[0]['vaez'] = torch.load(trainvaez_name)[1].detach()
    envs[1]['vaez'] = torch.load(testctvaez_name)[1].detach()
    envs[2]['vaez'] = torch.load(testobsvaez_name)[1].detach()



mlp = MLP().cuda()


optimizer_causalrep = optim.Adam(mlp._main.parameters(), lr=flags.lr, weight_decay=1e-8)

for step in range(flags.steps):
    for i in range(len(envs)):
        env = envs[i]
        features, logits, probs, beta_hat, logit_hats, prob_hats = mlp(env[flags.mode_train_data], env[flags.mode_latent])
        labels = env['labels']
        env['nll'] = mean_nll(probs, env['labels'], mode=flags.mode) 
        env['nllhat'] = mean_nll(prob_hats, env['labels'], mode=flags.mode) 
        env['acc'] = mean_accuracy(probs, env['labels'])
        env['acchat'] = mean_accuracy(prob_hats, env['labels'])

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

        env['causalrep'] = ((features.std(dim=0) * beta[0][1:flags.num_features+1])**2).sum()

         # + 2 * env['covs']).sum()


        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.finallayer.parameters():
            weight_norm += w.norm().pow(2)

        env['l2penalty'] = flags.l2_reg * weight_norm

        if step % 500 == 0:
            print("\nnll", env['nll'], 
                "\nl2", env['l2penalty'], 
                "\ncausalrep", env['causalrep'])
                # "\nfeatureZr2", env['featureZr2'])
            sys.stdout.flush()

    train_l2penalty = torch.stack([envs[0]['l2penalty']])
    train_causalrep = torch.stack([envs[0]['causalrep']])
    train_nll = torch.stack([envs[0]['nll']]).mean() 
    train_acc = torch.stack([envs[0]['acc']]).mean()
    testct_nll = torch.stack([envs[1]['nll']]).mean()
    testct_acc = torch.stack([envs[1]['acc']]).mean()
    testobs_nll = torch.stack([envs[2]['nll']]).mean()
    testobs_acc = torch.stack([envs[2]['acc']]).mean()

    nll_loss = train_nll.clone() 
    # + train_l2penalty.clone()


    if step % 1 == 0:
        l1_penalty = F.softmax(mlp._main[0].weight,dim=1).abs().sum()

        train_causalrep_loss = -train_causalrep.clone() 
        # + 1e-3 * l1_penalty - 1e-2 * torch.log(1 - train_featureZr2)

    

        optimizer_causalrep.zero_grad()
        train_causalrep_loss.backward(retain_graph=True)
        optimizer_causalrep.step()


    if step % 100 == 0:

        train_features, train_y = mlp(envs[0][flags.mode_train_data], envs[0][flags.mode_latent])[0].clone().cpu().detach().numpy(), envs[0]['labels'].clone().cpu().detach().numpy()
        testct_features, testct_y = mlp(envs[1][flags.mode_train_data], envs[1][flags.mode_latent])[0].clone().cpu().detach().numpy(), envs[1]['labels'].clone().cpu().detach().numpy()
        testobs_features, testobs_y = mlp(envs[2][flags.mode_train_data], envs[2][flags.mode_latent])[0].clone().cpu().detach().numpy(), envs[2]['labels'].clone().cpu().detach().numpy()

        C_vals = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
        causalrep_alphas, causalrep_trainaccs, causalrep_testobsaccs, causalrep_testctaccs = [], [], [], []
        for C in C_vals:
            alpha = 1./C
            print('\ncausal-pred-w-features', 'C', C)

            # clf = LinearRegression()
            # clf = Ridge(alpha=alpha)
            clf = LogisticRegression(C=C, class_weight='auto', solver='lbfgs')
            clf.fit(train_features, train_y)
            resulttrain = classification_report((train_y > 0), (clf.predict(train_features) > 0), output_dict=True)
            resultct = classification_report((testct_y > 0), (clf.predict(testct_features) > 0), output_dict=True)
            resultobs = classification_report((testobs_y > 0), (clf.predict(testobs_features)> 0), output_dict=True)
            print('train',resulttrain['accuracy'])
            print('testobs',resultobs['accuracy'])
            print('testct',resultct['accuracy'])
            sys.stdout.flush()
            causalrep_trainaccs.append(resulttrain['accuracy'])
            causalrep_testobsaccs.append(resultobs['accuracy'])
            causalrep_testctaccs.append(resultct['accuracy'])
            causalrep_alphas.append(alpha)

        print("\n\n##### causal rep top words")
        feature_weights = torch.topk(F.softmax(mlp._main[0].weight,dim=1),20, axis=1)
        top_causal_words = feature_weights[1].detach().cpu().numpy()
        top_causal_weights = feature_weights[0].detach().cpu().numpy()
        for j in np.argsort(-np.abs(beta[0][1:(1+flags.num_features)].detach().cpu().numpy())):
        # for j in range(top_causal_words.shape[0]):
            print("feature", j)
            print("coefficient", beta[0][j+1])
            sort_causal_words = np.argsort(-top_causal_weights[j])[:20]

            print("top causal words", [id2term[i] for i in top_causal_words[j][sort_causal_words]], top_causal_weights[j][sort_causal_words]
            )

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


    if step % 10 == 0:
        print("itr", np.int32(step),
        # "train_causalrephat_loss", train_causalrep_loss.detach().cpu().numpy(),
        "train_causalrep", train_causalrep.detach().cpu().numpy(), 
        # "train_causalrephat", train_causalrephat.detach().cpu().numpy(), 
        "train_nll", train_nll.detach().cpu().numpy(),
        "train_acc", train_acc.detach().cpu().numpy(),
        "testct_acc", testct_acc.detach().cpu().numpy(),
        "testobs_acc", testobs_acc.detach().cpu().numpy())
        sys.stdout.flush()

print("step", step, "add causalrep_res")
res = pd.concat([pd.DataFrame(causalrep_res, index=[0]), res], axis=1)

# compare with naive

naive_alphas, naive_trainaccs, naive_testobsaccs, naive_testctaccs = [], [], [], []
for C in [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]:
    alpha = 1./C
    print('\nnaive-pred', 'C', C)

    # clf = LinearRegression()
    # clf = Ridge(alpha=alpha)
    clf = LogisticRegression(C=C, class_weight='auto', solver='lbfgs')
    clf.fit(envs[0][flags.mode_train_data].cpu().detach().numpy(), train_y)
    resulttrain = classification_report((train_y > 0), (clf.predict(envs[0][flags.mode_train_data].cpu().detach().numpy()) > 0), output_dict=True)
    resultct = classification_report((testct_y > 0), (clf.predict(envs[1][flags.mode_train_data].cpu().detach().numpy()) > 0), output_dict=True)
    resultobs = classification_report((testobs_y > 0), (clf.predict(envs[2][flags.mode_train_data].cpu().detach().numpy())> 0), output_dict=True)
    print('train',resulttrain['accuracy'])
    print('testobs',resultobs['accuracy'])
    print('testct',resultct['accuracy'])
    sys.stdout.flush()

    naive_weights = clf.coef_
    top_naive_words = np.argsort(-np.abs(naive_weights))[0,:20]
    top_coef = naive_weights[0,top_naive_words]

    print("top naive words", [id2term[i] for i in top_naive_words], top_coef)
    naive_trainaccs.append(resulttrain['accuracy'])
    naive_testobsaccs.append(resultobs['accuracy'])
    naive_testctaccs.append(resultct['accuracy'])
    naive_alphas.append(alpha)


naive_res = {}

assert len(naive_alphas) == len(naive_trainaccs)
assert len(naive_alphas) == len(naive_testobsaccs)
assert len(naive_alphas) == len(naive_testctaccs)
for item in ['naive_trainaccs', 'naive_testobsaccs', 'naive_testctaccs']:
    for i, alpha in enumerate(naive_alphas):
        curname = item + '_' + str(alpha)
        if item == 'naive_trainaccs':
            naive_res[curname] = naive_trainaccs[i]
        elif item == 'naive_testobsaccs':
            naive_res[curname] = naive_testobsaccs[i]
        elif item == 'naive_testctaccs':
            naive_res[curname] = naive_testctaccs[i]

res = pd.concat([pd.DataFrame(naive_res, index=[0]), res], axis=1)



res.to_csv(out_dir + '/reviews_text' + str(int(time.time()*1e6)) + '.csv')
