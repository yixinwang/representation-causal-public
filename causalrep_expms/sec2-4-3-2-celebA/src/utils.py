import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn, optim, autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image
from sklearn.linear_model import LogisticRegression, Ridge



def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()


def mean_nll_mc(probs, y, num_categories, mode="logistic"):
    # return nn.functional.binary_cross_entropy_with_logits(logits, y)
    # i=0
    # losses = nn.BCELoss()(probs[:,i], (y.T[0]==i).float())
    if num_categories > 2:
        y_onehot = F.one_hot(y.long(), num_categories)[:,0,:].float()
    else:
        y_onehot = y
    if mode == "linear":
        losses = nn.MSELoss()(probs, y_onehot)
    elif mode == "logistic":
        losses = nn.BCELoss()(probs, y_onehot)
    # print(y_onehot.shape, probs.shape, losses.shape)

    # losses = torch.Tensor([nn.BCELoss()(
    # probs[:,i], (y.T[0]==i).float()) for i in range(y.shape[1])])
    # return nn.NLLLoss()(probs, y.T[0])
    return losses.mean()

def mean_accuracy_mc(probs, y, num_categories):
    # preds = probs.argmax(dim=1)
    # return ((preds - y.T[0]).abs() < 1e-2).float().mean()
    # i=0
    preds = (probs > 0.5).float()
    # yi = (y.T[0]==i).float()
    if num_categories > 2:
        y_onehot = F.one_hot(y.long(), num_categories)[:,0,:].float()
    else:
        y_onehot = y
    # print(preds.shape, y_onehot.shape)
    return ((preds - y_onehot).abs() < 1e-2).float().mean()


def compute_prob(logits, mode="logistic"):
    if mode == "linear":
        probs = torch.max(torch.stack([logits,torch.zeros_like(logits)],dim=2),dim=2)[0]
        probs = torch.min(torch.stack([probs,torch.ones_like(probs)],dim=2),dim=2)[0]
    elif mode == "logistic":
        probs = nn.Sigmoid()(logits)
    return probs

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

def mean_accuracy_np(probs, y):
    preds = (probs > 0.5)
    return (np.abs(preds - y) < 1e-2).mean()