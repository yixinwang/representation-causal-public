import argparse
import numpy as np
import pandas as pd
import time
import random
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch import nn, optim, autograd
from torch.autograd import Variable
from torchvision.utils import save_image
from scipy.ndimage.interpolation import shift


from sklearn.linear_model import LogisticRegression, Ridge
from utils import cov, compute_prob, mean_nll_mc, mean_accuracy_mc, mean_accuracy_np
from vae import VAE, vae_loss_function, train_vae, test_vae

randseed = int(time.time()*1e7%1e8)
print("random seed: ", randseed)
random.seed(randseed)
np.random.seed(randseed)
torch.manual_seed(randseed)

out_dir = 'unsupervised_out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

parser = argparse.ArgumentParser(description='Colored and Shifted MNIST Unsupervised')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_reg', type=float, default=10) #logistic 1 # linear 0.1
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--baselinesteps', type=int, default=101)
parser.add_argument('--mode', type=str, default="linear", choices=["linear", "logistic"])
parser.add_argument('--z_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--num_features', type=int, default=5)
parser.add_argument('--input_dim', type=int, default=2*14*14)
parser.add_argument('--vae_epochs', type=int, default=51)
parser.add_argument('--spurious_corr', type=float, default=0.9)
parser.add_argument('--alter_freq', type=int, default=50)
parser.add_argument('--num_train', type=int, default=10000)
flags = parser.parse_args()


print('Flags:')
for k,v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

final_trainood_accs = []
final_testood_accs = []
final_trainood_baselineaccs = []
final_testood_baselineaccs = []
final_trainood_baselinevaeaccs = []
final_testood_baselinevaeaccs = []
final_trainood_baselinecontrastiveaccs = []
final_testood_baselinecontrastiveaccs = []

final_traindownstream_accs = []
final_testdownstream_accs = []
final_traindownstream_baselineaccs = []
final_testdownstream_baselineaccs = []
final_traindownstream_baselinevaeaccs = []
final_testdownstream_baselinevaeaccs = []
final_traindownstream_baselinecontrastiveaccs = []
final_testdownstream_baselinecontrastiveaccs = []

for restart in range(flags.n_restarts):
    print("Restart", restart)

    # Load MNIST, make train/val splits, and shuffle train set examples

    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    size = 50000
    mnist_train = (mnist.data[:size], mnist.targets[:size])
    mnist_val = (mnist.data[size:], mnist.targets[size:])


    def shift_image(image, dx, dy):
        image = image.reshape((28, 28))
        shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
        return shifted_image.reshape([-1])

    ##################
    # generating data
    ##################

    print("\n##################")
    print("generating data")
    print("##################")

    def make_environment(images, labels, e):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a-b).abs() # Assumes both inputs are either 0 or 1

        subset = [(labels==1) | (labels==8)]
        images = images[subset]
        labels = labels[subset]

        X_train = images
        y_train = labels
        idx_train = torch.arange(len(y_train))

        X_train_augmented = [image.reshape([-1]) for image in images]
        y_train_augmented = [label for label in labels]
        id_train_augmented = [idx for idx in idx_train]

        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            for image, label, idx in zip(X_train, y_train, idx_train):
                 X_train_augmented.append(torch.from_numpy(shift_image(image, dx, dy)))
                 y_train_augmented.append(label)
                 id_train_augmented.append(idx)

        images = torch.stack(X_train_augmented,dim=0).long()
        labels = torch.Tensor(y_train_augmented).long()
        idxs = torch.Tensor(id_train_augmented).long()

        images = images.reshape((-1, 28, 28))[:, ::2, ::2]

        # above, 2x subsample for computational convenience

        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (labels == 1).float()
        labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
        images = images.view(-1, flags.input_dim)
        return {
            'images': (images.float() / 255.).cuda(),
            'labels': labels[:, None].cuda(),
            'colors': colors[:, None].cuda(),
            'idxs': idxs[:, None].cuda()
        }


    num_train = flags.num_train

    envs = [
        make_environment(mnist_train[0][:num_train], mnist_train[1][:num_train], 1-flags.spurious_corr),
        make_environment(mnist_val[0][:num_train], mnist_val[1][:num_train], 0.9)
    ]

    num_train_ids = torch.max(envs[0]['idxs'].max(), envs[1]['idxs'].max()) + 1
    print(num_train_ids) # number of different training ids = number of different output classes


    ##################
    # set up representation functions
    ##################

    print("\n##################")
    print("set up representation functions")
    print("##################")

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.num_features = flags.num_features
            lin1 = nn.Linear(flags.input_dim, flags.hidden_dim)
            lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
            lin3 = nn.Linear(flags.hidden_dim, self.num_features)
            lin4 = nn.Linear(flags.z_dim, flags.hidden_dim)
            lin5 = nn.Linear(flags.hidden_dim, 1)
            for i,lin in enumerate([lin1, lin2, lin3, lin4, lin5]):
                nn.init.xavier_uniform_(lin.weight, 1.)
                nn.init.zeros_(lin.bias)
                print("layer", i, lin.weight.abs().mean())
                # initialization to be too larg values will create optimization problems
                while lin.weight.abs().mean() > 0.1:
                    nn.init.xavier_uniform_(lin.weight, 1.)
                    print("layer", i, lin.weight.abs().mean())
            self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3, nn.ReLU(True), nn.BatchNorm1d(flags.num_features, affine=False))
            # _tvaez maps the VAE latent to one-dimensional outcome
            # for classification
            self._tvaez = nn.Sequential(lin4, nn.ReLU(True), lin5, nn.ReLU(True)) 
            # self.betas = torch.zeros([self.num_features+1, 1]).cuda()
            self.finallayer = nn.Linear(flags.num_features + 1, num_train_ids.item())
        def forward(self, input, vaez):
            features = self._main(input)
            logits = self.finallayer(torch.cat([features, self._tvaez(vaez)],dim=1))
            probs = compute_prob(logits, mode=flags.mode)
            return features, logits, probs

    class contrastiveMLP(nn.Module):
        def __init__(self):
            super(contrastiveMLP, self).__init__()
            self.num_features = flags.num_features
            lin1 = nn.Linear(flags.input_dim, flags.hidden_dim)
            lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
            lin3 = nn.Linear(flags.hidden_dim, self.num_features)
            for i,lin in enumerate([lin1, lin2, lin3]):
                nn.init.xavier_uniform_(lin.weight, 1.)
                nn.init.zeros_(lin.bias)
                print("layer", i, lin.weight.abs().mean())
                # initialization to be too larg values will create optimization problems
                while lin.weight.abs().mean() > 0.1:
                    nn.init.xavier_uniform_(lin.weight, 1.)
                    print("layer", i, lin.weight.abs().mean())
            self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3, nn.ReLU(True), nn.BatchNorm1d(flags.num_features, affine=False))
            # _tvaez maps the VAE latent to one-dimensional outcome
            # for classification
            # self._tvaez = nn.Sequential(lin4, nn.ReLU(True), lin5, nn.ReLU(True)) 
            # self.betas = torch.zeros([self.num_features+1, 1]).cuda()
            self.finallayer = nn.Linear(flags.num_features, num_train_ids.item())
        def forward(self, input):
            features = self._main(input)
            logits = self.finallayer(torch.cat([features],dim=1))
            probs = compute_prob(logits, mode=flags.mode)
            return features, logits, probs


    class baselineMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_features = flags.num_features
            lin1 = nn.Linear(flags.input_dim, flags.hidden_dim)
            lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
            lin3 = nn.Linear(flags.hidden_dim, self.num_features)
            for i,lin in enumerate([lin1, lin2, lin3]):
                nn.init.xavier_uniform_(lin.weight, 1.)
                nn.init.zeros_(lin.bias)
                print("baseline layer", i, lin.weight.abs().mean())
                # initialization to be too larg values will create optimization problems
                while lin.weight.abs().mean() > 0.1:
                    nn.init.xavier_uniform_(lin.weight, 1.)
                    print("layer", i, lin.weight.abs().mean())
            self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3, nn.ReLU(True), nn.BatchNorm1d(flags.num_features, affine=False))
            self.finallayer = nn.Linear(flags.num_features,  num_train_ids.item())
        def forward(self, input):
            features = self._main(input)
            logits = self.finallayer(features)
            probs = compute_prob(logits, mode="logistic")
            return features, logits, probs

    class baselinevaeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_features = flags.num_features
            lin4 = nn.Linear(flags.z_dim, flags.hidden_dim)
            lin5 = nn.Linear(flags.hidden_dim,  num_train_ids.item())
            for i,lin in enumerate([lin4, lin5]):
                nn.init.xavier_uniform_(lin.weight, 1.)
                nn.init.zeros_(lin.bias)
                print("baseline layer", i, lin.weight.abs().mean())
                # initialization to be too large values will create optimization problems
                while lin.weight.abs().mean() > 0.1:
                    nn.init.xavier_uniform_(lin.weight, 1.)
                    print("layer", i, lin.weight.abs().mean())
            self._tvaez = nn.Sequential(lin4, nn.ReLU(True), lin5, nn.ReLU(True))
            self.finallayer = nn.Linear(num_train_ids.item(),  num_train_ids.item())
        def forward(self, vaez):
            features = self._tvaez(vaez)
            logits = self.finallayer(features)
            probs = compute_prob(logits, mode="logistic")
            return features, logits, probs


    # neural network for downstream prediction task
    class testpredNet(nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(testpredNet, self).__init__()
            self.hidden = nn.Linear(n_feature, n_hidden)   # hidden layer
            self.predict = nn.Linear(n_hidden, n_output)   # output layer
            for lin in [self.hidden, self.predict]:
                nn.init.xavier_uniform_(lin.weight, 0.01)
                nn.init.zeros_(lin.bias)

        def forward(self, x):
            x = nn.ReLU(True)(self.hidden(x))      # activation function for hidden layer
            x = self.predict(x)             # linear output
            return x

    baselinemlp = baselineMLP().cuda()
    optimizer_baselinenll = optim.Adam(baselinemlp.parameters(), lr=flags.lr, weight_decay=0.1*flags.l2_reg)


    baselinetestprednet = testpredNet(n_feature=flags.num_features, n_hidden=128, n_output=1).cuda()     # define the network
    # print(net)  # net architecture
    optimizer_baselinetestpred = torch.optim.SGD(baselinetestprednet.parameters(), lr=10*flags.lr)

    loss_func = torch.nn.BCELoss() 

    baselinepred = LogisticRegression(C=0.01)

    ##################
    # baseline: naive regression for instance discrimination
    ##################

    print("\n##################")
    print("baseline")
    print("##################")

    for step in range(flags.baselinesteps):
        for i in range(len(envs)):
            env = envs[i]
            baselinefeatures, baselinelogits, baselineprobs = baselinemlp(env['images'])
            labels = env['idxs']
            env['baselinenll'] = mean_nll_mc(baselineprobs, env['idxs'], num_train_ids, mode="logistic")
            env['baselineacc'] = mean_accuracy_mc(baselineprobs, env['idxs'], num_train_ids)

            if i == 0: # training data
                baselinepred.fit(baselinefeatures.detach().cpu().numpy(), env['labels'].detach().cpu().numpy().T[0])

            env['skl_predacc'] = mean_accuracy_np(baselinepred.predict(baselinefeatures.detach().cpu().numpy()), env['labels'].detach().cpu().numpy().T[0])

        train_baselinenll = torch.stack([envs[0]['baselinenll']]).mean() 
        test_baselinenll = torch.stack([envs[1]['baselinenll']]).mean() 
        train_baselineacc = torch.stack([envs[0]['baselineacc']]).mean() 
        test_baselineacc = torch.stack([envs[1]['baselineacc']]).mean() 
        train_baselineskl_predacc = envs[0]['skl_predacc']
        test_baselineskl_predacc = envs[1]['skl_predacc']

        # skl_predacc is train on train, and test on testct

        # cf downstream is train on half of testobs, and test on half of test obs

        baselinenll_loss = train_baselinenll.clone() 
        # + train_l2penalty.clone()
        optimizer_baselinenll.zero_grad()
        baselinenll_loss.backward(retain_graph=True)
        optimizer_baselinenll.step()


        # below, currently only predict using the learned representation, consider including vaez too.
        baselinetestpred_trainX, baselinetestpred_trainy = baselinemlp(envs[1]['images'][::2])[0], envs[1]['labels'][::2]
        baselinetestpred_testX, baselinetestpred_testy = baselinemlp(envs[1]['images'][1::2])[0], envs[1]['labels'][1::2]
        baselineprediction = baselinetestprednet(baselinetestpred_trainX)     # input x and predict based on x
        baselineloss = loss_func(nn.Sigmoid()(baselineprediction), baselinetestpred_trainy)     # must be (1. nn output, 2. target)


        for t in range(200):
            optimizer_baselinetestpred.zero_grad()   # clear gradients for next train
            baselineloss.backward(retain_graph=True)         # backpropagation, compute gradients
            optimizer_baselinetestpred.step()        # apply gradients

        # downstream task; predict on a task where color is useless
        # we have to pick up the digit feature to succeed
        testpred_baselinetrainacc = mean_accuracy_mc(nn.Sigmoid()(baselinetestprednet(baselinetestpred_trainX)), envs[1]['labels'][::2], 2)
        testpred_baselinetestacc = mean_accuracy_mc(nn.Sigmoid()(baselinetestprednet(baselinetestpred_testX)), envs[1]['labels'][1::2], 2)


        if step % 10 == 0:
            print("itr", np.int32(step),
            "train_baselinenll", train_baselinenll.detach().cpu().numpy(),
            "train_baselineacc", train_baselineacc.detach().cpu().numpy(),
            "train_baselineskl_predacc", train_baselineskl_predacc,
            "testpred_baselinetrainacc", testpred_baselinetrainacc.detach().cpu().numpy(),
            "test_baselinenll", train_baselinenll.detach().cpu().numpy(),
            "test_baselineacc", test_baselineacc.detach().cpu().numpy(),
            "test_baselineskl_predacc", test_baselineskl_predacc,
            "testpred_baselinetestacc", testpred_baselinetestacc.detach().cpu().numpy())

    final_trainood_baselineaccs.append(train_baselineskl_predacc)
    final_testood_baselineaccs.append(test_baselineskl_predacc)
    final_traindownstream_baselineaccs.append(train_baselineacc.detach().cpu().numpy().item())
    final_testdownstream_baselineaccs.append(test_baselineacc.detach().cpu().numpy().item())



    ############
    # fit VAE
    ############
    print("fit VAE")

    train_loader = torch.utils.data.DataLoader(dataset=envs[0]['images'].view(-1, flags.input_dim), batch_size=flags.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=envs[1]['images'].view(-1, flags.input_dim), batch_size=flags.batch_size, shuffle=False)

    # build model
    vae = VAE(x_dim=flags.input_dim, h_dim1=flags.hidden_dim, h_dim2=flags.hidden_dim, z_dim=flags.z_dim)
    if torch.cuda.is_available():
            vae.cuda()

    optimizer_vae = optim.Adam(vae.parameters())

    for epoch in range(1, flags.vae_epochs):
        train_vae(vae, train_loader, optimizer_vae, epoch)
        test_vae(vae, test_loader)

    train_vae_recon, train_vae_mu, train_vae_logvar = vae(envs[0]['images'].view(-1, flags.input_dim))
    test_vae_recon, test_vae_mu, test_vae_logvar = vae(envs[1]['images'].view(-1, flags.input_dim))


    envs[0]['vaez'] = train_vae_mu.detach()
    envs[1]['vaez'] = test_vae_mu.detach()


    ##################
    # baselinevae (use vae features only)
    ##################

    print("\n##################")
    print("baseline VAE")
    print("##################")

    baselinevaemlp = baselinevaeMLP().cuda()
    optimizer_baselinevaenll = optim.Adam(baselinevaemlp.parameters(), lr=flags.lr, weight_decay=0.1*flags.l2_reg)

    baselinevaetestprednet = testpredNet(n_feature=num_train_ids.item(), n_hidden=128, n_output=1).cuda()     # define the network
    # print(net)  # net architecture
    optimizer_baselinevaetestpred = torch.optim.SGD(baselinevaetestprednet.parameters(), lr=10*flags.lr)


    baselinevaepred = LogisticRegression(C=0.01)

    for step in range(flags.baselinesteps):
        for i in range(len(envs)):
            env = envs[i]
            baselinevaefeatures, baselinevaelogits, baselinevaeprobs = baselinevaemlp(env['vaez'])
            labels = env['idxs']
            env['baselinevaenll'] = mean_nll_mc(baselinevaeprobs, env['idxs'], num_train_ids, mode="logistic")
            env['baselinevaeacc'] = mean_accuracy_mc(baselinevaeprobs, env['idxs'], num_train_ids)

            if i == 0: # training data
                baselinevaepred.fit(env['vaez'].detach().cpu().numpy(), env['labels'].detach().cpu().numpy().T[0])

            env['skl_predacc'] = mean_accuracy_np(baselinevaepred.predict(env['vaez'].detach().cpu().numpy()), env['labels'].detach().cpu().numpy().T[0])

        train_baselinevaenll = torch.stack([envs[0]['baselinevaenll']]).mean() 
        test_baselinevaenll = torch.stack([envs[1]['baselinevaenll']]).mean() 
        train_baselinevaeacc = torch.stack([envs[0]['baselinevaeacc']]).mean() 
        test_baselinevaeacc = torch.stack([envs[1]['baselinevaeacc']]).mean() 
        train_baselinevaeskl_predacc = envs[0]['skl_predacc']
        test_baselinevaeskl_predacc = envs[1]['skl_predacc']

        baselinevaenll_loss = train_baselinevaenll.clone() 
        # + train_l2penalty.clone()
        optimizer_baselinevaenll.zero_grad()
        baselinevaenll_loss.backward(retain_graph=True)
        optimizer_baselinevaenll.step()


        # below, currently only predict using the learned representation, consider including vaez too.
        baselinevaetestpred_trainX, baselinevaetestpred_trainy = baselinevaemlp(envs[1]['vaez'][::2])[0], envs[1]['labels'][::2]
        baselinevaetestpred_testX, baselinevaetestpred_testy = baselinevaemlp(envs[1]['vaez'][1::2])[0], envs[1]['labels'][1::2]
        baselinevaeprediction = baselinevaetestprednet(baselinevaetestpred_trainX)     # input x and predict based on x
        baselinevaeloss = loss_func(nn.Sigmoid()(baselinevaeprediction), baselinevaetestpred_trainy)     # must be (1. nn output, 2. target)


        for t in range(200):
            optimizer_baselinevaetestpred.zero_grad()   # clear gradients for next train
            baselinevaeloss.backward(retain_graph=True)         # backpropagation, compute gradients
            optimizer_baselinevaetestpred.step()        # apply gradients

        # downstream task; predict on a task where color is useless
        # we have to pick up the digit feature to succeed
        testpred_baselinevaetrainacc = mean_accuracy_mc(nn.Sigmoid()(baselinevaetestprednet(baselinevaetestpred_trainX)), envs[1]['labels'][::2], 2)
        testpred_baselinevaetestacc = mean_accuracy_mc(nn.Sigmoid()(baselinevaetestprednet(baselinevaetestpred_testX)), envs[1]['labels'][1::2], 2)


        if step % 10 == 0:
            print("itr", np.int32(step),
            "train_baselinevaenll", train_baselinevaenll.detach().cpu().numpy(),
            "train_baselinevaeacc", train_baselinevaeacc.detach().cpu().numpy(),
            "train_baselinevaeskl_predacc", train_baselinevaeskl_predacc,
            "testpred_baselinevaetrainacc", testpred_baselinevaetrainacc.detach().cpu().numpy(),
            "test_baselinevaenll", train_baselinevaenll.detach().cpu().numpy(),
            "test_baselinevaeacc", test_baselinevaeacc.detach().cpu().numpy(),
            "test_baselinevaeskl_predacc", test_baselinevaeskl_predacc,
            "testpred_baselinevaetestacc", testpred_baselinevaetestacc.detach().cpu().numpy())

    final_trainood_baselinevaeaccs.append(train_baselinevaeskl_predacc)
    final_testood_baselinevaeaccs.append(test_baselinevaeskl_predacc)
    final_traindownstream_baselinevaeaccs.append(train_baselinevaeacc.detach().cpu().numpy().item())
    final_testdownstream_baselinevaeaccs.append(test_baselinevaeacc.detach().cpu().numpy().item())

    ##################
    # causal_REP
    ##################

    print("\n##################")
    print("CAUSAL REP")
    print("##################")

    mlp = MLP().cuda()

    testprednet = testpredNet(n_feature=flags.num_features, n_hidden=128, n_output=1).cuda()     # define the network
    # print(net)  # net architecture
    optimizer_testpred = torch.optim.SGD(testprednet.parameters(), lr=10*flags.lr)
    loss_func = torch.nn.BCELoss() 


    # optimizer = optim.SGD(mlp.parameters(), lr=flags.lr)
    optimizer_nll = optim.Adam(list(mlp._tvaez.parameters()) + list(mlp.finallayer.parameters()), lr=10*flags.lr)
    optimizer_causalrep = optim.Adam(mlp._main.parameters(), lr=10*flags.lr)
    pred = LogisticRegression(C=0.001)


    for step in range(flags.steps):
        for i in range(len(envs)):
            env = envs[i]
            features, logits, probs = mlp(env['images'], env['vaez'])
            labels = env['idxs']
            env['nll'] = mean_nll_mc(probs, env['idxs'], num_train_ids, mode=flags.mode) 
            env['acc'] = mean_accuracy_mc(probs, env['idxs'], num_train_ids)

            # random sample one of the outcomes
            sample_out = 10
            outidxs = torch.randperm(num_train_ids)[:sample_out]
            causalrep_is = torch.zeros(sample_out).cuda()
            for out_i, outidx in enumerate(outidxs):
                env['covs'] = (2 * cov(torch.cat([features * mlp.finallayer.weight[outidx,:flags.num_features], torch.unsqueeze((mlp.finallayer.weight[outidx, flags.num_features:] * mlp._tvaez(env['vaez'])).sum(axis=1), 1)], dim=1))[-1][:-1]).sum()

                causalrep_is[out_i] = ((features.std(dim=0) * mlp.finallayer.weight[outidx,:flags.num_features])**2).sum() 

                # + env['covs'].sum()
            env['causalrep'] = causalrep_is.sum()



            # one can learn highly correlated features in the representations too, which can be an issue.

            y = features - features.mean(dim=0)
            X = torch.unsqueeze((mlp._tvaez(env['vaez']) - mlp._tvaez(env['vaez']).mean())[:,0], 1)
            beta = [torch.matmul(
                torch.matmul(
                    torch.inverse(1e-2*torch.eye(X.shape[1]).cuda()+
                        torch.matmul(
                            torch.transpose(X, 0, 1),
                            X)),
                    torch.transpose(X, 0, 1)),
                y[:,j]) for j in range(y.shape[1])]
            r2s = torch.Tensor([1 - (((X*(beta[j])).T[0] - y[:,j])**2).sum() / (y[:,j]**2 + 1e-8).sum() for j in range(y.shape[1])]).mean()

            env['featureZr2'] = r2s.cuda()

            weight_norm = torch.tensor(0.).cuda()
            for w in mlp.parameters():
                weight_norm += w.norm().pow(2)

            env['l2penalty'] = flags.l2_reg * weight_norm


            if i == 0: # training data
                pred.fit(features.detach().cpu().numpy(), env['labels'].detach().cpu().numpy().T[0])

            env['skl_predacc'] = mean_accuracy_np(pred.predict(features.detach().cpu().numpy()), env['labels'].detach().cpu().numpy().T[0])

            if step % flags.alter_freq == 0:
                print("\nnll", env['nll'], 
                    "\nl2", env['l2penalty'], 
                    "\ncausalrep", env['causalrep'], 
                    "\nfeatureZr2", env['featureZr2'])


        # below, currently only predict using the learned representation, consider including vaez too.
        testpred_trainX, testpred_trainy = mlp(envs[1]['images'][::2], envs[1]['vaez'][::2])[0], envs[1]['labels'][::2]
        testpred_testX, testpred_testy = mlp(envs[1]['images'][1::2], envs[1]['vaez'][1::2])[0], envs[1]['labels'][1::2]
        prediction = testprednet(testpred_trainX)     # input x and predict based on x
        loss = loss_func(nn.Sigmoid()(prediction), testpred_trainy)     # must be (1. nn output, 2. target)

        for t in range(200):
            optimizer_testpred.zero_grad()   # clear gradients for next train
            loss.backward(retain_graph=True)         # backpropagation, compute gradients
            optimizer_testpred.step()        # apply gradients

        train_l2penalty = torch.stack([envs[0]['l2penalty']])
        train_causalrep = torch.stack([envs[0]['causalrep']])
        train_featureZr2 = torch.stack([envs[0]['featureZr2']])
        train_nll = torch.stack([envs[0]['nll']]).mean() 
        train_acc = torch.stack([envs[0]['acc']]).mean()
        train_skl_predacc = envs[0]['skl_predacc']
        test_nll = torch.stack([envs[1]['nll']]).mean()
        test_acc = torch.stack([envs[1]['acc']]).mean()
        test_featureZr2 = torch.stack([envs[1]['featureZr2']])
        test_skl_predacc = envs[1]['skl_predacc']


        # downstream task; predict on a task where color is useless we
        # have to pick up the digit feature to succeed

        # split the observational test data in half. train on half and
        # predict on the other half.
        testpred_trainacc = mean_accuracy_mc(nn.Sigmoid()(testprednet(testpred_trainX)), envs[1]['labels'][::2], 2)
        testpred_testacc = mean_accuracy_mc(nn.Sigmoid()(testprednet(testpred_testX)), envs[1]['labels'][1::2], 2)

        nll_loss = train_nll.clone()
         # + train_l2penalty.clone() 

        optimizer_nll.zero_grad()
        nll_loss.backward(retain_graph=True)
        optimizer_nll.step()

        test_acc = envs[1]['acc']
        if step % flags.alter_freq == 0:
            train_causalrep_loss = -train_causalrep.clone() - 1e-4* torch.log(1 - train_featureZr2)
            optimizer_causalrep.zero_grad()
            train_causalrep_loss.backward()
            optimizer_causalrep.step()

        if step % 10 == 0:
            print("itr", np.int32(step),
            "train_causalrep", train_causalrep.detach().cpu().numpy(), 
            "nll_loss", nll_loss.detach().cpu().numpy(),
            "train_nll", train_nll.detach().cpu().numpy(),
            "train_acc", train_acc.detach().cpu().numpy(),
            "test_acc", test_acc.detach().cpu().numpy(),
            "train_skl_predacc", train_skl_predacc,
            "test_skl_predacc", test_skl_predacc,
            "train_featureZr2", train_featureZr2.detach().cpu().numpy(),
            "test_featureZr2", test_featureZr2.detach().cpu().numpy())

    final_trainood_accs.append(train_skl_predacc)
    final_testood_accs.append(test_skl_predacc)
    final_traindownstream_accs.append(train_acc.detach().cpu().numpy().item())
    final_testdownstream_accs.append(test_acc.detach().cpu().numpy().item())


    ##################
    # contrastive learning
    ##################
    print("\n##################")
    print("contrastive learning")
    print("\n##################")
    

    tau = 2

    contrastive_mlp = contrastiveMLP().cuda() # representation function

    contrastive_testprednet = testpredNet(n_feature=flags.num_features, n_hidden=128, n_output=1).cuda()     # define the network
    # print(net)  # net architecture
    optimizer_contrastive_testpred = torch.optim.SGD(contrastive_testprednet.parameters(), lr=10*flags.lr)


    # optimizer = optim.SGD(mlp.parameters(), lr=flags.lr)
    optimizer_contrastive_nll = optim.Adam(list(contrastive_mlp.finallayer.parameters()), lr=flags.lr, weight_decay=0.1)
    optimizer_contrastive = optim.Adam(contrastive_mlp._main.parameters(), lr=flags.lr, weight_decay=0.1)
    contrastive_pred = LogisticRegression(C=0.01)



    for step in range(flags.steps):
        # if step % 10 == 0:
            # print('\n', step)
        for i in range(len(envs)):
            env = envs[i]
            features, logits, probs = contrastive_mlp(env['images'])
            labels = env['idxs']
            env['nll'] = mean_nll_mc(probs, env['idxs'], num_train_ids, mode=flags.mode) 
            env['acc'] = mean_accuracy_mc(probs, env['idxs'], num_train_ids)
            subset_idx = torch.randperm(len(labels))[:100]
            subsetlabels = labels[subset_idx]
            subsetfeatures = features[subset_idx]

            sim_examples_feas = torch.stack([features[torch.where(labels == label)[0][torch.randperm(len(torch.where(labels == label.item())[0]))[0]]] for label in subsetlabels])

            dissim_examples_feass = []
            # randomly select num_neg negative examples
            num_neg = 5
            for _ in range(5):
                dissim_examples_feass.append(torch.stack([features[torch.where(labels != label)[0][torch.randperm(len(torch.where(labels != label)[0]))[0]]] for label in subsetlabels]))

            contrastive_numerator = torch.exp((subsetfeatures * sim_examples_feas).sum(axis=1) / tau)
            contrastive_denominator = torch.stack([torch.exp((subsetfeatures * dissim_examples_feas).sum(axis=1) / tau) for dissim_examples_feas in dissim_examples_feass]).sum(axis=0)
            env['contrastive_loss'] = -torch.log(contrastive_numerator / contrastive_denominator).sum()

            contrastive_weight_norm = torch.tensor(0.).cuda()
            for w in contrastive_mlp.parameters():
                contrastive_weight_norm += w.norm().pow(2)

            env['l2penalty'] = flags.l2_reg * contrastive_weight_norm


            if i == 0: # training data
                pred.fit(features.detach().cpu().numpy(), env['labels'].detach().cpu().numpy().T[0])

            env['skl_predacc'] = mean_accuracy_np(pred.predict(features.detach().cpu().numpy()), env['labels'].detach().cpu().numpy().T[0])

            if step % flags.alter_freq == 0:
                print(
                    "\nl2", env['l2penalty'], 
                    "\ncontrastive", env['contrastive_loss']
                    )


        # below, currently only predict using the learned representation, consider including vaez too.
        contrastive_testpred_trainX, contrastive_testpred_trainy = contrastive_mlp(envs[1]['images'][::2])[0], envs[1]['labels'][::2]
        contrastive_testpred_testX, contrastive_testpred_testy = contrastive_mlp(envs[1]['images'][1::2])[0], envs[1]['labels'][1::2]
        contrastive_prediction = contrastive_testprednet(contrastive_testpred_trainX)     # input x and predict based on x
        contrastivepred_loss = loss_func(nn.Sigmoid()(contrastive_prediction), contrastive_testpred_trainy)     # must be (1. nn output, 2. target)

        for t in range(200):
            optimizer_contrastive_testpred.zero_grad()   # clear gradients for next train
            contrastivepred_loss.backward(retain_graph=True)         # backpropagation, compute gradients
            optimizer_contrastive_testpred.step()        # apply gradients


        contrastive_train_l2penalty = torch.stack([envs[0]['l2penalty']])
        train_contrastive_loss = torch.stack([envs[0]['contrastive_loss']])
        contrastive_train_nll = torch.stack([envs[0]['nll']]).mean() 
        contrastive_train_acc = torch.stack([envs[0]['acc']]).mean()
        contrastive_train_skl_predacc = envs[0]['skl_predacc']
        contrastive_test_nll = torch.stack([envs[1]['nll']]).mean()
        contrastive_test_acc = torch.stack([envs[1]['acc']]).mean()
        contrastive_test_skl_predacc = envs[1]['skl_predacc']


        # downstream task; predict on a task where color is useless we
        # have to pick up the digit feature to succeed

        # split the observational test data in half. train on half and
        # predict on the other half.
        contrastive_testpred_trainacc = mean_accuracy_mc(nn.Sigmoid()(testprednet(contrastive_testpred_trainX)), envs[1]['labels'][::2], 2)
        contrastive_testpred_testacc = mean_accuracy_mc(nn.Sigmoid()(testprednet(contrastive_testpred_testX)), envs[1]['labels'][1::2], 2)

        contrastive_nll_loss = contrastive_train_nll.clone()
         # + train_l2penalty.clone() 

        optimizer_contrastive_nll.zero_grad()
        contrastive_nll_loss.backward(retain_graph=True)
        optimizer_contrastive_nll.step()

        contrastive_test_acc = envs[1]['acc']

        if step % flags.alter_freq == 0:
            train_contrastive = train_contrastive_loss.clone() 

            optimizer_contrastive.zero_grad()
            train_contrastive.backward()
            optimizer_contrastive.step()

        if step % 10 == 0:
            print("itr", np.int32(step),
            "train_contrastive", train_contrastive.detach().cpu().numpy(), 
            "nll_loss", contrastive_nll_loss.detach().cpu().numpy(),
            "train_nll", contrastive_train_nll.detach().cpu().numpy(),
            "train_acc", contrastive_train_acc.detach().cpu().numpy(),
            "test_acc", contrastive_test_acc.detach().cpu().numpy(),
            "train_skl_predacc", contrastive_train_skl_predacc,
            "test_skl_predacc", contrastive_test_skl_predacc)

    final_trainood_baselinecontrastiveaccs.append(contrastive_train_skl_predacc)
    final_testood_baselinecontrastiveaccs.append(contrastive_test_skl_predacc)
    final_traindownstream_baselinecontrastiveaccs.append(contrastive_train_acc.detach().cpu().numpy().item())
    final_testdownstream_baselinecontrastiveaccs.append(contrastive_test_acc.detach().cpu().numpy().item())


# skl_predacc is train on train, and test on testct

# cf downstream is train on half of testobs, and test on half of test obs


print('Final trainood baseline acc (mean/std across restarts so far):')
print(np.mean(final_trainood_baselineaccs), np.std(final_trainood_baselineaccs))
print('Final testood baseline acc (mean/std across restarts so far):')
print(np.mean(final_testood_baselineaccs), np.std(final_testood_baselineaccs))

print('Final trainood baselinevae acc (mean/std across restarts so far):')
print(np.mean(final_trainood_baselinevaeaccs), np.std(final_trainood_baselinevaeaccs))
print('Final testood baselinevae acc (mean/std across restarts so far):')
print(np.mean(final_testood_baselinevaeaccs), np.std(final_testood_baselinevaeaccs))

print('Final trainood baselinecontrastive acc (mean/std across restarts so far):')
print(np.mean(final_trainood_baselinecontrastiveaccs), np.std(final_trainood_baselinecontrastiveaccs))
print('Final testood baselinecontrastive acc (mean/std across restarts so far):')
print(np.mean(final_testood_baselinecontrastiveaccs), np.std(final_testood_baselinecontrastiveaccs))

print('Final trainood acc (mean/std across restarts so far):')
print(np.mean(final_trainood_accs), np.std(final_trainood_accs))
print('Final testood acc (mean/std across restarts so far):')
print(np.mean(final_testood_accs), np.std(final_testood_accs))

print(final_trainood_baselineaccs)
print(final_testood_baselineaccs)
print(final_trainood_baselinevaeaccs)
print(final_testood_baselinevaeaccs)
print(final_trainood_baselinecontrastiveaccs)
print(final_testood_baselinecontrastiveaccs)
print(final_trainood_accs)
print(final_testood_accs)



print('Final traindownstream baseline acc (mean/std across restarts so far):')
print(np.mean(final_traindownstream_baselineaccs), np.std(final_traindownstream_baselineaccs))
print('Final testdownstream baseline acc (mean/std across restarts so far):')
print(np.mean(final_testdownstream_baselineaccs), np.std(final_testdownstream_baselineaccs))

print('Final traindownstream baselinevae acc (mean/std across restarts so far):')
print(np.mean(final_traindownstream_baselinevaeaccs), np.std(final_traindownstream_baselinevaeaccs))
print('Final testdownstream baselinevae acc (mean/std across restarts so far):')
print(np.mean(final_testdownstream_baselinevaeaccs), np.std(final_testdownstream_baselinevaeaccs))

print('Final traindownstream baselinecontrastive acc (mean/std across restarts so far):')
print(np.mean(final_traindownstream_baselinecontrastiveaccs), np.std(final_traindownstream_baselinecontrastiveaccs))
print('Final testdownstream baselinecontrastive acc (mean/std across restarts so far):')
print(np.mean(final_testdownstream_baselinecontrastiveaccs), np.std(final_testdownstream_baselinecontrastiveaccs))


print('Final traindownstream acc (mean/std across restarts so far):')
print(np.mean(final_traindownstream_accs), np.std(final_traindownstream_accs))
print('Final testdownstream acc (mean/std across restarts so far):')
print(np.mean(final_testdownstream_accs), np.std(final_testdownstream_accs))

print(final_traindownstream_baselineaccs)
print(final_testdownstream_baselineaccs)
print(final_traindownstream_baselinevaeaccs)
print(final_testdownstream_baselinevaeaccs)
print(final_traindownstream_baselinecontrastiveaccs)
print(final_testdownstream_baselinecontrastiveaccs)
print(final_traindownstream_accs)
print(final_testdownstream_accs)

if not os.path.exists("./res"):
    try:
        os.makedirs("./res", 0o700)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

outfile = out_dir + '/unsupervised_CMNIST' + str(int(time.time()*1e6)) + '.csv'

result = pd.DataFrame({
    'causalrep_trainood_accs': np.array(final_trainood_accs),
    'causalrep_testood_accs': np.array(final_testood_accs),
    'naive_trainood_accs': np.array(final_trainood_baselineaccs),
    'naive_testood_accs': np.array(final_testood_baselineaccs),
    'naive_vae_trainood_accs': np.array(final_trainood_baselinevaeaccs),
    'naive_vae_testood_accs': np.array(final_testood_baselinevaeaccs),
    'naive_contrastive_trainood_accs': np.array(final_trainood_baselinevaeaccs),
    'naive_contrastive_testood_accs': np.array(final_testood_baselinevaeaccs),
    'causalrep_traindownstream_accs': np.array(final_traindownstream_accs),
    'causalrep_testdownstream_accs': np.array(final_testdownstream_accs),
    'naive_traindownstream_accs': np.array(final_traindownstream_baselineaccs),
    'naive_testdownstream_accs': np.array(final_testdownstream_baselineaccs),
    'naive_vae_traindownstream_accs': np.array(final_traindownstream_baselinevaeaccs),
    'naive_vae_testdownstream_accs': np.array(final_testdownstream_baselinevaeaccs),
    'naive_contrastive_traindownstream_accs': np.array(final_traindownstream_baselinecontrastiveaccs),
    'naive_contrastive_testdownstream_accs': np.array(final_testdownstream_baselinecontrastiveaccs),    'hidden_dim': np.repeat(flags.hidden_dim, flags.n_restarts),
    'l2_reg': np.repeat(flags.l2_reg, flags.n_restarts),
    'lr': np.repeat(flags.lr, flags.n_restarts),
    'n_restarts': np.repeat(flags.n_restarts, flags.n_restarts),
    'mode': np.repeat(flags.mode, flags.n_restarts),
    'steps': np.repeat(flags.steps, flags.n_restarts),
    'z_dim': np.repeat(flags.z_dim, flags.n_restarts),
    'batch_size': np.repeat(flags.batch_size, flags.n_restarts),
    'num_features': np.repeat(flags.num_features, flags.n_restarts),
    'input_dim': np.repeat(flags.input_dim, flags.n_restarts),
    'vae_epochs': np.repeat(flags.vae_epochs, flags.n_restarts),
    'spurious_corr': np.repeat(flags.spurious_corr, flags.n_restarts),
    'alter_freq': np.repeat(flags.alter_freq, flags.n_restarts),
    'randseed': np.repeat(randseed, flags.n_restarts),
    })

result.to_csv(outfile)

