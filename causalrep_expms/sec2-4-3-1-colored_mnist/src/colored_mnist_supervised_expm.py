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

from sklearn.linear_model import LogisticRegression, Ridge
from utils import cov, compute_prob, mean_nll, mean_accuracy, mean_accuracy_np
from vae import VAE, vae_loss_function, train_vae, test_vae

randseed = int(time.time()*1e7%1e8)
print("random seed: ", randseed)
random.seed(randseed)
np.random.seed(randseed)
torch.manual_seed(randseed)

out_dir = 'supervised_out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

parser = argparse.ArgumentParser(description='Colored MNIST Supervised')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_reg', type=float, default=1.)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--steps', type=int, default=1001)
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--mode', type=str, default="linear", choices=["linear", "logistic"])
parser.add_argument('--z_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_features', type=int, default=20)
parser.add_argument('--input_dim', type=int, default=2*14*14)
parser.add_argument('--vae_epochs', type=int, default=101)
parser.add_argument('--spurious_corr', type=float, default=0.8)
parser.add_argument('--alter_freq', type=int, default=100)
flags = parser.parse_args()

print('Flags:')
for k,v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

final_train_accs = []
final_test_accs = []
final_train_baselineaccs = []
final_test_baselineaccs = []
final_train_baselinevaeaccs = []
final_test_baselinevaeaccs = []

for restart in range(flags.n_restarts):
    print("Restart", restart)

    # Load MNIST, make train/val splits, and shuffle train set examples

    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    size = 50000
    mnist_train = (mnist.data[:size], mnist.targets[:size])
    mnist_val = (mnist.data[size:], mnist.targets[size:])


    def make_environment(images, labels, e):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a-b).abs() # Assumes both inputs are either 0 or 1
        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]

        subset = [(labels==1) | (labels==8)]
        images = images[subset]
        labels = labels[subset]
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
            'colors': colors[:, None].cuda()
        }

    envs = [
        make_environment(mnist_train[0], mnist_train[1], 1-flags.spurious_corr),
        make_environment(mnist_val[0], mnist_val[1], 0.9)
    ]


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
            self.finallayer = nn.Linear(flags.num_features + 1, 1)
        def forward(self, input, vaez):
            features = self._main(input)
            logits = self.finallayer(torch.cat([features, self._tvaez(vaez)],dim=1))
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
            self.finallayer = nn.Linear(flags.num_features, 1)
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
            lin5 = nn.Linear(flags.hidden_dim, 1)
            for i,lin in enumerate([lin4, lin5]):
                nn.init.xavier_uniform_(lin.weight, 1.)
                nn.init.zeros_(lin.bias)
                print("baseline layer", i, lin.weight.abs().mean())
                # initialization to be too larg values will create optimization problems
                while lin.weight.abs().mean() > 0.1:
                    nn.init.xavier_uniform_(lin.weight, 1.)
                    print("layer", i, lin.weight.abs().mean())
            self._tvaez = nn.Sequential(lin4, nn.ReLU(True), lin5, nn.ReLU(True))
            self.finallayer = nn.Linear(1, 1)
        def forward(self, vaez):
            features = self._tvaez(vaez)
            logits = self.finallayer(features)
            probs = compute_prob(logits, mode="logistic")
            return features, logits, probs



    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(flags.num_features, 1)
        def forward(self, x):
            return self.fc(x)

    def initNet(layer):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)



    baselinemlp = baselineMLP().cuda()
    optimizer_baselinenll = optim.Adam(baselinemlp.parameters(), lr=flags.lr, weight_decay=0.1*flags.l2_reg)

    # baseline
    for step in range(flags.steps):
        for i in range(len(envs)):
            env = envs[i]
            baselinefeatures, baselinelogits, baselineprobs = baselinemlp(env['images'])
            labels = env['labels']
            env['baselinenll'] = mean_nll(baselineprobs, env['labels'], mode="logistic")
            env['baselineacc'] = mean_accuracy(baselineprobs, env['labels'])
        train_baselinenll = torch.stack([envs[0]['baselinenll']]).mean() 
        test_baselinenll = torch.stack([envs[1]['baselinenll']]).mean() 
        train_baselineacc = torch.stack([envs[0]['baselineacc']]).mean() 
        test_baselineacc = torch.stack([envs[1]['baselineacc']]).mean() 

        baselinenll_loss = train_baselinenll.clone() 
        # + train_l2penalty.clone()
        optimizer_baselinenll.zero_grad()
        baselinenll_loss.backward(retain_graph=True)
        optimizer_baselinenll.step()

        if step % 10 == 0:
            print("itr", np.int32(step),
            "train_baselinenll", train_baselinenll.detach().cpu().numpy(),
            "train_baselineacc", train_baselineacc.detach().cpu().numpy(),
            "test_baselinenll", test_baselinenll.detach().cpu().numpy(),
            "test_baselineacc", test_baselineacc.detach().cpu().numpy())


    final_train_baselineaccs.append(train_baselineacc.detach().cpu().numpy().item())
    final_test_baselineaccs.append(test_baselineacc.detach().cpu().numpy().item())


    # fit VAE

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

    # baselinevae (use vae features only)

    baselinevaemlp = baselinevaeMLP().cuda()
    optimizer_baselinevaenll = optim.Adam(baselinevaemlp.parameters(), lr=flags.lr, weight_decay=0.1*flags.l2_reg)

    for step in range(flags.steps):
        for i in range(len(envs)):
            env = envs[i]
            baselinevaefeatures, baselinevaelogits, baselinevaeprobs = baselinevaemlp(env['vaez'])
            labels = env['labels']
            env['baselinevaenll'] = mean_nll(baselinevaeprobs, env['labels'], mode="logistic")
            env['baselinevaeacc'] = mean_accuracy(baselinevaeprobs, env['labels'])
        train_baselinevaenll = torch.stack([envs[0]['baselinevaenll']]).mean() 
        test_baselinevaenll = torch.stack([envs[1]['baselinevaenll']]).mean() 
        train_baselinevaeacc = torch.stack([envs[0]['baselinevaeacc']]).mean() 
        test_baselinevaeacc = torch.stack([envs[1]['baselinevaeacc']]).mean() 

        baselinevaenll_loss = train_baselinevaenll.clone() 
        # + train_l2penalty.clone()
        optimizer_baselinevaenll.zero_grad()
        baselinevaenll_loss.backward(retain_graph=True)
        optimizer_baselinevaenll.step()

        if step % 10 == 0:
            print("itr", np.int32(step),
            "train_baselinevaenll", train_baselinevaenll.detach().cpu().numpy(),
            "train_baselinevaeacc", train_baselinevaeacc.detach().cpu().numpy(),
            "test_baselinevaenll", train_baselinevaenll.detach().cpu().numpy(),
            "test_baselinevaeacc", test_baselinevaeacc.detach().cpu().numpy())


    final_train_baselinevaeaccs.append(train_baselinevaeacc.detach().cpu().numpy().item())
    final_test_baselinevaeaccs.append(test_baselinevaeacc.detach().cpu().numpy().item())

    # causal_REP

    mlp = MLP().cuda()
    net = Net().cuda() # final classification net

    optimizer_net = optim.Adam(net.parameters(), lr=flags.lr, weight_decay=flags.l2_reg)

    optimizer_nll = optim.Adam(list(mlp._tvaez.parameters()) + list(mlp.finallayer.parameters()), lr=10*flags.lr, weight_decay=flags.l2_reg)
    optimizer_causalrep = optim.Adam(mlp._main.parameters(), lr=flags.lr, weight_decay=1e-6)
    pred = LogisticRegression(C=0.01)


    print("weight mean", mlp._main[0].weight.abs().mean())


    for step in range(flags.steps):
        # if step % 10 == 0:
            # print('\n', step)

        for i in range(len(envs)):
            
            env = envs[i]
            features, logits, probs = mlp(env['images'], env['vaez'])
            labels = env['labels']
            vaez = env['vaez']
            env['nll'] = mean_nll(probs, env['labels'], mode=flags.mode) 
            env['acc'] = mean_accuracy(probs, env['labels'])

            env['covs'] = cov( torch.cat([features, mlp._tvaez(env['vaez'])], dim=1))[-1][:-1]

            env['causalrep'] = ((features.std(dim=0) * mlp.finallayer.weight[0][:flags.num_features])**2).sum()

                # + 2 * mlp.finallayer.weight[0][flags.num_features:] * mlp.finallayer.weight[0][:flags.num_features] * env['covs']).sum()

            # one can learn highly correlated features in the representations too, which can be an issue.

            y = features - features.mean(dim=0)
            X = mlp._tvaez(env['vaez']) - mlp._tvaez(env['vaez']).mean()
            beta = [torch.matmul(
                torch.matmul(
                    torch.inverse(1e-8*torch.eye(X.shape[1]).cuda()+
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
                pred.fit(features.detach().cpu().numpy(), labels.detach().cpu().numpy().T[0])

            if step % 100 == 0:
                if i == 0: # training data
                    initNet(net.fc)
                    running_loss = 0.0
                    loss_net = nn.BCELoss()(nn.Sigmoid()(net(features)), labels)
                    loss_net.backward(retain_graph=True)
                    for _ in range(1000):
                        optimizer_net.step()
            env['predacc'] = mean_accuracy(nn.Sigmoid()(net(features)), labels)

            env['skl_predacc'] = mean_accuracy_np(pred.predict(features.detach().cpu().numpy()), labels.detach().cpu().numpy().T[0])

            if step % flags.alter_freq == 0:
                print("\nnll", env['nll'], 
                    "\nl2", env['l2penalty'], 
                    "\ncausalrep", env['causalrep'], 
                    "\nfeatureZr2", env['featureZr2'])

        train_l2penalty = torch.stack([envs[0]['l2penalty']])
        train_causalrep = torch.stack([envs[0]['causalrep']])
        train_featureZr2 = torch.stack([envs[0]['featureZr2']])
        train_nll = torch.stack([envs[0]['nll']]).mean() 
        train_acc = torch.stack([envs[0]['acc']]).mean()
        train_predacc = torch.stack([envs[0]['predacc']]).mean()
        train_skl_predacc = envs[0]['skl_predacc']
        test_nll = torch.stack([envs[1]['nll']]).mean()
        test_acc = torch.stack([envs[1]['acc']]).mean()
        test_featureZr2 = torch.stack([envs[1]['featureZr2']])
        test_predacc = torch.stack([envs[1]['predacc']]).mean()
        test_skl_predacc = envs[1]['skl_predacc']

        nll_loss = train_nll.clone() 
        # + train_l2penalty.clone()
        optimizer_nll.zero_grad()
        nll_loss.backward(retain_graph=True)
        optimizer_nll.step()
        # print(nll_loss)


        test_acc = envs[1]['acc']
        if step % flags.alter_freq == 0:
            train_causalrep_loss = -train_causalrep.clone() - 1e-1* torch.log(1 - train_featureZr2)

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
            "train_predacc", train_predacc.detach().cpu().numpy(),
            "test_predacc", test_predacc.detach().cpu().numpy(),
            "train_skl_predacc", train_skl_predacc,
            "test_skl_predacc", test_skl_predacc,
            "train_featureZr2", train_featureZr2.detach().cpu().numpy(),
            "test_featureZr2", test_featureZr2.detach().cpu().numpy())


    final_train_accs.append(train_skl_predacc)
    final_test_accs.append(test_skl_predacc)


print('Final train baseline acc (mean/std across restarts so far):')
print(np.mean(final_train_baselineaccs), np.std(final_train_baselineaccs))
print('Final test baseline acc (mean/std across restarts so far):')
print(np.mean(final_test_baselineaccs), np.std(final_test_baselineaccs))


print('Final train baselinevae acc (mean/std across restarts so far):')
print(np.mean(final_train_baselinevaeaccs), np.std(final_train_baselinevaeaccs))
print('Final test baselinevae acc (mean/std across restarts so far):')
print(np.mean(final_test_baselinevaeaccs), np.std(final_test_baselinevaeaccs))


print('Final train acc (mean/std across restarts so far):')
print(np.mean(final_train_accs), np.std(final_train_accs))
print('Final test acc (mean/std across restarts so far):')
print(np.mean(final_test_accs), np.std(final_test_accs))

print('final_train_baselineaccs', final_train_baselineaccs)
print('final_test_baselineaccs', final_test_baselineaccs)
print('final_train_baselinevaeaccs', final_train_baselinevaeaccs)
print('final_test_baselinevaeaccs', final_test_baselinevaeaccs)
print('final_train_accs', final_train_accs)
print('final_test_accs', final_test_accs)

if not os.path.exists("./res"):
    try:
        os.makedirs("./res", 0o700)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

outfile = out_dir + '/supervised_CMNIST' + str(int(time.time()*1e6)) + '.csv'

result = pd.DataFrame({
    'causalrep_train_accs': np.array(final_train_accs),
    'causalrep_test_accs': np.array(final_test_accs),
    'naive_train_accs': np.array(final_train_baselineaccs),
    'naive_test_accs': np.array(final_test_baselineaccs),
    'naive_vae_train_accs': np.array(final_train_baselinevaeaccs),
    'naive_vae_test_accs': np.array(final_test_baselinevaeaccs),
    'hidden_dim': np.repeat(flags.hidden_dim, flags.n_restarts),
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

