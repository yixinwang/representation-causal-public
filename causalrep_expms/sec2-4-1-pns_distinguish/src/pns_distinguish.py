import numpy as np
import pandas as pd
import numpy.random as npr
from operator import xor
import seaborn as sns
import matplotlib.pyplot as plt   


sns.set(font_scale = 1.3)
sns.set_style('white')

def pns(z, y):
    y_giv_z = np.sum(z * y) / np.sum(z)
    y_giv_nz = np.sum((1-z) * y) / np.sum(1-z)
    lb = np.maximum(0, y_giv_z - y_giv_nz)
    ub = np.minimum(y_giv_z, 1-y_giv_nz)
    return lb, ub

def cond_pns(z0, y, cond):
    # cond is the column index of z1 this is being flipped; all other columns are being held fixed
    z_full = z0.copy()
    nz_full = z0.copy()
    nz_full[:,cond] = 1 - nz_full[:,cond]

    z = np.prod(z_full, axis=1)
    nz = np.prod(nz_full, axis=1)

    y_giv_z = np.sum(z * y) / np.sum(z)
    y_giv_nz = np.sum((nz) * y) / np.sum(nz)
    lb = np.maximum(0, y_giv_z - y_giv_nz)
    ub = np.minimum(y_giv_z, 1-y_giv_nz)
    return lb, ub


def pn(z, y):
    y_giv_z = np.sum(z * y) / np.sum(z)
    pns_lb, pns_ub = pns(z,y)
    lb = pns_lb / y_giv_z
    ub = pns_ub / y_giv_z
    return lb, ub


def ps(z, y):
    ny_giv_nz = 1 - np.sum((1-z) * y) / np.sum(1-z)
    pns_lb, pns_ub = pns(z,y)
    lb = pns_lb / ny_giv_nz
    ub = pns_ub / ny_giv_nz
    return lb, ub


def eval_pns_pn_ps(x, y, lower=True):
    pns_l_y, pns_u_y = pns(x,y)
    pn_l_y, pn_u_y = pn(x,y)
    ps_l_y, ps_u_y = ps(x,y)
    if lower:
        return pns_l_y, pn_l_y, ps_l_y
    else:
        return pns_u_y, pn_u_y, ps_u_y


def eval_cond_pns(x, y, lower=True):
    pns_l_ys, pns_u_ys = [], []
    for i in range(x.shape[1]):
        pns_l_y, pns_u_y = cond_pns(x,y,i)
        pns_l_ys.append(pns_l_y)
        pns_u_ys.append(pns_u_y)
    if lower:
        return pns_l_ys
    else:
        return pns_u_ys



def gen_z(p, noise_p, n_samples):
    z1 = npr.binomial(1, p=p, size=n_samples)
    z2 = z1^(npr.binomial(1, p=noise_p, size=n_samples))
    # print("correlation z1, z2\n", np.corrcoef(z1, z2))
    # as noise_p increase, z1 and z2 are less highly correlated.
    return z1, z2


def gen_y(z1, z2, noise_y, n_samples):
    y1 = z1^(npr.binomial(1, p=noise_y, size=n_samples))
    y2 = (z1 * z2)^(npr.binomial(1, p=noise_y, size=n_samples))
    # z1 is necessary and sufficient for y1.
    # z1 is necessary but insufficient for y2.
    # z2 is neither necessary nor sufficient for y1. 
    # z2 is necessary but insufficient for y2.
    # z1 * z2 is necessary and sufficient for y2.
    return y1, y2


# params = {'p': 0.5, # p(z1=1)
#         'noise_p': 0.2, # p(z2|z1)
#         'noise_y': 0.1, # p(y|z1,z2)
#         'n_samples': 1000,
#         'n_trials': 100}


def calc_all(p, noise_p, noise_y, n_samples, n_trials=100):
    res_all = []
    params = {'p': p, # p(z1=1)
            'noise_p': noise_p, # p(z2|z1)
            'noise_y': noise_y, # p(y|z1,z2)
            'n_samples': n_samples,
            'n_trials': 100}
    for _ in range(params['n_trials']):
        res = pd.DataFrame(params, index=[0])
        z1, z2 = gen_z(params['p'], params['noise_p'], params['n_samples'])
        y1, y2 = gen_y(z1, z2, params['noise_y'], params['n_samples'])

        params['z1_z2_corr'] = np.corrcoef(z1, z2)[0,1]

        dat = {'z1': z1, 'z2': z2, \
                'z1*z2': z1*z2, \
                'y1': y1, 'y2': y2}

        # print(params)

        for y_eval in ['y1', 'y2']: 
            for z_eval in ['z1', 'z2', 'z1*z2']:
                y_val = dat[y_eval]
                z_val = dat[z_eval]
                name = y_eval + '_' + z_eval + '_'
                # print("y_eval", y_eval, "z_eval", z_eval)
                out = eval_pns_pn_ps(z_val, y_val)
                # print("pns, pn, ps", out)
                res[name+"pns"] = out[0]
                res[name+"pn"] = out[1]
                res[name+"ps"] = out[2]

            # print("y_eval", y_eval)
            cond_out = eval_cond_pns(np.column_stack([dat['z1'], dat['z2']]), y_val)
            # print("conditional pns", cond_out)
            res['cond_pns_' + y_eval + '_' + 'z1'] = cond_out[0]
            res['cond_pns_' + y_eval + '_' + 'z2'] = cond_out[1]
        # print(res)
        res_all.append(res)
    res_all_pd = pd.concat(res_all)
    return res_all_pd


res_all = []

# for p in np.linspace(0,1,11):
for p in [0.4]:
    for noise_p in np.linspace(0,1,11):
        # for noise_y in np.linspace(0,1,11):
        for noise_y in [0.2]:
            for n_samples in [10000]:
                res = calc_all(p, noise_p, noise_y, n_samples)
                res_all.append(res)
                    


res_all_pd = pd.concat(res_all)

res_all_pd.to_csv('pns_distinguish_res.csv')

# compare conditional pns

cond_col = [col for col in res_all_pd if col.startswith('cond')]
fig, ax = plt.subplots()
fig.set_size_inches(3,2)
ax = sns.catplot(data=res_all_pd[cond_col], palette="pastel",kind="box")
ax.set_xticklabels([r'PNS($Z_1, Y_1 | Z_2$)', r'PNS($Z_2, Y_1 | Z_1$)', r'PNS($Z_1, Y_2 | Z_2$)', r'PNS($Z_2, Y_2 | Z_1$)'], rotation=30)
plt.tight_layout()  
plt.savefig('cond_pns.pdf') 


# plot ps, pn, pns wrt correlation

for y_eval, y_eval_name in zip(['y1', 'y2'], [r'$Y_1$', r'$Y_2$']): 
    for z_eval, z_eval_name in zip(['z1', 'z2', 'z1*z2'], [r'$Z_1$', r'$Z_2$', r'$Z_1 & Z_2$']):
        fig, axs = plt.subplots(1,3, figsize=(12, 3))
        name = y_eval + '_' + z_eval + '_'
        cond_col = [col for col in res_all_pd if col.startswith(name)]
        for i,idx in enumerate(['pn', 'ps', 'pns']):
            sns.scatterplot(ax=axs[i], x='z1_z2_corr', y=name + idx, data=res_all_pd, alpha=0.5, color='brown')
            axs[i].set_xlabel(r'Corr($Z_1, Z_2$)')
            axs[i].set_ylabel(idx.upper()+'('+z_eval_name+','+y_eval_name+')')
        plt.tight_layout()  
        plt.savefig(name[:-1]+'.pdf')  
