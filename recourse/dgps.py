import torch
import torch.distributions as dist
from torch.distributions import Categorical, MixtureSameFamily

from recourse.scm import SCM, Seq
from recourse.distutils import ShiftedBinomial
from typing import Any, Callable, Optional
from sklearn.metrics import r2_score

import logging
logger = logging.getLogger(__name__)

## simple settings

def get_binomial_linear_additive():
    desc_1 = r'x_1 := \epsilon_1, \epsilon_1 \sim \mathcal{N}(0, 1)'
    f_1 = lambda par, eps: eps
    f_1_inv = lambda par, x: x
    d_1 = ShiftedBinomial(8, 0.5, 0) # number of values, p, mu
    seq_1 = Seq('x1', [], f_1, f_1_inv, d_1, description=desc_1)
    
    desc_y = r'y := x_1 + \epsilon_, \epsilon_y \sim \mathcal{N}(0, 0.1)'
    f_y = lambda par, eps: (eps + par[:, 0])
    f_y_inv = lambda par, x: (x - par[:, 0])
    d_y = ShiftedBinomial(2, 0.5, 0)
    seq_y = Seq('y', ['x1'], f_y, f_y_inv, d_y, description=desc_y)
    
    desc_2 = r'x_2 := y + x_1 + \epsilon_2, \epsilon_2 \sim \mathcal{N}(0, 0.1)'
    f_2 = lambda par, eps: eps + par[:, 0] + par[:, 1]
    f_2_inv = lambda par, x: x - par[:, 0] - par[:, 1]
    d_2 = ShiftedBinomial(2, 0.5, 0)
    seq_2 = Seq('x2', ['y', 'x1'], f_2, f_2_inv, d_2, description=desc_2)
    
    seqs = [seq_1, seq_y, seq_2]
    scm = SCM(seqs)
    return scm

def get_binomial_linear_multiplicative():
    desc_1 = r'x_1 := \epsilon_1, \epsilon_1 \sim \mathcal{G}(0.5)'
    f_1 = lambda par, eps: eps
    f_1_inv = lambda par, x: x
    d_1 = ShiftedBinomial(5, 0.5, 5*0.5+1) # the offset ensure that the values are positive
    seq_1 = Seq('x1', [], f_1, f_1_inv, d_1, description=desc_1)
    
    desc_y = r'y := x_1 * \epsilon_y, \epsilon_y \sim \mathcal{G}(0.95)'
    f_y = lambda par, eps: (eps * par[:, 0])
    f_y_inv = lambda par, x: x / par[:, 0]
    d_y = ShiftedBinomial(1, 0.5, 1*0.5+1)
    seq_y = Seq('y', ['x1'], f_y, f_y_inv, d_y, description=desc_y)
    
    desc_2 = r'x_2 := y * x_1 * \epsilon_2, \epsilon_2 \sim \mathcal{G}(0.95)'
    g = lambda par: torch.sqrt(par[:, 0] * par[:, 1]) 
    f_2 = lambda par, eps: eps * par[:, 0] * par[:, 1]
    f_2_inv = lambda par, x: x / par[:, 0] / par[:, 1]
    d_2 = ShiftedBinomial(1, 0.5, 1*0.5+1)
    seq_2 = Seq('x2', ['y', 'x1'], f_2, f_2_inv, d_2, description=desc_2)
    
    seqs = [seq_1, seq_y, seq_2]
    scm = SCM(seqs)
    return scm

def get_binomial_nonlinear_additive():
    desc_1 = r'x_1 := \epsilon_1, \epsilon_1 \sim \mathcal{G}(0.5)'
    f_1 = lambda par, eps: eps
    f_1_inv = lambda par, x: x
    d_1 = ShiftedBinomial(8, 0.5, 0)
    seq_1 = Seq('x1', [], f_1, f_1_inv, d_1, description=desc_1)
    
    desc_y = r'y := x_1^2\epsilon_y, \epsilon_y \sim \mathcal{G}(0.95)'
    g_y = lambda par: par[:, 0]**2
    f_y = lambda par, eps: eps + g_y(par)
    f_y_inv = lambda par, y: y - g_y(par)
    d_y = ShiftedBinomial(2, 0.5, 0)
    seq_y = Seq('y', ['x1'], f_y, f_y_inv, d_y, description=desc_y)
    
    desc_2 = r'x_2 := (y + x_1)^2 + \epsilon_2, \epsilon_2 \sim \sim \mathcal{G}(0.95)'
    g_2 = lambda par: (par[:, 0] + 0.1*par[:, 1])**2
    f_2 = lambda par, eps: eps + g_2(par)
    f_2_inv = lambda par, x: x - g_2(par)
    d_2 = ShiftedBinomial(2, 0.5, 0)
    seq_2 = Seq('x2', ['y', 'x1'], f_2, f_2_inv, d_2, description=desc_2)
    
    seqs = [seq_1, seq_y, seq_2]
    scm = SCM(seqs)
    return scm

def get_binomial_nonlinear_multiplicative():
    desc_1 = r'x_1 := \epsilon_1, \epsilon_1 \sim \sim \mathcal{G}(0.5)'
    f_1 = lambda par, eps: eps
    f_1_inv = lambda par, x: x
    mixing = Categorical(torch.tensor([0.5, 0.5]))
    components = ShiftedBinomial(
        torch.tensor([2, 4]),
        torch.tensor([0.5, 0.5]),
        torch.tensor([1+1, 4*0.5+2])
    )
    d_1 = dist.MixtureSameFamily(mixing, components)
    # d_1 = ShiftedBinomial(5, 0.5, 5*0.5+1) # the offset ensure that the values are positive
    seq_1 = Seq('x1', [], f_1, f_1_inv, d_1, description=desc_1)
    
    desc_y = r'y := x_1^2 * \epsilon_y, \epsilon_y \sim \sim \mathcal{G}(0.95)'
    f_y = lambda par, eps: eps * par[:, 0]**2
    f_y_inv = lambda par, y: y / par[:, 0]**2
    d_y = ShiftedBinomial(2, 0.5, 2*0.5+1)
    seq_y = Seq('y', ['x1'], f_y, f_y_inv, d_y, description=desc_y)
    
    desc_2 = r'x_2 := (y +  x_1)^2 * \epsilon_2, \epsilon_2 \sim \mathcal{G}(0.5)'
    f_2 = lambda par, eps: eps * (par[:, 0] + par[:, 1])**2
    f_2_inv = lambda par, x2: x2 / (par[:, 0] + par[:, 1])**2
    d_2 = ShiftedBinomial(2, 0.5, 2*0.5+1)
    seq_2 = Seq('x2', ['y', 'x1'], f_2, f_2_inv, d_2, description=desc_2)
    
    seqs = [seq_1, seq_y, seq_2]
    scm = SCM(seqs)
    return scm
    

simple_scms = {
    'binomial_linear_additive': get_binomial_linear_additive(),
    'binomial_linear_multiplicative': get_binomial_linear_multiplicative(),
    'binomial_nonlinear_additive': get_binomial_nonlinear_additive(),
    'binomial_nonlinear_multiplicative': get_binomial_nonlinear_multiplicative(),
}
    
## more complicated settings
    
# def get_binomial_nonlinear_sigmoid():
#     desc_1 = r'x_1 := \epsilon_1, \epsilon_1 \sim \mathcal{G}(0.5)'
#     f_1 = lambda par, eps: eps
#     f_1_inv = lambda par, x1: x1
#     d_1 = dist.Geometric(0.5)
#     seq_1 = Seq('x1', [], f_1, f_1_inv, d_1, description=desc_1)
    
#     desc_y = r'y := \sigma(x_1^2 + \epsilon_y), \epsilon_y \sim \mathcal{G}(0.95)'
#     f_y = lambda par, eps: torch.sigmoid(eps + par[:, 0]**2)
#     f_y_inv = lambda par, y: (torch.logit(y) - par[:, 0]**2)
#     d_y = dist.Geometric(0.95)
#     seq_y = Seq('y', ['x1'], f_y, f_y_inv, d_y, description=desc_y)
    
#     desc_2 = r'x_2 := \sigma((y + 0.1 x_1)^2 + \epsilon_2), \sim \mathcal{G}(0.95)'
#     f_2 = lambda par, eps: torch.sigmoid(eps + (par[:, 0] + par[:, 1])**2)
#     f_2_inv = lambda par, x2: torch.logit(x2) - (par[:, 0] + par[:, 1])**2
#     d_2 = dist.Geometric(0.95)
#     seq_2 = Seq('x2', ['y', 'x1'], f_2, f_2_inv, d_2, description=desc_2)
    
#     seqs = [seq_1, seq_y, seq_2]
#     scm = SCM(seqs)
#     return scm

def get_binomial_linear_invertiblepolynomial():
    desc_1 = r'x_1 := \epsilon_1, \epsilon_1 \sim \sim \mathcal{G}(0.5)'
    f_1 = lambda par, eps: eps
    f_1_inv = lambda par, x: x
    d_1 = ShiftedBinomial(4, 0.5, 2-3)
    seq_1 = Seq('x1', [], f_1, f_1_inv, d_1, description=desc_1)
    
    desc_y = r'y := (x_1 + \epsilon_y)^3, \epsilon_y \sim \sim \mathcal{G}(0.95)'
    f_y = lambda par, eps: torch.pow((eps + par[:, 0]), 3)
    f_y_inv = lambda par, y: y.sign() * torch.pow(y.abs(), 1/3) - par[:, 0]
    d_y = ShiftedBinomial(2, 0.5, 1)
    seq_y = Seq('y', ['x1'], f_y, f_y_inv, d_y, description=desc_y)
    
    desc_2 = r'x_2 := (- y + x_1 + \epsilon_2)^3, \epsilon_2 \sim \mathcal{G}(0.95)'
    f_2 = lambda par, eps: torch.pow(eps + par[:, 0] - par[:, 1], 3)
    f_2_inv = lambda par, x2: x2.sign() * torch.pow(x2.abs(), 1/3) - par[:, 0] + par[:, 1]
    d_2 = ShiftedBinomial(1, 0.5, 0.5)
    seq_2 = Seq('x2', ['y', 'x1'], f_2, f_2_inv, d_2, description=desc_2)
    
    seqs = [seq_1, seq_y, seq_2]
    scm = SCM(seqs)
    return scm

def get_binomial_linear_noninvertiblepolynomial():
    desc_1 = r'x_1 := \epsilon_1, \epsilon_1 \sim \mathcal{G}(0.5)'
    f_1 = lambda par, eps: eps
    f_1_inv = lambda par, x: x
    
    d_1 = ShiftedBinomial(4, 0.5, 0)
    seq_1 = Seq('x1', [], f_1, f_1_inv, d_1, description=desc_1)
    
    desc_y = r'y := (x_1 + \epsilon_y)^2, \epsilon_y \sim \mathcal{G}(0.95)'
    f_y = lambda par, eps: (2*par[:, 0] - eps)**2
    f_y_inv = None
    #f_y_inv = lambda par, y: (torch.pow(y, 1/2) - par[:, 0]**2)
    d_y = ShiftedBinomial(4, 0.5, 0)
    seq_y = Seq('y', ['x1'], f_y, f_y_inv, d_y, description=desc_y)
    
    desc_2 = r'x_2 := (y + x_1 + \epsilon_2)^2, \epsilon_2 \sim \mathcal{G}(0.95)'
    f_2 = lambda par, eps: (10*par[:, 1] + 80 - eps)**2
    f_2_inv = None
    #f_2_inv = lambda par, x: torch.pow(x, 1/2) - (par[:, 0] * par[:, 1])**2
    d_2 = ShiftedBinomial(4, 0.5, 0)
    seq_2 = Seq('x2', ['y', 'x1'], f_2, f_2_inv, d_2, description=desc_2)
    
    seqs = [seq_1, seq_y, seq_2]
    scm = SCM(seqs)
    return scm


## collections

invertible_scms = {
    'binomial_linear_additive': get_binomial_linear_additive(),
    'binomial_linear_multiplicative': get_binomial_linear_multiplicative(),
    'binomial_nonlinear_additive': get_binomial_nonlinear_additive(),
    'binomial_nonlinear_multiplicative': get_binomial_nonlinear_multiplicative(),
    # 'binomial_nonlinear_sigmoid': get_binomial_nonlinear_sigmoid(),
    'binomial_linear_invertiblepolynomial': get_binomial_linear_invertiblepolynomial()
}
    
noninvertible_scms = {
    'binomial_linear_noninvertiblepolynomial': get_binomial_linear_noninvertiblepolynomial()
}


shortnames_scms = {
    'binomial_linear_additive': 'LinAdd',
    'binomial_linear_multiplicative': 'LinMult',
    'binomial_nonlinear_additive': 'NlinAdd',
    'binomial_nonlinear_multiplicative': 'NlinMult',
    'binomial_linear_invertiblepolynomial': 'LinCubic',
    'binomial_linear_noninvertiblepolynomial': 'LinQuad'
}

## exctract recourse problem from scm

import pandas as pd
from typing import Optional
import numpy as np

def get_rec_setup(scm: SCM, target_name: str, N_rec: int, N_fit: int, thresh_pred: float, model, model_kwargs: Optional[dict[str, Any]]=None):
    obss, epss = scm.sample(sample_shape=(N_fit*10,))

    X, y = obss.drop(target_name, axis=1), obss[target_name]
    
    # half the population should be qualified
    thresh_label = y.median()
    
    def fn_l(y):
        return (y >= thresh_label)
    
    y_bin = fn_l(y)
    y_bin = y_bin.astype(int)
    
    # the costs should be proportional to the std of the features of should increase with the position in the graph
    variances = X.var(axis=0)
    order = (pd.Series(range(len(scm.get_nodes_ordered())), index=list(reversed(scm.get_nodes_ordered()))) + 1) * 10
    costs =  order**2 / variances
    costs = costs.drop(target_name)
    costs = costs / costs.sum()
    cost_dict = costs.to_dict()
    
    # fitting the model
    ixs = np.random.choice(X.shape[0], N_fit, replace=False)
    X_fit = X.iloc[ixs, :]
    y_bin_fit = y_bin.iloc[ixs]
    model.fit(X_fit, y_bin_fit, **(model_kwargs or {}))
    
    def predict(X, return_raw=False):
        if not return_raw:
            return model.predict_proba(X)[:, 1] >= thresh_pred
        else:
            return model.predict_proba(X)[:, 1], model 
    
    ## getting a batch of rejected individuals
    obss, epss = scm.sample(sample_shape=(N_rec*10,)) # *10 to make sure that we have enough rejected individuals
    obss, epss = scm.get_sample()
    
    X, y = obss.drop(target_name, axis=1), obss[target_name]
    y_pred = predict(X)
    
    # get the rejected individuals
    rejected = (y_pred == False)
    rejected_idx = np.where(rejected)[0][:N_rec]
    
    obss_rej = obss.iloc[rejected_idx, :].copy()
    epss_rej = epss.iloc[rejected_idx, :].copy()
    X_rej = obss_rej.drop(target_name, axis=1)
        
    obss_rej['y_pred'] = y_pred[rejected_idx] 
    obss_rej['l'] = fn_l(obss_rej[target_name])
    
    assert rejected_idx.shape[0] == N_rec, "Not enough rejected individuals"
    
    return scm, target_name, thresh_label, fn_l, cost_dict, predict, X_rej, (obss_rej, epss_rej)

