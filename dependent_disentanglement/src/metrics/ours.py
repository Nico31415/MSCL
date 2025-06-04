import torch
from torch.distributions import Normal, MixtureSameFamily, Categorical


STD = 0.1

#==========Estimate marginal and conditional===========
def estimate_p_z_given_y(mu_z, logvar_z, y):
    y_queries = torch.unique(y)
    mixtures = {}
    for y_query in y_queries:
        idx = (y == y_query)
        mu_y = mu_z[idx]
        std_z = (logvar_z[idx] / 2).exp()
        mix = Categorical(logits=torch.ones(len(mu_y)))
        comp = Normal(loc=mu_y, scale=std_z)
        mixtures[y_query.item()] = MixtureSameFamily(mix, comp)
    return mixtures

def estimate_p_z(mu_z, logvar_z):
    std_z = (logvar_z / 2).exp()
    mix = Categorical(logits=torch.ones(len(mu_z)))
    comp = Normal(loc=mu_z, scale=std_z)
    mixture = MixtureSameFamily(mix, comp)
    return mixture

#==========Estimate Information Theory quantities===========
def estimate_mi_zj_x(mu_zj, logvar_zj, p_zj, n_samples=10):
    log_dif = 0
    for mu_k, logvar_k in zip(mu_zj, logvar_zj):
        p_zj_given_x = Normal(mu_k, (logvar_k / 2).exp())
        samples = p_zj_given_x.rsample((n_samples,))
        log_p_zj_given_x = p_zj_given_x.log_prob(samples)
        log_p_zj = p_zj.log_prob(samples)
        log_dif += (log_p_zj_given_x - log_p_zj).mean()
    return log_dif / mu_zj.shape[0]

def estimate_mi_zj_yi(mu_zj, logvar_zj, yi, p_zj, p_zj_given_yi, n_samples=10):
    log_dif = 0
    for k, (mu_k, logvar_k) in enumerate(zip(mu_zj, logvar_zj)):
        p_zj_given_x = Normal(mu_k, (logvar_k / 2).exp())
        samples = p_zj_given_x.rsample((n_samples,))
        log_p_zj_given_yi = p_zj_given_yi[yi[k].item()].log_prob(samples)
        log_p_zj = p_zj.log_prob(samples)
        log_dif += (log_p_zj_given_yi - log_p_zj).mean()
    return log_dif / mu_zj.shape[0]

def estimate_ent_yi(yi):
    n_classes = int(yi.max()) + 1
    counts = torch.bincount(yi.int(), minlength=n_classes).float()
    probs = counts / counts.sum()
    return -torch.sum(probs * torch.log(probs + 1e-9))


#==========Estimate Minimality===========
def estimate_minimality_ij(mu_zj, logvar_zj, yi, n_samples=10):
    p_zj = estimate_p_z(mu_zj, logvar_zj)
    p_zj_given_yi = estimate_p_z_given_y(mu_zj, logvar_zj, yi)
    mi_zj_yi = estimate_mi_zj_yi(
        mu_zj, logvar_zj, yi, p_zj, p_zj_given_yi, n_samples=n_samples)
    mi_zj_x = estimate_mi_zj_x(
        mu_zj, logvar_zj, p_zj)
    return (mi_zj_yi / mi_zj_x).item()

def estimate_minimality(y, z, n_samples=10, return_matrix=False):
    minimality = torch.zeros((y.shape[1], z.shape[1]))
    y, z = torch.Tensor(y), torch.Tensor(z)
    for i in range(y.shape[1]):
        for j in range(z.shape[1]):
            yi, zj = y[:,i], z[:,j]
            mu_zj = (zj - zj.mean()) / zj.std()
            logvar_zj = torch.log(torch.full((zj.shape[0],), STD**2)) 
            minimality[i,j] = estimate_minimality_ij(
                mu_zj, logvar_zj, yi, n_samples)
    minimality = torch.clamp(minimality, min=0.0, max=1.0)
    if return_matrix:
        return minimality
    else:
        return minimality.max(axis=0)[0].mean()

#==========Estimate Sufficiency===========
def estimate_sufficiency_ij(mu_zj, logvar_zj, yi, n_samples=10):
    p_zj = estimate_p_z(mu_zj, logvar_zj)
    p_zj_given_yi = estimate_p_z_given_y(mu_zj, logvar_zj, yi)
    mi_zj_yi = estimate_mi_zj_yi(
        mu_zj, logvar_zj, yi, p_zj, p_zj_given_yi, n_samples=n_samples)
    ent_yi = estimate_ent_yi(yi)
    return (mi_zj_yi / ent_yi).item()

def estimate_sufficiency(y, z, n_samples=10, return_matrix=False):
    sufficiency = torch.zeros((y.shape[1], z.shape[1]))
    y, z = torch.Tensor(y), torch.Tensor(z)
    for i in range(y.shape[1]):
        for j in range(z.shape[1]):
            yi, zj = y[:,i], z[:,j]
            mu_zj = (zj - zj.mean()) / zj.std()
            logvar_zj = torch.log(torch.full((zj.shape[0],), STD**2)) 
            sufficiency[i,j] = estimate_sufficiency_ij(
                mu_zj, logvar_zj, yi, n_samples)
    sufficiency = torch.clamp(sufficiency, min=0.0, max=1.0)
    if return_matrix:
        return sufficiency
    else:
        return sufficiency.max(axis=1)[0].mean()
    
#==========Estimate Minimality & Sufficiency===========
import numpy as np

def estimate_mi_all(mu_zj, logvar_zj, yi, p_zj, p_zj_given_yi, n_samples):
    log_dif_yi, log_dif_x, log_dif_x_yi = 0, 0, 0
    for k, (mu_k, logvar_k) in enumerate(zip(mu_zj, logvar_zj)):
        p_zj_given_x = Normal(mu_k, (logvar_k / 2).exp())
        samples = p_zj_given_x.rsample((n_samples,))
        log_p_zj_given_yi = p_zj_given_yi[yi[k].item()].log_prob(samples)
        log_p_zj_given_x = p_zj_given_x.log_prob(samples)
        log_p_zj = p_zj.log_prob(samples)
        log_dif_yi += (log_p_zj_given_yi - log_p_zj).mean()
        log_dif_x += (log_p_zj_given_x - log_p_zj).mean()
        log_dif_x_yi += (log_p_zj_given_x - log_p_zj_given_yi).mean()
    mi_zj_yi = log_dif_yi / mu_zj.shape[0]
    mi_zj_x = log_dif_x / mu_zj.shape[0]
    mi_zj_x_yi = log_dif_x_yi / mu_zj.shape[0]
    return mi_zj_yi, mi_zj_x, mi_zj_x_yi


def H_conditional(logvar):  # logvar: (N, d)
    return 0.5 * np.log(2 * np.pi * np.e * np.exp(logvar))
    
def estimate_min_suf_ij(mu_zj, logvar_zj, yi, n_samples):
    n_classes = len(list(set(yi.tolist())))
    yi = torch.round(yi * (n_classes - 1)).long()
    p_zj = estimate_p_z(mu_zj, logvar_zj)
    p_zj_given_yi = estimate_p_z_given_y(mu_zj, logvar_zj, yi)
    mi_zj_yi, mi_zj_x, mi_zj_x_yi = estimate_mi_all(
        mu_zj, logvar_zj, yi, p_zj, p_zj_given_yi, n_samples=n_samples)
    ent_yi = estimate_ent_yi(yi)
    print(mi_zj_yi, mi_zj_x, ent_yi)
    return (mi_zj_yi / mi_zj_x).item(), (mi_zj_yi / ent_yi).item()

def estimate_min_suf(y, z, n_samples=10, return_matrix=False):
    minimality = torch.zeros((y.shape[1], z.shape[1]))
    sufficiency = torch.zeros((y.shape[1], z.shape[1]))
    y, z = torch.Tensor(y), torch.Tensor(z)
    for i in range(y.shape[1]):
        for j in range(z.shape[1]):
            yi, zj = y[:,i], z[:,j]
            mu_zj = zj# (zj - zj.mean()) / zj.std()
            logvar_zj = torch.log(torch.full((zj.shape[0],), STD**2)) 
            minimality[i,j], sufficiency[i,j] = estimate_min_suf_ij(
                mu_zj, logvar_zj, yi, n_samples)
    minimality = torch.clamp(minimality, min=0.0, max=1.0).numpy()
    sufficiency = torch.clamp(sufficiency, min=0.0, max=1.0).numpy()
    if return_matrix:
        return minimality, sufficiency
    else:
        return minimality.max(axis=0)[0].mean(), sufficiency.max(axis=1)[0].mean()