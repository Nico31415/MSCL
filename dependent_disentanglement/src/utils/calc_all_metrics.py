import random
import numpy as np

from src.metrics import *
from .calc_accuracy import get_acc_classifier


def calc_dis_metrics(results, n_samples):
    idxs = random.choices(range(results['y'].shape[0]), k=n_samples)
    y_cat = np.zeros((n_samples, results['y'].shape[1]))
    for i in range(results['y'].shape[1]):
        classes = list(set(results['y'][:,i].numpy()))
        y_cat[:,i] = np.array([classes.index(i) for i in results['y'][idxs,i][:,None].numpy()])
    z = results['z'][idxs].numpy()
    factor_val = z_min_var(y_cat, z)
    mig_val = mig(y_cat, z)
    sap_val = sap(y_cat, z)
    modularity_val = modularity(y_cat, z)
    dci_val = dci(results['y'][idxs], z)
    irs_val = irs(results['y'][idxs], z)
    min_val, suf_val = estimate_min_suf(results['y'][idxs], z)
    return {
        "factor_vae":       factor_val,
        "mig":              mig_val,
        "sap":              sap_val,
        "modularity":       modularity_val,
        "disentanglement":  dci_val[0],
        "completeness":     dci_val[1],
        "irs":              irs_val,
        "minimality":       min_val,
        "sufficiency":      suf_val
    }

def calc_accs_ratios(results):
    accs_ratios = np.zeros(results['y'].shape[1])
    idxs_100 =  random.choices(range(results['y'].shape[0]), k=100)
    idxs_10000 = random.choices(range(results['y'].shape[0]), k=10000)
    for i in range(results['y'].shape[1]):
        classes = list(set(results['y'][:,i].numpy()))
        accs_100 = get_acc_classifier(
            results['z'][idxs_100].numpy(), 
            np.array([classes.index(i) for i in results['y'][idxs_100,i][:,None].numpy()]), 
            "random_forest")
        accs_10000 = get_acc_classifier(
            results['z'][idxs_10000].numpy(), 
            np.array([classes.index(i) for i in results['y'][idxs_10000,i][:,None].numpy()]), 
            "random_forest")
        accs_ratios[i] = accs_100 / accs_10000
    return accs_ratios.mean()

def calc_accs(results, n_samples):
    idxs = random.choices(range(results['y'].shape[0]), k=n_samples)
    accs = np.zeros((results['y'].shape[1], results['z'].shape[1]))
    accs_all = np.zeros(results['y'].shape[1])
    for i in range(results['y'].shape[1]):
        classes = list(set(results['y'][:,i].numpy()))
        y_i = np.array([classes.index(i) for i in results['y'][idxs,i][:,None].numpy()])
        accs_all[i] = get_acc_classifier(results['z'][idxs].numpy(), y_i, "random_forest")
        for j in range(results['z'].shape[1]):
            accs[i,j] = get_acc_classifier(results['z'][idxs,j][:,None].numpy(), y_i, "random_forest")
    return accs_all.mean(), accs.max(axis=1).mean()


def get_all_metrics(results, n_samples=2000):
    dis_metrics = calc_dis_metrics(results, n_samples)
    accs_ratios = calc_accs_ratios(results)
    accs, accs_all = calc_accs(results, n_samples)
    return {**dis_metrics, 'accs_ratios': accs_ratios, 'accs': accs, 'accs_all': accs_all}