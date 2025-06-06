"""
https://github.com/ubisoft/ubisoft-laforge-disentanglement-metrics/blob/main/src/metrics/mig.py
"""

import numpy as np

from sklearn.preprocessing import minmax_scale

from .utils import get_bin_index, get_mutual_information

    
def mig(factors, codes, continuous_factors=True, nb_bins=10):
    ''' MIG metric from R. T. Q. Chen, X. Li, R. B. Grosse, and D. K. Duvenaud,
        “Isolating sources of disentanglement in variationalautoencoders,”
        in NeurIPS, 2018.
    
    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param nb_bins:                         number of bins to use for discretization
    '''
    # count the number of factors and latent codes
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]
    
    # quantize factors if they are continuous
    if continuous_factors:
        factors = minmax_scale(factors)  # normalize in [0, 1] all columns
        factors = get_bin_index(factors, nb_bins)  # quantize values and get indexes
    
    # quantize latent codes
    codes = minmax_scale(codes)  # normalize in [0, 1] all columns
    codes = get_bin_index(codes, nb_bins)  # quantize values and get indexes

    # compute mutual information matrix
    mi_matrix = np.zeros((nb_factors, nb_codes))
    for f in range(nb_factors):
        for c in range(nb_codes):
            mi_matrix[f, c] = get_mutual_information(factors[:, f], codes[:, c])

    # compute the mean gap for all factors
    sum_gap = 0
    for f in range(nb_factors):
        mi_f = np.sort(mi_matrix[f, :])
        # get diff between highest and second highest term and add it to total gap
        sum_gap += mi_f[-1] - mi_f[-2]
    
    # compute the mean gap
    mig_score = sum_gap / nb_factors
    
    return mig_score