import numpy as np
from sklearn import ensemble, linear_model
from sklearn.preprocessing import StandardScaler


def get_regressor(reg_type="linear"):
    if reg_type=="linear":
        return linear_model.LinearRegression()
    elif reg_type=="sgd":
        return linear_model.SGDRegressor()
    elif reg_type=="lasso":
        return linear_model.Lasso()
    elif reg_type=="gradient_booster":
        return ensemble.GradientBoostingRegressor()
    else:
        raise ValueError
        

def to2axis(x):
    if len(x.shape)==1:
        return x.reshape(-1, 1)
    else:
        return x



def preds_regressor(x, y, reg_type):
    reg = get_regressor(reg_type)
    reg.fit(to2axis(x), y)
    preds = reg.predict(to2axis(x))
    return preds



def standardarize_columns(data):
    data_proc = np.zeros_like(data)
    for i in range(data.shape[1]):
        data_proc[:,i] = StandardScaler().fit_transform(data[:,i][:,None])[:,0]
    return data_proc


def calc_minimality(factors, codes, reg_type="gradient_booster"):
    n_factors, n_codes = factors.shape[1], codes.shape[1]
    minimality = np.zeros((n_codes, n_factors))
    factors = standardarize_columns(factors)
    codes = standardarize_columns(codes)
    error_j = 1
    for j in range(n_codes):
        for i in range(n_factors):
            preds_ji = preds_regressor(factors[:,i], codes[:,j], reg_type)
            error_ji = ((codes[:,j]-preds_ji)**2).mean()
            minimality[j,i] = max(1 - error_ji/error_j, 0)
    return minimality.max(axis=1).mean()


def calc_sufficiency(factors, codes, reg_type="gradient_booster"):
    n_factors, n_codes = factors.shape[1], codes.shape[1]
    sufficiency = np.zeros((n_factors, n_codes))
    factors = standardarize_columns(factors)
    codes = standardarize_columns(codes)
    error_i = 1
    for i in range(n_factors):
        for j in range(n_codes):
            preds_ij = preds_regressor(codes[:,j], factors[:,i], reg_type)
            error_ij = ((factors[:,i]-preds_ij)**2).mean()
            sufficiency[i,j] = max(1 - error_ij/error_i, 0)
    return sufficiency.max(axis=1).mean()


def calc_factors_invariance(factors, codes, reg_type="linear"):
    n_factors, n_codes = factors.shape[1], codes.shape[1]
    factors_invariance = np.zeros((n_codes, n_factors))
    factors = standardarize_columns(factors)
    codes = standardarize_columns(codes)
    for j in range(n_codes):
        preds_j = preds_regressor(factors, codes[:,j], reg_type)
        for i in range(n_factors):
            preds_ji = preds_regressor(factors[:,i], codes[:,j], reg_type)
            error_ji = ((preds_j-preds_ji)**2).mean()
            factors_invariance[j,i] = max(1 - error_ji, 0)
    return factors_invariance.max(axis=1).mean()


def calc_representations_invariance(factors, codes, reg_type="linear"):
    n_factors, n_codes = factors.shape[1], codes.shape[1]
    representations_invariance = np.zeros((n_factors, n_codes))
    factors = standardarize_columns(factors)
    codes = standardarize_columns(codes)
    for i in range(n_factors):
        preds_i = preds_regressor(codes, factors[:,i], reg_type)
        for j in range(n_codes):
            preds_ij = preds_regressor(codes[:,j], factors[:,i], reg_type)
            error_ij = ((preds_i-preds_ij)**2).mean()
            representations_invariance[i,j] = max(1 - error_ij, 0)
    return representations_invariance.max(axis=1).mean()


def calc_explicitness(factors, codes, reg_type="linear"):
    n_factors = factors.shape[1]
    explicitness = np.zeros(n_factors)
    factors = standardarize_columns(factors)
    codes = standardarize_columns(codes)
    for i in range(n_factors):
        preds_i = preds_regressor(codes, factors[:,i], reg_type)
        error_i = ((factors[:,i]-preds_i)**2).mean()
        explicitness[i] = max(1 - error_i, 0)
    return explicitness.mean()