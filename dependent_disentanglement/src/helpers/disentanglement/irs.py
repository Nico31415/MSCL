import numpy as np

from sklearn.preprocessing import minmax_scale

from .utils import get_bin_index


def irs(factors, codes, continuous_factors=True, nb_bins=10, diff_quantile=1.):
    ''' IRS metric from R. Suter, D. Miladinovic, B. Schölkopf, and S. Bauer,
        “Robustly disentangled causal mechanisms: Validatingdeep representations for interventional robustness,”
        in ICML, 2019.
    
    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param nb_bins:                         number of bins to use for discretization
    :param diff_quantile:                   float value between 0 and 1 to decide what quantile of diffs to select
                                            use 1.0 for the version in the paper
    '''
    # quantize factors if they are continuous
    if continuous_factors:
        factors = minmax_scale(factors)  # normalize in [0, 1] all columns
        factors = get_bin_index(factors, nb_bins)  # quantize values and get indexes
    
    # remove constant dimensions
    codes = _drop_constant_dims(codes)
    
    if not codes.any():
        irs_score = 0.0
    else:
        # count the number of factors and latent codes
        nb_factors = factors.shape[1]
        nb_codes = codes.shape[1]
        
        # compute normalizer
        max_deviations = np.max(np.abs(codes - codes.mean(axis=0)), axis=0)
        cum_deviations = np.zeros([nb_codes, nb_factors])
        for i in range(nb_factors):
            unique_factors = np.unique(factors[:, i], axis=0)
            assert(unique_factors.ndim == 1)
            nb_distinct_factors = unique_factors.shape[0]
            
            for k in range(nb_distinct_factors):
                # compute E[Z | g_i]
                match = factors[:, i] == unique_factors[k]
                e_loc = np.mean(codes[match, :], axis=0)

                # difference of each value within that group of constant g_i to its mean
                diffs = np.abs(codes[match, :] - e_loc)
                max_diffs = np.percentile(diffs, q=diff_quantile*100, axis=0)
                cum_deviations[:, i] += max_diffs
            
            cum_deviations[:, i] /= nb_distinct_factors
        
        # normalize value of each latent dimension with its maximal deviation
        normalized_deviations = cum_deviations / max_deviations[:, np.newaxis]
        irs_matrix = 1.0 - normalized_deviations
        disentanglement_scores = irs_matrix.max(axis=1)
        
        if np.sum(max_deviations) > 0.0:
            irs_score = np.average(disentanglement_scores, weights=max_deviations)
        else:
            irs_score = np.mean(disentanglement_scores)
    
    return irs_score


def _drop_constant_dims(codes):
    ''' Drop constant dimensions of latent codes
    
    :param codes:       latent codes associated to the dataset of factors
                        each column is a latent code and each line is a data point
    '''
    # check we have a matrix
    if codes.ndim != 2:
        raise ValueError("Expecting a matrix.")

    # compute variances and create mask
    variances = codes.var(axis=0)
    mask = variances > 0.
    if not np.all(mask):
        print(f'WARNING -- Collapsed latent dimensions detected -- mask = {mask}')
    
    return codes[:, mask]


def compute_irs(factors,
                codes,
                diff_quantile=0.99):
  """Computes the Interventional Robustness Score.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    diff_quantile: Float value between 0 and 1 to decide what quantile of diffs
      to select (use 1.0 for the version in the paper).

  Returns:
    Dict with IRS and number of active dimensions.
  """

  ys_discrete = get_bin_index(factors.T, 20)
  active_mus = _drop_constant_dims(codes.T)

  if not active_mus.any():
    irs_score = 0.0
  else:
    irs_score = scalable_disentanglement_score(ys_discrete.T, active_mus.T,
                                               diff_quantile)["avg_score"]

  score_dict = {}
  score_dict["IRS"] = irs_score
  score_dict["num_active_dims"] = np.sum(active_mus)
  return irs_score


def _drop_constant_dims(ys):
  """Returns a view of the matrix `ys` with dropped constant rows."""
  ys = np.asarray(ys)
  if ys.ndim != 2:
    raise ValueError("Expecting a matrix.")

  variances = ys.var(axis=1)
  active_mask = variances > 0.
  return ys[active_mask, :]


def scalable_disentanglement_score(gen_factors, latents, diff_quantile=0.99):
  """Computes IRS scores of a dataset.

  Assumes no noise in X and crossed generative factors (i.e. one sample per
  combination of gen_factors). Assumes each g_i is an equally probable
  realization of g_i and all g_i are independent.

  Args:
    gen_factors: Numpy array of shape (num samples, num generative factors),
      matrix of ground truth generative factors.
    latents: Numpy array of shape (num samples, num latent dimensions), matrix
      of latent variables.
    diff_quantile: Float value between 0 and 1 to decide what quantile of diffs
      to select (use 1.0 for the version in the paper).

  Returns:
    Dictionary with IRS scores.
  """
  num_gen = gen_factors.shape[1]
  num_lat = latents.shape[1]

  # Compute normalizer.
  max_deviations = np.max(np.abs(latents - latents.mean(axis=0)), axis=0)
  cum_deviations = np.zeros([num_lat, num_gen])
  for i in range(num_gen):
    unique_factors = np.unique(gen_factors[:, i], axis=0)
    assert unique_factors.ndim == 1
    num_distinct_factors = unique_factors.shape[0]
    for k in range(num_distinct_factors):
      # Compute E[Z | g_i].
      match = gen_factors[:, i] == unique_factors[k]
      e_loc = np.mean(latents[match, :], axis=0)

      # Difference of each value within that group of constant g_i to its mean.
      diffs = np.abs(latents[match, :] - e_loc)
      max_diffs = np.percentile(diffs, q=diff_quantile*100, axis=0)
      cum_deviations[:, i] += max_diffs
    cum_deviations[:, i] /= num_distinct_factors
  # Normalize value of each latent dimension with its maximal deviation.
  normalized_deviations = cum_deviations / max_deviations[:, np.newaxis]
  irs_matrix = 1.0 - normalized_deviations
  disentanglement_scores = irs_matrix.max(axis=1)
  if np.sum(max_deviations) > 0.0:
    avg_score = np.average(disentanglement_scores, weights=max_deviations)
  else:
    avg_score = np.mean(disentanglement_scores)

  parents = irs_matrix.argmax(axis=1)
  score_dict = {}
  score_dict["disentanglement_scores"] = disentanglement_scores
  score_dict["avg_score"] = avg_score
  score_dict["parents"] = parents
  score_dict["IRS_matrix"] = irs_matrix
  score_dict["max_deviations"] = max_deviations
  return score_dict