import copy as cp
import numpy as np
import scipy.stats as sts
from scipy.ndimage import rotate
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from skimage.transform import resize
from sklearn.metrics import mutual_info_score
from sklearn import svm
from sklearn import ensemble


def auto_rotate(auto, angle):
    auto_rot = rotate(auto, angle)

    return auto_rot


def get_circle(cell, x, y, radius, ring=False):
    circle = []

    rows = [y + p for p in range(-int(radius) + 1, int(radius))]
    cols = [x + p for p in range(-int(radius) + 1, int(radius))]
    for row in rows:
        row_sq = (row - y) ** 2
        for col in cols:
            col_sq = (col - x) ** 2
            cond = (radius / 2) ** 2 < row_sq + col_sq < radius ** 2 if ring else row_sq + col_sq < radius ** 2
            if cond:
                circle.append(cell[row, col])
    return circle


def get_corr_rot(auto, angle, radius, ring=False):
    auto_rot = auto_rotate(auto, angle)

    y, x = [int((x - 1) / 2) for x in auto.shape]
    y_1, x_1 = [int((x - 1) / 2) for x in auto_rot.shape]

    c1 = get_circle(auto, x, y, radius, ring=ring)
    c2 = get_circle(auto_rot, x_1, y_1, radius, ring=ring)

    return np.corrcoef(c1, c2)[0, 1]


def get_grid_score(auto, radius, ring=False):
    angles_1 = [60, 120]
    angles_2 = [30, 90, 150]

    corrs_1, corrs_2 = [], []
    for angle in angles_1:
        corrs_1.append(get_corr_rot(auto, angle, radius, ring=ring))

    for angle in angles_2:
        corrs_2.append(get_corr_rot(auto, angle, radius, ring=ring))

    return min(corrs_1) - max(corrs_2)


def detect_peaks(image, thresh=None):
    """
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    if thresh is not None:
        image = cp.deepcopy(image)
        # breakpoint()
        image[image < thresh * np.max(image)] = 0.0

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    y_peaks, x_peaks = np.where(detected_peaks)

    return detected_peaks, y_peaks, x_peaks


def get_radius(y_peaks, x_peaks, y_c, x_c, radius_lim):
    dists = []
    for b, a in zip(y_peaks, x_peaks):
        dists.append(np.sqrt((x_c - a) ** 2 + (y_c - b) ** 2))
    dists.sort()

    try:
        radius = np.mean(dists[5:7])
    except IndexError:
        radius = 8

    if np.isnan(radius):  # if no peaks
        radius = 8
    if radius >= radius_lim:
        radius = int(radius_lim * 0.75)

    return radius


def get_6_clostest_peaks(auto, radius, thresh=None):
    y_c, x_c = [int((x - 1) / 2) for x in auto.shape]

    _, y_peaks, x_peaks = detect_peaks(auto, thresh=thresh)
    # find x, y coordinates of 6 closest peaks
    y_x_ = []
    dists = []
    for b, a in zip(y_peaks, x_peaks):
        dist = np.sqrt((x_c - a) ** 2 + (y_c - b) ** 2)
        if 0 < dist <= radius:
            y_x_.append([b, a])
            dists.append(dist)
    # sort by distance
    idx = np.argsort(dists)
    idx = idx[:6]
    if len(dists) > 0:
        return [y_x_[a] for a in idx], np.mean(np.sort(dists)[:6])
    else:
        return [], np.nan


def get_field_size(auto, y_c, x_c):
    # central field size

    y_f, x_f = np.where(auto <= auto[y_c, x_c] / 2)
    dists_f = []
    for b, a in zip(y_f, x_f):
        dists_f.append(np.sqrt((x_c - a) ** 2 + (y_c - b) ** 2))
    dists_f.sort()

    if len(dists_f) > 0:
        field_size = dists_f[0]

        return field_size
    else:
        return 0.0


def elliptical_fit(y_x_):
    a = np.zeros((5, 5))
    skip_rand = np.random.randint(0, 6)
    counter = 0
    for i, (y, x) in enumerate(y_x_):
        if i == skip_rand:
            continue
        a[counter, 0] = x ** 2
        a[counter, 1] = x * y
        a[counter, 2] = y ** 2
        a[counter, 3] = x
        a[counter, 4] = y
        counter += 1

    f = 1
    a, b, c, d, e = np.dot(np.matmul(np.linalg.inv(np.matmul(a.T, a)), a.T), f * np.array([1, 1, 1, 1, 1]))
    discrim = b * b - 4 * a * c
    m1 = 2 * (a * e * e + c * d * d - b * d * e + discrim * f)
    semi_major = -np.sqrt(m1 * (a + c + np.sqrt((a - c) ** 2 + b * b))) / discrim
    semi_minor = -np.sqrt(m1 * (a + c - np.sqrt((a - c) ** 2 + b * b))) / discrim

    theta = np.arctan((c - a - np.sqrt((a - c) ** 2 + b * b)) / b)

    return theta, semi_major, semi_minor


def ellipse_correct(auto, theta, semi_major, semi_minor):
    auto_rot = auto_rotate(auto, +theta * 180 / np.pi)

    if np.abs(semi_minor) <= 1e-8 or np.isnan(semi_minor):
        raise ValueError('semi minor axis is nan or too small')
    if np.isnan(semi_major) or np.abs(semi_major) > 10000:
        raise ValueError('semi major axis is nan or too big')
    ratio = semi_major / semi_minor

    y, x = np.shape(auto_rot)
    auto_rot = resize(auto_rot, (int(y * ratio), x))
    auto_correct = auto_rotate(auto_rot, -theta * 180 / np.pi)

    y_diff, x_diff = [np.maximum(int((x - y) / 2), 1) for x, y in zip(np.shape(auto_correct), np.shape(auto))]
    auto_correct = auto_correct[y_diff:-y_diff, x_diff:-x_diff]
    return auto_correct


def grid_score_scale_analysis(auto, fit_ellipse=True, ring=False):
    theta = np.nan
    # find scale limit guess
    radius_lim = min(np.shape(auto)) / 2 + 5
    # get center of auto
    y_c, x_c = [int((x - 1) / 2) for x in auto.shape]
    # detect local peaks
    detected_peaks, y_peaks, x_peaks = detect_peaks(auto, thresh=0.3)
    # plt.scatter(x_peaks, y_peaks, c='w', s=3)
    # get spatial scale
    radius = get_radius(y_peaks, x_peaks, y_c, x_c, radius_lim) + get_field_size(auto, y_c, x_c)
    # get 6 local peaks
    y_x_, grid_scale = get_6_clostest_peaks(auto, radius, thresh=0.3)
    # print(y_x_)
    # fit an ellipse
    if fit_ellipse:
        try:
            theta, semi_major, semi_minor = elliptical_fit(y_x_)
            # correct for ellipse
            auto_correct = ellipse_correct(auto, theta, semi_major, semi_minor)
            # print(str(i))
            if np.min(auto_correct.shape) == 0:
                raise ValueError('auto got squished to zero')
        except (np.linalg.LinAlgError, ValueError, OverflowError):  # as e
            # plt.title('n/a')
            # print(str(i) + ' didnt fit ellipse because of ' + str(e))
            return np.nan, np.nan, np.nan

        # find scale limit guess
        radius_lim = max(np.shape(auto_correct)) / 2 + 5
        # get new center
        y_c, x_c = [int((x - 1) / 2) for x in auto_correct.shape]
        # get peaks
        detected_peaks, y_peaks, x_peaks = detect_peaks(auto_correct, thresh=0.3)
        # find scale
        radius = get_radius(y_peaks, x_peaks, y_c, x_c, radius_lim) + get_field_size(auto_correct, y_c, x_c)
    else:
        auto_correct = auto

    g_s_possible = []
    for radius_ in [radius - 6, radius - 5, radius - 4, radius - 3, radius - 2, radius - 1, radius, radius + 1,
                    radius + 2, radius + 3]:
        try:
            grid_score = get_grid_score(auto_correct, radius_, ring=ring)
            g_s_possible.append(grid_score)
        except IndexError:
            pass
    if len(g_s_possible) > 0:
        grid_score = np.nanmax(g_s_possible)
        return grid_score, grid_scale, theta
        # plt.title(str(grid_score)[:5] + '  ' + str(grid_scale)[:4])
    else:
        # plt.title('n/a')
        return np.nan, np.nan, np.nan


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information."""
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = mutual_info_score(ys[j, :], mus[i, :])
    return m


def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = mutual_info_score(ys[j, :], ys[j, :])
    return h


def histogram_discretize(target, num_bins=20):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(
            target[i, :], num_bins)[1][:-1])
    return discretized


def modularity(mutual_information):
    """Computes the modularity from mutual information."""
    # Mutual information has shape [num_codes, num_factors].
    squared_mi = np.square(mutual_information)
    max_squared_mi = np.max(squared_mi, axis=1)
    numerator = np.sum(squared_mi, axis=1) - max_squared_mi
    denominator = max_squared_mi * (squared_mi.shape[1] - 1.)
    delta = numerator / denominator
    modularity_score = 1. - delta
    index = (max_squared_mi == 0.)
    modularity_score[index] = 0.
    return np.mean(modularity_score)


def compute_score_matrix(mus, ys, mus_test, ys_test, continuous_factors):
    """Compute score matrix as described in Section 3."""
    num_latents = mus.shape[0]
    num_factors = ys.shape[0]
    score_matrix = np.zeros([num_latents, num_factors])
    for i in range(num_latents):
        for j in range(num_factors):
            mu_i = mus[i, :]
            y_j = ys[j, :]
            if continuous_factors:
                # Attribute is considered continuous.
                cov_mu_i_y_j = np.cov(mu_i, y_j, ddof=1)
                cov_mu_y = cov_mu_i_y_j[0, 1] ** 2
                var_mu = cov_mu_i_y_j[0, 0]
                var_y = cov_mu_i_y_j[1, 1]
                if var_mu > 1e-12:
                    score_matrix[i, j] = cov_mu_y * 1. / (var_mu * var_y)
                else:
                    score_matrix[i, j] = 0.
            else:
                # Attribute is considered discrete.
                mu_i_test = mus_test[i, :]
                y_j_test = ys_test[j, :]
                classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
                classifier.fit(mu_i[:, np.newaxis], y_j)
                pred = classifier.predict(mu_i_test[:, np.newaxis])
                score_matrix[i, j] = np.mean(pred == y_j_test)
    return score_matrix


def compute_avg_diff_top_two(matrix):
    sorted_matrix = np.sort(matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])


def compute_importance_gbt(x_train, y_train, x_test, y_test):
    """Compute importance based on gradient boosted trees."""
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors],
                                 dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        model = ensemble.GradientBoostingClassifier()
        model.fit(x_train.T, y_train[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - sts.entropy(importance_matrix.T + 1e-11,
                            base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code * code_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - sts.entropy(importance_matrix + 1e-11,
                            base=importance_matrix.shape[0])


def completeness(importance_matrix):
    """"Compute completeness of the representation."""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)


def compute_mig(mus_1, fact_1, mus_2, fact_2, num_bins=20, dataset='s', remove_unused=True, compute_dci=False,
                neuron_used=None):
    """Computes score based on both training and testing codes and fact_1."""
    # extract factors
    fact_1 = np.stack([val for key, val in fact_1.items() if key not in ['image', 'input']], axis=0)
    fact_2 = np.stack([val for key, val in fact_2.items() if key not in ['image', 'input']], axis=0)

    assert fact_1.shape[0] == fact_2.shape[0]

    num_used_latents = mus_1.shape[0]
    if remove_unused:
        # remove when mu = zero
        not_zero = np.mean(mus_1, axis=1) != 0
        if np.sum(not_zero) >= 2:
            mus_1 = mus_1[not_zero, :]
            mus_2 = mus_2[not_zero, :]
            if neuron_used is not None:
                neuron_used = neuron_used[not_zero]

        num_used_latents = mus_1.shape[0]

        if neuron_used is not None:
            # neuron_grads = neuron_grads > 0.001
            num_used_latents = sum(neuron_used)
            if num_used_latents >= 2:
                mus_1 = mus_1[neuron_used, :]
                mus_2 = mus_2[neuron_used, :]

    score_dict = {'num used latents': num_used_latents}
    discretized_mus = histogram_discretize(mus_1, num_bins=num_bins)
    discretized_facts = histogram_discretize(fact_1, num_bins=num_bins)
    m = discrete_mutual_info(discretized_mus, discretized_facts)
    assert m.shape[0] == mus_1.shape[0]
    assert m.shape[1] == fact_1.shape[0]
    # m is [num_latents, num_factors]

    # compute mig
    entropy = discrete_entropy(discretized_facts)
    sorted_m = np.sort(m, axis=0)[::-1]
    score_dict["discrete_mig"] = np.mean(
        np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))

    # compute mil
    # find maximum MI between a single latent and all fact_1, and divide by the total MI between latent and fact_1
    # possible should normalise m by entropy of each factor...
    # m_ = m / entropy[None, :]
    score = np.max(m, axis=1) / np.sum(m, axis=1)
    min_mil = 1.0 / m.shape[1]
    score_dict['discrete_mil'] = (np.mean(score) - min_mil) / (1 - min_mil)

    # compute modularity
    score_dict['discrete_mod'] = modularity(m)

    # compute dci (is slow)
    if compute_dci:
        """Computes score based on both training and testing codes and factors."""
        importance_matrix, train_err, test_err = compute_importance_gbt(mus_1, fact_1, mus_2, fact_2)
        score_dict["informativeness_train"] = train_err
        score_dict["informativeness_test"] = test_err
        score_dict["disentanglement"] = disentanglement(importance_matrix)
        score_dict["completeness"] = completeness(importance_matrix)

    # compute SAP
    """Computes score based on both training and testing codes and fact_1."""
    continuous_factors = True
    score_matrix = compute_score_matrix(mus_1, fact_1, mus_2, fact_2, continuous_factors)
    # Score matrix should have shape [num_latents, num_factors].
    assert score_matrix.shape[0] == mus_1.shape[0]
    assert score_matrix.shape[1] == fact_1.shape[0]
    score_dict["SAP_score"] = compute_avg_diff_top_two(score_matrix)

    return score_dict, (m, entropy)
