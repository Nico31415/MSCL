#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: XXX
"""

import importlib
import os
import gzip

import numpy as np
import pandas as pd
import copy as cp
import tensorflow as tf
import matplotlib.pyplot as plt
import data_utils as d_u

try:
    from tbparse import SummaryReader
except ModuleNotFoundError:
    pass

import model_utils

interpolation_method = 'None'
# fontsize = 25
# linewidth = 4
# labelsize = 20

COLOR = "black"
plt.rcParams["text.color"] = COLOR
plt.rcParams["axes.labelcolor"] = COLOR
plt.rcParams["xtick.color"] = COLOR
plt.rcParams["ytick.color"] = COLOR

fontsize = 15
linewidth = 3
labelsize = 15
legendsize = 12
legend_loc = (0.05, 0.15)
legend_ncol = 2
window_len = 10
dpi = 300
cs = ['white', 'pink']


def get_model(model_path, script_path, save_path, model_type='invar', use_old_scripts=True):
    """
    Load a trained model from a previous training run and save outputs
    """
    # Make sure there is a trained model for the requested index (training iteration)
    if os.path.isfile(model_path + '.index'):
        if model_type == 'invar':
            model_name = 'model_invariant'
            model_script_path = '/model_invariant.py'
        elif model_type == 'subspaces':
            model_name = 'subspaces'
            model_script_path = '/model_subspaces.py'
        elif model_type == 'subspaces_vae':
            model_name = 'subspaces_vae'
            model_script_path = '/model_subspaces.py'
        else:
            raise ValueError('incorrect model type given')

        # Load model module from stored scripts of trained model
        spec_model = importlib.util.spec_from_file_location(model_name, script_path + model_script_path) \
            if use_old_scripts else importlib.util.spec_from_file_location(model_name, model_script_path)
        stored = importlib.util.module_from_spec(spec_model)
        spec_model.loader.exec_module(stored)
        # Load data_utils from stored scripts of trained model
        spec_data_utils = importlib.util.spec_from_file_location("data_utils", script_path + '/data_utils.py') \
            if use_old_scripts else importlib.util.spec_from_file_location("data_utils", 'data_utils.py')
        stored_data_utils = importlib.util.module_from_spec(spec_data_utils)
        spec_data_utils.loader.exec_module(stored_data_utils)
        # Load model_utils from stored scripts of trained model
        spec_model_utils = importlib.util.spec_from_file_location("model_utils", script_path + '/model_utils.py') \
            if use_old_scripts else importlib.util.spec_from_file_location("model_utils", 'model_utils.py')
        stored_model_utils = importlib.util.module_from_spec(spec_model_utils)
        spec_model_utils.loader.exec_module(stored_model_utils)
        # Load the parameters of the model
        params = stored_model_utils.DotDict(np.load(save_path + '/params.npy', allow_pickle=True).item())
        # Create a new  model with the loaded parameters
        if model_type == 'invar':
            try:
                model = stored.PathIntegrator(params)
            except:
                model = stored.PatternLearner(params)
        elif model_type == 'subspaces':
            model = stored.SubspaceNet(params)
        elif model_type == 'subspaces_vae':
            model = stored.SubspaceVAE(params)
        else:
            raise ValueError('incorrect model type given')
        # Load the model weights after training
        model.load_weights(model_path)
    else:
        print('Error: no trained model found for ' + model_path)
        # Return None to indicate error
        model, params = None, None
        stored_data_utils, stored_model_utils = None, None
    # Return loaded and trained model
    return model, params, stored_model_utils, stored_data_utils


def set_directories(date, run, base_path='../Summaries/'):
    """
    Returns directories for storing data during a model training run from a given previous training run
    """

    # Initialise all paths
    run_path = base_path + date + '/run' + str(run) + '/'
    train_path = run_path + 'train'
    model_path = run_path + 'model'
    save_path = run_path + 'save'
    script_path = run_path + 'script'
    # Return folders to new path
    return run_path, train_path, model_path, save_path, script_path


def load_numpy_gz(file_name):
    try:
        return np.load(file_name, allow_pickle=True)
    except FileNotFoundError:
        f = gzip.GzipFile(file_name + '.gz', "r")
        return np.load(f, allow_pickle=True)


def get_ratemap(cell, n_locs_dim_, indices_):
    rate_map = np.zeros((n_locs_dim_, n_locs_dim_))
    rate_map[indices_[:, 1], indices_[:, 0]] = cell
    return rate_map[::-1, :]


def autocorr2d_no_nans(image, torus=False):
    # for square_worlds
    # DOESNT WORK FOR HEX WORLDS

    y_lim, x_lim = np.shape(image)

    if torus:
        auto = np.zeros_like(image)
        for y_shift in range(-int(y_lim / 2), np.ceil(y_lim / 2).astype(int)):
            for x_shift in range(-int(x_lim / 2), np.ceil(x_lim / 2).astype(int)):
                im_new = np.roll(image, y_shift, axis=0)
                im_new = np.roll(im_new, x_shift, axis=1)
                # correlate
                auto[y_shift + int(y_lim / 2), x_shift + int(x_lim / 2)] = \
                    np.corrcoef(image.flatten(), im_new.flatten())[0][1]

        return auto

    auto = np.zeros((2 * y_lim - 1, 2 * x_lim - 1))
    section_1, section_2 = None, None
    for y_shift in range(-y_lim + 1, y_lim):
        for x_shift in range(-x_lim + 1, x_lim):
            # y shift
            if y_shift == 0:
                section_1 = image
                section_2 = image
            elif y_shift > 0:  # shift down
                section_1 = image[y_shift:, :]
                section_2 = image[:-y_shift, :]
            elif y_shift < 0:  # shift up
                section_1 = image[:y_shift, :]
                section_2 = image[-y_shift:, :]
            # x_shift
            if x_shift == 0:
                section_1 = section_1
                section_2 = section_2
            elif x_shift > 0:  # shift right
                section_1 = section_1[:, x_shift:]
                section_2 = section_2[:, :-x_shift]
            elif x_shift < 0:  # shift left
                section_1 = section_1[:, :x_shift]
                section_2 = section_2[:, -x_shift:]

            if section_1.size == 0 or section_2.size == 0:
                print(y_lim, x_lim, y_shift, x_shift)

            not_allowed = np.logical_or(np.isnan(section_1), np.isnan(section_2))

            auto[y_shift + y_lim - 1, x_shift + x_lim - 1] = \
                np.corrcoef(section_1[~not_allowed].flatten(), section_2[~not_allowed].flatten())[0][1]

    return auto


def square_auto_cell(cell, y_, x_):
    auto = autocorr2d_no_nans(np.reshape(cell, (y_, x_)))

    mask = np.ones_like(auto)
    ys, xs = np.shape(mask)

    radius_sq = (3 / 4) * y_ ** 2
    # print(ys, xs)
    for y in range(ys):
        for x in range(xs):
            if (y - y_ + 1) ** 2 + (x - x_ + 1) ** 2 > radius_sq:  # 3/4 for hexagon sides as closer at 30 degrees
                mask[y, x] = np.nan
    auto = auto * mask
    radius = np.sqrt(radius_sq)

    y_indent = int(np.floor(ys / 2) - np.floor(radius)) - 1
    x_indent = int(np.floor(xs / 2) - np.floor(radius)) - 1

    if x_indent == 0 and y_indent == 0:
        auto = auto
    elif x_indent == 0:
        auto = auto[y_indent: -y_indent, :]
    elif y_indent == 0:
        auto = auto[:, x_indent: -x_indent]
    else:
        auto = auto[y_indent: -y_indent, x_indent: -x_indent]

    return auto


def generate_semicircle(center_x, center_y, radius, stepsize=0.1):
    """
    generates coordinates for a semicircle, centered at center_x, center_y
    """

    x = np.arange(center_x, center_x + radius + stepsize, stepsize)
    y = np.sqrt(radius ** 2 - x ** 2)

    # since each x value has two corresponding y-values, duplicate x-axis.
    # [::-1] is required to have the correct order of elements for plt.plot.
    x = np.concatenate([x, x[::-1]])

    # concatenate y and flipped y.
    y = np.concatenate([y, -y[::-1]])

    return x, y + center_y


def get_tensorboard_df(data, paths, min_steps=300):
    data['dfs'] = None
    info = data['info']
    labels = []
    dfs_all = []
    cutoff = 99999
    reader = None
    for key, value in info.items():
        labels.append(key)
        dfs = []
        for date, runs in value.items():
            for run in runs:
                for path in paths:
                    try:
                        log_dir = path + date + "/run" + str(run)
                        reader = SummaryReader(log_dir, pivot=True)
                    except ValueError:
                        pass
                if len(reader.tensors) >= min_steps:
                    dfs.append(reader.tensors)
                    print(key + ", date: " + date + ", run: " + str(run) + ', steps: ' + str(len(reader.tensors)))
                    cutoff = np.minimum(cutoff, len(reader.tensors))
                else:
                    print(key + ", date: " + date + ", run: " + str(run) + ', steps: VOID (' + str(
                        len(reader.tensors)) + ')')
        dfs_all.append(dfs)
    dfs_all_, labels_ = [], []
    for dfs, lab in zip(dfs_all, labels):
        if len(dfs) > 0:
            dfs_all_.append(dfs)
            labels_.append(lab)

    data['dfs'] = dfs_all_
    data['labels'] = labels_
    data['cutoff'] = cutoff

    return data


def show_df_names(df, to_print=True):
    names = []
    for col in df[0][0]:
        if "weights" not in col:
            if to_print:
                print(col)
            names.append(col)
    return names


def plot_train_curve(data__, metric_id, metric_label, save_path, label_keep=-1, ylim=(0, 1), cutoff=None,
                     legend_loc_=None, figsize=None, legend_ncol_=None):
    if legend_ncol_ is None:
        legend_ncol_ = legend_ncol
    if figsize:
        plt.figure(figsize=figsize)

    step = data__['dfs'][0][0]['step'][:data__['cutoff']]
    dfs_all_acc = [[a[metric_id][:data__['cutoff']] for a in x] for x in data__['dfs']]
    dfs_all_acc_mean = [pd.concat(a, axis=1).mean(axis=1).rolling(window=window_len, min_periods=0).mean() for a in
                        dfs_all_acc]
    dfs_all_acc_sem = [pd.concat(a, axis=1).sem(axis=1).rolling(window=window_len, min_periods=0).mean() for a in
                       dfs_all_acc]

    for mean, sem, label in zip(dfs_all_acc_mean, dfs_all_acc_sem, data__['labels']):
        plt.plot(step, mean, label=label[:label_keep], linewidth=linewidth)
        plt.fill_between(step, mean - sem, mean + sem, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=legend_loc_ if legend_loc_ else legend_loc, fontsize=legendsize,
               ncol=legend_ncol_)
    plt.xlabel('Gradient Step', fontsize=fontsize)
    plt.ylabel(metric_label, fontsize=fontsize)
    plt.xlim(0, cutoff if cutoff else step.iloc[data__['cutoff'] - 1])
    plt.ylim(ylim)
    plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
    plt.show()
    return


def plot_training_curves_layers(data__, metric_id, metric_label, save_path, label_keep=-1, ylim=(0, 1),
                                legend_ncol_=None, legend_loc_=None):
    if legend_ncol_ is None:
        legend_ncol_ = legend_ncol
    names = show_df_names(data__['dfs'], to_print=False)
    step = data__['dfs'][0][0]['step'][:data__['cutoff']]

    for name in names:
        if metric_id not in name:
            continue
        else:
            label = 'Layer ' + str(name[-1:])

        dfs_all_acc = [[a[name][:data__['cutoff']] for a in x] for x in data__['dfs']]
        dfs_all_acc_mean = [pd.concat(a, axis=1).mean(axis=1).rolling(window=window_len, min_periods=0).mean() for a in
                            dfs_all_acc]
        dfs_all_acc_sem = [pd.concat(a, axis=1).sem(axis=1).rolling(window=window_len, min_periods=0).mean() for a in
                           dfs_all_acc]

        for i, (mean, sem, label_) in enumerate(zip(dfs_all_acc_mean, dfs_all_acc_sem, data__['labels'])):
            plt.figure(i)
            plt.plot(step, mean, label=label[:label_keep], linewidth=linewidth)
            plt.fill_between(step, mean - sem, mean + sem, alpha=0.3)

            plt.legend(loc='center left', bbox_to_anchor=legend_loc_ if legend_loc_ else legend_loc,
                       fontsize=legendsize, ncol=legend_ncol_)
            plt.xlabel('Gradient Step', fontsize=fontsize)
            plt.ylabel(metric_label, fontsize=fontsize)
            plt.xlim(0, step.iloc[data__['cutoff'] - 1])
            plt.ylim(ylim)

            plt.savefig(save_path + label_[-7:] + "_.png", dpi=300, bbox_inches='tight')
    plt.show()


def get_pattern_data(date, run, index, base_path):
    model_type = 'invar'
    # Get directories for the requested run
    run_path, train_path, model_path, save_path, script_path = set_directories(date, run, base_path=base_path)
    if index == None:
        index = max([int(x.split('invar_')[1].split('.index')[0]) for x in os.listdir(model_path) if 'index' in x])
        print(index)
    # Load model from file
    model, params, stored_mu, stored_du = get_model(model_path + '/invar_' + str(index), script_path, save_path,
                                                    use_old_scripts=True, model_type=model_type)

    import parameters
    par_new = parameters.default_params_pattern()
    for key in par_new.keys():
        try:
            params[key]
        except:
            params[key] = par_new[key]
    params.n_objects_min = 0

    # run model
    not_right_foramt = True
    inputs = None
    it, it_min, it_max = 0, 1, 10

    # if consistent_object_locations want to see what reps looks like when objects move...
    while not_right_foramt and it < it_max and (not params.consistent_object_locations or it < it_min):
        it += 1
        inputs = d_u.get_boundaries(params, old_boundary_option=False)
        if np.min([len(x) for x in inputs['object_states']]) == 0:
            not_right_foramt = False
    # print('objects orig', inputs['object_states'])
    env0 = np.argmin([len(x) for x in inputs['object_states']])
    env1 = np.argmax([len(x) for x in [y for i, y in enumerate(inputs['object_states']) if i != env0]])
    env1 = env1 + 1 if env1 >= env0 else env1
    # print(env0, env1)

    # if consistent_object_locations want to see what reps looks like when objects move...
    if params.consistent_object_locations:
        inputs_consistent = d_u.get_boundaries(params)
        # print('objects consistent', inputs_consistent['object_states'])

        inputs_consistent['masks'][env0] = inputs['masks'][env0]
        inputs_consistent['start_states'][env0] = inputs['start_states'][env0]
        inputs_consistent['object_states'][env0] = inputs['object_states'][env0]
        inputs_consistent['action_map'][env0] = inputs['action_map'][env0]
        inputs_consistent['loss_scale_map'][env0] = inputs['loss_scale_map'][env0]
        for p, om in enumerate(inputs_consistent['object_map']):
            inputs_consistent['object_map'][p][env0] = inputs['object_map'][p][env0]
        inputs_consistent['object_type'][env0] = inputs['object_type'][env0]
        inputs = cp.deepcopy(inputs_consistent)
    # print('objects', inputs['object_states'])
    # print('object_type')

    try:
        gs = model.get_all_reps(inputs)
    except:
        gs = model(inputs)
    g = gs[-1]

    # make grid maps
    mesh_vec_h = np.linspace(0, 2 * np.pi, params.height + 1)[:-1]
    mesh_vec_w = np.linspace(0, 2 * np.pi, params.width + 1)[:-1]

    X, Y = np.meshgrid(mesh_vec_w, mesh_vec_h)
    x_y = np.stack([X.flatten(), Y.flatten()], axis=1)

    info = model_utils.DotDict()
    info.env0 = env0
    info.env1 = env1
    info.index = index
    info.x_y = x_y

    return params, g, inputs, model, info


def plot_pattern_all_cells(inputs, g, params, info):
    env0 = info['env0']
    env1 = info['env1']
    index = info['index']
    x_y = info['x_y']
    save_path_ = info['save_path']

    s = 400 / np.sqrt(params.height * params.width)
    n = np.ceil(np.sqrt(params.ent_dim)).astype(int)
    fig_size_all_cells = (int(20 / np.sqrt((128 / params.ent_dim))), int(20 / np.sqrt((128 / params.ent_dim))))

    for batch in [env0, env1]:
        plt.figure(figsize=fig_size_all_cells)
        for i in range(params.ent_dim):
            g_to_plot = g.numpy()[batch, ...]
            plt.subplot(n, n, i + 1)
            ax = plt.gca()
            max_min = np.max(np.abs(g.numpy()[:, :, i]))
            ax.scatter(x_y[:, 0], x_y[:, 1], c=g_to_plot[:, i], cmap='jet', s=s, marker='s', vmax=max_min,
                       vmin=0.0 if params.activation != 'none' \
                                   or 'non_negativity' in params.losses else -max_min)
            if 'action' in params.losses:
                for j, o_m in enumerate(inputs['object_map']):
                    for o_s in list(np.where(o_m[batch][:, 0] > 0)[0]):
                        ax.scatter((o_s % params.width) * (2 * np.pi / params.width),
                                   int(o_s / params.width) * (2 * np.pi / params.height), c=cs[j])

            ax.set_xticks([])
            ax.set_yticks([])
            ax.patch.set_edgecolor('black')
            ax.patch.set_linewidth('1')
            # plot starting location
            if params.init_type == 'learn_single':
                # don't understand the yx order here. Must have messed something up somewhere
                start_state = inputs['start_states'][batch]
                start_x = (start_state % params.width) * 2 * np.pi / params.height
                start_y = int(start_state / params.width) * 2 * np.pi / params.width
                plt.scatter(start_y, start_x, c='w')
        if batch == env1:
            plt.savefig(save_path_ + 'cells' + '_with_object_' + str(index) + ".png", dpi=dpi, bbox_inches='tight')
        elif batch == env0:
            plt.savefig(save_path_ + 'cells' + '_no_object_' + str(index) + ".png", dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_pattern_metrics(model, g, info):
    index = info.index
    save_path_ = info.save_path

    plt.figure(figsize=(4, 4))
    abs_cell_decode = tf.reduce_sum(tf.math.abs(model.decode_module.weight), axis=1).numpy()
    prob_cell_decode = abs_cell_decode / np.sum(abs_cell_decode)
    abs_cell_action = tf.reduce_sum(tf.math.abs(model.action_module.weight), axis=1)
    prob_cell_action = abs_cell_action / np.sum(abs_cell_action)
    l_ness = 1 - np.mean(
        np.logical_and(prob_cell_action > np.mean(prob_cell_action), prob_cell_decode > np.mean(prob_cell_decode)))
    plt.scatter(prob_cell_decode, prob_cell_action)
    plt.xlim([0, max(prob_cell_decode)])
    plt.ylim([0, max(prob_cell_action)])
    plt.xlabel('Spatial contribution', fontsize=fontsize)
    plt.ylabel('Object contribution', fontsize=fontsize)

    plt.savefig(save_path_ + 'MIL_' + str(index) + ".png", dpi=dpi, bbox_inches='tight')

    print('l ness: ', l_ness)
    cosine = tf.reduce_sum(abs_cell_decode * abs_cell_action) / tf.sqrt(
        tf.reduce_sum(abs_cell_decode ** 2) * tf.reduce_sum(abs_cell_action ** 2))
    print('cosine: ', cosine.numpy())

    # order cells by their contirbution to decode and action, then do rank correlation?
    import scipy.stats as sts
    rho, pval = sts.spearmanr(np.argsort(prob_cell_decode), np.argsort(prob_cell_action))
    print(rho)

    plt.figure(figsize=(4, 4))
    # equally can look at distribution of change-ability of cells over different environments.
    diff_over_envs = tf.reduce_mean((tf.expand_dims(g, axis=1) - tf.expand_dims(g, axis=0)) ** 2, axis=2).numpy()
    iu1 = np.triu_indices(diff_over_envs.shape[0], 1)
    diff_over_envs_mean = np.mean(diff_over_envs[iu1], axis=0)
    # divide different by typical firing of that cell
    diff_over_envs_mean = diff_over_envs_mean / tf.reduce_mean(g ** 2, axis=(0, 1)).numpy()
    _ = plt.hist(diff_over_envs_mean, bins=20)
    plt.xlim(0, 2)
    plt.xlabel('Changeability', fontsize=fontsize)

    plt.savefig(save_path_ + 'Changeability_' + str(index) + ".png", dpi=dpi, bbox_inches='tight')

    # equally can look at distribution of change-ability of cells over different environments
    # -> this time how correlated it is
    def zscore(g, axis=-1):
        demeaned = g - np.mean(g, axis=axis, keepdims=True)
        return demeaned / np.sqrt(np.mean(demeaned ** 2, axis=axis, keepdims=True))

    plt.figure(figsize=(4, 4))
    corr_means = []
    for i, g_ in enumerate(g.numpy()[:-1]):
        for g__ in g.numpy()[i + 1:]:
            corr_means.append(np.mean(zscore(g_, axis=0) * zscore(g__, axis=0), axis=0))
    corr_over_envs_mean = np.mean(corr_means, axis=0)
    _ = plt.hist(corr_over_envs_mean, bins=20, range=(0, 1))
    plt.xlim(0, 1)
    plt.xlabel('Spatial correlation', fontsize=fontsize)

    plt.savefig(save_path_ + 'Spatial_' + str(index) + ".png", dpi=dpi, bbox_inches='tight')
    plt.show()

    return diff_over_envs_mean, corr_over_envs_mean


def get_object_surround(model, params):
    inputs_ = d_u.get_boundaries(params)
    try:
        gs_ = model.get_all_reps(inputs_)
    except:
        gs_ = model(inputs_)
    g_ = gs_[-1]
    # print('object_locs: ', inputs_['object_states'])
    # print('object_types: ', inputs_['object_type'])

    object_all_type = []
    for object_type in params.object_types:

        # get object then get all surrounding locations
        object_info_all = []
        for batch in range(params.batch_size):
            if len([x for x in inputs_['object_type'][batch] if
                    x == object_type]) < 2:  # len(inputs_['object_states'][batch]) < 2:
                # print('not enough objects')
                continue
            else:
                pass
                # print('enough objects')

            object_info = []
            for o, o_type in zip(inputs_['object_states'][batch], inputs_['object_type'][batch]):
                if o_type != object_type:
                    continue

                rel_locs = {}
                # surrounding locations
                relative_locs_x = np.arange(params.n_locs) % params.width - o % params.width
                relative_locs_y = (np.arange(params.n_locs) / params.width).astype(int) - int(o / params.width)
                for i, (rel_loc_x, rel_locs_y) in enumerate(zip(relative_locs_x, relative_locs_y)):
                    key = str(rel_loc_x) + '_' + str(rel_locs_y)
                    rel_locs[key] = g_[batch, i, :].numpy()

                object_info.append(rel_locs)
            object_info_all.append(object_info)
        object_all_type.append(object_info_all)
        # print(' ')
    return object_all_type


def spatial_corr(info_0, info_1, params):
    common_rel = list(set(info_0.keys()) & set(info_1.keys()))
    corrs = []
    for cell in range(params.ent_dim):
        cell_0 = []
        cell_1 = []
        for rel in common_rel:
            cell_0.append(info_0[rel][cell])
            cell_1.append(info_1[rel][cell])
        corrs.append(np.corrcoef(cell_0, cell_1)[1, 0])

    return np.asarray(corrs)


def get_mean_spatial_corrs(object_all_type, params):
    mean_corrs_cells = []
    mean_corrs = []
    for object_info in object_all_type[0]:
        corrs = []
        for i, info_0 in enumerate(object_info):
            for j, info_1 in enumerate(object_info):
                if j > i:
                    corr = spatial_corr(info_0, info_1, params)
                else:
                    continue
                corrs.append(corr)

        mean_corrs_cells.append(np.nanmean(corrs, axis=0))
        mean_corrs.append(np.nanmean(corrs))

    return np.nanmean(mean_corrs), np.nanmean(mean_corrs_cells, axis=0)
