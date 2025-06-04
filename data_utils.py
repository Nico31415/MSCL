#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: XXX
"""

import tensorflow_datasets as tfds
import tensorflow as tf
import copy as cp
import datetime
import os
import logging
import numpy as np
import shutil
import glob

normalization_layer = tf.keras.layers.Rescaling(1. / 255)


def make_directories(base_path='../Summaries/'):
    """
    Creates directories for storing data during a model training run
    """

    # Get current date for saving folder
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    # Initialise the run and dir_check to create a new run folder within the current date
    run = 0
    dir_check = True
    # Initialise all pahts
    train_path, model_path, save_path, script_path, run_path = None, None, None, None, None
    # Find the current run: the first run that doesn't exist yet
    while dir_check:
        # Construct new paths
        run_path = base_path + date + '/run' + str(run) + '/'
        train_path = run_path + 'train'
        model_path = run_path + 'model'
        save_path = run_path + 'save'
        script_path = run_path + 'script'
        run += 1
        # And once a path doesn't exist yet: create new folders
        if not os.path.exists(train_path) and not os.path.exists(model_path) and not os.path.exists(save_path):
            try:
                os.makedirs(train_path)
                os.makedirs(model_path)
                os.makedirs(save_path)
                os.makedirs(script_path)
                dir_check = False
            except FileExistsError:
                continue

    # Save all python files in current directory to script directory
    files = glob.iglob(os.path.join('.', '*.py'))
    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, os.path.join(script_path, file))

    # Return folders to new path
    return run_path, train_path, model_path, save_path, script_path


def make_logger(run_path, name):
    """
    Creates logger so output during training can be stored to file in a consistent way
    """

    # Create new logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Remove anly existing handlers so you don't output to old files, or to new files twice
    # - important when resuming training existing model
    logger.handlers = []
    # Create a file handler, but only if the handler does
    handler = logging.FileHandler(run_path + name + '.log')
    handler.setLevel(logging.INFO)
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(handler)
    # Return the logger object
    return logger


def get_boundaries(pars, old_boundary_option=True):
    # if we want consistent object locations throughout training
    if pars.consistent_object_locations and old_boundary_option:
        # All objects need same location in all batches
        return cp.deepcopy(pars.objects_consistent)

    # boundary masks for each batch
    masks = np.ones((pars.batch_size, pars.n_locs, 1))
    start_states = np.zeros(pars.batch_size)

    heights, widths = [], []
    for batch in range(pars.batch_size):

        if batch > 0 and pars.consistent_object_locations:
            start_states[batch] = start_states[batch - 1]
            heights.append(heights[batch - 1])
            widths.append(widths[batch - 1])
            masks[batch] = masks[batch - 1]
            continue

        if pars.world_type == 'rectangle':
            heights.append(pars.height)
            widths.append(pars.width)

            start_states[batch] = np.random.choice(np.where(masks[batch, :, 0] > 0)[0])

        else:
            raise ValueError('boundaries not made for this world type')
    object_states, action_map, loss_scale_map, object_map, object_type = place_objects(pars, masks, heights, widths)

    return {'masks': masks,
            'start_states': start_states,
            'object_states': object_states,
            'action_map': action_map,
            'loss_scale_map': loss_scale_map,
            'object_map': object_map,
            'object_type': object_type,
            }


def place_objects(params, masks, heights, widths):
    # place objects in various positions
    object_states = [[] for _ in range(params.batch_size)]
    object_types = [[] for _ in range(params.batch_size)]
    action_map = np.zeros((params.batch_size, params.n_locs, params.action_dim))
    loss_scale_map = np.zeros((params.batch_size, params.n_locs, 1))
    object_map = [np.zeros((params.batch_size, params.n_locs, 1)) for _ in range(len(params.object_types))]

    for i, mask in enumerate(masks):

        if i > 0 and params.consistent_object_locations:
            object_states[i] = object_states[i - 1]
            action_map[i] = action_map[i - 1]
            loss_scale_map[i] = loss_scale_map[i - 1]
            for p, om in enumerate(object_map):
                object_map[p][i] = object_map[p][i - 1]
            object_types[i] = object_types[i - 1]
            continue

        action_maps_i, loss_scale_maps_i = [], []
        # find locations that are connected to and from. Then can place object at that location
        possible_object_states = np.where(mask > 0)[0]
        # make away from boundary
        possible_object_states = place_away_from_boundary(params, possible_object_states, heights[i], widths[i])
        for object_ in range(0, np.random.randint(params.n_objects_min, params.n_objects_max)):
            # make sure objects aren't too close together
            if object_ > 0:
                _, _, distances_to_object = distances(possible_object_states, object_states[i][-1], params)
                possible_object_states = possible_object_states[distances_to_object > params.min_object_distance]
            # add object location if possible
            if len(possible_object_states) > 0:
                object_states[i].append(np.random.choice(possible_object_states))
                # define object type
                object_type = np.random.choice(params.object_types)
                object_types[i].append(object_type)
                # place object in 'object map'
                object_map[params.object_types.index(object_type)][i, object_states[i][-1], 0] = 1.0
                # now return best actions for each location
                a_m, l_s_m = action_to_goal(np.arange(params.n_locs), object_states[i][-1], params)
                # reverse action map if object is repel
                if object_type == 'repel':
                    a_m = 1.0 - a_m
                action_maps_i.append(a_m)
                loss_scale_maps_i.append(l_s_m)

        if len(object_states[i]) < 1:
            continue

        # find locations influenced by which object
        distances_to_object = [distances(np.arange(params.n_locs), o_s, params)[2] for o_s in object_states[i]]
        object_influence = np.argmin(np.stack(distances_to_object, axis=0), axis=0)

        l_s_m_stacked = np.stack(loss_scale_maps_i, axis=0)
        loss_scale_map[i, :, 0] = l_s_m_stacked[object_influence, np.arange(object_influence.shape[0])]
        # actions towards object that influences the most
        action_map[i, ...] = np.stack(action_maps_i, axis=0)[object_influence, np.arange(object_influence.shape[0]), :]

    # return locations of objects, plus actions maps/potential for objects
    return object_states, action_map, loss_scale_map, object_map, object_types


def place_away_from_boundary(params, possible_object_states, height, width):
    width_boundary = np.logical_and(params.away_from_boundary <= possible_object_states % params.width,
                                    possible_object_states % params.width < width - params.away_from_boundary)
    height_boundary = np.logical_and(params.away_from_boundary <= possible_object_states / params.width,
                                     possible_object_states / params.width < height - params.away_from_boundary)
    possible_object_states = possible_object_states[np.logical_and(width_boundary, height_boundary)]

    return possible_object_states


def distances(s1, s2, params):
    x1 = s1 % params.width
    x2 = s2 % params.width

    y1 = np.floor(s1 / params.width)
    y2 = np.floor(s2 / params.width)

    dx = x2 - x1
    dy = y2 - y1

    return dx, dy, (dx ** 2 + dy ** 2) ** 0.5


def action_to_goal(s1, s2, params):
    # s1 is current state
    # s2 is goal
    # action is NOT one-hot - i.e. use sigmoid cross entropy
    dx, dy, _ = distances(s1, s2, params)

    # [right, left, up, down]
    action = np.zeros((*s1.shape, 4))
    # go right
    action[dx > 0, 0] = 1.0
    # go left
    action[dx < 0, 1] = 1.0
    # go up
    action[dy > 0, 2] = 1.0
    # go down
    action[dy < 0, 3] = 1.0

    try:
        scale = dx ** 2 + dy ** 2 < params.action_loss_scale ** 2
        # scale = np.exp(-(dx ** 2 + dy ** 2) / (params.action_loss_scale ** 2))
    except AttributeError:
        scale = dx ** 2 + dy ** 2 < params.action_loss_scale ** 2
        # scale = np.exp(-(dx ** 2 + dy ** 2) / (params.loss_scale ** 2))
    scale = np.float32(scale)

    return action, scale


def data_transform(x, weights, nonlinear=True):
    if isinstance(weights, list):
        for i, w in enumerate(weights):
            x = np.dot(x, w)
            if i < len(weights) - 1 and nonlinear:
                x = np.maximum(x, 0.0)
    else:
        x = np.dot(x, weights)
    return x


def sigmoid(x, beta=1.0):
    return 1.0 / (1.0 + np.exp(-beta * x))


def categorical_to_onehot(data):
    # x is list of data where each category has unique identifier
    n_categories = len(np.unique(data))
    categories_sorted = sorted(np.unique(data))
    data_onehot = []
    for x in data:
        x_one_hot = np.zeros(n_categories)
        category = categories_sorted.index(x)
        x_one_hot[category] = 1.0
        data_onehot.append(x_one_hot)

    return np.stack(data_onehot, axis=0)


def data_subspaces(par):
    # create dataset
    factor_all = []
    image_all = []
    dataset = None
    if par.dataset == 'dsprites':
        dataset = tfds.load('dsprites', split='train', shuffle_files=True, data_dir='../tensorflow_datasets')
    elif par.dataset == 'shapes3d':
        dataset = tfds.load('shapes3d', split='train', shuffle_files=True, data_dir='../tensorflow_datasets')
    elif par.dataset == 'factors':
        for i in range(par.n_data):
            factor = np.random.uniform(size=par.factor_dim)
            image = data_transform(factor, par.w_true_0, nonlinear=par.nonlinear_data)
            factor_all.append(factor)
            image_all.append(image)
    elif par.dataset == 'categorical':
        factors = np.eye(par.factor_dim)
        for i in range(par.n_data):
            factor = np.random.randint(par.factor_dim)
            factor_all.append(factors[factor])
            image_all.append(par.categorical_images[factor])
    else:
        raise ValueError('Incorrect dataset type')

    if par.dataset in ['factors', 'categorical']:
        factor_all = np.stack(factor_all, axis=0)
        image_all = np.stack(image_all, axis=0)

        if par.norm_synthetic_image:
            # make images have approx same variance/norm per output variable
            image_all = image_all * np.sqrt(image_all.shape[1] / np.sum(np.mean(image_all ** 2, axis=0)))

        factors = {'factor' + str(i): factor for i, factor in enumerate(factor_all.T)}

        if par.mix_input:
            input_all = np.matmul(factor_all, par.mix_rotation)
        else:
            input_all = factor_all

        if par.non_linear_activation_input:
            input_all = input_all ** 3

        if par.sigmoid_output:
            image_all = sigmoid(image_all)
            # normalise between 1 and 0
            image_all = (image_all - np.min(image_all, axis=0, keepdims=True)) / (
                    np.max(image_all, axis=0, keepdims=True) - np.min(image_all, axis=0, keepdims=True))
            if par.threshold_input:
                image_all[image_all >= 0.5] = 1.0
                image_all[image_all < 1.0] = 0.0

        dataset = tf.data.Dataset.from_tensor_slices({**factors, **{'image': image_all, 'input': input_all}})
    elif par.dataset in ['shapes3d']:
        # normalise data - for sigmoid cross entropy loss
        dataset = dataset.map(image_norm)

    if not par.graph_mode:
        dataset = dataset.take(par.metric_data_size + par.metric_data_size_2)  # for debugging

    # shuffle data
    dataset = dataset.shuffle(dataset.cardinality().numpy())

    dataset_batch = dataset.batch(par.batch_size)

    dataset_metric = [a for a in dataset.take(par.metric_data_size).batch(par.metric_data_size)][0]
    dataset_metric = {key: tf.cast(val, tf.float32) if key in ['image', 'input'] else tf.cast(val, tf.float32).numpy()
                      for key, val in dataset_metric.items() if 'value' not in key}

    dataset_ = dataset.skip(par.metric_data_size)
    dataset_metric_2 = [a for a in dataset_.take(par.metric_data_size_2).batch(par.metric_data_size_2)][0]
    dataset_metric_2 = {
        key: tf.cast(val, tf.float32) if key in ['image', 'input'] else tf.cast(val, tf.float32).numpy() for
        key, val in dataset_metric_2.items() if 'value' not in key}

    return dataset_batch, (dataset_metric, dataset_metric_2)


def image_norm(x):
    x['image'] = normalization_layer(x['image'])
    return x
