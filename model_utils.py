#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: X
"""

import functools
import tensorflow as tf
import scipy.stats as sts
import numpy as np


class DotDict(dict):
    # dot.notation access to dictionary attributes

    def __getattr__(*args):
        try:
            val = dict.__getitem__(*args)
            return DotDict(val) if type(val) is dict else val
        except KeyError:
            raise AttributeError()

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def to_dict(data):
        """
        Recursively transforms a dict to a dotted dictionary
        """
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, DotDict):
                    data[k] = dict(v)
                    DotDict.to_dict(data[k])
                elif isinstance(v, list):
                    data[k] = [DotDict.to_dict(i) for i in v]
        elif isinstance(data, list):
            return [DotDict.to_dict(i) for i in data]
        else:
            return data

        return dict(data)


def define_scope(func):
    """Creates a name_scope that contains all ops created by the function.
    The scope will default to the provided name or to the name of the function
    in CamelCase. If the function is a class constructor, it will default to
    the class name. It can also be specified with name='Name' at call time.

    Is helpful for debugging!
    """

    name_func = func.__name__
    if name_func == '__init__':
        name_func = func.__class__.__name__
    name_func = camel_case(name_func)

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        # Local, mutable copy of `name`.
        name_to_use = name_func

        with tf.name_scope(name_to_use):
            return func(*args, **kwargs)

    return _wrapper


def camel_case(name):
    """Converts the given name in snake_case or lowerCamelCase to CamelCase."""
    words = name.split('_')
    return ''.join(word.capitalize() for word in words)


def embeddings_tf_data(coordinates_train, coordinates_test):
    return tf.data.Dataset.from_tensor_slices(coordinates_train), tf.data.Dataset.from_tensor_slices(coordinates_test)


def threshold(x, thresh_min, thresh_max, thresh_slope=0.01):
    between_thresh = tf.minimum(tf.maximum(x, thresh_min), thresh_max)
    above_thresh = tf.maximum(x, thresh_max) - thresh_max
    below_thresh = tf.minimum(x, thresh_min) - thresh_min

    return between_thresh + thresh_slope * (above_thresh + below_thresh)


def is_alive(g, n_loc_min=0.0):
    alive = tf.cast(tf.reduce_sum(g ** 2, axis=2, keepdims=True) > 0, dtype=tf.float32)
    num_active_locs = tf.reduce_sum(alive, axis=1, keepdims=True)
    # threshold number of alive locations
    num_active_locs = tf.maximum(tf.squeeze(num_active_locs), n_loc_min)
    return tf.stop_gradient(alive), tf.stop_gradient(num_active_locs)


def get_mask_num_locs(g, inputs, final=True, n_loc_min=0.0):
    if final:
        num_active_locs = tf.reduce_sum(inputs['masks'], axis=(1, 2))
        masks = inputs['masks']
    else:
        alive, num_active_locs = is_alive(g, n_loc_min=n_loc_min)
        masks = alive

    return masks, num_active_locs


def inputs2tf(inputs):
    tf_inputs = {'masks': tf.constant(inputs['masks'], dtype=tf.float32),
                 'start_states': tf.constant(inputs['start_states'], dtype=tf.int32),
                 'action_map': tf.constant(inputs['action_map'], dtype=tf.float32),
                 'object_map': [tf.constant(x, tf.float32) for x in inputs['object_map']],
                 'loss_scale_map': tf.constant(inputs['loss_scale_map'], dtype=tf.float32),
                 }

    return tf_inputs


def make_summaries(gs, inputs, losses, model, par):
    summaries = {}
    g = gs[-1]
    g_stacked = tf.stack(gs)

    # losses
    for key, val in losses.items():
        summaries['losses/' + key] = val
        summaries['losses_scaled/' + key] = par.beta_loss[key] * val

    # accuracies
    if 'decode' in par.losses:
        summaries['accuracy/decode'] = model.decode_module.accuracy(g, inputs)
    if 'action' in par.losses:
        summaries['accuracy/action'] = model.action_module.accuracy(g, inputs)
    if 'object' in par.losses:
        summaries['accuracy/object'] = model.object_module.accuracy(g, inputs)

    # properties of trainable variables
    if par.summary_weights:
        summaries = trainable_weight_summaries(summaries, model.trainable_variables)

        # properties of final g
        summaries['variables/g_mean'] = tf.reduce_mean(g).numpy()
        summaries['variables/g_sq'] = tf.reduce_mean(g ** 2).numpy()
        summaries['variables/g_var'] = summaries['variables/g_sq'] - summaries['variables/g_mean'] ** 2

        # properties of pattern forming
        summaries['variables/g_seq_max'] = tf.reduce_max(g_stacked).numpy()
        summaries['variables/g_seq_min'] = tf.reduce_min(g_stacked).numpy()
        summaries['variables/g_fin_sq_sub_init_sq'] = tf.reduce_mean(gs[-1] ** 2).numpy() - tf.reduce_mean(
            gs[0] ** 2).numpy()
        summaries['variables/g_fin_sub_init_sq'] = tf.reduce_mean((gs[-1] - gs[0]) ** 2).numpy()
        summaries['variables/g_fin_init_var'] = tf.math.reduce_variance(
            tf.reduce_mean(g_stacked ** 2, axis=(1, 2, 3)).numpy())

    # metrics for 'factorisation'
    abs_decode, abs_action = None, None
    if 'decode' in par.losses:
        abs_decode = tf.math.abs(model.decode_module.weight)
        prob_decode = abs_decode / tf.reduce_sum(abs_decode, axis=0, keepdims=True)
        decode_entropy = tf.reduce_mean(tf.reduce_sum(-prob_decode * tf.math.log(prob_decode), axis=0))
        summaries['metrics/decode_entropy'] = decode_entropy.numpy()
    if 'action' in par.losses:
        abs_action = tf.math.abs(model.action_module.weight)
        prob_action = abs_action / tf.reduce_sum(abs_action, axis=0, keepdims=True)
        action_entropy = tf.reduce_mean(tf.reduce_sum(-prob_action * tf.math.log(prob_action), axis=0))
        summaries['metrics/action_entropy'] = action_entropy.numpy()
    if 'decode' in par.losses and 'action' in par.losses:
        abs_cell_decode = tf.reduce_sum(abs_decode, axis=1)
        prob_cell_decode = abs_cell_decode / tf.reduce_sum(abs_cell_decode)
        abs_cell_action = tf.reduce_sum(abs_action, axis=1)
        prob_cell_action = abs_cell_action / tf.reduce_sum(abs_cell_action)
        l_ness = 1.0 - tf.reduce_mean(tf.cast(tf.logical_and(prob_cell_action > tf.reduce_mean(prob_cell_action),
                                                             prob_cell_decode > tf.reduce_mean(prob_cell_decode)),
                                              tf.float32))
        summaries['metrics/l_ness'] = l_ness.numpy()
        rho, pval = sts.spearmanr(tf.argsort(prob_cell_decode).numpy(), tf.argsort(prob_cell_action).numpy())
        summaries['metrics/rank_corr'] = rho
        cosine = tf.reduce_sum(abs_cell_decode * abs_cell_action) / tf.sqrt(
            tf.reduce_sum(abs_cell_decode ** 2) * tf.reduce_sum(abs_cell_action ** 2))
        summaries['metrics/cosine'] = cosine.numpy()

    return summaries


def reparameterize(inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def reconstruction_loss(true, pred, binary=False):
    if binary:
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=true, logits=pred), axis=1)
    else:
        loss = tf.reduce_sum(0.5 * (true - pred) ** 2, axis=1)

    return tf.reduce_mean(loss, axis=0)


def ent_reg(hidden):
    ent_reg_ = [tf.reduce_mean(tf.reduce_sum(0.5 * hid ** 2, axis=1), axis=0) for hid in hidden]
    return tf.add_n(ent_reg_)


def nonneg_reg(hidden):
    nonneg_reg_ = [tf.reduce_mean(tf.reduce_sum(tf.nn.relu(-hid), axis=1), axis=0) for hid in hidden]
    return tf.add_n(nonneg_reg_)


def sparse_reg(hidden):
    sparse_reg_ = [tf.reduce_mean(tf.reduce_sum(tf.math.abs(hid), axis=1), axis=0) for hid in hidden]
    return tf.add_n(sparse_reg_)


def weight_reg(weights, exclude=('abcdefghijklmnopqrstuvwxyz',)):
    exclude_ = ('bias',) + exclude
    weight_reg = [tf.reduce_sum(tf.math.square(w)) for w in weights if np.all([y not in w.name for y in exclude_])]
    return tf.add_n(weight_reg)


def weight_reg_l1(weights, exclude=('abcdefghijklmnopqrstuvwxyz',)):
    exclude_ = ('bias',) + exclude
    weight_reg_l1 = [tf.reduce_sum(tf.math.abs(w)) for w in weights if np.all([y not in w.name for y in exclude_])]
    return tf.add_n(weight_reg_l1)


def log_normal_pdf(sample, mean, logvar):
    log2pi = tf.math.log(2. * np.pi)
    return -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)


def make_summaries_subspaces(gradient_time, summary_time, step_time, losses, metrics, model, betas, data, rec, par):
    data = tf.reshape(data, (data.shape[0], -1))
    rec = tf.reshape(rec, (rec.shape[0], -1))
    summaries = {'extras/gradient_time': gradient_time,
                 'extras/summary_time': summary_time,
                 'extras/step_time': step_time,
                 'accuracies/r2': 1.0 - tf.reduce_mean(tf.reduce_sum((data - rec) ** 2, axis=1) / tf.reduce_sum(
                     (data - tf.reduce_mean(data, axis=1, keepdims=True)) ** 2, axis=1)).numpy(),
                 }

    # losses
    for key, val in losses.items():
        summaries['losses/' + key] = val
        summaries['losses_scaled/' + key] = betas[key] * val

    if isinstance(metrics, list):
        for i, m in enumerate(metrics):
            for key, val in m.items():
                summaries['metrics/' + key + '_' + str(i)] = val
    else:
        for key, val in metrics.items():
            summaries['metrics/' + key] = val

    if par.summary_weights:
        summaries = trainable_weight_summaries(summaries, model.trainable_variables)

    return summaries


def neuron_loss_importance(model, hidden, loss_orig, target, start=1):
    # find which cells have an impact on final loss
    loss_change_all = []
    for j, hid in enumerate(hidden):
        loss_change = []
        for i in range(hid.shape[-1]):
            hid_lesion = tf.identity(hid)
            col_to_zero = [i]  # <-- column numbers you want to be zeroed out
            tnsr_shape = tf.shape(hid_lesion)
            mask = [tf.one_hot(col_num * tf.ones((tnsr_shape[0],), dtype=tf.int32), tnsr_shape[-1])
                    for col_num in col_to_zero]
            mask = tf.reduce_sum(mask, axis=0)
            mask = tf.cast(tf.logical_not(tf.cast(mask, tf.bool)), tf.float32)

            hid_lesion = hid_lesion * mask
            activities_lesion, _ = model(hid_lesion, index=j + start)
            loss_new = reconstruction_loss(target, activities_lesion[-1])
            loss_change.append(loss_new / (loss_orig + 1e-8))

        loss_change = tf.stack(loss_change)
        loss_change_all.append(loss_change.numpy())

    return loss_change_all


def important_neuroms(model, inputs, targets, pars, begin_layer=1):
    # BY NEURON EFFECT ON RECONSTRUCTION (by approx gradient)
    with tf.GradientTape() as tape:
        try:
            xs_, logits = model(inputs, training=False)
            loss_value, losses_ = model.get_losses(targets, xs_, pars.betas, logits)
        except ValueError:
            xs_ = model(inputs, training=False)
            loss_value, losses_ = model.get_losses(targets, xs_, pars.betas)
    gradients = tape.gradient(losses_['rec'], model.trainable_variables)
    neuron_grads = [tf.reduce_sum(tf.math.abs(x), axis=1).numpy() for x in gradients[begin_layer:] if len(x.shape) > 1]
    thresh_div = 4.5
    neuron_importances_gradient = [x > np.mean(np.sort(x)[::-1][:pars.factor_dim]) / thresh_div for x in neuron_grads]

    # BY NEURON OUTPUT WEIGHT
    thresh_div = 4.5
    weights = [tf.math.abs(x) for x in model.trainable_variables if 'weight' in x.name][begin_layer:]
    neuron_weights = [np.sum(tf.math.abs(x).numpy(), axis=1) for x in weights]
    neuron_importances_weight = [x > np.mean(np.sort(x)[::-1][:pars.factor_dim]) / thresh_div for x in neuron_weights]

    return [np.logical_and(n_i_w, n_i_g) for n_i_w, n_i_g in
            zip(neuron_importances_weight, neuron_importances_gradient)]


def trainable_weight_summaries(summaries, trainable_variables):
    for x in trainable_variables:
        summaries['weights/' + x.name + '_mean'] = tf.reduce_mean(x).numpy()
        summaries['weights/' + x.name + '_sq'] = tf.reduce_mean(x ** 2).numpy()
        summaries['weights/' + x.name + '_var'] = summaries['weights/' + x.name + '_sq'] - summaries[
            'weights/' + x.name + '_mean'] ** 2

    return summaries


class GECO:
    def __init__(self, geco_pars, beta):
        super(GECO, self).__init__()

        self.threshold = geco_pars['threshold']
        self.alpha = geco_pars['alpha']
        self.gamma = geco_pars['gamma']
        self.moving_average = 0.0
        self.batch_index = 0
        self.beta = beta

    def update(self, loss):
        constraint = loss - self.threshold
        if self.batch_index == 0:
            self.moving_average = constraint
        else:
            self.moving_average = self.alpha * self.moving_average + (1 - self.alpha) * constraint
        constraint = constraint + tf.stop_gradient(self.moving_average - constraint)

        # update beta params
        self.beta = self.beta * tf.exp(self.gamma * constraint)
        self.batch_index += 1

        return self.beta
