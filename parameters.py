#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: XXX
"""

import model_utils
import data_utils as d_u
import numpy as np
import os
from plotting_functions import load_numpy_gz
from model_utils import DotDict as Dd
from scipy.stats import ortho_group


def default_params_pattern():
    params = model_utils.DotDict()

    # debugging
    params.graph_mode = True

    # environment params
    params.summary_weights = False
    params.world_type = 'rectangle'
    params.torus = False
    params.batch_size = 8
    params.height = 16
    params.width = 16
    params.height_min = params.height - 2
    params.width_min = params.width - 2
    params.n_locs = params.height * params.width
    params.opposite = model_utils.DotDict({'left': 'right',
                                           'right': 'left',
                                           'up': 'down',
                                           'down': 'up'}, )
    params.adjacencies = get_adjacencies(params)
    params.n_objects_min = 0
    params.n_objects_max = 8
    params.min_object_distance = 8
    params.away_from_boundary = 0
    params.object_types = ['attract']  # , 'repel']  # , 'repel']
    params.consistent_object_locations = False
    if params.consistent_object_locations:
        params.n_objects_min = 2

    params.activation = 'relu'  # 'relu'  # 'none', 'relu', 'leaky_relu'

    # init params
    params.init_type = 'learn_everywhere'  # 'learn_everywhere'
    # model params
    params.ent_dim = 128
    params.action_dim = 4
    params.action_loss_scale = params.min_object_distance / 2
    params.inverse_actions = False  # doesn't really make sense for non-linear activation
    params.group_common_train = False
    params.transition_bias = False if params.init_type == 'learn_single' else False
    params.thresh_ent_max = 2.5
    params.thresh_ent_min = -2.5
    params.second_error = 1.0
    params.leaky_relu_alpha = 0.01
    params.normalisation = 'none'
    # pattern dynamics params
    params.n_steps = params.height + params.width + 10  # 0
    if params.n_steps < params.height:
        params.init_type = 'learn_everywhere'
    params.beta_softmax = 32.0 if params.normalisation == 'space' else 2.0
    if params.activation == 'leaky_relu':
        params.beta_softmax *= 0.5
    params.softmax_target = 0.9
    params.use_object_pattern_update = True
    params.object_pattern_update_object_location_only = True
    params.eta_transition = 0.15
    params.eta_object = 1.0
    params.clip_pattern_gradients = True
    params.pattern_clip_val = 8.0
    # learning params
    params.learning_rate = 10e-4
    params.train_steps = 50000
    params.summary_interval = 10
    params.save_interval = 500
    # which losses to use, and their scaling values
    params.ent_dim_loss_divide = False
    params.losses = ['transition', 'decode', 'action', 'object', 'ent_reg', 'l2_weight_reg']
    # regularise action predictive weights ->
    params.beta_loss = model_utils.DotDict({
        'loss': 1.0,  # overall scaling of loss. pointless, but a hack to make a bit of code run.
        'transition': 40.0 / 128,
        'decode': 0.01,
        'action': 0.04,
        'object': 0.1,
        'ent_reg': 0.001 / 128,
        'non_negativity': 0.01 / 128,
        'l2_weight_reg': 1e-7,
    })

    if params.ent_dim_loss_divide:
        params.beta_loss.transition *= 128 / params.ent_dim
        params.beta_loss.ent_reg *= 128 / params.ent_dim
        params.beta_loss.non_negativity *= 128 / params.ent_dim
        params.beta_loss.orthogonal *= (128 / params.ent_dim) ** 2

    params.g_init_min = -2.0
    params.g_init_max = 2.0

    if params.consistent_object_locations:
        params.away_from_boundary = 2
        params.objects_consistent = d_u.get_boundaries(params, old_boundary_option=False)

    return params


def default_params_subspaces_net(experiment_type=None):
    params = model_utils.DotDict()
    params.graph_mode = True
    params.summary_interval = 1000
    params.save_interval = 20000
    params.summary_weights = False

    # synthetic data
    params.n_data = 200000
    params.experiment_type = 'shallow_lin_factors' if experiment_type is None else experiment_type
    params.net_type = 'deep'  # 'shallow', 'medium'
    params.nonlinear_data = False  # just for factors / 2 subspaces
    params.non_linear_activation_input = False  # just for factors / 2 subspaces
    params.sigmoid_output = False
    params.mix_input = True
    params.threshold_input = False
    params.norm_synthetic_image = False
    params.sigma = 0.01
    hidden_mult = 30
    if params.experiment_type == 'shallow_lin_factors':
        params.dataset = 'factors'  # 'dsprites' 'shapes3d', 'factors'
        params.factor_dim = 6  # input dim
        params.image_dim = 6  # output dim
        params.net_type = 'shallow'
    elif params.experiment_type == 'deep_lin_factors':
        params.dataset = 'factors'
        params.factor_dim = 6  # input dim
        params.image_dim = 6  # output dim
    elif params.experiment_type == 'deep_nonlin_factors':
        params.dataset = 'factors'
        params.factor_dim = 6  # input dim
        params.image_dim = 6  # output dim
        params.nonlinear_data = True
        params.non_linear_activation_input = True
    else:
        raise ValueError('Incorrect Experiment type given')

    # for factors and 2_subspaces synthetic data
    params.input_dim = params.factor_dim
    params.output_dim = params.image_dim
    params.mix_rotation = ortho_group.rvs(dim=params.input_dim) if params.mix_input else np.eye(params.input_dim)
    gen_dims = [params.factor_dim, params.image_dim]  # , int((params.factor_dim + params.image_dim) / 2)
    params.w_true_0 = [np.random.randn(x, y) for x, y in zip(gen_dims, gen_dims[1:])]
    params.w_true_1 = [np.random.randn(x, y) for x, y in zip(gen_dims, gen_dims[1:])]

    # training
    params.num_epochs = 4001
    params.grad_steps = 300000
    params.batch_size = 128
    params.metric_data_size = 10000
    params.metric_data_size_2 = 5000

    # model
    params.hidden_dim = hidden_mult * params.input_dim  # (10x or 20x works)
    if params.net_type == 'deep':
        params.network_size = [params.input_dim, params.hidden_dim + 8, params.hidden_dim + 6, params.hidden_dim + 4,
                               params.hidden_dim + 2, params.hidden_dim, params.output_dim]
    elif params.net_type == 'shallow':
        params.network_size = [params.input_dim, params.hidden_dim + 8, params.output_dim]
    elif params.net_type == 'medium':
        params.network_size = [params.input_dim, params.hidden_dim + 8, params.hidden_dim + 6, params.hidden_dim + 4,
                               params.output_dim]
    else:
        raise ValueError('Incorrec Network size')
    params.use_relu = True

    # losses
    # params.losses_apply_layer = [True for i, x in enumerate(params.network_size[1:-1])]
    params.losses = ['rec', 'weight', 'ent']
    params.betas = {'total': 1.0,  # hack to make summaries easier
                    'rec': 1.0,  # 1.0
                    'ent': 1e-3,  # 1e-2 1e-3
                    'weight': 1e-4,  # 1e-4
                    'weight_l1': 1e-1,
                    'nonneg': 2.0,  # 5e-1,  # 5e-1
                    'sparse': 1e-1,
                    }

    # optimiser
    params.learning_rate = 3e-3
    params.geco = False
    params.geco = {'threshold': 0.02,
                   'alpha': 0.9,
                   'gamma': 2e-4,
                   }

    return params


def default_params_subspaces_vae(experiment_type=None):
    params = model_utils.DotDict()
    params.graph_mode = True
    params.summary_interval = 1000
    params.save_interval = 20000
    params.summary_weights = False

    """ DATASET CONFIG """

    # for synthetic data
    params.n_data = 200000
    params.image_dim = 50
    params.nonlinear_data = True
    params.factor_dim = 3
    params.net_type = 'deep'  # 'shallow', 'deep', 'deep_shallow'

    params.experiment_type = 'ae_nonlinear_factors' if experiment_type is None else experiment_type

    if params.experiment_type == 'ae_linear_factors':
        params.dataset = 'factors'  # 'dsprites' 'shapes3d', 'factors'
        params.factor_dim = 6
        params.nonlinear_data = False
        params.net_type = 'shallow'
    elif params.experiment_type == 'ae_nonlinear_factors':
        params.dataset = 'factors'
        params.factor_dim = 6
    elif params.experiment_type == 'vae_shapes3d':
        params.dataset = 'shapes3d'
    elif params.experiment_type == 'vae_dsprites':
        params.dataset = 'dsprites'
    elif params.experiment_type == 'ae_categorical':
        params.dataset = 'categorical'
        params.n_data = 20000
        params.factor_dim = 6
        params.net_type = 'deep_shallow'
        params.categorical_images = [np.random.uniform(size=params.image_dim) for _ in range(params.factor_dim)]
    else:
        raise ValueError('Incorrect Experiment type given')

    params.factor_dim__ = params.factor_dim
    params.image_dim__ = params.image_dim
    params.image_shape = (64, 64, 1) if params.dataset == 'dsprites' else (64, 64, 3)

    # important for VAE
    params.norm_synthetic_image = True
    params.sigmoid_output = True

    # unimportant for VAE
    params.mix_input = False
    params.mix_rotation = ortho_group.rvs(dim=params.factor_dim__) if params.mix_input else np.eye(params.factor_dim__)
    params.non_linear_activation_input = False
    params.threshold_input = False

    gen_dims = [params.factor_dim, int((params.factor_dim + params.image_dim) / 2),
                params.image_dim]
    params.w_true_0 = [np.random.randn(x, y) for x, y in zip(gen_dims, gen_dims[1:])]
    params.w_true_1 = [np.random.randn(x, y) for x, y in zip(gen_dims, gen_dims[1:])]

    """ MODEL CONFIG """

    params.latent_dim = 10
    if params.net_type == 'shallow':
        params.encoder_size = []
        params.decoder_size = []
    elif params.net_type == 'deep':
        params.encoder_size = [10 * params.image_dim__, 5 * (params.image_dim__ + params.latent_dim),
                               10 * params.latent_dim]  # [10 * params.image_dim, 10 * params.latent_dim]
        params.decoder_size = [10 * params.latent_dim, 5 * (params.image_dim__ + params.latent_dim),
                               10 * params.image_dim__]  # [10 * params.latent_dim, 10 * params.image_dim]
    elif params.net_type == 'deep_shallow':
        params.encoder_size = [10 * params.image_dim__, 5 * (params.image_dim__ + params.latent_dim),
                               10 * params.latent_dim]  # [10 * params.image_dim, 10 * params.latent_dim]
        params.decoder_size = []
    else:
        raise ValueError('Incorrect Experiment type given')

    if params.dataset in ['dsprites', 'shapes3d']:
        params.sigmoid_output = True
        params.encoder = 'conv'  # 'conv' 'mlp'
        params.decoder = 'deconv'  # 'deconv' 'mlp'
    else:
        params.encoder = 'mlp'
        params.decoder = 'mlp'

    """ LOSSES CONFIG """
    params.relu_latent_mu = False
    params.losses = ['rec', 'ent', 'weight', 'nonneg']  # 'rec', 'kl', 'nonneg', 'weight', 'ent'
    if params.dataset in ['factors', 'categorical']:
        params.betas = {'total': 1.0,  # hack to make summaries easier
                        'rec': 1.0,
                        'kl': 5e-4,
                        'ent': 5e-3,  # 1e-2 1e-3
                        'weight': 1e-3,  # 1e-4
                        'weight_l1': 1e-4,
                        'nonneg': 5e-1,  # 5e-1
                        'sparse': 5e-1,
                        }
    elif params.dataset in ['dsprites', 'shapes3d']:
        params.betas = {'total': 1.0,  # hack to make summaries easier
                        'rec': 1.0,
                        'kl': 1.0,  # [1, 2, 4, 6, 8, 16]
                        'ent': 1.0,  # 1e-2 1e-3
                        'weight': 0.03,  # 1e-4
                        'nonneg': 100.0,  # 5e-1
                        'sparse': 1e-1,
                        }
    else:
        raise ValueError('Incorrect dataset type for VAE')
    params.exclude_weight_reg = ('conv', 'enc')  # ('abcdefghijklmnopqrstuvwxyz', ) 'enc', 'dec',
    params.sample = True if 'kl' in params.losses and params.betas['kl'] > 0 else False

    """ TRAINING CONFIG """

    params.num_epochs = 4001
    params.grad_steps = 500000
    params.batch_size = 64  # 128
    params.metric_data_size = 10000
    params.metric_data_size_2 = 5000

    """ OPTIMIZER CONFIG """

    params.learning_rate = 0.0001
    params.adam_beta_1 = 0.9
    params.adam_beta_2 = 0.999
    params.adam_epsilon = 1e-08

    params.geco = False
    params.geco = {'threshold': 0.02,
                   'alpha': 0.9,
                   'gamma': 2e-4,
                   }

    return params


def get_adjacencies(params):
    # A_ij -> is j connected to i
    states = int(params.width * params.height)
    adj = {key: np.zeros((states, states)) for key in params.opposite.keys()}

    if params.world_type == 'rectangle':
        for i in range(states):
            # up - down
            if i + params.width < states:
                adj['up'][i, i + params.width] = 1.0
                adj['down'][i + params.width, i] = 1.0
            # left - right
            if np.mod(i, params.width) != 0:
                adj['right'][i, i - 1] = 1.0
                adj['left'][i - 1, i] = 1.0

            if params.torus:
                if int(i / params.width) == 0:
                    adj['up'][i + states - params.width, i] = 1.0
                    adj['down'][i, i + states - params.width] = 1.0
                if np.mod(i, params.width) == 0:
                    adj['right'][i, i + params.width - 1] = 1.0
                    adj['left'][i + params.width - 1, i] = 1.0

        # add combined adjacency matric
        adj['adj_all'] = sum([a for a in adj.values()])
        adj['adj_all_norm_d_in'] = adj['adj_all'] / (np.sum(adj['adj_all'], axis=1, keepdims=True) + 1e-8)
        # add transition matrix
        adj['trans_all'] = adj['adj_all'] / (np.sum(adj['adj_all'], axis=0, keepdims=True) + 1e-8)

    return adj
