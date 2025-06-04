#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: XXX
"""
import tensorflow as tf
import model_utils as mu
from tensorflow.keras.models import Model

eps = 1e-8

initializer_orthog = tf.keras.initializers.Orthogonal()
initializer_identity = tf.keras.initializers.Identity()
initializer_zeros = tf.keras.initializers.Zeros()
initializer_trunc_normal = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None)
initializer_ones = tf.keras.initializers.Ones()
initializer_uniform = tf.keras.initializers.RandomUniform  # note this needs extra inputs!


class PatternLearner(Model):
    def __init__(self, params):
        super(PatternLearner, self).__init__()
        self.par = params

        self.adjacencies = {key: tf.constant(value, dtype=tf.float32) for key, value in self.par.adjacencies.items()}

        if 'action' in self.par.losses:
            self.action_module = ActionModule(self.par)
        if 'decode' in self.par.losses:
            self.decode_module = DecodeModule(self.par)
        if 'object' in self.par.losses:
            self.object_module = ObjectModule(self.par)
        self.regularisation = Regularisation(self.par)

        # initial_ent reps
        if self.par.init_type == 'learn_single':
            # only have initial ent rep at single location
            self.init_g_single = tf.Variable(
                initializer_uniform(minval=self.par.g_init_min, maxval=self.par.g_init_max)(
                    shape=(1, self.par.ent_dim)), trainable=True, name='init_g_single')
        elif self.par.init_type == 'learn_everywhere':
            # or initial ent rep at all locations
            self.init_g_everywhere = tf.Variable(
                initializer_uniform(minval=self.par.g_init_min, maxval=self.par.g_init_max)(
                    shape=(1, self.par.n_locs, self.par.ent_dim)), trainable=True, name='init_g_everywhere')

        self.transition_mask = 1

    @mu.define_scope
    def call(self, inputs, training=None):
        g = self.init_g(inputs)
        gs = [g]
        for i in range(self.par.n_steps):
            g = self.single_step(g, inputs)
            gs.append(g)
        return gs

    @mu.define_scope
    def init_g(self, inputs):
        if self.par.init_type == 'learn_single':
            g = tf.zeros((self.par.batch_size, self.par.n_locs, self.par.ent_dim))
            updates = self.init_g_single
            for i in range(self.par.batch_size):
                indices = tf.expand_dims(tf.stack([tf.constant(i), inputs['start_states'][i]], axis=0), axis=0)
                g = tf.tensor_scatter_nd_update(g, indices, updates)
        elif self.par.init_type == 'learn_everywhere':
            g = tf.tile(self.init_g_everywhere, (self.par.batch_size, 1, 1))
        else:
            raise ValueError('incorrect init type given')

        g = self.set_boundary(g, inputs['masks'])
        g = self.activation(g)
        g = self.threshold(g)
        g = self.normalise(g)

        return g

    @mu.define_scope
    def activation(self, g):
        if self.par.activation == 'relu':
            g = tf.nn.relu(g)
        elif self.par.activation == 'leaky_relu':
            g = tf.nn.leaky_relu(g, alpha=self.par.leaky_relu_alpha)
        elif self.par.activation == 'none':
            g = g
        else:
            raise ValueError('wrong activation given')
        return g

    @mu.define_scope
    def activation_deriv(self, g):
        if self.par.activation == 'relu':
            return tf.where(g > 0, tf.ones_like(g), tf.zeros_like(g))
        elif self.par.activation == 'leaky_relu':
            return tf.where(g > 0, tf.ones_like(g), self.par.leaky_relu_alpha * tf.ones_like(g))
        elif self.par.activation == 'none':
            return tf.ones_like(g)
        else:
            raise ValueError('wrong activation given')

    @mu.define_scope
    def threshold(self, g):
        return mu.threshold(g, self.par.thresh_ent_min, self.par.thresh_ent_max)

    @mu.define_scope
    def set_boundary(self, g, mask):
        return tf.where(mask > 0.5, g, tf.zeros_like(g))

    @mu.define_scope
    def normalise(self, g):
        if self.par.normalisation == 'none':
            return g

        alive, num_active_locs = mu.is_alive(g, n_loc_min=1.0)

        if self.par.normalisation == 'space':
            # normalise each spatial map (each grid cell)
            sum_sq = tf.reduce_sum(g ** 2, axis=1, keepdims=True)
            length = tf.sqrt(sum_sq * 10 / num_active_locs)
            g = g / (length + eps)
        elif self.par.normalisation == 'cells':
            # normalise each spatial location (across grid cell)
            sum_sq = tf.reduce_sum(g ** 2, axis=2, keepdims=True)
            length = tf.sqrt(sum_sq * 10 / self.par.ent_dim)
            g = g / (length + eps)
        elif self.par.normalisation == 'qr':
            q, _ = tf.linalg.qr(g, full_matrices=False)
        else:
            raise ValueError('wrong normalisation given')
        return g

    @mu.define_scope
    def roll(self, g, direc, transpose=False):
        # g is batch * n_locs x n_g

        # With an adjacency matrix for each action
        if transpose:
            return tf.matmul(tf.transpose(self.adjacencies[direc]), g)
        else:
            return tf.matmul(self.adjacencies[direc], g)

    @mu.define_scope
    def single_step(self, g_t, inputs):

        # find alive/dead locations
        alive, _ = mu.is_alive(g_t)

        # updates from predictions
        action_update, decode_update, object_update = 0, 0, 0
        if self.par.use_object_pattern_update and 'object' in self.par.losses:
            object_update = self.object_module.pattern_update(g_t, inputs, alive)
        g_t = g_t + action_update + decode_update + object_update

        # update from transition
        transition_update = self.transition_step(g_t, inputs)
        g = g_t + transition_update

        g = self.set_boundary(g, inputs['masks'])
        # apply activation
        g = self.threshold(g)

        # apply normalisation
        g = self.normalise(g)

        return g

    @mu.define_scope
    def get_losses(self, gs, inputs, final=True):
        g = gs[-1]
        masks, num_active_locs = mu.get_mask_num_locs(g, inputs, final=final, n_loc_min=1.0)

        loss_dict = dict()
        if 'transition' in self.par.losses:
            loss_dict['transition'], _, _ = self.loss(g, masks, num_active_locs)
        if 'decode' in self.par.losses:
            loss_dict['decode'], _ = self.decode_module.loss(g, masks, num_active_locs)
        if 'action' in self.par.losses:
            loss_dict['action'], _, _ = self.action_module.loss(g, inputs, masks, num_active_locs)
        if 'object' in self.par.losses:
            loss_dict['object'], _, _ = self.object_module.loss(g, inputs, masks, num_active_locs)
        if 'ent_reg' in self.par.losses:
            loss_dict['ent_reg'] = self.regularisation.l2_ent(g, masks, num_active_locs)
        if 'l2_weight_reg' in self.par.losses:
            loss_dict['l2_weight_reg'] = self.regularisation.l2_weight(self.trainable_variables)
        if 'l1_weight_reg' in self.par.losses:
            loss_dict['l1_weight_reg'] = self.regularisation.l1_weight(self.trainable_variables)
        if 'non_negativity' in self.par.losses:
            loss_dict['non_negativity'] = self.regularisation.non_negativity(g, masks, num_active_locs)

        # multiply by betas
        loss_ = sum([value * self.par.beta_loss[key_] for key_, value in loss_dict.items()])
        loss_dict['loss'] = loss_
        return loss_dict


class ActionModule(Model):
    def __init__(self, params):
        super(ActionModule, self).__init__()
        self.par = params

        self.weight = tf.Variable(initializer_orthog(shape=(self.par.ent_dim, self.par.action_dim)),
                                  trainable=True, name='action projection')
        self.bias = tf.Variable(initializer_zeros(shape=self.par.action_dim), trainable=True, name='action bias')

        self.beta = self.par.beta_softmax * tf.math.log(self.par.softmax_target / (1 - self.par.softmax_target))

    @mu.define_scope
    def predict(self, g):
        logits = tf.matmul(g, self.weight) + self.bias
        logits = self.beta * logits
        preds = tf.nn.sigmoid(logits)

        return logits, preds

    @mu.define_scope
    def pattern_update(self, g, inputs, alive):
        _, preds = self.predict(g)

        update = tf.matmul(inputs['action_map'] - preds, tf.transpose(self.weight, (1, 0)))
        if self.par.clip_pattern_gradients:
            update = tf.clip_by_norm(update, self.par.pattern_clip_val, axes=[2])
        # only apply update_action to states that have been reached by path int.
        eta = tf.where(alive > 0, self.par.eta_action, 0)
        update = eta * update

        if not self.par.graph_mode:
            self.check_dynamics_minimising(g + update, g, inputs, print_=True)

        return update

    @mu.define_scope
    def loss(self, g, inputs, masks, num_active_locs):
        # make predictions
        logits, _ = self.predict(g)
        # do sigmoid cross entropy
        loss_space = tf.nn.sigmoid_cross_entropy_with_logits(inputs['action_map'], logits)
        # ignore loss of inactive cells
        loss_space = loss_space * masks
        # scale by distance from goal
        loss_space = loss_space * inputs['loss_scale_map']
        # average over actions
        loss_average = tf.reduce_sum(loss_space, axis=2) / self.par.action_dim
        # average over locations
        loss_average = tf.reduce_sum(loss_average, axis=1) / num_active_locs
        # sum over batches
        loss = tf.reduce_mean(loss_average)

        return loss, loss_space, loss_average

    @mu.define_scope
    def accuracy(self, g, inputs):
        masks, _ = mu.get_mask_num_locs(g, inputs)
        _, preds = self.predict(g)
        # multiple action can be good at same time. As use sigmoid cross entropy
        # find components above and below 0.5, and see if they are same as
        thresholded_preds = tf.where(preds > 0.5, tf.ones_like(preds), tf.zeros_like(preds))
        accuracy = tf.cast(tf.equal(thresholded_preds, inputs['action_map']), dtype=tf.float32) / self.par.action_dim
        return tf.reduce_sum(accuracy * masks * inputs['loss_scale_map']) / (
                tf.reduce_sum(masks * inputs['loss_scale_map']) + eps)

    @mu.define_scope
    def check_dynamics_minimising(self, g, g_t, inputs, print_):
        # check if action loss actually getting minimised
        m, l_ = mu.get_mask_num_locs(g, inputs, n_loc_min=1.0)
        m_t, l_t = mu.get_mask_num_locs(g_t, inputs, n_loc_min=1.0)

        diff = self.loss(g, inputs, m, l_)[2] - self.loss(g_t, inputs, m_t, l_t)[2]
        if tf.reduce_any(diff > 0) and print_:
            print('action not decreasing energy')

        return diff


class DecodeModule(Model):
    def __init__(self, params):
        super(DecodeModule, self).__init__()
        self.par = params

        # need to project to something of dims n_locs
        self.weight = tf.Variable(initializer_orthog(shape=(self.par.ent_dim, self.par.n_locs)),
                                  trainable=True, name='decode matrix')
        self.bias = tf.Variable(initializer_zeros(shape=self.par.n_locs), trainable=True, name='decode bias')
        self.labels = tf.tile(tf.expand_dims(tf.range(0, self.par.n_locs, 1), axis=0), (self.par.batch_size, 1))

        true = tf.tile(tf.expand_dims(tf.eye(self.par.n_locs), axis=0), (self.par.batch_size, 1, 1))
        self.true = tf.cast(true, tf.float32)
        self.beta = self.par.beta_softmax * tf.math.log(
            (self.par.n_locs - 1.0) * self.par.softmax_target / (1 - self.par.softmax_target))

    @mu.define_scope
    def predict(self, g):
        logits = tf.matmul(g, self.weight) + self.bias

        logits = logits - tf.reduce_max(logits, axis=2, keepdims=True)
        logits = self.beta * logits
        preds = tf.nn.softmax(logits, axis=2)

        return logits, preds

    @mu.define_scope
    def pattern_update(self, g, inputs, alive):

        _, predict = self.predict(g)

        update = tf.matmul(self.true - predict, tf.transpose(self.weight, (1, 0)))
        if self.par.clip_pattern_gradients:
            update = tf.clip_by_norm(update, self.par.pattern_clip_val, axes=[2])

        # only apply update_action to states that have been reached by path int.
        eta = tf.where(alive > 0, self.par.eta_decode, 0)
        update = eta * update

        if not self.par.graph_mode:
            self.check_dynamics_minimising(g + update, g, inputs, print_=True)

        return update

    @mu.define_scope
    def loss(self, g, masks, num_active_locs):
        # make predictions
        logits, preds = self.predict(g)
        # compare to ground truth:
        loss_space = tf.nn.sparse_softmax_cross_entropy_with_logits(self.labels, logits)
        # normalise cross entropy with respect to length
        loss_space = loss_space / (1 + tf.math.log(tf.expand_dims(tf.maximum(num_active_locs, 2.0), axis=1) - 1.0))

        # ignore inactive locations
        loss_space = loss_space * tf.reshape(masks, (self.par.batch_size, self.par.n_locs))
        # average over locations
        loss_average = tf.reduce_sum(loss_space, axis=1) / num_active_locs
        # sum over batches
        loss = tf.reduce_mean(loss_average)

        return loss, {'sim_mat_demax': logits,
                      'softmax': preds,
                      'cross_ent': loss_space}

    @mu.define_scope
    def check_dynamics_minimising(self, g, g_t, inputs, print_=False):
        # check if decode loss actually getting minimised
        m, l_ = mu.get_mask_num_locs(g, inputs, n_loc_min=1.0)
        m_t, l_t = mu.get_mask_num_locs(g_t, inputs, n_loc_min=1.0)

        diff = self.loss(g, m, l_)[2] - self.loss(g_t, m_t, l_t)[2]
        if tf.reduce_any(diff > 0) and print_:
            print('decode not decreasing energy')

        return diff

    @mu.define_scope
    def accuracy(self, g, inputs):
        masks, _ = mu.get_mask_num_locs(g, inputs)
        _, preds = self.predict(g)  # , masks, num_active_locs
        accuracy = tf.cast(tf.equal(tf.cast(tf.argmax(preds, axis=2), dtype=tf.int32), self.labels), dtype=tf.float32)
        return tf.reduce_sum(accuracy * tf.squeeze(masks)) / (tf.reduce_sum(masks) + eps)


class ObjectModule(Model):
    def __init__(self, params):
        super(ObjectModule, self).__init__()
        self.par = params
        # need to project to something of dims 1
        self.weight = [tf.Variable(initializer_orthog(shape=(self.par.ent_dim, 1)), trainable=True,
                                   name='object matrix ' + str(o_type)) for o_type in self.par.object_types]
        self.bias = [tf.Variable(initializer_zeros(shape=1), trainable=True,
                                 name='object bias ' + str(o_type)) for o_type in self.par.object_types]

        self.beta = self.par.beta_softmax * tf.math.log(self.par.softmax_target / (1 - self.par.softmax_target))

        self.signal_weight = [tf.Variable(initializer_orthog(shape=(1, self.par.ent_dim)), trainable=True,
                                          name='object signal ' + str(o_type)) for o_type in self.par.object_types]

    @mu.define_scope
    def predict(self, g):
        logits = [tf.matmul(g, o_m) + o_b for o_m, o_b in zip(self.weight, self.bias)]
        logits = [self.beta * l_ for l_ in logits]
        preds = [tf.nn.sigmoid(l_) for l_ in logits]

        return logits, preds

    @mu.define_scope
    def pattern_update(self, g, inputs, alive):
        _, preds = self.predict(g)

        update = [tf.matmul(o_map - o_pred, tf.transpose(o_mat, (1, 0))) for o_map, o_pred, o_mat in
                  zip(inputs['object_map'], preds, self.weight)]

        if self.par.object_pattern_update_object_location_only:
            update = [a * b for a, b in zip(update, inputs['object_map'])]

        update = tf.add_n([u for u in update])
        if self.par.clip_pattern_gradients:
            update = tf.clip_by_norm(update, self.par.pattern_clip_val, axes=[2])
        # only apply update to states that have been reached by path int.
        eta = tf.where(alive > 0, self.par.eta_object, 0)
        update = eta * update

        if not self.par.graph_mode:
            self.check_dynamics_minimising(g + update, g, inputs, print_=True)

        return update

    @mu.define_scope
    def loss(self, g, inputs, masks, num_active_locs):
        # make predictions
        logits, _ = self.predict(g)
        # do sigmoid cross entropy
        loss_space = [tf.nn.sigmoid_cross_entropy_with_logits(o_m, l_) for o_m, l_ in zip(inputs['object_map'], logits)]
        # ignore loss of inactive cells
        loss_space = [l_s * masks for l_s in loss_space]
        # average over locations
        loss_average = [tf.reduce_sum(l_, axis=(1, 2)) / num_active_locs for l_ in loss_space]
        # sum over object types
        loss_average = tf.add_n(loss_average) / len(self.par.object_types)
        # sum over batches
        loss = tf.reduce_mean(loss_average)

        return loss, loss_space, loss_average

    @mu.define_scope
    def accuracy(self, g, inputs):
        masks, _ = mu.get_mask_num_locs(g, inputs)
        _, preds = self.predict(g)
        thresholded_preds = [tf.where(p > 0.5, tf.ones_like(p), tf.zeros_like(p)) for p in preds]
        accuracy = tf.add_n([tf.cast(tf.equal(t_p, o_m), dtype=tf.float32) / len(self.par.object_types) for t_p, o_m in
                             zip(thresholded_preds, inputs['object_map'])])
        return tf.reduce_sum(accuracy * masks) / (tf.reduce_sum(masks) + eps)

    @mu.define_scope
    def check_dynamics_minimising(self, g, g_t, inputs, print_=False):
        # check if object loss actually getting minimised
        m, l_ = mu.get_mask_num_locs(g, inputs, n_loc_min=1.0)
        m_t, l_t = mu.get_mask_num_locs(g_t, inputs, n_loc_min=1.0)

        diff = self.loss(g, inputs, m, l_)[2] - self.loss(g_t, inputs, m_t, l_t)[2]
        if tf.reduce_any(diff > 0) and print_:
            print('object not decreasing energy')

        return diff


class Regularisation(Model):
    def __init__(self, params):
        super(Regularisation, self).__init__()
        self.par = params

    @mu.define_scope
    def l2_ent(self, g, masks, num_active_locs):
        g = g * masks
        # (n_locs may be different for each environment)
        ent_reg = tf.reduce_sum(g ** 2, axis=(1, 2)) / num_active_locs
        # sum over batches
        ent_reg = tf.reduce_mean(ent_reg)
        return ent_reg

    @mu.define_scope
    def l2_weight(self, trainable_variables):
        # don't include init_g weights
        l2_weight_reg = tf.add_n(
            [tf.nn.l2_loss(v) for v in trainable_variables if 'bias' not in v.name and 'init_' not in v.name])
        return l2_weight_reg

    @mu.define_scope
    def l1_weight(self, trainable_variables):
        # don't include init_g weights
        l1_weight_reg = tf.add_n([tf.reduce_sum(tf.abs(v)) for v in trainable_variables if
                                  'bias' not in v.name and 'init_' not in v.name])
        return l1_weight_reg

    @mu.define_scope
    def non_negativity(self, g, masks, num_active_locs):
        g = g * masks
        non_negativity = tf.reduce_sum(tf.nn.relu(-g), axis=(1, 2)) / num_active_locs
        # sum over batches
        non_negativity = tf.reduce_mean(non_negativity)
        return non_negativity


class PathIntegrator(PatternLearner):
    def __init__(self, params):
        super().__init__(params)

        # group generator matrices
        group_matrix_shape = (self.par.ent_dim, self.par.ent_dim)
        self.right = tf.Variable(initializer_trunc_normal(shape=group_matrix_shape), trainable=True,
                                 name='group_right')
        self.left = tf.Variable(initializer_trunc_normal(shape=group_matrix_shape), trainable=True,
                                name='group_left') if not self.par.inverse_actions else 0.0
        self.up = tf.Variable(initializer_trunc_normal(shape=group_matrix_shape), trainable=True, name='group_up')
        self.down = tf.Variable(initializer_trunc_normal(shape=group_matrix_shape), trainable=True,
                                name='group_down') if not self.par.inverse_actions else 0.0
        self.common = tf.Variable(initializer_identity(shape=group_matrix_shape), trainable=self.par.group_common_train,
                                  name='group_common')

        self.actions = {'left': self.left,
                        'right': self.right,
                        'up': self.up,
                        'down': self.down,
                        }

        self.group_bias = tf.Variable(initializer_zeros(shape=self.par.ent_dim), trainable=self.par.transition_bias,
                                      name='group bias')

    @mu.define_scope
    def group_action(self, x, action_key, transpose=False):

        if action_key in ['left', 'down'] and self.par.inverse_actions:
            opposite_action = self.actions[self.par.opposite[action_key]] + self.common
            action = tf.linalg.pinv(opposite_action)
        else:
            action = self.actions[action_key] + self.common

        if transpose:
            return tf.matmul(x, tf.transpose(action))
        else:
            return tf.matmul(x, action)

    @mu.define_scope
    def transition_step(self, g, inputs):
        """
        iterative updates of cells at each location.
        Don't use updates from 'dead' locations i.e. locations with sum(g**2) = 0
        """

        # find alive/dead locations
        alive, _ = mu.is_alive(g)

        # path integration errors
        errors, alive_to, path_ints = self.get_path_errors(g, inputs['masks'])

        # get 'back errors' for path int pattern dynamics update. Dead cells don't contribute to back errors
        errors_back = {action_key: self.roll(
            alive * self.group_action(error * self.activation_deriv(path_ints[action_key]), action_key, transpose=True),
            action_key, transpose=True) for (action_key, error) in errors.items()}

        # if location dead - update quickly!
        eta = tf.where(alive > 0, self.par.eta_transition, 1.0)
        # work out effective number of actions inputting into each state
        n_effective_actions = tf.stop_gradient(tf.maximum(tf.add_n(alive_to.values()), 1.0))
        eta = tf.where(n_effective_actions > 0, eta / n_effective_actions, 0.0 * n_effective_actions)

        updates = {action_key: -error + self.par.second_error * errors_back[action_key] for (action_key, error)
                   in errors.items()}
        update = tf.add_n(updates.values())

        if self.par.clip_pattern_gradients:
            update_norm = tf.clip_by_norm(update, self.par.pattern_clip_val, axes=[2])
        else:
            update_norm = update

        # updates = eta * update
        updates = tf.where(alive > 0, eta * update_norm, eta * update)

        if not self.par.graph_mode:
            _ = self.check_transition_dynamics_minimising(g + updates, g, inputs, print_=True)

        return updates

    @mu.define_scope
    def get_path_errors(self, g, masks):

        # perform path integration predictions for each location having taken each action
        # don't apply activation just yet!
        path_ints = {action_key: self.roll(self.group_action(g, action_key) + self.group_bias, action_key) for
                     action_key in self.actions.keys()}

        # can't path integration to locations outside of boundary
        path_ints = {action_key: self.set_boundary(value, masks) for action_key, value in path_ints.items()}

        # Whether location got alive info from action - base on path_int so works with any world_type
        alive_to = {action_key: mu.is_alive(path_int)[0] for (action_key, path_int) in path_ints.items()}

        # apply activation to path ints then ...
        # get error of path integration predictions TO every location
        errors = {action_key: (g - self.activation(path_int)) * alive_to[action_key] for (action_key, path_int) in
                  path_ints.items()}

        return errors, alive_to, path_ints

    @mu.define_scope
    def loss(self, g, masks, num_active_locs):

        errors, _, _ = self.get_path_errors(g, masks)
        # errors: whether this state was predixtable from surrounding states
        loss_space = tf.add_n(
            [tf.reduce_sum(0.5 * (error ** 2), axis=2) for action_key, error in errors.items()])
        # average over locations
        loss_average = tf.reduce_sum(loss_space, axis=1) / num_active_locs
        # sum over batches
        loss = tf.reduce_mean(loss_average)

        return loss, loss_space, loss_average

    @mu.define_scope
    def check_transition_dynamics_minimising(self, g, g_t, inputs, print_=False):
        # check if path int loss actually getting minimised
        m, l_ = mu.get_mask_num_locs(g, inputs, n_loc_min=1.0)
        m_t, l_t = mu.get_mask_num_locs(g_t, inputs, n_loc_min=1.0)

        diff = self.loss(g, m, l_)[2] - self.loss(g_t, m_t, l_t)[2]
        if tf.reduce_any(diff > 0) and print_:
            print('path int not decreasing energy')

        return diff
