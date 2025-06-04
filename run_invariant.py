#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: XXX
"""
import tensorflow as tf
import time
import parameters
import model_invariant as model_invar
import os
import shutil
import glob
import data_utils
import model_utils
import numpy as np

# importlib.reload(model_invar)

# Create directories for storing all information about the current run
run_path, train_path, model_path, save_path, script_path = data_utils.make_directories('../Summaries_pattern/')
# Save all python files in current directory to script directory
files = glob.iglob(os.path.join('.', '*.py'))
for file in files:
    if os.path.isfile(file):
        shutil.copy2(file, os.path.join(script_path, file))

params = parameters.default_params_pattern()
# Save parameters
np.save(os.path.join(save_path, 'params'), dict(params))

model = model_invar.PathIntegrator(params)

# Create a logger to write log output to file
logger_sums = data_utils.make_logger(run_path, 'summaries')
# Create a tensor board to stay updated on training progress. Start tensorboard with tensorboard --logdir=runs
summary_writer = tf.summary.create_file_writer(train_path)

tf.config.run_functions_eagerly(not params.graph_mode)
optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)


@tf.function  # (experimental_compile=True)
def train_step(model_, inputs_):
    with tf.GradientTape() as tape:
        # Forward pass.
        gs_ = model_(inputs_, training=True)
        loss_dict_ = model.get_losses(gs_, inputs_)

    # Update the weights of the model to minimize the loss value.
    gradients = tape.gradient(loss_dict_['loss'], model_.trainable_weights)
    capped_grads = [(tf.clip_by_norm(grad, 2.0), var) if grad is not None else (grad, var) for grad, var in
                    zip(gradients, model_.trainable_variables)]
    optimizer.apply_gradients(capped_grads)

    return gs_, loss_dict_


start_time = time.time()
train_i = 0
for train_i in range(params.train_steps):
    # Get inputs
    inputs = data_utils.get_boundaries(params)
    inputs_tf = model_utils.inputs2tf(inputs)

    # Start back_prop
    gs, loss_dict = train_step(model, inputs_tf)

    if train_i % params.summary_interval == 0:
        # convert losses to numpy
        for key, val in loss_dict.items():
            loss_dict[key] = val.numpy()
        time_taken = time.time() - start_time
        msg = 'iteration={:.5f}, time={:.4f}'.format(train_i,
                                                     time_taken if train_i == 0 else time_taken / params.summary_interval)
        logger_sums.info(msg)
        msg = ''.join([key + '={' + key + ':.6f}, ' for key in loss_dict.keys()]).format(**loss_dict)
        logger_sums.info(msg)
        msg = ''.join([key + '={' + key + ':.6f}, ' for key in loss_dict.keys()]).format(
            **{key_: loss_dict[key_] * params.beta_loss[key_] for key_ in loss_dict.keys()})
        logger_sums.info(msg)
        start_time = time.time()

        summaries = model_utils.make_summaries(gs, inputs_tf, loss_dict, model, params)
        for key_, val_ in summaries.items():
            with summary_writer.as_default():
                tf.summary.scalar(key_, val_, step=train_i)
        summary_writer.flush()

    # Save model parameters which can be loaded later to analyse model
    if train_i % params.save_interval == 0 and train_i > 0:
        start_time = time.time()
        # data_utils.save_model_outputs(test_step, train_i, save_path, params)

        # save model checkpoint
        model.save_weights(model_path + '/invar_' + str(train_i))
        logger_sums.info(
            "save data time {:.2f}, train_i={:.2f}".format(time.time() - start_time, train_i, ))

model.save_weights(model_path + '/invar_' + str(train_i))
print('finished')
