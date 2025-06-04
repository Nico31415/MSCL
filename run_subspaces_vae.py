#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: XXX
"""
import numpy as np
import tensorflow as tf
import parameters
import model_subspaces
import model_utils as m_u
import cell_analyses as c_u
import data_utils as d_u
import plotting_functions as p_f
import os
import time
import sys

# reload old model
reload = False
if len(sys.argv) > 1:
    if len(sys.argv[1]) > 0:
        reload = True

if reload:
    date = sys.argv[1][:10]
    run = int(sys.argv[1].split('run')[1])
    print('RELOADING RUN: ', date, run)
    params_old = parameters.default_params_subspaces_vae()
    base_path = '/well/behrens/users/uqb299/Summaries_subspaces_vae/'  # '../Summaries_subspaces_vae/'  #
    run_path, train_path, model_path, save_path, script_path = p_f.set_directories(date, run, base_path=base_path)
    index = max([int(x.split('subspaces_')[1].split('.index')[0]) for x in os.listdir(model_path) if 'index' in x])
    grad_step = index + 1
    # Load previous packages
    model, params, m_u, d_u = p_f.get_model(model_path + '/subspaces_' + str(index), script_path, save_path,
                                            use_old_scripts=True, model_type='subspaces_vae')
    # run to new length
    params.grad_steps = params_old.grad_steps
    # load full model - contains old optimizer state
    model_full = tf.keras.models.load_model(model_path + '/subspaces_full_model')  # , custom_objects={"model": model})
    # Hacks to make optimizer state carry over...
    old_optimizer_state = [np.array([grad_step])[0]] + model_full.optimizer.get_weights()
    reset_opt_state = True
    del model_full, params_old
else:
    print('STARTING NEW RUN DATA')
    # get parameters
    params = parameters.default_params_subspaces_vae()
    # Create directories for storing all information about the current run
    run_path, train_path, model_path, save_path, script_path = d_u.make_directories('../Summaries_subspaces_vae/')
    # Save parameters
    np.save(os.path.join(save_path, 'params'), dict(params))
    # Initialise model & optimiser
    model = model_subspaces.SubspaceVAE(params)
    grad_step, reset_opt_state, old_optimizer_state = 0, False, None

# Create a logger to write log output to file
logger_sums = d_u.make_logger(run_path, 'summaries')
# Create a tensor board to stay updated on training progress. Start tensorboard with tensorboard --logdir=runs
summary_writer = tf.summary.create_file_writer(train_path)

tf.config.run_functions_eagerly(not params.graph_mode)


@tf.function  # (experimental_compile=True)
def train_step(model_, inputs, betas, reset_opt_state_=False):
    with tf.GradientTape() as tape:
        decoded, latents_ = model_(inputs, training=True)
        loss_value, losses_ = model_.get_losses(inputs, decoded, latents_, betas)
    gradients = tape.gradient(loss_value, model_.trainable_variables)
    # do this after getting metrics, so model is 'fresh' for metrics
    if reset_opt_state_:
        # ideally we want zero weight update. But will just get an update in momentum's direction. So probs OK.
        # Should reset weights after this update, but can't figure out how.
        # Really should work out how to better load in both model weights and optimizer states, over my hacks
        model_.optimizer.apply_gradients(zip([tf.zeros_like(x) for x in gradients], model_.trainable_variables))
    else:
        model_.optimizer.apply_gradients(zip(gradients, model_.trainable_variables))

    return losses_, (decoded, latents_)


@tf.function  # (experimental_compile=True)
def test_step(model_, inputs):
    decoded, latents_ = model_(inputs, training=False)
    return decoded, latents_


# collect data
ds_batch, (ds_metric, ds_metric_2) = d_u.data_subspaces(params)

# train
epoch_loss_avg, epoch, grad_step_time = None, 0, 0
break_inner = False
aa = time.time()
for epoch in range(params.num_epochs):
    msg = 'Epoch: {:03d}'.format(epoch)
    logger_sums.info(msg)
    for i, data in enumerate(ds_batch):
        # parse data
        image = tf.cast(data['image'], tf.float32)

        # Optimize the model
        bb = time.time()
        losses, xs = train_step(model, image, params.betas, reset_opt_state_=reset_opt_state)
        if reset_opt_state:
            model.optimizer.set_weights(old_optimizer_state)
            reset_opt_state = False
        grad_step_time += time.time() - bb

        if grad_step % params.summary_interval == 0:
            cc = time.time()
            step_time = (time.time() - aa) if grad_step == 0 else (time.time() - aa) / params.summary_interval

            # calculate disentangling metrics
            (_, probs), latents = test_step(model, ds_metric['image'])
            (_, mu, logvar) = [x.numpy().T for x in latents]
            neuron_used = np.mean(np.exp(logvar), axis=1) <= 0.5 if params.sample else None
            (_, probs_2), latents_2 = test_step(model, ds_metric_2['image'])
            (_, mu_2, logvar_2) = [x.numpy().T for x in latents_2]

            metrics, _ = c_u.compute_mig(mu, ds_metric, mu_2, ds_metric_2, dataset=params.dataset,
                                         neuron_used=neuron_used)

            # times for gradients and summaries
            summary_time = time.time() - cc
            gradient_time = grad_step_time if grad_step == 0 else grad_step_time / params.summary_interval

            # save to tensorboard
            summaries = m_u.make_summaries_subspaces(gradient_time, summary_time, step_time, losses, metrics, model,
                                                     params.betas, ds_metric['image'], probs, params)
            for key_, val_ in summaries.items():
                with summary_writer.as_default():
                    tf.summary.scalar(key_, val_, step=grad_step)
            summary_writer.flush()

            # log summaries
            msg = ('Step: {:06d}, Gradient Time: {:.4f}, Summary Time: {:.4f}, ' +
                   'Step Time: {:.4f}').format(grad_step, gradient_time, summary_time, step_time)
            logger_sums.info(msg)
            msg = 'Losses: ' + ''.join([key + '={' + key + ':.4f}, ' for key in losses.keys()]).format(**losses)
            logger_sums.info(msg)
            msg = 'Metrics: ' + ''.join([key + '={' + key + ':.3f}, ' for key in metrics.keys()]).format(**metrics)
            logger_sums.info(msg)

            aa = time.time()
            grad_step_time = 0.0

        # Save model parameters which can be loaded later to analyse model
        if grad_step % params.save_interval == 0:
            dd = time.time()
            # save model checkpoint
            model.save_weights(model_path + '/subspaces_' + str(grad_step))
            # save full model (no index as only want latest copy)
            model.save(model_path + '/subspaces_full_model')
            logger_sums.info('Step: {:06d}, save data time {:.2f}'.format(grad_step, time.time() - dd))

        if grad_step >= params.grad_steps:
            break_inner = True
            break
        grad_step += 1
    if break_inner:
        break

model.save_weights(model_path + '/subspaces_' + str(grad_step))
model.save(model_path + '/subspaces_full_model')
print('finished')
