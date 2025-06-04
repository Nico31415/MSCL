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
import os
import time

# Create directories for storing all information about the current run
run_path, train_path, model_path, save_path, script_path = d_u.make_directories('../Summaries_subspaces/')

params = parameters.default_params_subspaces_net()
# Save parameters
np.save(os.path.join(save_path, 'params'), dict(params))

# Create a logger to write log output to file
logger_sums = d_u.make_logger(run_path, 'summaries')
# Create a tensor board to stay updated on training progress. Start tensorboard with tensorboard --logdir=runs
summary_writer = tf.summary.create_file_writer(train_path)

model = model_subspaces.SubspaceNet(params)
optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
geco_reconstruction = m_u.GECO(params.geco, params.betas['rec'])
tf.config.run_functions_eagerly(not params.graph_mode)


@tf.function  # (experimental_compile=True)
def train_step(model_, inputs, images, betas):
    with tf.GradientTape() as tape:
        xs_, logits = model_(inputs, training=True)
        loss_value, losses_ = model_.get_losses(images, xs_, betas, logits)
    gradients = tape.gradient(loss_value, model_.trainable_variables)
    # do this after getting metrics, so model is 'fresh' for metrics
    optimizer.apply_gradients(zip(gradients, model_.trainable_variables))

    return losses_, xs_


@tf.function  # (experimental_compile=True)
def test_step(model_, inputs):
    xs_, _ = model_(inputs, training=False)
    return xs_


# collect data
ds_batch, (ds_metric, ds_metric_2) = d_u.data_subspaces(params)

# train
epoch_loss_avg, epoch, grad_step, grad_step_time = None, 0, 0, 0
break_inner = False
aa = time.time()
for epoch in range(params.num_epochs):
    msg = 'Epoch: {:03d}'.format(epoch)
    logger_sums.info(msg)
    for i, data in enumerate(ds_batch):
        # parse data
        input_ = tf.cast(data['input'], tf.float32)
        image_ = tf.cast(data['image'], tf.float32)

        # Optimize the model
        bb = time.time()
        losses, xs = train_step(model, input_, image_, params.betas)
        grad_step_time += time.time() - bb

        # geco update
        if params.geco:
            params.betas['ent'] = geco_reconstruction.update(losses['rec'])

        if grad_step % params.summary_interval == 0:
            cc = time.time()
            step_time = (time.time() - aa) if grad_step == 0 else (time.time() - aa) / params.summary_interval

            # calculate disentangling metrics
            xs = test_step(model, ds_metric['input'])
            prob = xs[-1].numpy()
            xs = [x.numpy().T for x in xs]
            xs_2 = test_step(model, ds_metric_2['input'])
            xs_2 = [x.numpy().T for x in xs_2]

            begin_layer = 1
            neuron_used = m_u.important_neuroms(model, ds_metric['input'], ds_metric['image'], params, begin_layer=1)
            metrics = [c_u.compute_mig(x, ds_metric, x_2, ds_metric_2, dataset=params.dataset, neuron_used=n_u)[0] for
                       (x, x_2, n_u) in zip(xs[begin_layer:], xs_2[begin_layer:], neuron_used)]
            # just to check that the neuron used is ok...
            metrics_without = [c_u.compute_mig(x, ds_metric, x_2, ds_metric_2, dataset=params.dataset)[0] for
                               (x, x_2) in zip(xs[begin_layer:], xs_2[begin_layer:])]
            metrics = [{**m, **{'wo_' + key: value for key, value in m_w.items()}} for m, m_w in
                       zip(metrics, metrics_without)]

            # times for gradients and summaries
            summary_time = time.time() - cc
            gradient_time = grad_step_time if grad_step == 0 else grad_step_time / params.summary_interval

            # save to tensorboard
            summaries = m_u.make_summaries_subspaces(gradient_time, summary_time, step_time, losses, metrics, model,
                                                     params.betas, ds_metric['image'], prob, params)
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
            msg = 'Metrics: ' + ''.join(
                [key[8:] + '={' + key + ':.4f}, ' for key in summaries.keys() if 'metric' in key]).format(
                **summaries)
            logger_sums.info(msg)

            aa = time.time()
            grad_step_time = 0.0

        # Save model parameters which can be loaded later to analyse model
        if grad_step % params.save_interval == 0:
            dd = time.time()
            # save model checkpoint
            model.save_weights(model_path + '/subspaces_' + str(grad_step))
            logger_sums.info('Step: {:06d}, save data time {:.2f}'.format(grad_step, time.time() - dd))

        if grad_step >= params.grad_steps:
            break_inner = True
            break
        grad_step += 1
    if break_inner:
        break

model.save_weights(model_path + '/subspaces_' + str(grad_step))
print('finished')
