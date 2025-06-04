#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: XXX
"""
import tensorflow as tf
import model_utils as m_u
import tensorflow.keras.layers as layers

"""
Build Model
"""
initializer_orthog = tf.keras.initializers.Orthogonal()
initializer_zeros = tf.keras.initializers.Zeros()
initializer_glorot_uniform = tf.keras.initializers.GlorotUniform()


class SubspaceNet(tf.keras.Model):
    def __init__(self, par):
        super(SubspaceNet, self).__init__()

        self.par = par
        self.network_size = par.network_size

        self.weights_ = [tf.Variable(par.sigma * initializer_glorot_uniform(shape=(in_dim, out_dim)), trainable=True,
                                     name='weight_' + str(i))
                         for i, (in_dim, out_dim) in enumerate(zip(par.network_size[:-1], par.network_size[1:]))]
        self.biases_ = [tf.Variable(initializer_zeros(shape=out_dim), trainable=True, name='bias_' + str(i))
                        for i, out_dim in enumerate(par.network_size[1:])]

    def call(self, x, training=None, index=0):
        xs = [x]
        num = len(self.weights_[index:])
        for i, (weight, bias) in enumerate(zip(self.weights_[index:], self.biases_[index:])):
            x = tf.matmul(x, weight) + bias
            if i < num - 1 and self.par.use_relu:
                x = tf.nn.relu(x)
            xs.append(x)

        logits = xs[-1]
        if self.par.sigmoid_output:
            xs[-1] = tf.nn.sigmoid(logits)

        return xs, logits

    def get_losses(self, true, xs, betas, logits):
        losses = {}
        hidden = xs[1:-1]
        if 'rec' in self.par.losses:
            losses['rec'] = m_u.reconstruction_loss(true, logits, binary=self.par.sigmoid_output)
        if 'ent' in self.par.losses:
            losses['ent'] = m_u.ent_reg(hidden)
        if 'nonneg' in self.par.losses:
            losses['nonneg'] = m_u.nonneg_reg(hidden)
        if 'sparse' in self.par.losses:
            losses['sparse'] = m_u.sparse_reg(hidden)
        if 'weight' in self.par.losses:
            losses['weight'] = m_u.weight_reg(self.trainable_weights)
        if 'weight_l1' in self.par.losses:
            losses['weight_l1'] = m_u.weight_reg_l1(self.trainable_weights)

        loss_total = tf.add_n([losses[key] * betas[key] for key in self.par.losses])

        losses['total'] = loss_total

        return loss_total, losses


class SubspaceVAE(tf.keras.Model):
    def __init__(self, par):
        super(SubspaceVAE, self).__init__()

        self.par = par

        if self.par.encoder == 'mlp':
            self.encoder = MLPEncoder(par)
        elif self.par.encoder == 'conv':
            self.encoder = ConvEncoder(par.image_shape, par.latent_dim)
        else:
            self.encoder = None
            raise ValueError('Not implemented yet')

        if self.par.decoder == 'mlp':
            self.decoder = MLPDecoder(par)
        elif self.par.decoder == 'deconv':
            self.decoder = DeconvDecoder(par.image_shape, par.latent_dim)
        else:
            self.decoder = None
            raise ValueError('Not implemented yet')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=par.learning_rate, beta_1=par.adam_beta_1,
                                                  beta_2=par.adam_beta_2, epsilon=par.adam_epsilon)

    def call(self, x, training=None):
        mu, logvar = self.encode(x)
        z = m_u.reparameterize((mu, logvar)) if self.par.sample else mu
        logits, probs = self.decode(z, apply_sigmoid=self.par.sigmoid_output)

        return (logits, probs), (z, mu, logvar)

    @tf.function
    def sample(self, n_samples=100, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(n_samples, self.par.latent_dim))
        return self.decode(eps, apply_sigmoid=self.par.sigmoid_output)

    def encode(self, x):
        mu, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        if self.par.relu_latent_mu:
            mu = tf.nn.relu(mu)
        return mu, logvar

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
        else:
            probs = logits
        return logits, probs

    def get_losses(self, x, decoded, latents, betas):
        logits, probs = decoded
        z, mu, logvar = latents

        if self.par.sigmoid_output:
            reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x)
        else:
            reconstruction_loss = 0.5 * (x - probs) ** 2

        if self.par.dataset in ['factors', 'categorical']:
            reconstruction_loss = -tf.reduce_sum(reconstruction_loss, axis=1)
        else:
            # reduce more if image
            reconstruction_loss = -tf.reduce_sum(reconstruction_loss, axis=[1, 2, 3])

        reconstruction_loss = -tf.reduce_mean(reconstruction_loss)
        # kl
        kl_loss = - 0.5 * (1 + logvar - tf.square(mu) - tf.exp(logvar))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        losses = {}
        if 'rec' in self.par.losses:
            losses['rec'] = reconstruction_loss
        if 'kl' in self.par.losses:
            losses['kl'] = kl_loss
        if 'ent' in self.par.losses:
            losses['ent'] = m_u.ent_reg([mu])
        if 'nonneg' in self.par.losses:
            losses['nonneg'] = m_u.nonneg_reg([mu])
        if 'sparse' in self.par.losses:
            losses['sparse'] = m_u.sparse_reg([mu])
        if 'weight' in self.par.losses:
            losses['weight'] = m_u.weight_reg(self.trainable_weights, exclude=self.par.exclude_weight_reg)
        if 'weight_l1' in self.par.losses:
            losses['weight_l1'] = m_u.weight_reg_l1(self.trainable_weights, exclude=self.par.exclude_weight_reg)

        loss_total = tf.add_n([losses[key] * betas[key] for key in self.par.losses])
        losses['total'] = loss_total

        return loss_total, losses


class MLPEncoder(tf.keras.Model):
    def __init__(self, par):
        super(MLPEncoder, self).__init__()

        self.mlp_encoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(e_s, activation=tf.nn.relu) for e_s in par.encoder_size] + [
                tf.keras.layers.Dense(par.latent_dim * 2)])

    def call(self, x):
        return self.mlp_encoder(x)


class MLPDecoder(tf.keras.Model):
    def __init__(self, par):
        super(MLPDecoder, self).__init__()

        self.mlp_decoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(d_s, activation=tf.nn.relu) for d_s in par.decoder_size] + [
                tf.keras.layers.Dense(par.image_dim__)])

    def call(self, x):
        return self.mlp_decoder(x)


class ConvEncoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim):
        super(ConvEncoder, self).__init__()

        encoder_inputs = tf.keras.Input(shape=input_shape)
        x0 = layers.Conv2D(32, 4, activation="relu", strides=2, padding="same", name='enc0_conv')
        x1 = layers.Conv2D(32, 4, activation="relu", strides=2, padding="same", name='enc1_conv')
        x2 = layers.Conv2D(64, 4, activation="relu", strides=2, padding="same", name='enc2_conv')
        x3 = layers.Conv2D(64, 4, activation="relu", strides=2, padding="same", name='enc3_conv')
        x4 = layers.Flatten()
        x5 = layers.Dense(256, activation="relu", name='enc4_dense')
        x6 = layers.Dense(2 * latent_dim, activation=None, name='enc5_dense')
        self.conv_encoder = tf.keras.Sequential([encoder_inputs, x0, x1, x2, x3, x4, x5, x6])

    def call(self, x):
        return self.conv_encoder(x)


class DeconvDecoder(tf.keras.Model):
    def __init__(self, image_shape, latent_dim):
        super(DeconvDecoder, self).__init__()

        self.image_shape = image_shape

        latent_inputs = tf.keras.Input(shape=(latent_dim,))
        x0 = layers.Dense(256, activation=tf.nn.relu, name='dec0_dense')
        x1 = layers.Dense(1024, activation=tf.nn.relu, name='dec1_dense')
        x2 = layers.Reshape((4, 4, 64))
        x3 = layers.Conv2DTranspose(64, 4, activation="relu", strides=2, padding="same", name='dec2_conv')
        x4 = layers.Conv2DTranspose(32, 4, activation="relu", strides=2, padding="same", name='dec3_conv')
        x5 = layers.Conv2DTranspose(32, 4, activation="relu", strides=2, padding="same", name='dec4_conv')
        x6 = layers.Conv2DTranspose(image_shape[2], 4, activation=None, strides=2, padding="same", name='dec5_conv')
        self.deconv_decoder = tf.keras.Sequential([latent_inputs, x0, x1, x2, x3, x4, x5, x6])

    def call(self, x):
        return tf.reshape(self.deconv_decoder(x), [-1] + list(self.image_shape))
