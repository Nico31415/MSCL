import torch
from torch import nn

from src.utils.initialization import weights_init
from .encoder.locatello import Encoder
from .decoder.locatello import Decoder


class Model(nn.Module):
    def __init__(self, img_size, latent_dim, encoder_decay=0., decoder_decay=0., **kwargs):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(Model, self).__init__()

        if list(img_size[1:]) not in [[64, 64]]:
            raise RuntimeError(
                "{} sized images not supported. Only ((None, 64, 64) or (None, 32, 32) supported."
                .format(img_size))

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.dist_nparams = 2
        self.encoder = Encoder(
            img_size, self.latent_dim, dist_nparams=self.dist_nparams)
        self.decoder = Decoder(
            img_size, self.latent_dim)
        self.model_name = 'vae_locatello'
        self.reset_parameters()
        if encoder_decay or decoder_decay:
            self.to_optim = [
                {'params': self.encoder.parameters(), 'weight_decay': encoder_decay}, 
                {'params': self.decoder.parameters(), 'weight_decay': decoder_decay}
            ]

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return {'samples_qzx': mean + std * eps}
        else:
            # Reconstruction mode
            return {'samples_qzx': mean}

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        stats_qzx = self.encoder(x)['stats_qzx']
        samples_qzx = self.reparameterize(*stats_qzx.unbind(-1))['samples_qzx']
        reconstructions = self.decoder(samples_qzx)['reconstructions']
        return {
            'reconstructions': reconstructions, 
            'stats_qzx': stats_qzx, 
            'samples_qzx': samples_qzx}

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_qzx(self, x):
        """
        Returns a sample z from the latent distribution q(z|x).

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        stats_qzx = self.encoder(x)['stats_qzx']
        samples_qzx = self.reparameterize(*stats_qzx.unbind(-1))['samples_qzx']
        return samples_qzx
