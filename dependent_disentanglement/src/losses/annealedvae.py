import torch

from .baseloss import BaseLoss
from .utils import _reconstruction_loss, _kl_normal_loss, linear_annealing

class Loss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    C_init : float, optional
        Starting annealed capacity C.

    C_fin : float, optional
        Final annealed capacity C.

    gamma : float, optional
        Weight of the KL divergence term.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Burgess, Christopher P., et al. "Understanding disentangling in
        $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
    """
    def __init__(self, C_init, C_fin, gamma, anneal_steps, log_components, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.C_init = C_init
        self.C_fin = C_fin
        self.anneal_steps = anneal_steps
        self.log_components = log_components

    def __call__(self, data, reconstructions, stats_qzx, is_train, **kwargs):
        self._pre_call(is_train)
        if isinstance(stats_qzx, torch.Tensor):
            stats_qzx = stats_qzx.unbind(-1)

        log_data = {}

        rec_loss = _reconstruction_loss(data, reconstructions, distribution=self.rec_dist)
        log_data['rec_loss'] = rec_loss.item()

        kl_loss = _kl_normal_loss(*stats_qzx, return_components=True)
        if self.log_components:
            log_data.update(
                {f'kl_loss_{i}': value.item() for i, value in enumerate(kl_loss)})
        kl_loss = kl_loss.sum()
        log_data['kl_loss'] = kl_loss.item()

        C = (linear_annealing(self.C_init, self.C_fin, self.n_train_steps,
                              self.anneal_steps) if is_train else self.C_fin)

        loss = rec_loss + self.gamma * (kl_loss - C).abs()
        log_data['loss'] = loss.item()

        return {'loss': loss, 'to_log': log_data}

    def attrs_to_chkpt(self):
        return {'n_train_steps': self.n_train_steps}
