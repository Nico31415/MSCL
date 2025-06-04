from .utils import RECON_DISTS

LOSS_LIST = ['betavae', 'annealedvae', 'vae', 'factorvae', 'betatcvae', 'adagvae', 'factorizedsupportvae', 'factorizedsupporttcvae']

def get_loss(device, name, **kwargs):
    """Return the correct loss function given the argparse arguments."""
    if name == "betavae":
        from .betavae import Loss
        return Loss(**kwargs)
    elif name == "vae":
        from .betavae import Loss
        return Loss(beta=1, **kwargs)
    elif name == "annealedvae":
        from .annealedvae import Loss
        return Loss(**kwargs)
    elif name == "factorvae":
        from .factorvae import Loss
        return Loss(device, **kwargs)
    elif name == "betatcvae":
        from .betatcvae import Loss
        return Loss(**kwargs)
    elif name == "adagvae":
        from .adagvae import Loss
        return Loss(**kwargs)
    elif name == 'factorizedsupportvae':
        from .factorizedsupportvae import Loss
        return Loss(**kwargs)
    elif name == 'factorizedsupporttcvae':
        from .factorizedsupporttcvae import Loss
        return Loss(**kwargs)
    err = "Unkown loss.name = {}. Possible values: {}"
    raise ValueError(err.format(name, LOSS_LIST))
