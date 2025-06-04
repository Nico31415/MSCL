MODEL_LIST = [
    'vae', 
    'vae_burgess', 
    'vae_chen_mlp', 
    'vae_locatello', 
    'vae_locatello_sbd',
    'vae_montero_small', 
    'vae_montero_large',
]

def get_model(device, name, img_size, **kwargs):
    if name not in MODEL_LIST:
        err = "Unkown model.name = {}. Possible values: {}"
        raise ValueError(err.format(name, MODEL_LIST))
    elif name == 'vae_burgess':
        from .vae_burgess import Model
        return Model(img_size, **kwargs).to(device)
    elif name == 'vae_chen_mlp':
        from .vae_chen_mlp import Model
        return Model(img_size, **kwargs).to(device)
    elif name == 'vae_locatello':
        from .vae_locatello import Model
        return Model(img_size, **kwargs).to(device)
    elif name == 'vae_locatello_sbd':
        from .vae_locatello_sbd import Model
        return Model(img_size, **kwargs).to(device)
    elif name == 'vae_montero_small':
        from .vae_montero_small import Model
        return Model(img_size, **kwargs).to(device)
    elif name == 'vae_montero_large':
        from .vae_montero_large import Model
        return Model(img_size, **kwargs).to(device)
    elif name == 'vae':
        from .vae_locatello import Model
        return Model(img_size, **kwargs).to(device)
