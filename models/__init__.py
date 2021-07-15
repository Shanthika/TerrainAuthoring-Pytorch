from .base import *
from .vanilla_vae import *
from .pix2pix import *


# Aliases
VAE = VanillaVAE
vae_models = {'VanillaVAE':VanillaVAE,}


pix2pix_model = {
    'Generator': Generator,
    'Discriminator': Discriminator
}
