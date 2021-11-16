"""
Loss Network (VGG16) and stuff
"""
import torch

from modules.pluralistic_model import base_function
from modules.pluralistic_model.external_function import GANLoss


def calc_loss():
    raise NotImplementedError


def discriminator_loss(netD, real, fake):
    """Calculate GAN loss for the discriminator"""
    gan_loss = GANLoss('lsgan')
    # Real
    D_real = netD(real)
    D_real_loss = gan_loss(D_real, True, True)
    # fake
    D_fake = netD(fake.detach())
    D_fake_loss = gan_loss(D_fake, False, True)
    # loss for discriminator
    D_loss = (D_real_loss + D_fake_loss) * 0.5
    return D_loss


def generator_loss(netD, real, fake, lambda_g):
    gan_loss = GANLoss('lsgan')
    l1_loss = torch.nn.L1Loss()
    """Calculate training loss for the generator"""
    base_function._freeze(netD)  # TODO: should unfreeze later
    D_fake = netD(fake)
    loss_ad_g = gan_loss(D_fake, True, False) * lambda_g
    loss_l1_g = l1_loss(fake, real)
    G_loss = loss_ad_g + loss_l1_g

    return G_loss
