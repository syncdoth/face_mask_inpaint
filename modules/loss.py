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


def dice_coeff(input, target, reduce_batch_first=False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(
            f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})'
        )

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input, target, reduce_batch_first=False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...],
                           reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input, target, multiclass=False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)