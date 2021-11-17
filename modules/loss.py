"""
Loss Network (VGG16) and stuff
"""
import torch
from torch import nn

from modules.pluralistic_model.external_function import GANLoss


class TotalLoss(nn.Module):
    # TODO: implement this class
    def __init__(self, lambda_g=1.0):
        super().__init__()
        self.gan_loss = GANLoss('lsgan')
        self.l1_loss = nn.L1Loss()
        self.lambda_g = lambda_g

    def perceptual_loss(self, gt_img, gen_img):
        raise NotImplementedError

    def style_loss(self, gen_img, src_img, src_mask):
        raise NotImplementedError

    def contextual_loss(self, gen_img, ref_img, src_mask):
        raise NotImplementedError

    def adv_loss(self, netD, real, fake):
        D_loss = self.discriminator_loss(netD, real, fake)
        G_loss = self.generator_loss(netD, real, fake)
        return D_loss, G_loss

    def discriminator_loss(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = self.gan_loss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.gan_loss(D_fake, False, True)

        D_loss = (D_real_loss + D_fake_loss) * 0.5
        return D_loss

    def generator_loss(self, netD, real, fake):
        """Calculate training loss for the generator"""
        D_fake = netD(fake)
        loss_ad_g = self.gan_loss(D_fake, True, False) * self.lambda_g
        loss_l1_g = self.l1_loss(fake, real)
        G_loss = loss_ad_g + loss_l1_g

        return G_loss

    def __call__(self, discriminator, src_img, gt_img, ref_img, gen_img, src_mask):
        D_loss, G_loss = self.adv_loss(discriminator, gt_img, gen_img)
        perc_loss = self.perceptual_loss(gt_img, gen_img)
        style_loss = self.style_loss(gen_img, src_img, src_mask)
        cx_loss = self.contextual_loss(gen_img, ref_img, src_mask)

        secondary_loss = perc_loss + style_loss + cx_loss

        return D_loss + secondary_loss, G_loss + secondary_loss


######################## Unet Losses #########################################
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
