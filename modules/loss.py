"""
Loss Network (VGG16) and stuff
"""
import torch
import torchvision
from torch import nn

from modules.pluralistic_model import base_function
from modules.pluralistic_model.external_function import GANLoss
from modules.pluralistic_model.external_function import StyleLoss
from modules.pluralistic_model.external_function import contextual_loss
from modules.model import scale_img


#Implementation mainly from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
class VGGLoss(torch.nn.Module):

    def __init__(self):
        super(VGGLoss, self).__init__()
        blocks = []
        pretrained_vgg = torchvision.models.vgg16(pretrained=True)
        blocks.append(pretrained_vgg.features[:4].eval())
        blocks.append(pretrained_vgg.features[4:9].eval())
        blocks.append(pretrained_vgg.features[9:16].eval())
        blocks.append(pretrained_vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def rescale_images(self, images):
        ret = []
        for img in images:
            if len(img.shape) < 4:
                img = img.unsqueeze(1)
                ret.append(scale_img(img, [224, 224]).squeeze(1))
            else:
                ret.append(scale_img(img, [224, 224]))

        return ret

    def forward(self, input, target, lossType='perceptual'):
        # Perceptual Loss : input = gen_img, target = gt_img
        # Style Loss : input = gen_img * src_mask, target = src_img,
        if (input.shape[-2], input.shape[-1]) == (224, 224):
            input, target = self.rescale_images([input, target])
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            dim = x.shape[1] * x.shape[2] * x.shape[3]  # C * H * W
            if lossType == 'perceptual':  #perceptual
                loss += torch.nn.functional.l1_loss(x, y) / dim
            elif lossType == 'style':  #style
                loss += StyleLoss(x, y) / (x.shape[1] * x.shape[1] * dim)
            elif (lossType == 'contextual' and i > 2):  # use 4th block only
                loss += contextual_loss(x, y) / dim
        return loss


class GANOptimizer(nn.Module):

    def __init__(self, optimizer_D, optimizer_G, lambda_g=0.01, debug=False):
        super().__init__()
        self.gan_loss = GANLoss('lsgan')
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGLoss()
        self.debug = debug  # if debug, return 0 for not implemented loss terms
        self.optimizer_D = optimizer_D
        self.optimizer_G = optimizer_G

        self.lambda_perc = 0.1
        self.lambda_style = 250
        self.lambda_cx = 1
        self.lambda_g = lambda_g

    def perceptual_loss(self, gt_img, gen_img):
        return self.vgg_loss(gen_img, gt_img, lossType='perceptual')

    def style_loss(self, gen_img, src_img, src_mask):
        src_mask = (1 - src_mask).unsqueeze(1)  # Yes inverse
        return self.vgg_loss(gen_img * src_mask, src_img, lossType='style')

    def contextual_loss(self, gen_img, ref_img, src_mask):
        src_mask = src_mask.unsqueeze(1)  # No inverse
        return self.vgg_loss(gen_img * src_mask,
                             ref_img * src_mask,
                             lossType='contextual')

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

    def generator_loss(self, netD, real, fake, freeze=True):
        """Calculate training loss for the generator"""
        if freeze:
            base_function._freeze(netD)  # need to freeze here: unfreeze after backward
        D_fake = netD(fake)
        loss_ad_g = self.gan_loss(D_fake, True, False) * self.lambda_g
        loss_l1_g = self.l1_loss(fake, real)
        G_loss = loss_ad_g + loss_l1_g

        return G_loss

    def __call__(self, discriminator, src_img, gt_img, ref_img, gen_img, src_mask):
        G_loss = self.generator_loss(discriminator, gt_img, gen_img, freeze=False)
        perc_loss = self.perceptual_loss(gt_img, gen_img) * self.lambda_perc
        style_loss = self.style_loss(gen_img, src_img, src_mask) * self.lambda_style
        cx_loss = self.contextual_loss(gen_img, ref_img, src_mask) * self.lambda_cx

        G_loss = G_loss + perc_loss + style_loss + cx_loss

        self.optimizer_G.zero_grad()
        G_loss.backward()
        self.optimizer_G.step()

        D_loss = self.discriminator_loss(discriminator, gt_img, gen_img)
        self.optimizer_D.zero_grad()
        D_loss.backward()
        self.optimizer_D.step()

        return D_loss, G_loss

    def calc_loss(self, discriminator, src_img, gt_img, ref_img, gen_img, src_mask):
        D_loss = self.discriminator_loss(discriminator, gt_img, gen_img)
        G_loss = self.generator_loss(discriminator, gt_img, gen_img, freeze=False)
        perc_loss = self.perceptual_loss(gt_img, gen_img)
        style_loss = self.style_loss(gen_img, src_img, src_mask)
        cx_loss = self.contextual_loss(gen_img, ref_img, src_mask)

        G_loss = G_loss + perc_loss + style_loss + cx_loss
        return D_loss, G_loss


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
