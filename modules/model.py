import torch.nn.functional as F
from torch import nn

from modules.example_guided_att import ExampleGuidedAttention
from modules.pluralistic_model import network


def scale_img(img, size):
    scaled_img = F.interpolate(img, size=size, mode='bilinear', align_corners=True)
    return scaled_img


class ReferenceFill(nn.Module):

    def __init__(self, mask_detector, encoder_params, decoder_params):
        """
        mask_params:
            n_channels,
            bilinear=True,
            threshold=0.5

        encoder_params: (default)
            ngf=32,
            img_f=128,
            layers=5,
            norm='none',
            activation='LeakyReLU',
            init_type='orthogonal'

        decoder_params: (default)
            ngf=32,
            img_f=256,
            layers=5,
            norm='instance',
            activation='LeakyReLU',
            init_type='orthogonal'
        """
        super().__init__()
        self.mask_detector = mask_detector
        self.src_encoder = network.define_e(**encoder_params)
        self.ref_encoder = network.define_e(**encoder_params)
        self.decoder = network.define_g(**decoder_params)

        self.attention = ExampleGuidedAttention(encoder_params['img_f'])

    def forward(self, src_image, ref_image, src_mask=None):
        """
        both have shape [N, 3, 218, 178]  (CelebA dataset images)
        """
        if src_mask is None:
            src_mask = self.mask_detector(src_image, mode='eval')
        src_features = self.src_encoder(src_image)
        ref_features = self.ref_encoder(ref_image)

        scaled_mask = scale_img(src_mask.unsqueeze(1), src_features.shape[-2:])
        enc_features = self.attention(scaled_mask, src_features, ref_features)
        dec_image = self.decoder(enc_features)
        scale_img(dec_image, src_image.shape[-2:])
        return dec_image
