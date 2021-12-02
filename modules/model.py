import torch
import torch.nn.functional as F
from torch import nn

from modules.example_guided_att import ExampleGuidedAttention
from modules.pluralistic_model import network
from modules.drn import *


def scale_img(img, size):
    scaled_img = F.interpolate(img, size=size, mode='bilinear', align_corners=True)
    return scaled_img


class ReferenceFill(nn.Module):

    def __init__(self, mask_detector, encoder_params, decoder_params, use_att=True):
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
        encoder_type = encoder_params.pop('type')
        if encoder_type == 'drn':
            self.src_encoder = drn_c_42(pretrained=True, out_map=True)
            self.src_encoder.fc = torch.nn.Conv2d(self.src_encoder.out_dim,
                                                  encoder_params['img_f'],
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0,
                                                  bias=True)
            self.ref_encoder = drn_c_42(pretrained=True, out_map=True)
            self.ref_encoder.fc = torch.nn.Conv2d(self.ref_encoder.out_dim,
                                                  encoder_params['img_f'],
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0,
                                                  bias=True)
            decoder_params['layers'] = 6
        elif encoder_type == 'pluralistic':
            self.src_encoder = network.define_e(**encoder_params)
            self.ref_encoder = network.define_e(**encoder_params)
        else:
            raise NotImplementedError
        self.decoder = network.define_g(**decoder_params)

        self.use_att = use_att
        if use_att:
            self.attention = ExampleGuidedAttention(encoder_params['img_f'])

    def forward(self, src_image, ref_image, src_mask=None):
        """
        both have shape [N, 3, 218, 178]  (CelebA dataset images)
        """
        if src_mask is None:
            src_mask = self.mask_detector(src_image, mode='eval')
        src_features = self.src_encoder(src_image)
        ref_features = self.ref_encoder(ref_image)

        if self.use_att:
            scaled_mask = scale_img(src_mask.unsqueeze(1), src_features.shape[-2:])
            enc_features = self.attention(scaled_mask, src_features, ref_features)
        else:
            enc_features = torch.cat([src_features, ref_features], dim=1)
        dec_image = self.decoder(enc_features)
        dec_image = scale_img(dec_image, src_image.shape[-2:])
        return dec_image
