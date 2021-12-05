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

    def __init__(self,
                 mask_detector,
                 encoder_params,
                 decoder_params,
                 use_att=True,
                 out_size=(256, 256)):
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
        self.encoder_type = encoder_params.pop('type')
        if self.encoder_type == 'drn':
            self.src_encoder = drn_c_42(pretrained=False, out_map=True)
            self.src_encoder.fc = torch.nn.Conv2d(self.src_encoder.out_dim,
                                                  encoder_params['img_f'],
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0,
                                                  bias=True)
            self.ref_encoder = drn_c_42(pretrained=False, out_map=True)
            self.ref_encoder.fc = torch.nn.Conv2d(self.ref_encoder.out_dim,
                                                  encoder_params['img_f'],
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0,
                                                  bias=True)
        elif self.encoder_type == 'pluralistic':
            # NOTE: 4 differences.
            # 1. no short+long attention in decoder
            # 2. prior from source, post from reference
            # 3. no kl divergence
            # 4. no disc_rec
            self.src_encoder = network.define_e(**encoder_params, encoder_type='src')
            self.ref_encoder = network.define_e(**encoder_params, encoder_type='ref')
        else:
            raise NotImplementedError
        self.decoder = network.define_g(**decoder_params)

        self.use_att = use_att
        if use_att:
            self.attention = ExampleGuidedAttention(encoder_params['img_f'])

        self.pool = nn.AdaptiveAvgPool2d(out_size)

    def forward(self, src_image, ref_image, src_mask=None, resize=True, no_prior=False):
        """
        both have shape [N, 3, 218, 178]  (CelebA dataset images)
        """
        if src_mask is None:
            src_mask = self.mask_detector(src_image, mode='eval')
        if self.encoder_type == 'drn':
            src_features = self.src_encoder(src_image)
            ref_features = self.ref_encoder(ref_image)
        elif self.encoder_type == 'pluralistic':
            src_dist, src_features = self.src_encoder(src_image)
            ref_dist, ref_features = self.ref_encoder(ref_image)

        if self.use_att:
            scaled_mask = scale_img(src_mask.unsqueeze(1), src_features.shape[-2:])
            enc_features = self.attention(scaled_mask, src_features, ref_features)
        else:
            scaled_mask = scale_img(src_mask.unsqueeze(1), src_features.shape[-2:])
            enc_features = (1 - scaled_mask) * src_features + scaled_mask * ref_features

        if self.encoder_type == 'drn' or no_prior:
            dec_image = self.decoder(enc_features)
        elif self.encoder_type == 'pluralistic':
            z = self.decoder.get_z(src_dist, ref_dist, return_zq=not self.use_att)
            dec_image = self.decoder(enc_features, z=z)

        if resize:
            dec_image = self.pool(dec_image)
        return dec_image
