import torch
from torch import nn

from example_guided_att import ExampleGuidedAttention
from mask_detector import MaskDetector


class ImageEncoder(nn.Module):

    def __init__(self, params):
        super().__init__()

    def forward(self, image, mask=None):
        pass


class ImageDecoder(nn.Module):

    def __init__(self, params):
        super().__init__()

    def forward(self, encoded_features):
        pass


class ImageDiscriminator(nn.Module):

    def __init__(self, params):
        super().__init__()

    def forward(self, real_image, fake_image):
        pass


class ReferenceFill(nn.Module):

    def __init__(self, mask_params, encoder_params, decoder_params):
        super().__init__()
        self.mask_detector = MaskDetector(**mask_params)
        self.src_encoder = ImageEncoder(encoder_params)
        self.ref_encoder = ImageEncoder(encoder_params)
        self.decoder = ImageDecoder(decoder_params)

        self.attention = ExampleGuidedAttention(encoder_params.out_channels)

    def forward(self, src_image, ref_image):
        src_mask = self.mask_detector(src_image, mode='eval')
        src_features = self.src_encoder(src_image, src_mask)
        ref_features = self.ref_encoder(ref_image)

        enc_features = self.attention(src_mask, src_features, ref_features)
        dec_image = self.decoder(enc_features)
        return dec_image
