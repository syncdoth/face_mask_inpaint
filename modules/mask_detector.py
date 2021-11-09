import torch
from torch import nn

from Pytorch_UNet.unet.unet_model import UNet


class MaskDetector(nn.Module):

    def __init__(self, n_channels, bilinear=True, threshold=0.5):
        super().__init__()
        self.model = UNet(n_channels, 1, bilinear=bilinear)
        self.threshold = threshold

    def forward(self, image, mode='train'):
        """
        image: torch.FloatTensor of [N, C, H, W]
        return: mask. torch.FloatTensor of [N, C, H, W]

        if mode is not train, return the threshold value.
        """
        output = self.model(image)
        probs = torch.sigmoid(output)[0]

        if mode == 'train':
            return probs

        return probs > self.threshold
