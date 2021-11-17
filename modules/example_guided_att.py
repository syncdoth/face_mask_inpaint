import torch
from torch import nn


class ExampleGuidedAttention(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels // 4, 1, bias=False)

    def apply_attention_map(self, att_map, features):
        N, C, H, W = features.shape
        pixels = features.reshape(N, C, -1)
        att_out = (pixels @ att_map.permute(0, 2, 1)).reshape(N, C, H, W)
        return att_out

    def forward(self, src_mask, src_feature, ref_feature):
        """
        src_mask: [N, 1, H, W]
        src_feature: [N, C, H, W]
        ref_feature: [N, C, H, W]
        """
        query = self.conv(src_feature)  # [N, C, H, W] -> [N, C/4, H, W]
        query = query.reshape(query.shape[0], query.shape[1], -1)  # [N, C/4, H * W]
        # [N, H * W, H * W]
        att_map = torch.softmax(query.permute(0, 2, 1) @ query, dim=-1)
        src_att = self.apply_attention_map(att_map, src_feature)
        ref_att = self.apply_attention_map(att_map, ref_feature)

        ex_guide_flow = (1 - src_mask) * ref_att + src_mask * ref_feature

        out = torch.cat([ex_guide_flow, src_att], dim=1)  # [N, C*2, H, W]
        return out
