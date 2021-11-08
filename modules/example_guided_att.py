import torch
from torch import nn


class ExampleGuidedAttention(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels / 4, 1, bias=False)

    def apply_attention_map(self, att_map, features):
        N, H, W, C = features.shape
        pixels = features.reshape(N, -1, C).permute(0, 2, 1)
        att_out = (pixels @ att_map).reshape(N, H, W, C)
        return att_out

    def forward(self, src_mask, src_feature, ref_feature):
        query = self.conv(src_feature)  # [N, H, W, C] -> [N, H, W, C/4]
        query = query.reshape(query.shape[0], -1, query.shape[-1])  # [N, H * W, C/4]
        # [N, H * W, H * W]
        att_map = torch.softmax(query @ query.permute(0, 2, 1), dim=-1)
        src_att = self.apply_attention_map(att_map, src_feature)
        ref_att = self.apply_attention_map(att_map, src_feature)

        src_mask = src_mask.unsqueeze(-1)  # [N, H, W, 1]

        ex_guide_flow = src_mask * ref_att + (1 - src_mask) * ref_feature

        out = torch.cat([ex_guide_flow, src_att], dim=-1)  # [N, H, W, C * 2]
        return out
