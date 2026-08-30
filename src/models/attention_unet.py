"""Phase 4 proposed model: Attention U-Net with a MobileNetV2 encoder.

- Encoder: `segmentation_models.pytorch`'s ImageNet-pretrained MobileNetV2
  (not hand-rolled), adapted to 8 input channels by smp's first-conv weight
  inflation.
- Decoder: hand-built U-Net decoder where every skip connection passes through
  an additive attention gate (Oktay et al., 2018) whose gating signal is the
  upsampled coarser decoder feature. This matches "attention gates on all skip
  connections" from the brief and the John & Zhang (2022) design.
- Loss is the same Dice + BCE used for the Phase 3 baseline.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Oktay-style additive attention gate.

    x: skip feature (F_l channels), g: gating signal from coarser decoder level
    (F_g channels). Returns x re-weighted by a learned spatial attention map.
    """

    def __init__(self, f_l: int, f_g: int, f_int: int) -> None:
        super().__init__()
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1, bias=False), nn.BatchNorm2d(f_int))
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=False), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        a = self.relu(self.w_x(x) + self.w_g(g))
        a = self.psi(a)
        return x * a


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionUNetMobileNetV2(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        classes: int = 1,
        encoder_weights: str | None = "imagenet",
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
    ) -> None:
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            "mobilenet_v2", in_channels=in_channels, depth=5, weights=encoder_weights,
        )
        enc_ch = list(self.encoder.out_channels)      # e.g. [8, 16, 24, 32, 96, 1280]

        # decoder consumes features[1:] (5 levels): deepest first
        skips = enc_ch[1:][::-1]                      # [1280, 96, 32, 24, 16]
        self.up_blocks = nn.ModuleList()
        self.att_gates = nn.ModuleList()             # one per decoder step that has a skip
        prev = skips[0]                               # 1280 (bottleneck)
        for i, dec_c in enumerate(decoder_channels):
            skip_c = skips[i + 1] if i + 1 < len(skips) else 0
            if skip_c > 0:
                self.att_gates.append(AttentionGate(f_l=skip_c, f_g=prev,
                                                    f_int=max(skip_c // 2, 8)))
            self.up_blocks.append(_DoubleConv(prev + skip_c, dec_c))
            prev = dec_c
        self.head = nn.Conv2d(prev, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)                       # 6 tensors, strides 1..32
        dec_feats = feats[1:][::-1]                   # [f/32, f/16, f/8, f/4, f/2]
        d = dec_feats[0]
        for i, up in enumerate(self.up_blocks):
            d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
            if i + 1 < len(dec_feats):
                skip = dec_feats[i + 1]
                if i < len(self.att_gates):
                    skip = self.att_gates[i](skip, d)
                if d.shape[-2:] != skip.shape[-2:]:
                    d = F.interpolate(d, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                d = torch.cat([skip, d], dim=1)
            d = up(d)
        return self.head(d)
