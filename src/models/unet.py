"""Standard U-Net (Phase 3 baseline).

Plain encoder/decoder U-Net with double-conv blocks, max-pool downsampling and
transpose-conv upsampling. No attention gates, no pretrained / MobileNetV2
encoder - those belong to the Phase 4 proposed model. Input is the 8-band
bi-temporal stack, output is a single-channel logit map (forest-loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        classes: int = 1,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.depth = depth
        chs = [base_channels * (2 ** i) for i in range(depth + 1)]  # e.g. 32,64,128,256,512

        self.downs = nn.ModuleList()
        prev = in_channels
        for c in chs[:-1]:
            self.downs.append(DoubleConv(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chs[-2], chs[-1])

        self.upconvs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(depth):
            in_c = chs[-1 - i]
            out_c = chs[-2 - i]
            self.upconvs.append(nn.ConvTranspose2d(in_c, out_c, 2, stride=2))
            self.ups.append(DoubleConv(out_c * 2, out_c))

        self.head = nn.Conv2d(chs[0], classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(self.depth):
            x = self.upconvs[i](x)
            skip = skips[-1 - i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i](x)
        return self.head(x)


def build_model(name: str, **kw) -> nn.Module:
    name = name.lower()
    if name in ("unet", "baseline_unet"):
        return UNet(**kw)
    if name in ("attention_unet_mnv2", "attention_unet", "att_unet"):
        from .attention_unet import AttentionUNetMobileNetV2

        allowed = {"in_channels", "classes", "encoder_weights", "decoder_channels"}
        return AttentionUNetMobileNetV2(**{k: v for k, v in kw.items() if k in allowed})
    raise ValueError(f"unknown model '{name}'")
