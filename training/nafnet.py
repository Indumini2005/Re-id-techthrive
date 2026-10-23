"""NAFNet - Nonlinear Activation Free Network for image restoration.

Reference: Chen et al., "Simple Baselines for Image Restoration" (ECCV 2022).

Deliberately kept free of custom CUDA ops and of any activation that TensorRT
dislikes - the whole graph is conv / pointwise-mul / pixel-shuffle, which is
why it quantizes to INT8 cleanly and exports to ONNX without surprises.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors, written with primitive ops so
    ONNX export stays simple."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        y = (x - mu) / torch.sqrt(var + self.eps)
        return y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Replaces GELU/ReLU: split the channels in half and multiply them.
    No activation function, hence 'nonlinear activation free'."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(
        self,
        c: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        dw_channel = c * dw_expand

        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)

        # Simplified Channel Attention: global pool + 1x1, no sigmoid.
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)

        self.sg = SimpleGate()
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        # Learned residual scales, initialized at zero so the block starts as
        # an identity map and training is stable from step 1.
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """UNet-shaped stack of NAFBlocks with a global residual connection."""

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 32,
        middle_blk_num: int = 1,
        enc_blk_nums: List[int] = (1, 1, 1, 8),
        dec_blk_nums: List[int] = (1, 1, 1, 1),
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        enc_blk_nums, dec_blk_nums = list(enc_blk_nums), list(dec_blk_nums)
        assert len(enc_blk_nums) == len(dec_blk_nums), "encoder/decoder depth mismatch"

        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2))
            )
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    # -- helpers -----------------------------------------------------------
    def _check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if mod_h or mod_w:
            x = F.pad(x, (0, mod_w, 0, mod_h))
        return x

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """inp: N,3,H,W float in [0,1] -> same shape, denoised."""
        _, _, h, w = inp.shape
        x = self._check_image_size(inp)
        padded = x

        x = self.intro(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + padded                       # global residual: predict the noise
        return x[:, :, :h, :w]


def build_nafnet(width=None, enc=None, middle=None, dec=None, drop_out_rate=0.0) -> NAFNet:
    """Builds a NAFNet from config.py defaults unless overridden."""
    import config

    return NAFNet(
        img_channel=3,
        width=width if width is not None else config.NAFNET_WIDTH,
        enc_blk_nums=enc if enc is not None else config.NAFNET_ENC_BLKS,
        middle_blk_num=middle if middle is not None else config.NAFNET_MIDDLE_BLKS,
        dec_blk_nums=dec if dec is not None else config.NAFNET_DEC_BLKS,
        drop_out_rate=drop_out_rate,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = NAFNet(width=32, enc_blk_nums=[1, 1, 1, 8], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
    dummy = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = net(dummy)
    print(f"params : {count_parameters(net) / 1e6:.2f}M")
    print(f"in/out : {tuple(dummy.shape)} -> {tuple(out.shape)}")
