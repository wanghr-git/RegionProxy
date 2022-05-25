import einops
import torch.nn as nn
import torch.nn.functional as F
import collections.abc
from itertools import repeat


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


# A conv block that bundles conv -> norm -> activation layers
class depthwise_conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(depthwise_conv, self).__init__()
        self.convmodule = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding, bias=bias, groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convmodule(x)


class pointwise_conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(pointwise_conv, self).__init__()
        self.convmodule = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convmodule(x)


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, nin, nout, kernels_per_layer=1):
        super(DepthwiseSeparableConvModule, self).__init__()
        self.depthwise = depthwise_conv(nin, nin * kernels_per_layer)
        self.pointwise = pointwise_conv(nin * kernels_per_layer, nout, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ProxyHead(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 region_res=(4, 4),):

        super(ProxyHead, self).__init__()

        self._device = None
        self.region_res = to_2tuple(region_res)

        self.mlp = nn.Sequential(nn.Sequential(nn.Linear(in_channels, num_classes)))
        self.affinity_head = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels, channels),
            nn.Conv2d(
                channels, 9 * self.region_res[0] * self.region_res[1], kernel_size=1)
        )

    def init_weights(self):
        super(ProxyHead, self).init_weights()
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=.0)
        assert all(self.affinity_head[-1].conv.bias == 0)

    def forward_affinity(self, x):
        self._device = x.device
        B, _, H, W = x.shape

        # get affinity
        x = x.contiguous()
        affinity = self.affinity_head(x)
        affinity = affinity.reshape(B, 9, *self.region_res, H, W)  # (B, 9, h, w, H, W)

        # handle borders
        affinity[:, :3, :, :, 0, :] = float('-inf')  # top
        affinity[:, -3:, :, :, -1, :] = float('-inf')  # bottom
        affinity[:, ::3, :, :, :, 0] = float('-inf')  # left
        affinity[:, 2::3, :, :, :, -1] = float('-inf')  # right

        affinity = affinity.softmax(dim=1)
        return affinity

    def forward_cls(self, x):
        self._device = x.device
        B, _, H, W = x.shape

        # get token logits
        token_logits = self.mlp(x.permute(0, 2, 3, 1).reshape(B, H * W, -1))  # (B, H * W, C)
        return token_logits

    def forward(self, inputs):
        # x_mid, x = self._transform_inputs(inputs)  # (B, C, H, W)
        x_mid, x = inputs
        B, _, H, W = x.shape

        affinity = self.forward_affinity(x_mid)
        token_logits = self.forward_cls(x)

        # classification per pixel
        token_logits = token_logits.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, C, H, W)
        token_logits = F.unfold(token_logits, kernel_size=3, padding=1).reshape(B, -1, 9, H, W)  # (B, C, 9, H, W)
        token_logits = einops.rearrange(token_logits, 'B C n H W -> B H W n C')  # (B, H, W, 9, C)

        affinity = einops.rearrange(affinity, 'B n h w H W -> B H W (h w) n')  # (B, H, W, h * w, 9)
        seg_logits = (affinity @ token_logits).reshape(B, H, W, *self.region_res, -1)  # (B, H, W, h, w, C)
        seg_logits = einops.rearrange(seg_logits, 'B H W h w C -> B C (H h) (W w)')  # (B, C, H * h, W * w)

        return seg_logits
