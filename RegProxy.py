import torch
import torch.nn as nn
from ViT import VisionTransformer
from proxy_head import ProxyHead


class RegProxy(nn.Module):
    def __init__(self, img_size, backbone_pretrained_path):
        super(RegProxy, self).__init__()
        self.backbone = VisionTransformer(img_size=img_size, out_indices=[2, 11])
        self.backbone.load_pretrained(backbone_pretrained_path)
        self.decode_head = ProxyHead(in_channels=768, channels=768, num_classes=2, region_res=(8, 8))

    def forward(self, x):
        out = self.backbone(x)
        out = self.decode_head(out)

        return out
