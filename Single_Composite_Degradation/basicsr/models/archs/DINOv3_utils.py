import torch
import torch.nn as nn
import torch.nn.functional as F
import re


REPO_DIR = "dinov3"
DINO_NAME = "dinov3_vitl16"
MODEL_TO_NUM_LAYERS = {
    "VITS": 12, "VITSP": 12, "VITB": 12,
    "VITL": 24, "VITHP": 32, "VIT7B": 40,
}


class DINOExtractor(nn.Module):

    def __init__(
        self,
        weights_path="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        extract_ids=(5, 11, 17, 23),    # Zero-based indices. For DINOv3 ViT-L/16 with 24 blocks, these correspond to the 6th, 12th, 18th, and 24th layers.
        device=None,
    ):
        super().__init__()

        self.model = torch.hub.load(
            REPO_DIR,
            DINO_NAME,
            source="local",
            weights=weights_path,
        ).eval()

        self.n_layers = MODEL_TO_NUM_LAYERS[
            re.sub(r"\d+", "", DINO_NAME.split("_")[-1]).upper()
        ]
        self.patch_size = int(re.findall(r"\d+", DINO_NAME.split("_")[-1])[-1])
        self.extract_ids = list(extract_ids)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: Input image tensor with shape [B, 3, H, W], expected to be in [0, 1].

        Returns:
            A list of DINOv3 feature maps extracted from selected intermediate layers.
            For DINOv3 ViT-L/16 with 512x512 input, each feature map has shape [B, 1024, 32, 32].
        """
        x = x.clamp(0, 1)
        x = F.interpolate(
            x, size=(512, 512), mode="bilinear", align_corners=False, antialias=True
        )
        x = (x - self.mean) / self.std

        with torch.no_grad():
            feats = self.model.get_intermediate_layers(
                x, n=self.n_layers, reshape=True, norm=True
            )

        return [feats[i] for i in self.extract_ids]


class DimAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, reduction=64):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, reduction, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(
                reduction, reduction, kernel_size=3, padding=1,
                groups=reduction, bias=False
            ),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Conv2d(reduction, out_dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.reduce(x)
        x = self.dw(x)
        return self.proj(x)


class SemanticAdapter(nn.Module):
    def __init__(self, in_dim=1024, out_dim=256, sizes=(32, 16, 8, 4), bottleneck=64):
        super().__init__()
        self.sizes = list(sizes)
        self.blocks = nn.ModuleList([
            DimAdapter(in_dim, out_dim, reduction=bottleneck)
            for _ in self.sizes
        ])

    def forward(self, feats):
        outs = []
        for i, x in enumerate(feats):
            x = F.interpolate(
                x,
                size=(self.sizes[i], self.sizes[i]),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            outs.append(self.blocks[i](x))
        return outs


class SemanticModulator(nn.Module):
    def __init__(self, dim, condition_dim=256):
        super().__init__()
        self.condition_proj = nn.Sequential(
            nn.Conv2d(condition_dim, dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim * 2, 1),
        )
        self.modulation_strength = nn.Parameter(torch.zeros(1))

    def forward(self, x, condition):
        if condition.shape[2:] != x.shape[2:]:
            condition = F.interpolate(
                condition, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        scale, shift = self.condition_proj(condition).chunk(2, dim=1)
        x_sft = x * (1 + scale) + shift
        return x + self.modulation_strength * (x_sft - x)