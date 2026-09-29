"""Project-local Ultralytics layers used by the underwater lamp detector."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class ImageNetNormalize(nn.Module):
    """Normalize Ultralytics' [0, 1] images for ImageNet-pretrained backbones."""

    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return images
        return (images - self.mean.to(dtype=images.dtype)) / self.std.to(dtype=images.dtype)


class UnderwaterLightAttention(nn.Module):
    """A lightweight channel-spatial gate for high-resolution lamp features."""

    def __init__(self, channels: int, reduction: int = 8, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        hidden = max(channels // reduction, 4)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channel = self.channel_mlp(features.mean((2, 3), keepdim=True))
        channel = channel + self.channel_mlp(features.amax((2, 3), keepdim=True))
        features = features * self.sigmoid(channel)
        spatial = torch.cat((features.mean(1, keepdim=True), features.amax(1, keepdim=True)), dim=1)
        return features * self.sigmoid(self.spatial(spatial))


class AdaptiveReceptiveFieldFusion(nn.Module):
    """Select a per-channel receptive field for blurred or haloed lamp features."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.branches = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, groups=channels, bias=False)
            for dilation in (1, 2, 3)
        )
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, len(self.branches) * channels, kernel_size=1, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        branch_features = torch.stack([branch(features) for branch in self.branches], dim=1)
        batch, _, channels, _, _ = branch_features.shape
        weights = self.selector(features).view(batch, len(self.branches), channels, 1, 1)
        return (branch_features * weights.softmax(dim=1)).sum(dim=1)


class UnderwaterP2Enhancer(nn.Module):
    """Keep P2 enhancement switches in one stable layer for fair ablations."""

    def __init__(self, channels: int, attention: bool = True, arf: bool = False) -> None:
        super().__init__()
        self.arf = AdaptiveReceptiveFieldFusion(channels) if arf else nn.Identity()
        self.attention = UnderwaterLightAttention(channels) if attention else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.attention(self.arf(features))


class UnderwaterCrossScaleGate(nn.Module):
    """Dynamically balance upsampled semantics and P2 spatial detail."""

    def __init__(self, channels: int, reduction: int = 8, enabled: bool = True) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("UnderwaterCrossScaleGate requires two equal channel branches.")
        self.enabled = enabled
        branch_channels = channels // 2
        hidden = max(channels // reduction, 8)
        if enabled:
            self.selector = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            )
            # Near-equal initial weights preserve the original fusion while a
            # tiny nonzero initialization lets both selector layers learn on
            # the first optimizer step.
            nn.init.normal_(self.selector[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.selector[-1].bias)
        else:
            self.selector = nn.Identity()
        self.branch_channels = branch_channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return features
        semantic, detail = features.split(self.branch_channels, dim=1)
        descriptor = torch.cat(
            (
                semantic.mean((2, 3), keepdim=True),
                detail.mean((2, 3), keepdim=True),
            ),
            dim=1,
        )
        batch = features.shape[0]
        weights = self.selector(descriptor).view(batch, 2, self.branch_channels, 1, 1)
        weights = weights.softmax(dim=1) * 2.0
        return torch.cat((semantic * weights[:, 0], detail * weights[:, 1]), dim=1)


class ResidualCoordinateAttention(nn.Module):
    """Coordinate attention with identity initialization for stable transfer."""

    def __init__(self, channels: int, reduction: int = 16, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        if enabled:
            hidden = max(8, channels // reduction)
            self.reduce = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.Hardswish(inplace=True),
            )
            self.height_gate = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
            self.width_gate = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
            self.residual_scale = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return features
        height, width = features.shape[-2:]
        pooled_height = features.mean(dim=3, keepdim=True)
        pooled_width = features.mean(dim=2, keepdim=True).transpose(2, 3)
        encoded = self.reduce(torch.cat((pooled_height, pooled_width), dim=2))
        height_features, width_features = torch.split(encoded, (height, width), dim=2)
        width_features = width_features.transpose(2, 3)
        attention = self.height_gate(height_features).sigmoid() * self.width_gate(width_features).sigmoid()
        return features + self.residual_scale * features * attention


class _TorchVisionFeatureShell(nn.Module):
    """Present a custom feature extractor in TorchVision's classifier layout."""

    def __init__(self, features: nn.Sequential) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.Identity()
        self.classifier = nn.Identity()


class _FasterViTFixedResolutionStage(nn.Module):
    """Run a fixed-resolution FasterViT stage while allowing YOLO stride probing."""

    def __init__(
        self,
        stage: nn.Module,
        expected_size: int,
        bootstrap_max_size: int,
        detector_imgsz: int,
    ) -> None:
        super().__init__()
        self.stage = stage
        self.expected_size = expected_size
        self.bootstrap_max_size = bootstrap_max_size
        self.detector_imgsz = detector_imgsz

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        height, width = features.shape[-2:]
        if (height, width) == (self.expected_size, self.expected_size):
            return self.stage(features)

        # Ultralytics constructs a detector with a 256x256 dummy image to infer
        # output strides. Bypassing only that tiny probe preserves the correct
        # tensor scales without pretending FasterViT supports arbitrary sizes.
        if max(height, width) <= self.bootstrap_max_size:
            return features

        raise RuntimeError(
            "FasterViT-0 was built for square "
            f"{self.detector_imgsz}x{self.detector_imgsz} detector inputs, but its "
            f"stage received {height}x{width} features. Train, validate and predict "
            f"with imgsz={self.detector_imgsz} and rectangular batching disabled."
        )


def _register_fastervit_torchvision_model() -> None:
    """Register FasterViT-0 so Ultralytics can consume it through TorchVision."""
    from torchvision.models._api import BUILTIN_MODELS, register_model

    model_name = "fastervit0_anyres"
    if model_name in BUILTIN_MODELS:
        return

    @register_model(name=model_name)
    def fastervit0_anyres(*, weights=None, progress: bool = True, **kwargs):
        del progress
        try:
            from fastervit import create_model
        except ImportError as exc:
            raise ImportError(
                "FasterViT backbone requires fastervit>=0.9.8. "
                "Install it with: python -m pip install \"fastervit>=0.9.8\""
            ) from exc

        resolution = kwargs.pop("resolution", (1280, 1280))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported FasterViT options: {unknown}")
        if len(resolution) != 2 or resolution[0] != resolution[1]:
            raise ValueError("The detector adapter currently requires a square FasterViT resolution.")
        detector_imgsz = int(resolution[0])
        if detector_imgsz != 1280:
            raise ValueError("fastervit0_anyres is configured for detector imgsz=1280.")

        checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        backbone = create_model(
            "faster_vit_0_any_res",
            resolution=list(resolution),
            window_size=[7, 7, 7, 7],
            pretrained=weights not in (None, False, "none", "None"),
            model_path=str(checkpoint_dir / "fastervit_0_224_1k.pth.tar"),
        )

        # FasterViT stages downsample internally. Split each stage from its
        # downsampler so TorchVision/Index can expose P2, P3, P4 and P5.
        feature_layers: list[nn.Module] = [backbone.patch_embed]
        for level_index, level in enumerate(backbone.levels):
            downsample = level.downsample
            level.downsample = None
            if level_index == 2:
                level = _FasterViTFixedResolutionStage(
                    level,
                    expected_size=detector_imgsz // 16,
                    bootstrap_max_size=16,
                    detector_imgsz=detector_imgsz,
                )
            elif level_index == 3:
                level = _FasterViTFixedResolutionStage(
                    level,
                    expected_size=detector_imgsz // 32,
                    bootstrap_max_size=8,
                    detector_imgsz=detector_imgsz,
                )
            feature_layers.append(level)
            if downsample is not None:
                feature_layers.append(downsample)
        feature_layers.append(backbone.norm)
        return _TorchVisionFeatureShell(nn.Sequential(*feature_layers))


def register_ultralytics_layers() -> None:
    """Expose local layers to Ultralytics' YAML parser without editing its package."""
    from ultralytics.nn import tasks

    _register_fastervit_torchvision_model()
    tasks.ImageNetNormalize = ImageNetNormalize
    tasks.UnderwaterLightAttention = UnderwaterLightAttention
    tasks.AdaptiveReceptiveFieldFusion = AdaptiveReceptiveFieldFusion
    tasks.UnderwaterP2Enhancer = UnderwaterP2Enhancer
    tasks.UnderwaterCrossScaleGate = UnderwaterCrossScaleGate
    tasks.ResidualCoordinateAttention = ResidualCoordinateAttention
