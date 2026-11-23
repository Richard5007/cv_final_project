"""Custom from-scratch CNN architecture for Stage 1 Steel Surface Defect Classification."""

from typing import Dict
import torch
import torch.nn as nn


CLASS_ORDER = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


class ConvBlock(nn.Module):
    """Single convolutional block: Conv2d -> optional BatchNorm2d -> ReLU -> optional MaxPool2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool = False,
        use_pool: bool = True,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not use_bn,
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SteelDefectCNN(nn.Module):
    """Self-designed from-scratch CNN for 1-channel grayscale steel defect images."""

    def __init__(
        self,
        num_classes: int = 6,
        use_bn: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_bn = use_bn
        self.dropout_p = dropout_p

        self.features = nn.Sequential(
            ConvBlock(1, 32, use_bn=use_bn, use_pool=True),
            ConvBlock(32, 64, use_bn=use_bn, use_pool=True),
            ConvBlock(64, 128, use_bn=use_bn, use_pool=True),
            ConvBlock(128, 256, use_bn=use_bn, use_pool=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        classifier_layers = [
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            classifier_layers.append(nn.Dropout(p=dropout_p))
        classifier_layers.append(nn.Linear(128, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}
