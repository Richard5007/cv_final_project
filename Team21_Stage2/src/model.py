"""Stage 2 Transfer Learning architectures and fine-tuning freezing strategies."""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torchvision import models


CLASS_ORDER = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PRETRAINED_SOURCES = {
    "densenet121": "torchvision.models.DenseNet121_Weights.IMAGENET1K_V1",
    "mobilenet_v3_large": "torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2",
}


def build_stage2_model(
    architecture: str,
    strategy: str = "full_finetuning",
    num_classes: int = 6,
    pretrained: bool = False,
) -> nn.Module:
    """Build Stage 2 transfer learning model.

    Always call with pretrained=False during offline inference so weights are loaded
    strictly from the local stage2_model.pth checkpoint without internet access.
    """
    arch = architecture.lower()
    if arch == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features  # 1024
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )
    elif arch == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features  # 1280
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    apply_finetuning_strategy(model, arch, strategy)
    return model


def apply_finetuning_strategy(model: nn.Module, architecture: str, strategy: str) -> None:
    """Freeze/unfreeze model parameters according to the selected fine-tuning strategy."""
    arch = architecture.lower()
    strat = strategy.lower()

    if strat == "feature_extraction":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif strat == "partial_finetuning":
        for param in model.parameters():
            param.requires_grad = False
        if arch == "densenet121":
            # Freeze conv0..transition3; unfreeze denseblock4, norm5, and classifier
            for name in ("denseblock4", "norm5"):
                submodule = getattr(model.features, name)
                for param in submodule.parameters():
                    param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif arch == "mobilenet_v3_large":
            # Freeze features[0:13]; unfreeze features[13:17] and classifier
            for idx in range(13, len(model.features)):
                for param in model.features[idx].parameters():
                    param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    elif strat == "full_finetuning":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")


def get_parameter_groups(
    model: nn.Module,
    strategy: str,
    head_lr: float = 1e-3,
    backbone_lr: float = 1.5e-4,
) -> List[Dict]:
    """Create optimizer parameter groups with differential learning rates for fine-tuning."""
    strat = strategy.lower()
    if strat == "feature_extraction":
        return [{"params": [p for p in model.classifier.parameters() if p.requires_grad], "lr": head_lr}]

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier"):
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    return groups


def get_layer_freeze_summary(model: nn.Module, architecture: str, strategy: str) -> Dict:
    """Return exact frozen/trainable parameter counts and top-level layer breakdown."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    arch = architecture.lower()
    frozen_modules: List[str] = []
    trainable_modules: List[str] = []

    if arch == "densenet121":
        for name, child in model.features.named_children():
            has_grad = any(p.requires_grad for p in child.parameters())
            (trainable_modules if has_grad else frozen_modules).append(f"features.{name}")
        has_cls_grad = any(p.requires_grad for p in model.classifier.parameters())
        (trainable_modules if has_cls_grad else frozen_modules).append("classifier")
    elif arch == "mobilenet_v3_large":
        for idx, child in enumerate(model.features):
            has_grad = any(p.requires_grad for p in child.parameters())
            (trainable_modules if has_grad else frozen_modules).append(f"features[{idx}]")
        has_cls_grad = any(p.requires_grad for p in model.classifier.parameters())
        (trainable_modules if has_cls_grad else frozen_modules).append("classifier")

    return {
        "architecture": arch,
        "strategy": strategy,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "frozen_params": int(total_params - trainable_params),
        "trainable_ratio_pct": float(100.0 * trainable_params / max(total_params, 1)),
        "frozen_modules": frozen_modules,
        "trainable_modules": trainable_modules,
    }
