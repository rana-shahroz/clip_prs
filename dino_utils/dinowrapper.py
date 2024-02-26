
from enum import Enum
from typing import Union, Optional
from dino_utils.hook import HookManager
import torch
import torch.nn as nn

from .make_dino import _make_dinov2_model, _DINOV2_BASE_URL, _make_dinov2_model_name

class Weights(Enum):
    IMAGENET1K = "IMAGENET1K"


def _make_dinov2_linear_classification_head(
    *,
    arch_name: str = "vit_large",
    patch_size: int = 14,
    embed_dim: int = 1024,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    num_register_tokens: int = 0,
    **kwargs,
):
    if layers not in (1, 4):
        raise AssertionError(f"Unsupported number of layers: {layers}")
    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    linear_head = nn.Linear((1 + layers) * embed_dim, 1_000)

    if pretrained:
        model_base_name = _make_dinov2_model_name(arch_name, patch_size)
        model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
        layers_str = str(layers) if layers == 4 else ""
        url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_linear{layers_str}_head.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        linear_head.load_state_dict(state_dict, strict=True)

    return linear_head


class DinoWrapper(nn.Module) : 
    def __init__(
        self, 
        name = "dinov2_vitb14_lc",
        hook : Optional[HookManager] = None, 
        **kwargs,
    ) -> None : 
        
        super().__init__()
        self.layers = 1
        self.hook_manager = hook or HookManager()
        if name == "dinov2_vitb14_lc" : 
            self.backbone = _make_dinov2_model(
                arch_name="vit_base",
                pretrained=True,
                num_register_tokens=0,
                interpolate_antialias=False,
                interpolate_offset=0.1,
                hook = self.hook_manager.fork('backbone'),
                **kwargs,
            )
            
            embed_dim = self.backbone.embed_dim
            patch_size = self.backbone.patch_size
            self.linear_head = _make_dinov2_linear_classification_head(
                arch_name="vit_base",
                patch_size=patch_size,
                embed_dim=embed_dim,
                layers=1,
                pretrained=True,
                weights=Weights.IMAGENET1K,
                num_register_tokens=False,
            )
        
    def forward(self, x) : 
        if self.layers == 1 : 
            x = self.backbone.forward_features(x)
            cls_token = x["x_norm_clstoken"]
            patch_tokens = x["x_norm_patchtokens"]
            
            linear_input = torch.cat([
                cls_token, 
                patch_tokens.mean(dim=1),
            ], dim = 1)
        elif self.layers == 4 : 
            x = self.backbone_get_intermediate_layers(x, n=4, return_class_token=True)
            linear_input = torch.cat([
                x[0][1],
                x[1][1],
                x[2][1],
                x[3][1],
                x[3][0].mean(dim=1)
            ], dim=1)
        else : 
            assert False, f"Unsupported number of layers: {self.layers}"
        
        return self.linear_head(linear_input)
    
    def encode_image(self, x) : 
        return self.backbone.forward_features(x)
    
    def get_predictions(self, representation) : 

        cls_token = representation["x_norm_clstoken"]
        patch_tokens = representation["x_norm_patchtokens"]
        
        linear_input = torch.cat([
            cls_token, 
            patch_tokens.mean(dim=1),
        ], dim = 1)
        
        return self.linear_head(linear_input)
    
    
    
def _make_dinov2_linear_classifier(
    *,
    arch_name: str = "vit_large",
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    **kwargs,
):
    backbone = _make_dinov2_model(
        arch_name=arch_name,
        pretrained=pretrained,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
        **kwargs,
    )

    embed_dim = backbone.embed_dim
    patch_size = backbone.patch_size
    linear_head = _make_dinov2_linear_classification_head(
        arch_name=arch_name,
        patch_size=patch_size,
        embed_dim=embed_dim,
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=num_register_tokens,
    )

    return DinoWrapper(backbone=backbone, linear_head=linear_head, layers=layers)


def dinov2_vits14_lc(
    *,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    **kwargs,
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-S/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_small",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )


def dinov2_vitb14_lc(
    *,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    **kwargs,
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-B/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_base",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )


def dinov2_vitl14_lc(
    *,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    **kwargs,
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-L/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_large",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )


def dinov2_vitg14_lc(
    *,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    **kwargs,
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-g/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_giant2",
        layers=layers,
        ffn_layer="swiglufused",
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )


def dinov2_vits14_reg_lc(
    *, layers: int = 4, pretrained: bool = True, weights: Union[Weights, str] = Weights.IMAGENET1K, **kwargs
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-S/14 backbone with registers (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_small",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitb14_reg_lc(
    *, layers: int = 4, pretrained: bool = True, weights: Union[Weights, str] = Weights.IMAGENET1K, **kwargs
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-B/14 backbone with registers (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_base",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitl14_reg_lc(
    *, layers: int = 4, pretrained: bool = True, weights: Union[Weights, str] = Weights.IMAGENET1K, **kwargs
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-L/14 backbone with registers (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_large",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitg14_reg_lc(
    *, layers: int = 4, pretrained: bool = True, weights: Union[Weights, str] = Weights.IMAGENET1K, **kwargs
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-g/14 backbone with registers (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_giant2",
        layers=layers,
        ffn_layer="swiglufused",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )
