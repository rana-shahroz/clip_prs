from functools import partial
import math
from typing import Sequence, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
import einops
import numbers 

from utils.hook import HookManager
# from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block


def make_2tuple(x) : 
    if isinstance(x, tuple) : 
        assert len(x) == 2
        return x
    
    assert isinstance(x, int)
    return (x, x)


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module : 
    if not depth_first and include_root : 
        fn(module=module, name=name)
    
    for child_name, child_module in module.named_children() : 
        child_name = ".".join((name, child_name)) if name else child_name 
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)

    if depth_first and include_root : 
        fn(module=module, name=name)
    
    return module



class MLP(nn.Module) : 
    def __init__ (
        self, 
        in_features: int, 
        hidden_features: Optional[int] = None, 
        out_features: Optional[int] = None, 
        act_layer: Callable[..., nn.Module] = nn.GELU, 
        drop: float = 0.0,
        bias: bool = True, 
        hook: Optional[HookManager] = None,
    ) -> None : 
        
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features
        
        self.hook = hook or HookManager()
        
        # c_fc layer
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        
        # gelu layer 
        self.act = act_layer()
        
        # c_proj layer
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        
        # Dropout layer
        self.drop = nn.Dropout(drop)
        
    
    def forward(
        self, 
        x: Tensor, 
    ) -> Tensor : 
        
        # Hooking the results with same names for ease
        x = self.hook('c_fc.post', ret=self.fc1(x))
        x = self.hook('gelu.post', ret=self.gelu(x))
        x = self.hook('dropout1.post', ret=self.drop(x))
        x = self.hook('c_proj.post', ret=self.c_proj(x))
        x = self.hook('dropout2.post', ret=self.drop(x))
        self.hook.finalize()
        return x
    
    
class PatchEmbed(nn.Module) : 
    """
    2D Image to patch Embeddings. [Does not use Patct dropout]
    (B, C, H, W) -> (B, N, D)
    ** NO need to Hook ** 
    
    Args: 
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """
    def __init__(
        self, 
        img_size: Union[int, Tuple[int, int]] = 224, 
        patch_size: Union[int, Tuple[int, int]] = 16, 
        in_chans: int = 3, 
        embed_dim: int = 768, 
        norm_layer: Optional[Callable] = None, 
        flatten_embedding: bool = True,
    ) -> None : 
        
        super().__init__()
        
        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )
        
        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        
        self.in_chans = in_chans 
        self.embed_dim = embed_dim
        
        self.flatten_embedding = flatten_embedding 
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    
    def forward(
        self, 
        x: Tensor, 
    ) -> Tensor : 
        _, _, H, W = x.shape 
        patch_H, patch_W = self.patch_size 
        
        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"
        
        x = self.proj(x) # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2). transpose(1, 2) # B HW C 
        x = self.norm(x)
        if not self.flatten_embedding : 
            x = x.reshape(-1, H, W, self.embed_dim) # B H W C 
        
        return x
    
    def flops(self) -> float : 
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None : 
            flops += Ho * Wo * self.embed_dim
            
        return flops