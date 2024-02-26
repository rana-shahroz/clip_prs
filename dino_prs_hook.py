import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import argparse
import datetime
import json
from pathlib import Path


class PRSLogger(object):
    def __init__(self, model, device, spatial: bool = True):
        self.current_layer = 0
        self.device = device
        self.attentions = []
        self.mlps = []
        self.spatial = spatial
        self.post_ln_std = None
        self.post_ln_mean = None
        self.model = model
        self.pre_blocks = []
        self.post_blocks = []
        self.mean_c = None
        self.weighted_c = None

    @torch.no_grad()
    def compute_attentions_spatial(self, ret):
        # assert len(ret.shape) == 5, "Verify that you use method=`head` and not method=`head_no_spatial`" # [b, n, m, h, d]
        # assert self.spatial, "Verify that you use method=`head` and not method=`head_no_spatial`"
        # bias_term = self.model.transformer.blocks[
        #     self.current_layer
        # ].attn.proj.bias
        # self.current_layer += 1
        return_value = ret.detach().cpu()  # This is only for the cls token
            # print(return_value.shape)
        self.attentions.append(
            return_value
        )  # [b, n, h, d]
        return ret

    @torch.no_grad()
    def compute_attentions_non_spatial(self, ret):
        # assert len(ret.shape) == 4, "Verify that you use method=`head_no_spatial` and not method=`head`" # [b, n, h, d]
        # assert not self.spatial, "Verify that you use method=`head_no_spatial` and not method=`head`"
        # bias_term = self.model.transformer.blocks[
        #     self.current_layer
        # ].attn.proj.bias
        # self.current_layer += 1
        return_value = ret.detach().cpu()  # This is only for the cls toke
        self.attentions.append(
            return_value
        )  # [b, h, d]
        return ret

    @torch.no_grad()
    def compute_mlps(self, ret):
        self.mlps.append(ret.detach().cpu())  # [b, d]
        return ret

    @torch.no_grad()
    def log_post_ln_mean(self, ret):
        self.post_ln_mean = ret.detach().cpu()  # [b, 1]
        # return torch.zeros((self.post_ln_mean.shape))
        return ret.to(self.device)

    @torch.no_grad()
    def log_post_ln_std(self, ret):
        self.post_ln_std = ret.detach().cpu()  # [b, 1]
        return ret

    def _normalize_mlps(self):
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]
        # This is just the normalization layer:
        normalization_term = (
            self.mlps.shape[1] # This should not be on tokens
        )
        # print(self.mlps.shape)
        # print(self.post_ln_mean[:, np.newaxis, :,  :, ].shape)
        mean_centered = (
            self.mlps
            - self.post_ln_mean[:, np.newaxis, :,  :, ]) / (len_intermediates * normalization_term)
        weighted_mean_centered = (
            self.model.backbone.norm.weight.detach().to(self.device) * mean_centered
        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[:, np.newaxis, :,  :, ].to(self.device)
        bias_term = (
            self.model.backbone.norm.bias.detach().to(self.device) / len_intermediates
        )
        post_ln = weighted_mean_by_std + bias_term
        return post_ln, mean_centered.sum(axis=1), weighted_mean_by_std.sum(axis=1)

#     def _normalize_attentions_spatial(self):
#         len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]  # 2*l + 1
#         normalization_term = (
#             self.attentions.shape[2] * self.attentions.shape[3]
#         )  # n * h
#         # This is just the normalization layer:
#         print(self.attentions.shape) 
#         print(self.post_ln_mean.shape)
# #         torch.Size([1, 12, 257, 768])
# # torch.Size([1, 257, 1])
#         # print(self.post_ln_mean.permute(0, 2, 1)[:, :, :, np.newaxis].shape)
        
#         mean_centered = self.attentions - self.post_ln_mean[:, np.newaxis, :,  :, 
#                                                             ].to(self.device) / (len_intermediates * normalization_term)
        
#         # print(self.attentions.shape)
        
#         # mean_centered = self.attentions - self.post_ln_mean[
#         #     :, :, np.newaxis, np.newaxis, np.newaxis
#         # ].to(self.device) / (len_intermediates * normalization_term)
        
#         # print(mean_centered.shape)
        
#         # print(mean_centered.shape)
#         weighted_mean_centered = (
#             self.model.backbone.norm.weight.detach().to(self.device) * mean_centered
#         )
#         weighted_mean_by_std = weighted_mean_centered / self.post_ln_std.permute(0, 2, 1)[:, :, :, np.newaxis
#         ].to(self.device)
        
#         bias_term = self.model.backbone.norm.bias.detach().to(self.device) / (
#             len_intermediates * normalization_term
#         )
#         post_ln = weighted_mean_by_std + bias_term
#         return post_ln

    def _normalize_attentions_non_spatial(self):
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]  # 2*l + 1
        normalization_term = (
            self.attentions.shape[1] # This should not be on tokens
        )  # h
        # This is just the normalization layer:
        mean_centered = self.attentions - self.post_ln_mean[:, np.newaxis, :,  :, 
                                ].to(self.device) / (len_intermediates * normalization_term)
        # [1, 257, 1] -> [1, 1, 257, 1] , dont use permute, just add the np.newaxis in the middle
        # [1, 12, 257, 784] - > attns
        # [1, 13, 257, 784] -> mlps
        weighted_mean_centered = (
            self.model.backbone.norm.weight.detach().to(self.device) * mean_centered
        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[:, np.newaxis, :,  :, ].to(self.device)
        bias_term = self.model.backbone.norm.bias.detach().to(self.device) / (
            len_intermediates * normalization_term
        )
        post_ln = weighted_mean_by_std + bias_term 
        # print(mean_centered.shape)
        return post_ln, mean_centered.sum(axis=1), weighted_mean_by_std.sum(axis=1)


    def log_pre_pre(self, ret) : 
        self.pre_blocks.append(ret.detach().cpu())
        return ret
    
    def log_pre_post(self, ret) : 
        self.post_blocks.append(ret.detach().cpu())
        return ret
    

    def log_mean_reduced(self, ret) :
        self.mean_c = ret.detach().cpu()
        # return torch.zeros((ret.shape))
        return ret
    
    def log_weighted_c(self, ret) :
        self.weighted_c = ret.detach().cpu()
        return ret
    
    def log_xs(self, ret): 
        self.xs = ret.detach().cpu()
        # self.dims = ret[1]
        return ret
        # return torch.ones((ret.shape))
    
    @torch.no_grad()
    def finalize(self):
        """We calculate the post-ln scaling, project it and normalize by the last norm."""
        self.attentions = torch.stack(self.attentions, axis=1).to(
            self.device
        )  # [b, l, n, h, d] X -> 
        # print(self.attentions.shape)
        self.mlps = torch.stack(self.mlps, axis=1).to(self.device)  # [b, l + 1, d]
        self.pre_blocks = torch.stack(self.pre_blocks, axis=1).to(self.device)
        
        
        self.post_blocks = torch.stack(self.post_blocks, axis =1).to(self.device)
        # print(self.post_blocks.shape)
        calculated_post_blocks = (self.attentions.sum(axis=1) + self.mlps.sum(axis=1))[:, np.newaxis, :, :, ]
        # print(self.post_blocks.shape)
        # print(calculated_post_blocks.shape)
        
        # print((self.post_blocks - calculated_post_blocks).sum())
        mc_mlp = self.mlps - self.post_ln_mean[:, np.newaxis, :, :,].to(self.device) / (25 )
        mc_attn = self.attentions - self.post_ln_mean[:, np.newaxis, :, :, ].to(self.device) / (25 * 12)
        
        # This is the same, which means that we should get the attentions + mlps to be equal to post blocks.
        mean_centered_post_blocs = self.post_blocks - self.post_ln_mean[:, np.newaxis, :,  :, 
                                ].to(self.device)
        
        mc_test = calculated_post_blocks - self.post_ln_mean[:, np.newaxis, :, :, ].to(self.device)
        print(((mc_mlp.sum(axis=1) + mc_attn.sum(axis=1)) - self.mean_c).sum())
        # print((mc_test - self.mean_c).sum())
        # print(self.mlps.shape)
        # if self.spatial:
        #     projected_attentions = self._normalize_attentions_spatial()
        # else:
        projected_attentions, mc_attn, wc_attn = self._normalize_attentions_non_spatial()
        projected_mlps, mc_mlp, wc_mlp = self._normalize_mlps()
        # print(self.xs)
        # print(self.xs.shape)
        # print(self.xs.mean(axis=-1, keepdim=True).shape)
        # print(self.dims)
        # print(self.mean_c.shape)
        # print(self.post_ln_mean)
        # print(self.xs.mean(axis=-1, keepdim=True) - self.post_ln_mean)
        # print(mc_attn + mc_mlp)
        # print(mc_attn + mc_mlp - self.mean_c)
        # print((mc_attn + mc_mlp  - self.mean_c).sum()) # 10000
        # print((wc_attn + wc_mlp - self.weighted_c).sum())
        return (
            self.attentions,
            self.mlps, 
            self.pre_blocks, 
            self.post_blocks,
            projected_attentions, 
            projected_mlp
        )
        
    def reinit(self):
        self.current_layer = 0
        self.attentions = []
        self.mlps = []
        self.post_ln_mean = None
        self.post_ln_std = None
        self.pre_blocks = []
        self.post_blocks = []
        self.mean_c = None
        self.weighted_c = None
        self.xs = None
        self.dims = None
        torch.cuda.empty_cache()


def hook_prs_logger(model, device, spatial: bool = True):
    """Hooks a projected residual stream logger to the model."""
    prs = PRSLogger(model, device, spatial=spatial)
    if spatial:
        model.hook_manager.register(
            "transformer.resblocks.*.attn.out.post", prs.compute_attentions_spatial
        )
    else:
        model.hook_manager.register(
            "transformer.resblocks.*.attn.out.post", prs.compute_attentions_non_spatial
        )
    model.hook_manager.register(
        "transformer.resblocks.*.mlp.c_proj.post", prs.compute_mlps
    )
    model.hook_manager.register("ln_pre_post", prs.compute_mlps)
    model.hook_manager.register("ln_post.mean", prs.log_post_ln_mean)
    model.hook_manager.register("ln_post.sqrt_var", prs.log_post_ln_std)
    return prs


def hook_prs_logger_dino(model, device, spatial:bool = True) : 
    prs = PRSLogger(model, device)
    if spatial:
        model.hook_manager.register(
            "backbone.transformer.resblocks.*.after_attn", prs.compute_attentions_spatial
        )
    else:
        model.hook_manager.register(
            "backbone.transformer.resblocks.*.after_attn", prs.compute_attentions_non_spatial
    )
    
    model.hook_manager.register(
        "backbone.transformer.resblocks.*.after_mlp", prs.compute_mlps
    )
        
    model.hook_manager.register("backbone.transformer.ln_pre_pre", prs.compute_mlps)
    
    model.hook_manager.register(
        "backbone.transformer.ln_post.mean", prs.log_post_ln_mean
    )
    
    model.hook_manager.register(
        "backbone.transformer.ln_post.sqrt_var", prs.log_post_ln_std
    )
    
    model.hook_manager.register(
        "backbone.transformer.ln_pre_post", prs.log_pre_post
    )
    model.hook_manager.register(
        "backbone.transformer.ln_pre_pre", prs.log_pre_pre
    )
    
    model.hook_manager.register(
        "backbone.transformer.ln_post.mean_reduced", prs.log_mean_reduced,
    )
    
    model.hook_manager.register(
        "backbone.transformer.ln_post.weighted_mean", prs.log_weighted_c,
    )
    
    model.hook_manager.register(
        "backbone.transformer.ln_post.x", prs.log_xs,
    )
    return prs
    
    
    
# First try to find the correct dims for the normalization term
# Also correct the np.newaxis instead of everything else
# Make sure every step is correct on the way. 
#   Start from the pre normalization [ls and no ls check pls]
#   Then check the normalization 
#   Mean-reduced term : mean-centered term
#   renorm.post term : after post_ln things.
#   Work with non-spatial only pls




# 1. Try to set in the intervention, set logs function, return = same shape but 1s.
# 2. Try making it zeros for debugging as well. 
# Do it asap