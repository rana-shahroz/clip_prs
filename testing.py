# import torch

# from utils.factory import create_model_and_transforms, get_tokenizer

# device = "cuda:0"

# # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
# model, _, preprocess = create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
# print(model)

import torch
import torchvision
from utils.factory import create_model_and_transforms, get_tokenizer
import tqdm
from utils.make_dino import dinov2_vitb14
from utils.dinowrapper import DinoWrapper
from enum import Enum
from typing import Any, Dict, TypeVar, Optional
from prs_hook import hook_prs_logger_dino
import os
import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
import tqdm
from prs_hook import hook_prs_logger
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet, ImageFolder


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
    
    
class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)
    
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

output_dir = "./output_dir"

x, _, preprocess = create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
device = "cuda:0"
model = DinoWrapper().to(device)
prs = hook_prs_logger_dino(model, device)

ds = torchvision.datasets.ImageNet(root='./imagenet', split="val", transform=preprocess)
dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=10)

attention_results = []
mlp_results = []
cls_to_cls_results = []

for i, (image, _) in enumerate(tqdm.tqdm(dataloader)):
    with torch.no_grad():
        prs.reinit()
        representation = model.encode_image(
            image.to(device)
        )['x_norm_clstoken']
        attentions, mlps = prs.finalize(representation)
        attentions = attentions.detach().cpu().numpy()  # [b, l, n, h, d]
        mlps = mlps.detach().cpu().numpy()  # [b, l+1, d]
        attention_results.append(
            np.sum(attentions, axis=2)
        )  # Reduce the spatial dimension
        mlp_results.append(mlps)
        cls_to_cls_results.append(
            np.sum(attentions[:, :, 0], axis=2)
        )  # Store the cls->cls attention, reduce the heads
        
        break 

print(representation.shape)
print(attentions.sum(axis = (1,2, 3)).shape)
print(mlps.sum(axis=1).shape)    
rep2 = attentions + mlps 

print(representation.detach().cpu() - torch.from_numpy(rep2))
with open(
    os.path.join(output_dir, f"dino_attn.npy"), "wb"
) as f:
    np.save(f, np.concatenate(attention_results, axis=0))
with open(
    os.path.join(output_dir, f"dino_mlp.npy"), "wb"
) as f:
    np.save(f, np.concatenate(mlp_results, axis=0))
with open(
    os.path.join(output_dir, f"dino_cls_attn.npy"), "wb"
) as f:
    np.save(f, np.concatenate(cls_to_cls_results, axis=0))