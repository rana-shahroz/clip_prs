import torch

from utils.factory import create_model_and_transforms, get_tokenizer

device = "cuda:0"

# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
model, _, preprocess = create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
print(model)