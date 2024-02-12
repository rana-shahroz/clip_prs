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

model, _, preprocess = create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')

device = "cuda:0"
dinov2_vitb14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
ds = torchvision.datasets.ImageNet(root='./imagenet', split="val", transform=preprocess)
dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=10)

print(model)
print("*********\n\n")
print(dinov2_vitb14_lc)
# for i, (image, _) in enumerate(tqdm.tqdm(dataloader)):
#     with torch.no_grad():
#         # prs.reinit()
#         representation = dinov2_vitb14_lc(image)
#         # rep = model(image)
#         rep1 = model.encode_image(image)
#     break

# print(model)
# print(dinov2_vitb14_lc)
# print(representation.shape)
# print(representation)
# print(rep1.shape)
# print(rep1)
# print(dinov2_vitb14_lc)