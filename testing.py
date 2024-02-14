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
from enum import Enum
from typing import Any, Dict, TypeVar, Optional


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
    



model, _, preprocess = create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')

dino = dinov2_vitb14()
dino.load_state_dict(torch.load("./dinovitb14.pt"))
# print(dino)
device = "cuda:0"
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc').to(device)
# lin_head = lin_head.linear_head

# print(lin_head)
# state_dict = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# torch.save(state_dict.state_dict(), "./dinovitb14.pt")
# print(state_dict)

# print(dino)
# # device = "cuda:0"
# # dinov2_vitb14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
ds = torchvision.datasets.ImageNet(root='./imagenet', split="val", transform=preprocess)
dataloader = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=False, num_workers=10)

# # print(model)
# # print("*********\n\n")
# # print(dinov2_vitb14_lc)
def run_validate(loader, base_progress=0):
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm((loader))):
            # i = base_progress + i
            # if args.gpu is not None and torch.cuda.is_available():
            #     images = images.cuda(args.gpu, non_blocking=True)
            # if torch.backends.mps.is_available():
            #     images = images.to('mps')
            #     target = target.to('mps')
            # if torch.cuda.is_available():
            #     target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            images = images.to(device)
            target = target.to(device)
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
model.eval()
criterion = torch.nn.CrossEntropyLoss().to(device)
run_validate(dataloader)

print(top1.summary())
print(top5.summary())
# # print(model)
# # print(dinov2_vitb14_lc)
# print(representation.shape)
# print(representation)
# print(rep1.shape)
# print(rep1)
# print(dinov2_vitb14_lc)