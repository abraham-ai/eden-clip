import torch 
import torchvision

Augmentations = torch.nn.Sequential(
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomAffine(24, (.1, .1))#, fill=0)
    )