import random
import torch 
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import kornia.augmentation as K


def get_augmentations_0(config, device):
    return torch.nn.Sequential(
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomAffine(24, (.1, .1))#, fill=0)
    )

def get_augmentations_1(config, device):
    return torch.nn.Sequential(
        T.RandomHorizontalFlip(),
        T.RandomAffine(21, (.1, .1)),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01)
    )

def get_augmentations_2(config, device):
    cut_pow = 1.0
    cut_size = 224
    cutn = config['cutn']
    aug_lists = [
        ['Af', 'Pe', 'Ji', 'Er'],
        ['Ji', 'Et', 'Af', 'Sh'],
        ['Sh', 'Pe'],
        ['Ro', 'Gn', 'Pe', 'Er'],
        ['Af', 'Et', 'Ts', 'Ji', 'Er', 'Gn'],
        ['Ji', 'Gn', 'Ts'],
        ['Et', 'Sh', 'Ro', 'Af', 'Re']
    ]
    aug_list = random.choice(aug_lists)
    augment_list = []
    for item in aug_list:
        if item == 'Ji':
            augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.05, p=0.5))
        elif item == 'Sh':
            augment_list.append(K.RandomSharpness(sharpness=0.4, p=0.7))
        elif item == 'Gn':
            augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1., p=0.5))
        elif item == 'Pe':
            augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
        elif item == 'Ro':
            augment_list.append(K.RandomRotation(degrees=15, p=0.7))
        elif item == 'Af':
            augment_list.append(K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'))
        elif item == 'Et':
            augment_list.append(K.RandomElasticTransform(p=0.7))
        elif item == 'Ts':
            augment_list.append(K.RandomThinPlateSpline(scale=0.3, same_on_batch=False, p=0.7))    # No
        elif item == 'Cr':
            augment_list.append(K.RandomCrop(size=(cut_size, cut_size), p=0.5))
        elif item == 'Er':
            augment_list.append(K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7))
        elif item == 'Re':
            augment_list.append(K.RandomResizedCrop(size=(cut_size, cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5))
    return torch.nn.Sequential(*augment_list)


def get_augmentations(config, device):
    return get_augmentations_2(config, device)
