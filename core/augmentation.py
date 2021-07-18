import torch 
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF





def gaussian_sigma(x):
    return 0.3 * ((x - 1) * 0.5 - 1) + 0.8

def random_resize(x, y, z):
    uni_scale = torch.rand((1,)).mul(y-1) + 1
    bi_scale  = torch.rand((2,)).mul(z)
    return F.interpolate(x, scale_factor=(uni_scale.item() + bi_scale[0].item(),uni_scale.item() + bi_scale[1].item()), mode='bilinear', align_corners=False, recompute_scale_factor=False)

def random_size_crop(x, y, z, v=0):
    crop_size = (torch.rand((1,)).mul(z-y) + y).int().item()
    x_var = (torch.rand((1,)).mul(v)).int().item()
    y_var = (torch.rand((1,)).mul(v)).int().item()
    return T.RandomCrop((crop_size - y_var, crop_size - x_var))(x)

def random_noise_scale(x, y, z, v, device):
    noise_size = torch.rand((1,)).mul(z-y).add(y).int().item()
    noise_size_bi = torch.rand((2,)).mul(v).sub(v/2).int()
    noise = torch.randn((1,3,noise_size+noise_size_bi[0].item(),noise_size+noise_size_bi[1].item())).to(device)
    return F.interpolate(noise, size=(x.shape[-2], x.shape[-1]), mode='bicubic', align_corners=False)

def gaussian_blur_scales(x):
    if x.shape[-1] >= 336:
        return TF.gaussian_blur(x, 5)
    else:
        return TF.gaussian_blur(x, 3)


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
    return T.Compose([
        T.Lambda(lambda x: x - torch.randn_like(x).mul(0.02)),
        T.Pad(25, padding_mode='reflect'),
        #T.RandomHorizontalFlip(),
        #T.ColorJitter(brightness=0.02, contrast=0.01, saturation=0.02, hue=0.01),
        #T.RandomAffine(24, (.1, .1)),
        T.RandomChoice([
            T.Lambda(lambda x: x),
            T.RandomChoice([
                T.GaussianBlur( 3, (gaussian_sigma( 3)*0.75,gaussian_sigma( 3))),
                T.GaussianBlur( 5, (gaussian_sigma( 5)*0.75,gaussian_sigma( 5))),
                T.GaussianBlur( 7, (gaussian_sigma( 7)*0.75,gaussian_sigma( 7)))
            ]),
        ]),
        T.RandomChoice([
            T.Lambda(lambda x: x - random_noise_scale(x,  16,  32, 8, device).mul(0.02)),
            T.Lambda(lambda x: x - random_noise_scale(x,  32,  64, 8, device).mul(0.02)),
            T.Lambda(lambda x: x - random_noise_scale(x,  64, 128, 8, device).mul(0.02)),
            T.Lambda(lambda x: x - random_noise_scale(x, 128, 256, 8, device).mul(0.02)),
        ]),
        T.RandomRotation(15),
        T.Lambda(lambda x: x + torch.randn_like(x).mul(0.02)),
        T.Compose([
            T.Lambda(lambda x: random_size_crop(x, config.width, config.height, 16)),
            T.RandomChoice([
                T.Lambda(lambda x: gaussian_blur_scales(x)),
            ]),
            T.Lambda(lambda x: F.interpolate(x, (224, 224), mode='bicubic', align_corners=False)),
        ]),
        T.Lambda(lambda x: x + torch.randn_like(x).mul(0.02)),
        T.RandomHorizontalFlip(0.125)
        #T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def get_augmentations(config, device):
    return get_augmentations_0(config, device)