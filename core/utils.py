import torch 
import numpy as np 
import PIL
from .constants import normalization

def lerp(low, high, val):
    res = low * (1.0 - val) + high * val
    return res

    
def slerp(low, high, val):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    epsilon = 1e-7
    omega = (low_norm*high_norm).sum(1)
    omega = torch.acos(torch.clamp(omega, -1 + epsilon, 1 - epsilon))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def postprocess(img, pre_scaled=True):
    img = np.array(img)[:,:,:]
    img = np.transpose(img, (1, 2, 0))
    if not pre_scaled:
        img = scale(img, 48*4, 32*4)
    img = (255.0 * img).astype(np.uint8)
    return img

def augment(into, up_noise, scaler, device, cutn=32, config = None, augs = None):
    # global up_noise, scaler
    sideX, sideY, channels = config.size[0], config.size[1], 3
    into = torch.nn.functional.pad(into, (sideX//2, sideX//2, sideX//2, sideX//2), mode='constant', value=0)
    into = augs(into)
    p_s = []
    for ch in range(cutn):
        # size = torch.randint(int(.5*sideX), int(1.9*sideX), ())
        size = int(torch.normal(1.2, .3, ()).clip(.43, 1.9) * sideX)
        if ch > cutn - 4:
            size = int(sideX*1.4)
        offsetx = torch.randint(0, int(sideX*2 - size), ())
        offsety = torch.randint(0, int(sideX*2 - size), ())
        apper = into[:, :, offsetx:offsetx + size, offsety:offsety + size]
        apper = torch.nn.functional.interpolate(apper, (int(224*scaler), int(224*scaler)), mode='bilinear', align_corners=True)
        p_s.append(apper)
    into = torch.cat(p_s, 0)
    into = into + up_noise*torch.rand((into.shape[0], 1, 1, 1)).to(device)*torch.randn_like(into, requires_grad=False)
    return into

def model(x, taming_transformers):
    o_i2 = x
    o_i3 = taming_transformers.model.post_quant_conv(o_i2)
    i = taming_transformers.model.decoder(o_i3)
    return i


def ascend_txt(lats,config, taming_transformers, up_noise , scaler, cutn, augs, perceptor, device):
    # global lats
    out = model(lats(), taming_transformers = taming_transformers)
    into = augment((out.clip(-1, 1) + 1) / 2, up_noise , scaler, cutn, config, augs, device = device)
    into = normalization(into)
    iii = perceptor.encode_image(into)    


    t_losses = [-t['weight'] * torch.cosine_similarity(t['embedding'], iii, -1)
                for t in config.text_inputs]
    
    i_losses = [-i['weight'] * torch.cosine_similarity(i['embedding'], iii, -1)
                for i in config.image_inputs]

    all_losses = t_losses + i_losses
    return all_losses

def make_image(lats, taming_transformers):
    with torch.no_grad():
        alnot = (model(lats(), taming_transformers = taming_transformers).cpu().clip(-1, 1) + 1) / 2
        img = postprocess(alnot.cpu()[0])
    img = PIL.Image.fromarray(img.astype(np.uint8)).convert('RGB')
    return img
