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
    sideX, sideY = config.size[0], config.size[1]
    min_side = min(sideX, sideY)
    into = torch.nn.functional.pad(
        into, 
        (min_side//2, min_side//2, min_side//2, min_side//2),
        mode='constant', 
        value=0
    )
    into = augs(into)
    p_s = []
    for ch in range(cutn):
        size = int(torch.normal(1.2, .3, ()).clip(.43, 1.9) * min_side)
        if ch > cutn - 4:
            size = int(min(sideX, sideY)  * 1.4)
        offsetx = torch.randint(0, int(sideX * 2 - size), ())
        offsety = torch.randint(0, int(sideY * 2 - size), ())
        apper = into[:, :, offsetx:offsetx + size, offsety:offsety + size]
        if apper.shape[2] == 0 or apper.shape[3] == 0:
            apper = into
        apper = torch.nn.functional.interpolate(apper, (int(224*scaler), int(224*scaler)), mode='bilinear', align_corners=True)
        p_s.append(apper)
        del apper

    into = torch.cat(p_s, 0)
    into = into + up_noise * torch.rand((into.shape[0], 1, 1, 1)).to(device) * torch.randn_like(into, requires_grad=False)
    del p_s
    return into


def transformer_forward_pass(x, model, device):
    o_i2 = x
    o_i3 = model.post_quant_conv(o_i2)
    i = model.decoder(o_i3)
    return i


# def ascend_txt(lats,config, transformer_model, up_noise , scaler, cutn, augs, perceptor, device):

#     out = transformer_forward_pass(lats(), model = transformer_model, device = device)

#     into = augment((out.clip(-1, 1) + 1) / 2, up_noise, scaler, device, cutn, config, augs)   
#     into = normalization(into)
#     iii = perceptor.encode_image(into)


#     t_losses = [-t['weight'] * torch.cosine_similarity(t['embedding'], iii, -1)
#                 for t in config.text_inputs]
    
#     i_losses = [-i['weight'] * torch.cosine_similarity(i['embedding'], iii, -1)
#                 for i in config.image_inputs]

#     all_losses = t_losses + i_losses
#     return all_losses

def make_image(lats, model, device):
    with torch.no_grad():
        alnot = (transformer_forward_pass(lats(), model = model, device = device).cpu().clip(-1, 1) + 1) / 2
        img = postprocess(alnot.cpu()[0])
    img = PIL.Image.fromarray(img.astype(np.uint8)).convert('RGB')
    return img
