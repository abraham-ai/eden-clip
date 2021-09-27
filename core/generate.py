import clip
import torch
import PIL
import imageio
import numpy as np
from tqdm import tqdm

from .pars import Pars
from .easy_dict import EasyDict
from .constants import normalization
from .augmentation import get_augmentations
from .utils import (
    augment, 
    postprocess, 
    make_image,
    transformer_forward_pass
)

def generate(config: dict, perceptor, preprocess, model, augmentations, device, img = None, progress = None, progress_step_size = None):

    assert 'text_inputs' in config or 'image_inputs' in config, 'Error: no text or image inputs'
    
    config = EasyDict(config)
    
    config.text_inputs = config.text_inputs if 'text_inputs' in config else []
    config.image_inputs = config.image_inputs if 'image_inputs' in config else []
    config.width = config.width if 'width' in config else 512
    config.height = config.height if 'height' in config else 512
    config.size = (config.width, config.height)
    config.batch_size = config.batch_size if 'batch_size' in config else 1
    config.learning_rate = config.learning_rate if 'learning_rate' in config else 0.1
    config.lr_decay_after = config.lr_decay_after if 'lr_decay_after' in config else 400
    config.lr_decay_rate = config.lr_decay_rate if 'lr_decay_rate' in config else 0.995
    config.up_noise = config.up_noise if 'up_noise' in config else 0.11
    config.weight_decay = config.weight_decay if 'weight_decay' in config else 0.1
    config.cutn = config.cutn if 'cutn' in config else 32
    config.num_iterations = config.num_iterations if 'num_iterations' in config else 1000

    scaler = 1.0
    dec = config.weight_decay
    up_noise = config.up_noise
    cutn = config.cutn
    lr_decay_after = config.lr_decay_after
    lr_decay_rate = config.lr_decay_rate

    if img is not None:
        img = PIL.Image.fromarray(img)
        img = img.resize((config.width, config.height), PIL.Image.BILINEAR)
        img = np.array(img)

    lats = Pars(img, config, model, device = device).to(device)    
    mapper = [lats.normu]
    optimizer = torch.optim.AdamW([{'params': mapper, 
                                    'lr': config.learning_rate}], 
                                  weight_decay=dec)

    for text_input in config.text_inputs:
        tx = clip.tokenize(text_input['text'])
        tx_embedding = perceptor.encode_text(tx.to(device)).detach().clone()
        text_input['embedding'] = tx_embedding
        
    for image_input in config.image_inputs:
        img_embedding = (torch.nn.functional.interpolate(torch.tensor(imageio.imread(image_input['path'])).unsqueeze(0).permute(0, 3, 1, 2), (224, 224)) / 255).to(device)[:,:3]
        img_embedding = normalization(img_embedding)
        img_embedding = perceptor.encode_image(img_embedding.to(device)).detach().clone()
        image_input['embedding'] = img_embedding
        
    # optimize
    for itt in tqdm(range(config.num_iterations)):

        out = transformer_forward_pass(lats(), model = model, device = device)
        into = augment((out.clip(-1, 1) + 1) / 2, up_noise, scaler, device, cutn, config, augmentations)   
        into = normalization(into)
        iii = perceptor.encode_image(into)

        t_losses = [-t['weight'] * torch.cosine_similarity(t['embedding'], iii, -1)
                    for t in config.text_inputs]
        
        i_losses = [-i['weight'] * torch.cosine_similarity(i['embedding'], iii, -1)
                    for i in config.image_inputs]

        all_losses = t_losses + i_losses

        loss = sum(all_losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update learning rate and weight decay
        if itt > lr_decay_after: 
            for g in optimizer.param_groups:
                g['lr'] *= lr_decay_rate
                g['lr'] = max(g['lr'], .1)
            dec *- lr_decay_rate

        if torch.abs(lats()).max() > 5:
            for g in optimizer.param_groups:
                g['weight_decay'] = dec
        else:
            for g in optimizer.param_groups:
                g['weight_decay'] = 0

        '''
        update progress 
        '''
        if progress_step_size != None and progress != None:
            progress.update(progress_step_size)
                
    # clean up
    for text_input in config.text_inputs:
        text_input['embedding'] = None
    for image_input in config.image_inputs:
        image_input['embedding'] = None

    # final result
    img = np.array(make_image(lats = lats, model = model, device = device))
    return img