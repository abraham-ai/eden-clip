import clip
import torch
import imageio
import numpy as np
from tqdm import tqdm

from .pars import Pars
from .easy_dict import EasyDict
from .constants import normalization
from .augmentation import Augmentations
from .utils import (
    augment, 
    postprocess, 
    ascend_txt, 
    make_image
)

def generate(config, perceptor, preprocess, model, device, img = None):

    assert 'text_inputs' in config or 'image_inputs' in config, 'Error: no text or image inputs'
    
    config = EasyDict(config)
    
    config.text_inputs = config.text_inputs if 'text_inputs' in config else []
    config.image_inputs = config.image_inputs if 'image_inputs' in config else []
    config.width = config.width if 'width' in config else 512
    config.height = config.height if 'height' in config else 512
    config.size = (config.width, config.height)
    config.batch_size = config.batch_size if 'batch_size' in config else 1
    config.learning_rate = config.learning_rate if 'learning_rate' in config else 0.3
    config.lr_decay_after = config.lr_decay_after if 'lr_decay_after' in config else 400
    config.lr_decay_rate = config.lr_decay_rate if 'lr_decay_rate' in config else 0.995
    config.up_noise = config.up_noise if 'up_noise' in config else 0.11
    config.weight_decay = config.weight_decay if 'weight_decay' in config else 0.1
    config.cutn = 2 #config.cutn if 'cutn' in config else 32
    config.num_iterations = config.num_iterations if 'num_iterations' in config else 1000


    scaler = 1.0
    dec = config.weight_decay
    up_noise = config.up_noise
    cutn = config.cutn
    lr_decay_after = config.lr_decay_after
    lr_decay_rate = config.lr_decay_rate
    save_frame = 0
        
    lats = Pars(img, config, model, device = device).to(device)    
    mapper = [lats.normu]
    optimizer = torch.optim.AdamW([{'params': mapper, 
                                    'lr': config.learning_rate}], 
                                  weight_decay=dec)
    
    for text_input in config.text_inputs:
        tx = clip.tokenize(text_input['text'])

        """
        messing up GPU ids here
        """
        tx_embedding = perceptor.encode_text(tx.to(device)).detach().clone()
        text_input['embedding'] = tx_embedding
        
    for image_input in config.image_inputs:
        img_embedding = (torch.nn.functional.interpolate(torch.tensor(imageio.imread(image_input['path'])).unsqueeze(0).permute(0, 3, 1, 2), (224, 224)) / 255).to(device)[:,:3]
        img_embedding = normalization(img_embedding)
        img_embedding = perceptor.encode_image(img_embedding.to(device)).detach().clone()
        image_input['embedding'] = img_embedding
        
    # optimize
    for itt in tqdm(range(config.num_iterations)):
        
        loss1 = ascend_txt(
            lats = lats, 
            config = config, 
            transformer_model = model, 
            up_noise = up_noise, 
            scaler = scaler, 
            cutn = cutn, 
            augs = Augmentations,
            perceptor = perceptor,
            device = device
        )

        loss = sum(loss1)
        loss = loss.mean()
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
                

    img = np.array(make_image(lats = lats, model = model, device = device))
    return img
