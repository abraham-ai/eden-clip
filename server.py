import torch
import imageio
import numpy as np
from tqdm import tqdm

import clip

from core.pars import Pars
from core.easy_dict import EasyDict
from core.constants import normalization
from core.augmentation import Augmentations
from core.utils import (
    augment, 
    postprocess, 
    model, 
    ascend_txt, 
    make_image
)

from ml4a.models import taming_transformers

from eden.block import BaseBlock
from eden.datatypes import Image
from eden.hosting import host_block 


eden_block = BaseBlock()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# is this necessary?
torch.multiprocessing.set_start_method('spawn', force=True)


def generate(inputs, img=None, callback=None , **kwargs):
    global perceptor
    global up_noise, scaler, dec, cutn
    global lats

    current_config = {
        'text_inputs': [{
            'text': inputs['prompt'], 
            'weight': 10.0
        }],
        'size': (inputs['width'], inputs['height']),
        'num_iterations': inputs['iters'],
        'weight_decay': inputs['weight_decay'],
        'learning_rate': inputs['learning_rate'],
        'lr_decay_after': inputs['learning_rate'],
        'lr_decay_rate': inputs['lr_decay_rate'],
        'batch_size': 1,
        'cutn': 24,
    }


    assert 'text_inputs' in current_config or 'image_inputs' in config, 'Error: no text or image inputs'
    

    config = EasyDict(current_config)
    
    config.text_inputs = config.text_inputs if 'text_inputs' in config else []
    config.image_inputs = config.image_inputs if 'image_inputs' in config else []
    config.size = config.size if 'size' in config else (512, 512)
    config.batch_size = config.batch_size if 'batch_size' in config else 1
    config.learning_rate = config.learning_rate if 'learning_rate' in config else 0.3
    config.lr_decay_after = config.lr_decay_after if 'lr_decay_after' in config else 400
    config.lr_decay_rate = config.lr_decay_rate if 'lr_decay_rate' in config else 0.995
    config.up_noise = config.up_noise if 'up_noise' in config else 0.11
    config.weight_decay = config.weight_decay if 'weight_decay' in config else 0.1
    config.cutn = config.cutn if 'cutn' in config else 32
    config.num_iterations = config.num_iterations if 'num_iterations' in config else 1000

    print(config)   
    
    
    


    scaler = 1.0
    dec = config.weight_decay
    up_noise = config.up_noise
    cutn = config.cutn
    lr_decay_after = config.lr_decay_after
    lr_decay_rate = config.lr_decay_rate
    save_frame = 0
    
    
    torch.cuda.empty_cache()
    
    
    lats = Pars(img, config).cuda()    
    mapper = [lats.normu]
    optimizer = torch.optim.AdamW([{'params': mapper, 
                                    'lr': config.learning_rate}], 
                                  weight_decay=dec)
    
    for text_input in config.text_inputs:
        tx = clip.tokenize(text_input['text'])
        tx_embedding = perceptor.encode_text(tx.cuda()).detach().clone()
        text_input['embedding'] = tx_embedding
        
    for image_input in config.image_inputs:
        img_embedding = (torch.nn.functional.interpolate(torch.tensor(imageio.imread(image_input['path'])).unsqueeze(0).permute(0, 3, 1, 2), (224, 224)) / 255).cuda()[:,:3]
        img_embedding = normalization(img_embedding)
        img_embedding = perceptor.encode_image(img_embedding.cuda()).detach().clone()
        image_input['embedding'] = img_embedding
        
    # optimize
    for itt in tqdm(range(config.num_iterations)):
        
        loss1 = ascend_txt(
            lats= lats, 
            config = config, 
            taming_transformers = taming_transformers, 
            up_noise = up_noise, 
            scaler = scaler, 
            cutn = cutn, 
            augs = Augmentations,
            perceptor = perceptor
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
                

    img = np.array(make_image(lats = lats, taming_transformers = taming_transformers))
    return {
        'creation': Image(img)
    }

my_args = {
    'prompt': 'hello world',
    'width': 256,
    'height': 256,
    'iters': 10,
    'weight_decay': 0.1,
    'learning_rate': 0.1,
    'lr_decay_after': 400,
    'lr_decay_rate': 0.995
}


perceptor, preprocess = None, None
@eden_block.setup
def setup_models():
    global perceptor, preprocess 
    perceptor, preprocess = clip.load('ViT-B/32', jit=False)
    perceptor = perceptor.eval()
    taming_transformers.setup('vqgan')
    


@eden_block.run(
    args = my_args
)
def run(config):
    img = generate(config, None)
    return img

host_block(
    eden_block,
    port = 5000
)