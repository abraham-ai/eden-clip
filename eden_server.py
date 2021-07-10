import os
import glob
import json
import copy
import torch 
import clip
import PIL

from ml4a.models import taming_transformers
from core.generate import generate
from eden.block import BaseBlock
from eden.datatypes import Image
from eden.hosting import host_block

eden_block = BaseBlock()
eden_block.max_gpu_mem = 1.0

def get_models(config):

    gpu_idx = int(config['__gpu__'].replace("cuda:", ""))
    # setup_models(gpu_idx)

    perceptor, preprocess = clip.load('ViT-B/32', jit=False, device = 'cuda:' + str(gpu_idx))
    perceptor = perceptor.eval()
    print("SETUP CLIP ON cuda:{}".format(gpu_idx))

    taming_transformers.gpu = gpu_idx
    taming_transformers.setup('imagenet')
    print("SETUP TAMING ON cuda:{}".format(gpu_idx))
    print(f'setup complete, transformer on: {id(taming_transformers)}')

    return taming_transformers, perceptor, preprocess

my_args = {
    'prompt': 'hello world',
    'width': 256,
    'height': 256,
    'num_octaves': 3,
    'octave_scale': 2.0,
    'num_iterations': 10,
    'weight_decay': 0.1,
    'learning_rate': 0.1,
    'lr_decay_after': 400,
    'lr_decay_rate': 0.995
}    

@eden_block.run(
    args = my_args
)
def run(config):

    print("the config \n ")
    print(config)
    print(f"gpu for {config['username']}  is ", config['__gpu__'])

    taming_transformers, perceptor, preprocess = get_models(config = config)

    model = taming_transformers.model
    model.post_quant_conv = model.post_quant_conv.to(config['__gpu__'])
    model.decoder = model.decoder.to(config['__gpu__'])

    # from copy import deepcopy

    # taming_transformers.model = deepcopy(taming_transformers.model)

    config['num_iterations'] = [100, 300, 300]
    config['text_inputs'] = [{
        'text': config['prompt'],
        'weight': 10.0
    }]
    

    try:

        width, height = config['width'], config['height']
        octave_scale, num_octaves = config['octave_scale'], config['num_octaves']

        img = None

        for octave in range(3):
            config_octave = config.copy()

            config_octave['width'] = int(width * (octave_scale ** -(num_octaves-octave-1)))
            config_octave['height'] = int(height * (octave_scale ** -(num_octaves-octave-1)))
            config_octave['num_iterations'] = config['num_iterations'][octave]
            
            img = generate(config_octave, perceptor = perceptor, preprocess = preprocess, model = model, device = config['__gpu__'], img = img)

    except Exception as e:

        print('prompt', config['prompt'])
        raise Exception(str(e))

    return {
        'creation': Image(img),
        'config': config
    }

host_block(
    eden_block,
    port = 5656,
    max_num_workers = 4
)
