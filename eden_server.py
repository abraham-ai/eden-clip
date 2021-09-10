import os
import glob
import json
import time
import random
import copy
import torch 
import PIL
from omegaconf import OmegaConf

import clip
import taming.modules.losses
from taming.models.vqgan import VQModel

from core.prompt import get_permuted_prompts
from core.generate import generate
from core.augmentation import get_augmentations

from eden.block import BaseBlock
from eden.datatypes import Image
from eden.hosting import host_block

eden_block = BaseBlock()


def get_models(config):
    print(config)

    # load CLIP
    clip_model = config['clip_model']
    assert clip_model in ['ViT-B/32', 'ViT-B/16'], \
        'No CLIP model named {}. Available models are: ViT-B/32, ViT-B/16'
    perceptor, preprocess = clip.load(clip_model, jit=False, device = config['__gpu__'])
    perceptor = perceptor.eval()

    # load VQGAN
    model_name = config['model_name']
    assert model_name in ['imagenet', 'wikiart'], \
        'No model named {}. Available models are: imagenet, wikiart'.format(model_name)
    if model_name == 'imagenet':
        checkpoint = 'pretrained/imagenet/last.ckpt'
        config_file = 'pretrained/imagenet/model.yaml'
    elif model_name == 'wikiart':
        checkpoint = 'pretrained/wikiart/wikiart_16384.ckpt'
        config_file = 'pretrained/wikiart/wikiart_16384.yaml'
    taming_config = OmegaConf.load(config_file)
    model = VQModel(**taming_config.model.params)
    sd = torch.load(checkpoint, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)

    model = model.eval()
    model = model.to(config['__gpu__'])
    #torch.set_grad_enabled(False)

    print("Setup CLIP on {}".format(config['__gpu__']))
    print("Setup VQGAN on {}".format(config['__gpu__']))

    return model, perceptor, preprocess


my_args = {
    'model_name': 'imagenet',
    'clip_model': 'ViT-B/32',
    'text_input': 'hello world',
    'width': 256,
    'height': 256,
    'num_octaves': 3,
    'octave_scale': 2.0,
    'num_iterations': [100, 200, 300]
}    
@eden_block.run(
    args = my_args,
    progress = True
)
def run(config):

    config['weight_decay'] = 0.05 + 0.1 * random.random()
    config['learning_rate'] = 0.05 + 0.05 * random.random()
    config['lr_decay_rate'] = 0.96 + 0.039 * random.random()
    config['cutn'] = [48, 36, 30]
    config['batch_size'] = 1
    config['text_inputs'] = get_permuted_prompts(config['text_input'], 2)
    print("config: \n", config)

    model, perceptor, preprocess = get_models(config = config)
    augmentations = get_augmentations(config, config['__gpu__'])

    #model = taming_transformers.model
    model.post_quant_conv = model.post_quant_conv.to(config['__gpu__'])
    model.decoder = model.decoder.to(config['__gpu__'])

    try:
        width, height = config['width'], config['height']
        octave_scale, num_octaves = config['octave_scale'], config['num_octaves']
        progress = config['__progress__']

        '''
        This block uses a heuristic to estimate the progress_step_size at each octave
        '''
        #total_progress = [n*(config['octave_scale']**(2*i)) for i, n in enumerate(config['num_iterations'])]
        iter_per_sec = [4.962, 3.826, 1.964]  # determined empirically for sizes 128, 256, 512
        total_progress = [n/i for i, n in zip(iter_per_sec, config['num_iterations'])]
        progress_step_sizes = [t/(n * sum(total_progress)) for n, t in zip(config['num_iterations'], total_progress)]

        img = None

        for octave in range(config['num_octaves']):
            config_octave = config.copy()

            config_octave['width'] = int(width * (octave_scale ** -(num_octaves-octave-1)))
            config_octave['height'] = int(height * (octave_scale ** -(num_octaves-octave-1)))
            config_octave['num_iterations'] = config['num_iterations'][octave]
            config_octave['cutn'] = config['cutn'][octave]
            config_octave['lr_decay_after'] = int(config_octave['num_iterations'] * 0.5)

            progress_step_size = progress_step_sizes[octave]

            t0 = time.time()

            img = generate(
                config_octave, 
                perceptor = perceptor, 
                preprocess = preprocess, 
                model = model, 
                augmentations = augmentations, 
                device = config['__gpu__'], 
                img = img,
                progress = progress, 
                progress_step_size = progress_step_size
            )

            t1 = time.time()

            print('octave {}, {}x{}, {} iterations, {} = {} iters/sec'.format(octave, config_octave['width'], config_octave['height'], config_octave['num_iterations'], t1-t0, float(config_octave['num_iterations'])/(t1-t0)))

    except Exception as e:
        raise Exception(str(e))

    return {
        'creation': Image(img),
    }

host_block(
    block = eden_block,
    port = 5656,
    max_num_workers = 2,
    redis_port = 6379,
    exclude_gpu_ids = []
)