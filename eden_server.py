import os
import glob
import json
import torch 
import clip
import PIL

from core.generate import generate
from ml4a.models import taming_transformers

from eden.block import BaseBlock
from eden.datatypes import Image
from eden.hosting import host_block

RESULTS_DIR =  'abraham_results'  ## '../static/results'

eden_block = BaseBlock()

perceptor, preprocess = None, None
perceptor, preprocess = clip.load('ViT-B/32', jit=False)
perceptor = perceptor.eval()

taming_setup = False
DEVICE = None

# is this necessary?
torch.multiprocessing.set_start_method('spawn', force=True)

#@eden_block.setup
def setup_models(gpu_idx):
    global taming_transformers
    print("SETUP TAMING ON {}".format(gpu_idx))
    taming_transformers.gpu = gpu_idx
    taming_transformers.setup('vqgan')
    taming_setup = True
    print('setup complete')
    
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

@eden_block.run(
    args = my_args
)
def run(config):
    print("the config \n ")
    print(config)

    print(f"gpu for {config['username']}  is ", config['__gpu__'])
    gpu_idx = int(config['__gpu__'].replace("cuda:", ""))
    setup_models(gpu_idx)

    img = generate(config, perceptor =  perceptor, preprocess = preprocess, taming_transformers = taming_transformers, device = config['__gpu__'])

    print(sorted(glob.glob(f'{RESULTS_DIR}/*')), ' sorted thingy')
    try:
        last_idx = int(sorted(glob.glob(f'{RESULTS_DIR}/*'))[-1].split('/')[-1])
    except:
        last_idx = 0
    idx = 1 + last_idx
    output_dir = f'{RESULTS_DIR}/%04d'%idx
    print('last idx', last_idx)
    image_path = '{}/{}'.format(output_dir, 'image.jpg')
    config_path = '{}/{}'.format(output_dir, 'config.json')        
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with open(config_path, 'w') as outfile:
        json.dump(config, outfile)

    PIL.Image.fromarray(img).save(image_path)
    

    return {
        'creation': Image(img)
    }

host_block(
    eden_block,
    port = 5656,
    max_num_workers = 4
)
