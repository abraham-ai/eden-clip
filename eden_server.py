import os
import glob
import json
import copy
import torch 
import clip
import PIL
from dotenv import load_dotenv

from core.generate import generate

from eden.block import BaseBlock
from eden.datatypes import Image
from eden.hosting import host_block

eden_block = BaseBlock()
eden_block.max_gpu_mem = 1.0


pretrained_vqgan = {
    'imagenet': {
        'checkpoint': 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1',
        'config': 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1'
    },
    'wikiart': {
        'checkpoint': 'http://mirror.io.community/blob/vqgan/wikiart_16384.ckpt',
        'config': 'http://mirror.io.community/blob/vqgan/wikiart_16384.yaml'
    },
}

def get_models(config):

    import taming.modules.losses
    from taming.models.vqgan import VQModel
    from omegaconf import OmegaConf
    from ml4a.utils import downloads

    print("the config is ")
    print(config)

    # load CLIP
    perceptor, preprocess = clip.load('ViT-B/32', jit=False, device = config['__gpu__'])
    perceptor = perceptor.eval()

    # load VQGAN
    model_name = 'imagenet'
    
    checkpoint = downloads.download_data_file(
        pretrained_vqgan[model_name]['checkpoint'],
        'taming-transformers/{}/checkpoint.ckpt'.format(model_name))

    config_file = downloads.download_data_file(
        pretrained_vqgan[model_name]['config'],
        'taming-transformers/{}/config.yaml'.format(model_name))

    taming_config = OmegaConf.load(config_file)
    model = VQModel(**taming_config.model.params)
    sd = torch.load(checkpoint, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)

    model = model.eval()
    model = model.to(config['__gpu__'])
    #torch.set_grad_enabled(False)

    print("SETUP CLIP ON {}".format(config['__gpu__']))
    print("SETUP TAMING ON {}".format(config['__gpu__']))

    return model, perceptor, preprocess


my_args = {
    'text_inputs': [{
        'text': 'hello world',
        'weight': 10.0
    }],
    'width': 256,
    'height': 256,
    'num_octaves': 3,
    'octave_scale': 2.0,
    'num_iterations': [10, 20, 30],
    'weight_decay': 0.1,
    'learning_rate': 0.1,
    'lr_decay_after': 400,
    'lr_decay_rate': 0.995
}    
@eden_block.run(
    args = my_args,
    progress = True
)
def run(config):

    print("config: \n")
    print(config)
    print(f"gpu for {config['username']}  is ", config['__gpu__'])

    model, perceptor, preprocess = get_models(config = config)

    #model = taming_transformers.model
    model.post_quant_conv = model.post_quant_conv.to(config['__gpu__'])
    model.decoder = model.decoder.to(config['__gpu__'])

    print(config)
    
    try:
        width, height = config['width'], config['height']
        octave_scale, num_octaves = config['octave_scale'], config['num_octaves']

        img = None

        num_total_steps = sum(config['num_iterations'])
        progress = config['__progress__']
        progress_step_size = 1/num_total_steps

        for octave in range(3):
            config_octave = config.copy()

            config_octave['width'] = int(width * (octave_scale ** -(num_octaves-octave-1)))
            config_octave['height'] = int(height * (octave_scale ** -(num_octaves-octave-1)))
            config_octave['num_iterations'] = config['num_iterations'][octave]
            config_octave['lr_decay_after'] = int(config_octave['num_iterations'] * 0.5)

            img = generate(config_octave, perceptor = perceptor, preprocess = preprocess, model = model, device = config['__gpu__'], img = img, progress = progress, progress_step_size = progress_step_size)

        # load_dotenv()
        # RESULTS_DIR = os.environ['RESULTS_DIR']  
        # try:
        #     last_idx = int(sorted(glob.glob(f'{RESULTS_DIR}/*'))[-1].split('/')[-1])
        # except:
        #     last_idx = 0
        # idx = 1 + last_idx
        # output_dir = f'{RESULTS_DIR}/%04d'%idx
        # image_path = '{}/{}'.format(output_dir, 'image.jpg')
        # config_path = '{}/{}'.format(output_dir, 'config.json')        

        # if not os.path.isdir(output_dir):
        #     os.mkdir(output_dir)

        # with open(config_path, 'w') as outfile:
        #     json.dump(config, outfile)

        # PIL.Image.fromarray(img).save(image_path)


    except Exception as e:
        raise Exception(str(e))

    return {
        'creation': Image(img),
    }

host_block(
    eden_block,
    port = 5454,
    max_num_workers = 2
)
