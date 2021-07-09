import torch 
import clip

from core.generate import generate
from ml4a.models import taming_transformers

from eden.block import BaseBlock
from eden.datatypes import Image
from eden.hosting import host_block

eden_block = BaseBlock()

perceptor, preprocess = None, None
perceptor, preprocess = clip.load('ViT-B/32', jit=False)
perceptor = perceptor.eval()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# is this necessary?
torch.multiprocessing.set_start_method('spawn', force=True)

taming_transformers.setup('vqgan')
print('setup complete')

@eden_block.setup
def setup_models():
    pass
    
    
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
    img = generate(config, perceptor =  perceptor, preprocess = preprocess, taming_transformers = taming_transformers, device = config['__gpu__'])
    return {
        'creation': Image(img)
    }

host_block(
    eden_block,
    port = 5000
)