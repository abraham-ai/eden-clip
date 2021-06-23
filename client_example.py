from eden.client import Client
from eden.datatypes import Image

c = Client(url = 'http://dc8f1a4c31c1.ngrok.io', username= 'eden_clip_client',timeout= 990000)

setup_response = c.setup()

config = {
    'prompt': 'abraham',
    'width': 256,
    'height': 256,
    'iters': 500,
    'weight_decay': 0.1,
    'learning_rate': 0.1,
    'lr_decay_after': 400,
    'lr_decay_rate': 0.995
}
run_response = c.run(config)

pil_image = run_response['output']['creation']
pil_image.save('saved_from_server.png')