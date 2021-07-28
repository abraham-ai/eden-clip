# eden-clip

This is the first generator running on the [Abraham website](https://www.abraham.ai/create).

The active generator is largely adapted from the LatentVisions notebooks series by [@advadnoun](https://twitter.com/advadnoun/), which shows how to combine [CLIP](https://github.com/openai/CLIP) and [VQGAN](https://github.com/CompVis/taming-transformers) to generate images from text. Additional contributions are sourced from [@RiversHaveWings](https://twitter.com/RiversHaveWings) and [@hotgrits](https://twitter.com/torridgristle). Further improvements have been learned from the long tail of people openly experimenting with CLIP online and providing various recommendations on how to structure prompts, choose hyper-parameters, and other insights.

# how to run

hosting openAI CLIP's text to image pipeline with [eden](https://github.com/abraham-ai/eden)

Running locally through eden (no need to clone the repo):

1. (Highly recommended but optional) Make a `venv`

```
python3 -m venv env-eden
```

2. Activate the `venv`

```
source env-eden/bin/activate
```

3. Install Redis.

```
sudo apt-get install redis-server
```

and [configure it properly](https://www.digitalocean.com/community/tutorials/how-to-install-and-secure-redis-on-ubuntu-20-04).

4. Download pretrained models for [taming-transformers](https://github.com/CompVis/taming-transformers).

```
sh download_models.sh
```

4. Install eden from the `token` branch source

```
pip install git+https://github.com/abraham-ai/eden.git --no-deps
```

5. Run the following snippet in a file:

```python
from eden.github import GithubSource

g = GithubSource(url = "https://github.com/abraham-ai/eden-clip.git")

if __name__ == '__main__':
    g.build_and_run()
```

6. Hosting the model online would require `ngrok`. Note that we're running on [`port = 5454`](https://github.com/abraham-ai/eden-clip/blob/b819465478775118f883eabdc2f46ac665414c4f/server.py#L50) by default.

```
ngrok http 5454
```

7. Copy paste the ngrok URL you got into the snippet below. Then you can run it pretty much from anywhere. 

```python
from eden.client import Client
from eden.datatypes import Image

c = Client(url = 'YOUR_NGROK_OR_LOCALHOST_URL', username= 'eden_clip_client', timeout= 990000)

config = {
    'model_name': 'imagenet',
    'text_inputs': [
            {
        'text': 'blue',
        'weight': 10.0,
        },
            {
        'text': 'mushroom',
        'weight': 20.0,
        },
    ],
    'width': 256,
    'height': 256,
    'num_octaves': 3,
    'octave_scale': 2.0,
    'num_iterations': [20, 50, 100],
    'weight_decay': 0.1,
    'learning_rate': 0.1,
    'lr_decay_after': 400,
    'lr_decay_rate': 0.995
}   

run_response = c.run(config)

## one eternity later

resp = c.await_results(token = run_response['token'], show_progress = True)  

if resp['status'] == 'complete':
    pil_image = resp['output']['creation']
    pil_image.save('saved_from_server.png')
```
