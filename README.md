# eden-clip

[![Docker CI](https://github.com/abraham-ai/eden-clip/actions/workflows/docker-ci.yml/badge.svg)](https://github.com/abraham-ai/eden-clip/actions/workflows/docker-ci.yml)

This is the first generator running on the [Abraham website](https://www.abraham.ai/create).

The active generator is largely adapted from the LatentVisions notebooks series by [@advadnoun](https://twitter.com/advadnoun/), which shows how to combine [CLIP](https://github.com/openai/CLIP) and [VQGAN](https://github.com/CompVis/taming-transformers) to generate images from text. Additional contributions are sourced from [@RiversHaveWings](https://twitter.com/RiversHaveWings) and [@hotgrits](https://twitter.com/torridgristle). Further improvements have been learned from the long tail of people openly experimenting with CLIP online and providing various recommendations on how to structure prompts, choose hyper-parameters, and other insights.

## Hosting from your local machine

In order to set it up and download all dependencies, run the following command in a new `venv`

```
sh setup.sh
```

An then to run the server: 

```
python3 eden_server.py -n 1 -p 5656 -rh localhost -rp 6379
```
- `-n`: number of workers to be run in parallel (defaults to `1`)
- `-p`: port to be exposed (defaults to `5656`)
- `-rh`: redis host where queue metadata/results would be stored (defaults to `localhost`)
- `-rp`: redis port (defaults to `6379`)

## Setting up a client and using it

Copy paste the ngrok URL you got into the snippet below. Then you can run it pretty much from anywhere. 

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
token = run_response['token']
```

Now in order to check the status of your task or obtain the results, you can run: 

```python
resp = c.fetch(token = token)
```
## Running with `nvidia-docker`

> In case you don't have `nvidia-docker`, it can be installed [from here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Building from Dockerfile
```
nvidia-docker build . --file Dockerfile --tag eden-clip
```

Running on `localhost:5656`
```
nvidia-docker run --gpus all -p 5656:5656 --network="host" eden-clip -n 1 -p 5656 -rh localhost -rp 6379
```
