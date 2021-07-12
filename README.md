# eden-clip
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

3. Install eden from the `token` branch source

```
pip install git+https://github.com/abraham-ai/eden.git --no-deps
```

4. Run the following snippet in a file:

```python
from eden.github import GithubSource

g = GithubSource(url = "https://github.com/Mayukhdeb/eden-clip.git")

if __name__ == '__main__':
    g.build_and_run()
```

5. Hosting the model online would require `ngrok`. Note that we're running on [`port = 5000`](https://github.com/Mayukhdeb/eden-clip/blob/b819465478775118f883eabdc2f46ac665414c4f/server.py#L50) by default.

```
ngrok http 5000
```

6. Copy paste the ngrok URL you got into the snippet below. Then you can run it pretty much from anywhere. 

```python
from eden.client import Client
from eden.datatypes import Image

c = Client(url = 'YOUR_NGROK_OR_LOCALHOST_URL', username= 'eden_clip_client', timeout= 990000)

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

## one eternity later

resp = c.fetch(token = run_response['token'])  ## might return {'status': 'running'}

if resp['status'] == 'complete':
    pil_image = resp['output']['creation']
    pil_image.save('saved_from_server.png')
```
