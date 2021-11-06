from eden.client import Client

c = Client(url = 'url_to_host', username= 'eden_clip_client')

config = {
  "model_name": "imagenet",
  "clip_model": "ViT-B/32",
  "text_input": "Garden of Eden; Beautiful",
  "width": 1024,
  "height": 256,
  "num_octaves": 3,
  "octave_scale": 2,
  "num_iterations": [
    200,
    200,
    100
  ]
}  

run_response = c.run(config)
token = run_response['token']

output = c.await_results(token = token, fetch_interval = 1, show_progress = False)
output['output']['creation'].save(f'{token}.png')