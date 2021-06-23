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

3. Install eden

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


