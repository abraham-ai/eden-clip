pip install https://github.com/openai/CLIP/archive/refs/heads/main.zip --no-deps
pip install git+https://github.com/abraham-ai/eden.git

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install omegaconf==2.1.2
pip install ftfy==6.1.1
pip install regex==2022.3.15
pip install pytorch_lightning
pip install imageio
pip install kornia

mkdir -p pretrained/imagenet
wget -nv -O pretrained/imagenet/last.ckpt https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1
wget -nv -O pretrained/imagenet/model.yaml https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1
# mkdir -p pretrained/wikiart
# wget -O pretrained/wikiart/wikiart_16384.ckpt http://mirror.io.community/blob/vqgan/wikiart_16384.ckpt
# wget -O pretrained/wikiart/wikiart_16384.yaml http://mirror.io.community/blob/vqgan/wikiart_16384.yaml

echo "Setup complete!"