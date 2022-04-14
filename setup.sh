pip install --upgrade pip
pip install https://github.com/openai/CLIP/archive/refs/heads/main.zip --no-deps
pip install taming-transformers==0.0.1 --no-deps
pip install git+https://github.com/abraham-ai/eden.git
sh download_models.sh
pip install -r requirements.txt
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html # did not work in requirements.txt
echo "Setup complete!"