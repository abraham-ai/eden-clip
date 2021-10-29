pip install --upgrade pip
pip install https://github.com/openai/CLIP/archive/refs/heads/main.zip --no-deps
pip install taming-transformers==0.0.1 --no-deps
pip install git+https://github.com/abraham-ai/eden.git
sh download_models.sh
pip install -r requirements.txt
echo "Setup complete!"