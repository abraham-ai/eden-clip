sudo apt install python3.8-venv
python3 -m venv env-eden-clip
source env-eden-clip/bin/activate
pip install --upgrade pip
pip install https://github.com/openai/CLIP/archive/refs/heads/main>
pip install git+https://github.com/abraham-ai/eden.git
sh download_models.sh
pip install -r requirements.txt
echo "Setup complete!"
