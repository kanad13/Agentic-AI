pwd
git config --global user.email 'kunalpathak13@gmail.com'
pip3 install -r ./.devcontainer/requirements.txt
python ./.devcontainer/test.py

# python3 -m venv "langchain_agent_env"
# source "langchain_agent_env"/bin/activate
# pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
# pip install -r .devcontainer/requirements.txt
# pip list
# pip freeze > requirements.txt
# sudo pip uninstall -r requirements.txt -y
# gradio misc/present/gradio/test-gradio.py

#!pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Install pandas and other packages
#!pip3 install -r requirements.txt

# Verify the installation by importing the libraries
#import torch
#import torchvision
#import torchaudio

#print("PyTorch version:", torch.__version__)
#print("Torchvision version:", torchvision.__version__)
#print("Torchaudio version:", torchaudio.__version__)
