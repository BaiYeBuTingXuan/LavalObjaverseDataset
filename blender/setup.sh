conda create -env laval-obajverse-dataset python=3.11
conda activate laval-objaverse-dataset
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip3 install tyro wandb objaverse kornia pillow tqdm opencv-python