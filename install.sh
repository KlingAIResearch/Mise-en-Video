#!/usr/bin/env bash

set -e

# Run after:
#   conda create -n mev python=3.10 -y
#   conda activate mev

conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -y -c conda-forge suitesparse

python -m pip install torch==2.5.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install pytorch-lightning
python -m pip install \
  "git+https://github.com/facebookresearch/detectron2.git@a59f05630a8f205756064244bf5beb8661f96180"
python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@stable"
python -m pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.5.1+cu118.html

python -m pip install --no-build-isolation \
  chumpy@git+https://github.com/mattloper/chumpy

python -m pip install \
  einops \
  evo \
  gdown \
  google-genai \
  huggingface_hub \
  "imageio[ffmpeg]" \
  loguru \
  numpy==2.2.6 \
  open3d \
  opencv-python \
  openpyxl \
  pillow \
  plyfile \
  pulp \
  pycocotools \
  pyrender \
  pytorch-minimize \
  safetensors \
  scikit-image \
  scipy \
  smplx \
  supervision \
  timm==0.6.7 \
  trimesh
