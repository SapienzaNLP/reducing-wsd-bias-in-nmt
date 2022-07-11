#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda create -n mt-wsd python=3.8 -y
conda activate mt-wsd

read -p "Enter cuda version (e.g. 10.1 or none to avoid installing cuda support - 11.0 if using a 3090): " cuda_version
if [ $cuda_version == "none" ]; then
  conda install -y pytorch cpuonly -c pytorch
else
  conda install -y pytorch cudatoolkit=$cuda_version -c pytorch
fi

pip install -r requirements.txt