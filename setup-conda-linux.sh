#!/bin/bash -i

conda env create pmaps
conda activate pmaps
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install -c conda-forge opencv
pip install cvpubsubs matplotlib tox-conda
