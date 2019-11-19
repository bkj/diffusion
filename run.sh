#!/bin/bash

# run.sh

# --
# Setup env

conda create -n diffusion_env python==3.7 pip -y
conda activate diffusion_env

# --
# Install dependencies

conda install -y pandas
conda install -y -c pytorch faiss-cpu==1.5.3
conda install -y joblib==0.13.2
conda install -y tqdm==4.35.0
conda install -y h5py
conda install -y scipy==1.3.1
conda install -y scikit-learn==0.21.3
conda install -y tqdm

# --
# Download data

mkdir -p data

wget http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/oxford5k/gnd_oxford5k.pkl \
    -O data/gnd_oxford5k.pkl

wget http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/paris6k/gnd_paris6k.pkl \
    -O data/gnd_paris6k.pkl

cd data
ln -s gnd_oxford5k.pkl gnd_oxford105k.pkl
ln -s gnd_paris6k.pkl gnd_paris106k.pkl
cd ..

for dataset in oxford5k paris6k; do
    for feature in siamac resnet; do
        wget ftp://ftp.irisa.fr/local/texmex/corpus/diffusion/data/${dataset}_${feature}.mat \
            -O data/${dataset}_${feature}.mat
    done
done

# --
# Convert matlab to python

python mat2npy.py --dataset_name oxford5k --feature_type resnet --mat_dir data
python mat2npy.py --dataset_name paris6k --feature_type resnet --mat_dir data

# --
# Run

python rank.py                                                \
    --query_path       data/query/oxford5k_resnet_glob.npy    \
    --gallery_path     data/gallery/oxford5k_resnet_glob.npy  \
    --gnd_path         data/gnd_oxford5k.pkl                  \
    --dataset_name     oxford5k                               \
    --n_trunc          1000


python rank.py                                               \
    --query_path       data/query/paris6k_resnet_glob.npy    \
    --gallery_path     data/gallery/paris6k_resnet_glob.npy  \
    --gnd_path         data/gnd_paris6k.pkl                  \
    --dataset_name     paris6k                               \
    --n_trunc          1000

