#!/bin/bash

# run-ts.sh

conda activate diffusion_env

# --
# Download adta

mkdir -p data/ucr
wget 'https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/UCRArchive_2018.zip' \
    -O data/ucr/UCRArchive_2018.zip

cd data/ucr
unzip UCRArchive_2018.zip
# pwd: someone
mv UCRArchive_2018/* ./
rm -r UCRArchive_2018 UCRArchive_2018.zip
cd ../../

# --
# Run

python ts-classifier.py