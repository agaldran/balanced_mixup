#!/usr/bin/env bash

mkdir data
wget https://datasets.simula.no/hyper-kvasir/hyper-kvasir-labeled-images.zip
unzip hyper-kvasir-labeled-images.zip -d data/
rm hyper-kvasir-labeled-images.zip

python prepare_endo_data.py

rm -r data/labeled-images/
