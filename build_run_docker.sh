#!/bin/bash
# Build container
sudo docker build . --tag tensorflow-jupyter-gpu:v1.0
# Get interactive terminal in container
sudo docker run --gpus all -it -v `pwd`:`pwd` -w `pwd` tensorflow-jupyter-gpu:v1.0
# Run main.sh
