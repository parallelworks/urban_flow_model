#!/bin/bash
sudo docker build . --tag tensorflow-jupyter-gpu:v1.0
sudo docker run -it -v `pwd`:`pwd` -w `pwd` tensorflow-jupyter-gpu:v1.0 main.sh
