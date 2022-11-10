#!/bin/bash
dir_to_mount=$(pwd)
docker run -it --ipc=host --gpus all -v $HOME/.cache:/root/.cache -v $dir_to_mount:/workspace gpt
