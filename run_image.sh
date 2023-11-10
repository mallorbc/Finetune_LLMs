#!/bin/bash
dir_to_mount=$(pwd)
docker run -it --ipc=host --gpus all -v $HOME/.cache:/root/.cache -v $HOME/.zshrc:/root/.zshrc -v $HOME/.zsh_history:/root/.zsh_history -v $dir_to_mount:/workspace gpt
