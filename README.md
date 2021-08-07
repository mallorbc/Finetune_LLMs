# finetune_GPT_NEO

## Overview

This repo contains code to download and convert GPT-J-6B into a Pytorch usable format.  Additionally, it contains code to fine-tune GPT-J-6B with a famous quotes dataset.

```/quotes_dataset``` contains the dataset properly formatted for fine-tuning. See repo for making this dataset [here](https://github.com/mallorbc/GPT_Neo_quotes_dataset)

```/gptneo6B``` contains code to download and convert the model into a format to use with Pytorch

```/finetuning_repo``` contains code orginally from the repo [here](https://github.com/Xirider/finetune-gpt2xl) that I have modified to work with GPT-J-6B

## Walkthrough

See the video [here](google.com) for a video tutorial

1. First create a conda envrionment and enter the environment
2. Run the ```./install_requirements.sh``` script
3. First you want to download the model and get the model into a usable format using the ```/gptneo6B``` folder
4. Then you want to copy the data from ```train.csv``` and ```validation.csv``` from ```/quotes_dataset``` to the ```/finetuning_repo``` folder
5. Run the finetuning code with appropriate flags to fine tune the model

