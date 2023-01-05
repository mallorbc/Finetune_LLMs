# Finetune_GPTNEO_GPTJ6B

## Overview

This repo contains code to fine-tune GPT-J-6B with a famous quotes dataset. Originally, the repo downloaded and converted the model weights when GPTJ was not yet added to huggingface transformer package.  That code can still be seen under the branch ```original_youtube```

```/quotes_dataset``` contains the dataset properly formatted for fine-tuning. See repo for making this dataset [here](https://github.com/mallorbc/GPT_Neo_quotes_dataset)

```/finetuning_repo``` contains code orginally from the repo [here](https://github.com/Xirider/finetune-gpt2xl) that I have modified to work with GPT-J-6B

## Old Video Walkthroughs

See the video for orignal repo code [here](https://www.youtube.com/watch?v=fMgQVQGwnms&ab_channel=Blake) for a video tutorial.

A more updated video for using the Huggingface model can be seen[here](https://www.youtube.com/watch?v=bLMbnHunL_E&t=75s)

1. First create a conda envrionment and enter the environment
2. Run the ```./install_requirements.sh``` script
3. Then you want to copy the data from ```train.csv``` and ```validation.csv``` from ```/quotes_dataset``` to the ```/finetuning_repo``` folder
4. Run the finetuning code with appropriate flags to fine tune the model. See ```example_run.txt```

## Updated Docker Walkthrough

The updated walkthrough uses nvidia docker to take the headache out of much of the process.

### Requirements
1. A sufficient Nvidia GPU(typically at least 24GB of VRAM and support for fp16).  If using cloud offerings I reccomend A100.  Though it costs more its speed and VRAM make up for it.
2. Use a Linux machine.  I reccommend Ubuntu
3. Sufficiently modern version of docker(when in doubt update to latest)
4. nvidia-docker to allow GPU passthrough the the docker container. See install guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
5. Make sure you have the lastest nvidia drivers installed. Check out the tool [here](https://www.nvidia.com/download/index.aspx)
