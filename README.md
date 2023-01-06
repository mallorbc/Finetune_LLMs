# Finetune_GPTNEO_GPTJ6B

## Overview

This repo contains code to fine-tune GPT-J-6B with a famous quotes dataset. Originally, the repo downloaded and converted the model weights when GPTJ was not yet added to huggingface transformer package.  That code can still be seen under the branch ```original_youtube```

```/quotes_dataset``` contains the dataset properly formatted for fine-tuning. See repo for making this dataset [here](https://github.com/mallorbc/GPT_Neo_quotes_dataset)

```/finetuning_repo``` contains code orginally from the repo [here](https://github.com/Xirider/finetune-gpt2xl) that I have modified to work with GPT-J-6B

## Old Video Walkthroughs

See the old video for orignal repo code [here](https://www.youtube.com/watch?v=fMgQVQGwnms&ab_channel=Blake) for a video tutorial.

A more updated video for using the Huggingface model can be seen [here](https://www.youtube.com/watch?v=bLMbnHunL_E&t=75s)

1. First create a conda envrionment and enter the environment
2. Run the ```./install_requirements.sh``` script
3. Then you want to copy the data from ```train.csv``` and ```validation.csv``` from ```/quotes_dataset``` to the ```/finetuning_repo``` folder
4. Run the finetuning code with appropriate flags to fine tune the model. See ```example_run.txt``` inside the ```finetuning_repo```

## Updated Docker Walkthrough

The updated walkthrough uses nvidia docker to take the headache out of much of the process.

### Requirements
1. A sufficient Nvidia GPU(typically at least 24GB of VRAM and support for fp16).  If using cloud offerings I reccomend A100.  Though it costs more its speed and VRAM make up for it.
2. Use a Linux machine.  I reccommend Ubuntu
3. Sufficiently modern version of docker(when in doubt update to latest)
4. nvidia-docker to allow GPU passthrough the the docker container. See install guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
5. Make sure you have the lastest nvidia drivers installed. Check out the tool [here](https://www.nvidia.com/download/index.aspx)

#### Cuda Drivers Example

If you have a 64 bit Linux system, and need drivers for an A100, you can run a command like this to get setup.

```wget https://us.download.nvidia.com/tesla/515.86.01/NVIDIA-Linux-x86_64-515.86.01.run```

You will then run the downloaded program with sudo.

```chmod 777 NVIDIA-Linux-x86_64-515.86.01.run```

```sudo ./NVIDIA-Linux-x86_64-515.86.01.run```

### Usage

1. First, build the docker image by running ```build_image.sh```.  If you recieve an error about not being able to find the docker image, update to a newer cuda version.  The images are periodically depreacated.  Then open a PR so you can fix this issue for others.  Building the docker image can take many minutes.
2. Run ```run_image.sh```.  This script runs the docker image that was just built and mounts the current directory to ```/workspace``` inside of the docker container.  All GPUs in the system will be passed through.  Additionally, to prevent downloading models each time this container is ran, your ```.cache``` will also be passed through.
3. This image can now be used for finetuning a model with GPUs, or for using DeepSpeed inference.  Navigate to another folder for more information
