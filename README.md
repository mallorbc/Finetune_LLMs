# Finetune LLMs

## Overview

This repo contains code to fine-tune Large Language Models(LLMs) with a famous quotes dataset. 

The supported methods of Finetuning are DeepSpeed, Lora, or QLora.

Originally, the repo downloaded and converted the model weights for GPTJ when it was not yet added to Huggingface transformer package.  That code can still be seen under the branch ```original_youtube```.

```/quotes_dataset``` contains the dataset properly formatted for fine-tuning. See repo for making this dataset [here](https://github.com/mallorbc/GPT_Neo_quotes_dataset)

```/finetuning_repo``` contains code originally from the repo [here](https://github.com/Xirider/finetune-gpt2xl) that I have modified to work with more models and with more methods.

## Professional Assistance

If are in need of paid professional help, that is available through this [email](mailto:blakecmallory@gmail.com) 

## Old Video Walkthroughs(Don't Use Under Normal Conditions)

See the old video for the original repo code [here](https://www.youtube.com/watch?v=fMgQVQGwnms&ab_channel=Blake) for a video tutorial.

A more updated video for using the Huggingface model can be seen [here](https://www.youtube.com/watch?v=bLMbnHunL_E&t=75s)

Go to the ```original_youtube``` branch is you want to see the code, but I HIGHLY recommend you use more modern methods

## Updated Docker Walkthrough(Use This Under Normal Conditions)

The updated walkthrough uses Nvidia-docker to take the headache out of much of the process.

### Requirements
1. A sufficient Nvidia GPU(typically at least 24GB of VRAM and support for fp16).  If using cloud offerings I recommend A100.  Though it costs more its speed and VRAM make up for it.
2. Use a Linux machine.  I recommend Ubuntu
3. Sufficiently modern version of docker(when in doubt update to latest)
4. Nvidia-docker to allow GPU passthrough to the docker container. See the install guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
5. Make sure you have the latest Nvidia drivers installed. Check out the tool [here](https://www.nvidia.com/download/index.aspx)

#### Cuda Drivers Example

If you have a 64-bit Linux system and need drivers for an A100, you can run a command like this to get set up.

```wget https://us.download.nvidia.com/tesla/515.86.01/NVIDIA-Linux-x86_64-515.86.01.run```

You will then run the downloaded program with sudo.

```chmod 777 NVIDIA-Linux-x86_64-515.86.01.run```

```sudo ./NVIDIA-Linux-x86_64-515.86.01.run```

### Usage

1. First, build the docker image by running ```build_image.sh```.  If you receive an error about not being able to find the docker image, update to a newer Cuda version.  The images are periodically deprecated.  Then open a PR so you can fix this issue for others.  Building the docker image can take many minutes.
2. Run ```run_image.sh```.  This script runs the docker image that was just built and mounts the current directory to ```/workspace``` inside of the docker container.  All GPUs in the system will be passed through.  Additionally, to prevent downloading models each time this container is run, your ```.cache``` will also be passed through.
3. This image can now be used for finetuning a model with GPUs, or for using DeepSpeed inference.  Navigate to another folder for more information
