FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

SHELL [ "/bin/bash","-c" ]
#Used for GPU setup
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility

RUN apt update -y \
&& apt upgrade -y

RUN apt install wget -y \
&& apt install git -y \ 
&& apt install libaio-dev -y \
&& apt install libaio1 -y 

RUN apt install python3.9 -y \
&& apt install python3-pip -y \
&& apt install python-is-python3 -y

RUN apt install zsh -y 

SHELL [ "/bin/zsh","-c" ]

RUN sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"


RUN pip install --upgrade pip setuptools wheel

RUN pip install ninja datasets pytest

RUN pip install torch torchvision torchaudio

RUN pip install datasets 

RUN pip install git+https://github.com/huggingface/transformers.git
RUN pip install git+https://github.com/huggingface/accelerate.git
RUN pip install git+https://github.com/huggingface/peft.git

RUN pip install sentencepiece

RUN pip install einops

RUN pip install triton

RUN pip install git+https://github.com/microsoft/DeepSpeed.git@v0.14.0

# RUN pip install git+https://github.com/microsoft/DeepSpeed-MII.git


RUN pip install wandb

RUN pip install protobuf==3.20.1

RUN pip install bitsandbytes

RUN pip install scipy

RUN pip install --upgrade torch

RUN pip install trl

RUN pip install ninja packaging

RUN MAX_JOBS=4 pip install flash-attn --no-build-isolation

RUN pip install rich

RUN pip install text-generation

WORKDIR /workspace

CMD ["zsh"]
