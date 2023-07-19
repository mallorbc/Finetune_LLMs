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

RUN pip install --upgrade pip setuptools wheel

RUN pip install ninja

RUN pip install torch torchvision torchaudio

RUN pip install datasets \ 
&& pip install transformers \ 
&& pip install accelerate

# RUN pip install git+https://github.com/huggingface/transformers.git
# RUN pip install git+https://github.com/huggingface/accelerate.git
RUN pip install git+https://github.com/huggingface/peft.git

RUN pip install sentencepiece

RUN pip install einops

RUN pip install triton==1.0.0

RUN pip install git+https://github.com/microsoft/DeepSpeed.git@v0.9.3

RUN pip install git+https://github.com/microsoft/DeepSpeed-MII.git

RUN pip install wandb

RUN pip install protobuf==3.20.*

WORKDIR /workspace

CMD ["bash"]
