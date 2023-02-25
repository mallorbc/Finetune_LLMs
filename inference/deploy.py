import mii
import os
from transformers import AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m","--model_name", type=str, default="EleutherAI/gpt-j-6B")
parser.add_argument("-tp","--tensor_parallel", type=int, default=1)

# model_name = "EleutherAI/gpt-neo-2.7B"

args = parser.parse_args()
model_name = args.model_name
tensor_parallel = args.tensor_parallel

# dtype = "int8"
dtype = "fp16"

skip_model_check = True
deployment_name = "deployment"
max_tokens = 2048

mii_configs = {"tensor_parallel": tensor_parallel, "dtype": dtype,"skip_model_check": skip_model_check, "max_tokens": max_tokens}
mii.deploy(task="text-generation",
           model=model_name,
           deployment_name=deployment_name,
           mii_config=mii_configs)