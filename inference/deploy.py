import mii
import os
from transformers import AutoModelForCausalLM
# model_name = "EleutherAI/gpt-neo-2.7B"
model_name = "EleutherAI/gpt-j-6B"

# model_name = AutoModelForCausalLM.from_pretrained(model_name)

# dtype = "int8"
dtype = "fp16"

skip_model_check = True

mii_configs = {"tensor_parallel": 1, "dtype": dtype,"skip_model_check": skip_model_check}
mii.deploy(task="text-generation",
           model=model_name,
           deployment_name="deployment",
           mii_config=mii_configs)