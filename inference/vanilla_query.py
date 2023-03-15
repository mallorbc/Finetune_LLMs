from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m","--model_name", type=str, default="decapoda-research/llama-7b-hf")
args = parser.parse_args()

model_name = args.model_name

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16).cuda()

prompt = "love:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
output = model.generate(input_ids, max_length=100,min_length=10,do_sample=True,)
print(tokenizer.decode(output[0], skip_special_tokens=True))
