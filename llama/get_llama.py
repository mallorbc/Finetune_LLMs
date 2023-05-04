from transformers import AutoTokenizer, AutoModelForCausalLM
import os

#13B model
model_name = "TheBloke/vicuna-13B-1.1-HF"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
output_path = "./local_vicuna13B"
os.makedirs(output_path, exist_ok=True)
tokenizer.save_pretrained(output_path)
model.save_pretrained(output_path)

#7B model
model_name = "TheBloke/vicuna-7B-1.1-HF"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
output_path = "./local_vicuna7B"
os.makedirs(output_path, exist_ok=True)
tokenizer.save_pretrained(output_path)
model.save_pretrained(output_path)
