import torch
from peft import PeftModel
import os
from transformers import AutoModelForCausalLM,AutoTokenizer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bm","--base_model", type=str, required=True,help="Give the name of the model as it appears on the HuggingFace Hub")
    parser.add_argument("-lm","--lora_model", type=str, required=True,help="Give the path to the Lora model")
    parser.add_argument("-o","--output", type=str, default="merged_model",help="Give the path to the output folder")
    
    args = parser.parse_args()
    args.output = os.path.realpath(args.output)

    BASE_MODEL = args.base_model
    LORA_WEIGHTS = os.path.realpath(args.lora_model)


    os.makedirs("offload", exist_ok=True)


    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        offload_folder="offload", 
        trust_remote_code=True,
    )
        
    model = PeftModel.from_pretrained(
        model, 
        LORA_WEIGHTS, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        offload_folder="offload", 

    )

    os.makedirs(args.output, exist_ok=True)
    
    model = model.merge_and_unload()
    model.save_pretrained(args.output)

    tokenizer = AutoTokenizer.from_pretrained(LORA_WEIGHTS)
    tokenizer.save_pretrained(args.output)