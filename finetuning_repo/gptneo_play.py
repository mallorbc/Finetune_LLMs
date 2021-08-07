# credit to Niels Rogge - https://github.com/huggingface/transformers/issues/10704

from transformers import GPTNeoForCausalLM, AutoTokenizer,GPT2Tokenizer
import numpy as np
import torch
import torch.nn as nn
import os
import re
import argparse
clear = lambda: os.system('clear')

def fix_text(text):
    if text.find("&amp") != -1:
        text = text.replace("&amp","&")
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser("tool for runiing a pretrained gpt neo")
    parser.add_argument("-m","--model",default="finetuned/checkpoint-450",type=str)
    parser.add_argument("-min","--min_output",default=10,type=int)
    parser.add_argument("-max","--max_output",default=300,type=int)
    parser.add_argument("-temp","--temp_output",default=0.9,type=float)
    parser.add_argument("-n","--num_output",default=1,type=int)
    parser.add_argument("-sixB",default=None,type=int)
    args = parser.parse_args()
    default_min_len = args.min_output
    default_max_len = args.max_output
    default_temp = args.temp_output
    default_number_of_outputs = args.num_output
    gpt6b = args.sixB
    model_to_load = args.model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if gpt6b is not None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = GPTNeoForCausalLM.from_pretrained(model_to_load).half().to("cuda")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_to_load)
        model = GPTNeoForCausalLM.from_pretrained(model_to_load).to("cuda")
    # model = model.half()

    model.eval()
    print("model loaded")

    while True:
        clear()
        prompt = str(input("What prompt do you want to give the AI? "))
        prompt = "<|endoftext|>" + prompt
        try: 
            min_output_length = int(input("Minimum output length? "))
        except:
            min_output_length = default_min_len
        try: 
            max_output_length = int(input("Maximum output length? "))
        except:
            max_output_length = default_max_len
        try:
            temp = float(input("What temperature to use? "))
        except:
            temp = default_temp
        try:
            number_of_outputs = int(input("Number of outputs? "))
        except:
            number_of_outputs = default_number_of_outputs


        encoding = tokenizer(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            generated_ids = model.generate(**encoding,do_sample=True, temperature=temp, max_length=max_output_length,min_length=min_output_length,num_return_sequences=number_of_outputs)
        generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)
        for text in generated_texts:
            text = fix_text(text)
            print()
            print(text)

        delay=str(input())
        if delay == "0":
            break
