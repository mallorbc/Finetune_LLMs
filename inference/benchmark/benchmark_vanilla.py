from transformers import AutoTokenizer
import time
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--num_samples", type=int, default=5)
    parser.add_argument('-m',"--model_name", type=str, default='EleutherAI/gpt-j-6B')
    args = parser.parse_args()
    

    model = AutoModelForCausalLM.from_pretrained(args.model_name,torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    input_text = ""
    tokens_to_try = [1,2,4,8,16,32,64,128,256,512,1024,2048]
    for i in range(len(tokens_to_try)):
        number_of_tokens = tokens_to_try[i]
        total_times = []
        for i in range(args.num_samples):
            start = time.time()
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()
            result = model.generate(input_ids, do_sample=False, max_length=number_of_tokens,min_length=number_of_tokens)
            end = time.time()
            total_times.append(end-start)
        print(f"Number of tokens: {number_of_tokens} | Average time: {np.mean(total_times)} | Standard Deviation: {np.std(total_times)}")
