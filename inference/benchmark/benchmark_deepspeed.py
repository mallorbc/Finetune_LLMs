import os
import deepspeed
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import time
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',"--model_name", type=str, default='EleutherAI/gpt-j-6B')
    parser.add_argument("-n","--num_samples", type=int, default=5)
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    model_name = args.model_name

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=local_rank,torch_dtype=torch.float16)

    generator.model = deepspeed.init_inference(generator.model,
                                            mp_size=world_size,
                                            dtype=torch.half,
                                            replace_method='auto',
                                            max_tokens=2048,
                        replace_with_kernel_inject=True)
    torch.cuda.synchronize()
    input_text = "<|endoftext|> Hi"
    tokens_to_try = [8,16,32,64,128,256,512,1024,2048]
    total_number_of_runs = 0
    try:
        for i in range(len(tokens_to_try)):
            number_of_tokens = tokens_to_try[i]
            total_times = []
            for i in range(args.num_samples):
                start = time.time()
                string = generator(input_text,do_sample=False, max_length=number_of_tokens,min_length=number_of_tokens)
                if i == args.num_samples-1:
                    print(string)
                end = time.time()
                total_times.append(end-start)
                total_number_of_runs += 1

            print(f"Run number: {total_number_of_runs} | Number of tokens: {number_of_tokens} | Average time: {np.mean(total_times)} | Standard Deviation: {np.std(total_times)}")
    except Exception as e:
        print(e)
        print(f"Run Number: {total_number_of_runs}")



    # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    #     print(string)