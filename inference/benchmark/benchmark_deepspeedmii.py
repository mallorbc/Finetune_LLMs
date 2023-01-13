import mii
from transformers import AutoTokenizer
import time
import argparse
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--num_samples", type=int, default=5)
    args = parser.parse_args()

    generator = mii.mii_query_handle("deployment")

    input_text = "<|endoftext|> Hi"
    tokens_to_try = [8,16,32,64,128,256,512,1024,2048]
    total_number_of_runs = 0
    try:
        for i in range(len(tokens_to_try)):
            number_of_tokens = tokens_to_try[i]
            total_times = []
            for i in range(args.num_samples):
                start = time.time()
                result = generator.query({"query": [""]}, do_sample=False, max_length=number_of_tokens,min_length=number_of_tokens)
                if i == args.num_samples-1:
                    print(result)                                               
                total_number_of_runs += 1
                end = time.time()
                total_times.append(end-start)
            print(f"Number of tokens: {number_of_tokens} | Average time: {np.mean(total_times)} | Standard Deviation: {np.std(total_times)}")
    except Exception as e:
        print(e)
        print(f"Run Number: {total_number_of_runs}")
