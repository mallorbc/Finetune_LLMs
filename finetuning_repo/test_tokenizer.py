from transformers import AutoTokenizer
import argparse
import os
import pandas as pd
import time
from utils import get_logger
logger = get_logger("test_tokenizer","info")

parser = argparse.ArgumentParser()

parser.add_argument("-f", "--file", help="input file",required=True,type=str)
parser.add_argument("--custom_tokens",help="input file",default=None,type=str)
parser.add_argument("-m", "--model", help="model name",default="meta-llama/Llama-2-7b-hf",type=str)
parser.add_argument("-t", "--text", help="text to tokenize",required=True,type=str)
args = parser.parse_args()

access_token = os.getenv("HF_TOKEN", "")


tokenizer = AutoTokenizer.from_pretrained(args.model,token=access_token,add_eos_token=False,add_special_tokens=False,add_bos_token=False,trust_remote_code=True)

if args.custom_tokens is not None:
    with open(args.custom_tokens) as f:
        token = f.readlines()
    tokens = [x.strip() for x in token]
    tokenizer.add_tokens(tokens,special_tokens=True)



text = args.text


test_tokens = tokenizer.encode(text)
logger.info("Looking for %s",test_tokens)

test_tokens_size = len(test_tokens)

df = pd.read_csv(args.file)

for index, row in df.iterrows():
    text = row["text"]
    tokens = tokenizer.encode(text)
    for i in range(len(tokens)):
        chunk = tokens[i:i+test_tokens_size]
        print(chunk)
        decoded_chunk = tokenizer.decode(chunk)
        print(decoded_chunk)
        print("\n\n")
        if chunk == test_tokens:
            logger.info("Found match")
            print(chunk)
            print(decoded_chunk)
            print(text)
            quit()

    logger.info("Looked for %s",test_tokens)
