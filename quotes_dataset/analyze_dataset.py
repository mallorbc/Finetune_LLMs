from transformers import AutoTokenizer
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("-m","--model_name", type=str, default="EleutherAI/gpt-j-6B")
parser.add_argument("-d","--dataset", type=str, default="./llama/")
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.model_name)
dataset_dir = os.path.realpath(args.dataset)

train_file = os.path.join(dataset_dir, "train.csv")
valid_file = os.path.join(dataset_dir, "validation.csv")

train_df = pd.read_csv(train_file)
valid_df = pd.read_csv(valid_file)
combined_df = pd.concat([train_df, valid_df])

#loop through each row under text column and tokenize using iterows
all_sizes = []
for index, row in tqdm(combined_df.iterrows(), total=len(combined_df)):
    text = row['text']
    tokenized_text = tokenizer(text, return_tensors="np")
    tokenized_size = np.shape(tokenized_text['input_ids'])[-1]
    all_sizes.append(tokenized_size)

print("Max size: ", max(all_sizes))
print("Min size: ", min(all_sizes))
print("Mean size: ", np.mean(all_sizes))
print("Median size: ", np.median(all_sizes))



