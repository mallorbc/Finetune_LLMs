import pandas as pd
import os

train_df = pd.read_csv('./GPT/train.csv')
valid_df = pd.read_csv('./GPT/validation.csv')

old_eos = "<|endoftext|>"

new_bos = "<s>"
new_eos = "</s>"

#loop through each row under text column and replace <|endoftext|> with </s>
for i in range(len(train_df)):
    old_text = train_df['text'][i]
    new_text = old_text.replace(old_eos,"")
    new_text = new_bos + new_text + new_eos
    # print(new_text)
    train_df['text'][i] = new_text



for i in range(len(valid_df)):
    old_text = train_df['text'][i]
    new_text = old_text.replace(old_eos,"")
    new_text = new_bos + new_text + new_eos
    valid_df['text'][i] = new_text

#save the new csv files
output_path = './llama/'
output_path = os.path.realpath(output_path)
os.makedirs(output_path, exist_ok=True)

train_path = os.path.join(output_path, 'train.csv')
valid_path = os.path.join(output_path, 'validation.csv')
train_df.to_csv(train_path, index=False)
valid_df.to_csv(valid_path, index=False)