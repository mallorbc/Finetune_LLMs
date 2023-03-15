import pandas as pd
import os

train_df = pd.read_csv('./GPT/train.csv')
valid_df = pd.read_csv('./GPT/validation.csv')

#loop through each row under text column and replace <|endoftext|> with </s>
for i in range(len(train_df)):
    train_df['text'][i] = train_df['text'][i].replace('<|endoftext|>', '')

for i in range(len(valid_df)):
    valid_df['text'][i] = valid_df['text'][i].replace('<|endoftext|>', '')

#save the new csv files
output_path = './llama/'
output_path = os.path.realpath(output_path)
os.makedirs(output_path, exist_ok=True)

train_path = os.path.join(output_path, 'train.csv')
valid_path = os.path.join(output_path, 'validation.csv')
train_df.to_csv(train_path, index=False)
valid_df.to_csv(valid_path, index=False)