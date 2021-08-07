from transformers import GPTNeoForCausalLM, AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()


model = GPTNeoForCausalLM.from_pretrained(args.model).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(args.model)


while True:
    text = input("\n\nInput text to prompt the model: ")
    text = str(text)
    if len(text) == 0:
        continue
    ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

    # add the length of the prompt tokens to match with the mesh-tf generation
    max_length = 400 + ids.shape[1]

    gen_tokens = model.generate(
        ids,
        do_sample=True,
        min_length=max_length,
        max_length=max_length,
        temperature=0.9,
        use_cache=True
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print("Text generated:")
    print(gen_text)
