import os
import deepspeed
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',"--model_name", type=str, default='EleutherAI/gpt-j-6B')
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

    input_text = "DeepSpeed is"
    string = generator(input_text, do_sample=True, max_length=2047,min_length=2047, top_k=50, top_p=0.95, temperature=0.9)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(string)