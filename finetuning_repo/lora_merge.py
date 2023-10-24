import torch
from peft import PeftModel
import os
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,AutoConfig
import argparse
from utils import get_logger
import transformers
from typing import Dict
import json
from typing import List, Optional

logger = get_logger("merge", "info")

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    custom_tokens:Optional[List[str]]=None,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """

    if len(list(special_tokens_dict.keys())) >0 or custom_tokens is not None:
        logger.info("Resizing tokenizer and embedding...")
        logger.info("Special tokens dict: %s", special_tokens_dict)
        logger.info("Custom tokens: %s", custom_tokens)
    else:
        return False
    num_new_tokens = len(list(special_tokens_dict.keys())) + (0 if custom_tokens is None else len(custom_tokens))
    logger.info("Number of new tokens: %d", num_new_tokens)
    if len(list(special_tokens_dict.keys())) > 0:
        tokenizer.add_special_tokens(special_tokens_dict)
    if custom_tokens is not None:
        tokenizer.add_tokens(custom_tokens,special_tokens=True)

    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bm","--base_model", type=str, default=None,help="Give the name of the model as it appears on the HuggingFace Hub")
    parser.add_argument("-lm","--lora_model", type=str, required=True,help="Give the path to the Lora model")
    parser.add_argument("-o","--output", type=str, required=True,help="Give the path to the output folder")
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--context_size", type=int, default=None, help="Context size during fine-tuning")
    parser.add_argument("--custom_tokens",type=str,default=None)
    parser.add_argument("--pad_token_id",type=int,default=None)

    
    args = parser.parse_args()
    args.output = os.path.realpath(args.output)

    LORA_WEIGHTS = os.path.realpath(args.lora_model)

    if args.base_model is not None:
        BASE_MODEL = args.base_model
        logger.info("Using base model %s", BASE_MODEL)
    else:
        adapter_config_path = os.path.join(LORA_WEIGHTS,"adapter_config.json")
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        BASE_MODEL = adapter_config["base_model_name_or_path"]
        logger.info("Base model not given, using %s", BASE_MODEL)

    if args.cpu:
        device_map = {"": "cpu"}
        logger.info("Using CPU")
        logger.warning("This will be slow, use GPUs with enough VRAM if possible")
    else:
        device_map = "auto"
        logger.info("Using Auto device map")
        logger.warning("Make sure you have enough GPU memory to load the model")


    os.makedirs("offload", exist_ok=True)

    config = AutoConfig.from_pretrained(BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        model_max_length=args.context_size if args.context_size is not None else config.max_position_embeddings,
    )   

    if args.custom_tokens is not None:
        with open(os.path.realpath(args.custom_tokens)) as f:
            custom_tokens = f.readlines()
        custom_tokens = [x.strip() for x in custom_tokens]
    else:
        custom_tokens = None

    #THIS IS A HACK TO GET THE PAD TOKEN ID NOT TO BE EOS
    #good one for LLama is 18610
    if args.pad_token_id is not None:
        logger.info("Using pad token id %d", args.pad_token_id)
        tokenizer.pad_token_id = args.pad_token_id
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(args.pad_token_id)


    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN



    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=device_map,
        offload_folder="offload", 
        trust_remote_code=True,
        quantization_config=None
    )
    
    added_tokens = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
        custom_tokens=custom_tokens
    )
    logger.info("Loading Lora model...")
        
    lora_model = PeftModel.from_pretrained(
        model, 
        LORA_WEIGHTS, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=device_map,
        offload_folder="offload", 

    )

    os.makedirs(args.output, exist_ok=True)
    logger.info("Merging model...")
    lora_model = lora_model.merge_and_unload()
    logger.info("Merge complete, saving model to %s ...", args.output)

    lora_model.save_pretrained(args.output)
    logger.info("Model saved")

    tokenizer.save_pretrained(args.output)