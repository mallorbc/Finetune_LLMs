import torch
from peft import PeftModel
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
)
import argparse
from utils import get_logger
import transformers
from typing import Dict, List, Optional
import json
import sys

logger = get_logger("merge", "info")

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    custom_tokens: Optional[List[str]] = None,
) -> bool:
    """
    Resize tokenizer and embedding if new special/custom tokens are provided.

    Note: This is the unoptimized version that may make your embedding size
    not divisible by 64.
    """
    if len(special_tokens_dict) == 0 and custom_tokens is None:
        return False

    logger.info("Resizing tokenizer and embedding...")
    if special_tokens_dict:
        logger.info("Special tokens dict: %s", special_tokens_dict)
    if custom_tokens:
        logger.info("Custom tokens: %s", custom_tokens)

    num_new_tokens = len(special_tokens_dict) + (len(custom_tokens) if custom_tokens else 0)
    logger.info("Number of new tokens: %d", num_new_tokens)

    # Add tokens
    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)
    if custom_tokens:
        tokenizer.add_tokens(custom_tokens, special_tokens=True)

    model.resize_token_embeddings(len(tokenizer))

    # Initialize new embeddings as average of existing ones
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_avg
        output_embeddings[-num_new_tokens:] = output_avg

    return True


def main():
    parser = argparse.ArgumentParser(description="Merge base and LoRA models")
    parser.add_argument(
        "-bm",
        "--base_model",
        type=str,
        default=None,
        help="Name of the base model on Hugging Face Hub",
    )
    parser.add_argument(
        "-lm", "--lora_model", type=str, required=True, help="Path to the LoRA model"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to save merged model"
    )
    parser.add_argument("--cpu", action="store_true", default=False, help="Force CPU usage")
    parser.add_argument("--context_size", type=int, default=None, help="Context size")
    parser.add_argument("--custom_tokens", type=str, default=None, help="Path to custom tokens file")
    parser.add_argument("--pad_token_id", type=int, default=None, help="Pad token ID override")

    args = parser.parse_args()
    args.output = os.path.realpath(args.output)
    LORA_WEIGHTS = os.path.realpath(args.lora_model)

    # Resolve base model
    if args.base_model:
        BASE_MODEL = args.base_model
        logger.info("Using base model: %s", BASE_MODEL)
    else:
        adapter_config_path = os.path.join(LORA_WEIGHTS, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            logger.error("adapter_config.json not found and no base model provided.")
            sys.exit(1)
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        BASE_MODEL = adapter_config["base_model_name_or_path"]
        logger.info("Base model not given, using from adapter config: %s", BASE_MODEL)

    # Device map
    if args.cpu:
        device_map = {"": "cpu"}
        logger.info("Using CPU (slow)")
    else:
        device_map = "auto"
        logger.info("Using Auto device map")

    os.makedirs("offload", exist_ok=True)

    # Load tokenizer
    config = AutoConfig.from_pretrained(BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        model_max_length=args.context_size if args.context_size else config.max_position_embeddings,
    )

    # Load custom tokens
    custom_tokens = None
    if args.custom_tokens:
        with open(os.path.realpath(args.custom_tokens)) as f:
            custom_tokens = [line.strip() for line in f if line.strip()]

    # Optional pad token override
    if args.pad_token_id is not None:
        logger.info("Using pad token id %d", args.pad_token_id)
        tokenizer.pad_token_id = args.pad_token_id
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(args.pad_token_id)

    # Ensure required special tokens
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # Load base model
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map=device_map,
        offload_folder="offload",
        trust_remote_code=True,
    )

    # Resize embeddings if needed
    _ = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
        custom_tokens=custom_tokens,
    )

    # Load and merge LoRA model
    logger.info("Loading LoRA model...")
    lora_model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map=device_map,
        offload_folder="offload",
    )

    os.makedirs(args.output, exist_ok=True)
    logger.info("Merging model...")
    lora_model = lora_model.merge_and_unload()

    logger.info("Saving merged model to %s", args.output)
    lora_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info("✅ Merge complete. Model and tokenizer saved.")


if __name__ == "__main__":
    main()
