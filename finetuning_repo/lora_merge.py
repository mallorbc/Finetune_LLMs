import torch
from peft import PeftModel
import os
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,AutoConfig
import argparse
from utils import get_logger
from accelerate import Accelerator
logger = get_logger("merge", "info")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bm","--base_model", type=str, required=True,help="Give the name of the model as it appears on the HuggingFace Hub")
    parser.add_argument("-lm","--lora_model", type=str, required=True,help="Give the path to the Lora model")
    parser.add_argument("-o","--output", type=str, default="merged_model",help="Give the path to the output folder")
    parser.add_argument("--cpu", action="store_true", default=False)

    
    args = parser.parse_args()
    args.output = os.path.realpath(args.output)

    BASE_MODEL = args.base_model
    LORA_WEIGHTS = os.path.realpath(args.lora_model)
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


    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=device_map,
        offload_folder="offload", 
        trust_remote_code=True,
        quantization_config=None
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

    tokenizer = AutoTokenizer.from_pretrained(LORA_WEIGHTS)
    tokenizer.save_pretrained(args.output)