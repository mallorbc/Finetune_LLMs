import argparse
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,TrainingArguments,AutoConfig
from datasets import Dataset
import torch
import logging
import os
from peft import LoraConfig, TaskType,get_peft_model,prepare_model_for_kbit_training
import pandas as pd





from utils import get_logger
logger = get_logger("finetune", "info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("-t","--token", type=str, default=None)
    parser.add_argument("--split_model", action="store_true",default=False)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    parser.add_argument("-lr","--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_steps",type=int, default=10)
    parser.add_argument("--eval_steps",type=int, default=10)
    parser.add_argument("--save_steps",type=int, default=10)
    parser.add_argument("-e","--epochs",  type=int,default=1)
    parser.add_argument("-b","--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)

    parser.add_argument("-tf","--train_file", type=str, required=True)
    parser.add_argument("-vf","--validation_file", type=str, required=True)
    parser.add_argument("-s","--save_limit", type=int, default=1)
    
    parser.add_argument("--use_int4", action="store_true", default=False)
    parser.add_argument("--use_int8", action="store_true", default=False)
    parser.add_argument("--disable_lora", action="store_true", default=False)
    parser.add_argument("--disable_flash_attention", action="store_true", help="Disable flash attention", default=False)

    
    parser.add_argument("--pad_token_id", default=None, type=int, help="The end of sequence token.")
    parser.add_argument("--add_eos_token", action="store_true", help="Add EOS token to tokenizer", default=False)
    parser.add_argument("--add_bos_token",  action="store_true", help="Add BOS token to tokenizer", default=False)

    parser.add_argument("--train_dataset_ratio",default=1.0,type=float,help="Ratio of the training dataset to use")
    parser.add_argument("--validation_dataset_ratio",default=1.0,type=float,help="Ratio of the validation dataset to use")
    args = parser.parse_args()

    if args.token is None:
        access_token = os.getenv("HF_TOKEN", "")
    else:
        access_token = args.token

    config_kwargs = {
        "trust_remote_code": True if args.trust_remote_code else None,
        "token":access_token
    }
    config = AutoConfig.from_pretrained(args.model_name, **config_kwargs)

    config.use_cache = False
    config_dict = config.to_dict()
    model_type = config_dict["model_type"]


    use_flash_attention = False

    if not args.disable_flash_attention and model_type != "llama":
        logger.info("Model is not llama, disabling flash attention...")
    elif args.disable_flash_attention and model_type == "llama":
        logger.info("Model is llama, could be using flash attention...")
    elif not args.disable_flash_attention and torch.cuda.get_device_capability()[0] >= 8:
        from llama_patch import replace_attn_with_flash_attn
        logger.info("Using flash attention for llama...")
        replace_attn_with_flash_attn()
        use_flash_attention = True





    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = "GPT_finetuning"

    if args.split_model:
        logger.info("Splitting the model across all available devices...")
        kwargs = {"device_map":"auto"}
    else:
        kwargs = {"device_map":None}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=access_token,trust_remote_code=args.trust_remote_code,add_eos_token=args.add_eos_token,add_bos_token=args.add_bos_token,use_fast=True)
    #THIS IS A HACK TO GET THE PAD TOKEN ID NOT TO BE EOS
    #good one for LLama is 18610
    if args.pad_token_id is not None:
        logger.info("Using pad token id %d", args.pad_token_id)
        tokenizer.pad_token_id = args.pad_token_id

    if tokenizer.pad_token_id is None:
        logger.info("Pad token id is None, setting to eos token id...")
        tokenizer.pad_token_id = tokenizer.eos_token_id


    block_size = args.block_size
    logger.info("Using a block size of %d", block_size)

    if args.use_int4:
        logger.info("Using int4 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        optimizer = "adamw_bnb_8bit"
    elif args.use_int8:
        logger.info("Using int8 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        optimizer = "adamw_bnb_8bit"
    else:
        logger.info("Using no quantization")
        bnb_config = None
        optimizer = "adamw_torch"


    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
    )



    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=access_token,quantization_config=bnb_config,trust_remote_code=args.trust_remote_code,torch_dtype=torch_dtype,config=config, **kwargs)


    if use_flash_attention:
        from llama_patch import llama_forward_with_flash_attn
        assert model.model.layers[0].self_attn.forward.__doc__ == llama_forward_with_flash_attn.__doc__, "Model is not using flash attention"

    if not args.disable_lora:
        logger.info("Using LORA...")
        if args.use_int4 or args.use_int8:
            logger.info("Preparing model for kbit training...")
            model = prepare_model_for_kbit_training(model)
            if use_flash_attention:
                from llama_patch import upcast_layer_for_flash_attention
                logger.info("Upcasting flash attention layers...")
                model = upcast_layer_for_flash_attention(model, torch_dtype)
        logger.info("Getting PEFT model...")
        model = get_peft_model(model, peft_config)
    else:
        logger.info("Using Full Finetuning")


    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        num_train_epochs=args.epochs,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.log_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size*2,
        optim=optimizer,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        weight_decay=args.weight_decay,
        report_to="wandb",
        load_best_model_at_end=True,
        save_total_limit=args.save_limit,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
    )

    train_df = pd.read_csv(args.train_file)
    if args.train_dataset_ratio < 1.0:
        train_df = train_df.sample(frac=args.train_dataset_ratio)
    train_dataset = Dataset.from_pandas(train_df)
    validation_df = pd.read_csv(args.validation_file)
    if args.validation_dataset_ratio < 1.0:
        validation_df = validation_df.sample(frac=args.validation_dataset_ratio)
    validation_dataset = Dataset.from_pandas(validation_df)


    # get trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        dataset_text_field="text",
        max_seq_length=block_size,
        tokenizer=tokenizer,
    )

    # train
    trainer.train()