# New Guide Using Docker

The options for finetuning are either using nothing(only good for small models), DeepSpeed(good for full finetuning of large models), FSDP for full finetuning of large models, or the use of Lora or Qlora.  The way you run the code will change slightly based on what option you want to use, but the options are mostly the same.

Full fine-tuning is typically slightly better than using Lora or Qlora, but the requirements of the hardware are much smaller when using something like QLora.  

In terms of hardware requirements, the order is something like the following:

1. Regular finetuning(full finetuning)
2. Finetuning with DeepSpeed(full finetuning)
3. Lora(we first freeze the weights of the LLM and then train a smaller model that learns the weight gradients that are later applied to the model)
4. QLora(we train quantize then freeze the weights of the LLM and then train a smaller model that learns the weight gradients that are later applied to the model)

## DeepSpeed

DeepSpeed is an easy-to-use deep learning optimization software suite that enables unprecedented scale and speed for Deep Learning Training and Inference.  Learn more about DeepSpeed [here](https://github.com/microsoft/DeepSpeed) and [here](https://www.deepspeed.ai/) and [here](https://huggingface.co/docs/transformers/main_classes/deepspeed)

Each stage of DeepSpeed Zero requires less GPU resources at the cost of speed.  Use 1->3 as you experience lack of memory or need more resources to train.

```ds_config_stage1.json```, ```ds_config_stage2.json``` and ```ds_config_stage3.json``` are all configs that I have used depending on what system I am on and what model I am trying to train. 

## QLora and Lora

Lora is a method where rather than updating all the weights of the model, we train a separate model that learns the rank-decomposition of some of the modules in a model(typically attention or linear layers) and updates those model weights that represent this rank decomposition of the newly learned objective.  The base model does not change at all.

We can then apply the reconstructed weight updates to the base model weights to have the final model.

The idea is that we have two matrixes of 100x100 and 100x100, if we multiply them we get a 100x100 matrix.  Research has shown that LLMs have low rank.  Having a low rank means that the matrix can be represented by a smaller matrix without losing information since the larger matrix has linearly dependent matrixes. 

This means that we can represent two matrixes as two other matrixes of a lower rank.  Let's say 10.  That makes two matrics of 10x100 and 100X10 making a 10x10 matrix after the matrix multiplication, which requires fewer resources to train and since LLMs have low rank, should still be able to be reconstructed.

Lora works with quantization as well.  We can load the base model with either 16 bits precisions, bitsandbytes int8, or bitsandbytes int4. Int4 Lora is also known as QLora.

It's also likely worthwhile looking at the QLora repo and reading the paper to see how to best use this tech. QLora is less than 3 months old at this time of writing.

Lora is a bit faster than Qlora, so first try just using Lora, or using Lora with Deepspeed before using Qlora.

## Wandb

Wandb is a tool used to track and visualize all the pieces of your machine-learning pipeline, from datasets to production models.  See more [here](https://github.com/wandb/wandb) and [here](https://wandb.ai/)

This is a tool that can give you a nice web interface to track the loss values while training.  Alternatively, you could use a log file which I will go over under running.

To disable wandb, set environment variable ```WANDB_DISABLED``` to ```true```.

To do so run this:

```export WANDB_DISABLED=true```

To use, be sure to log in using ```wandb login```

After logging in, you will be given a link that gives you a token, copy and paste that token into the terminal.

By default any new run will be under the project name ```GPT_finetuning``` but you can change that by setting ```WANDB_PROJECT``` to your own value.

Example:

```export WANDB_PROJECT=Testing```

## Preparing A Dataset

Preparing a dataset is a very important step to finetuning.  There are several ways to go about preparing the dataset.  Those include the corpus model or individual entries mode.  

Corpus mode is easier, you just have many pieces of text and arrange the dataset into a csv file under a column called ```text```.  This would be good for things like code, books, etc.  You could also use this mode for allowing few-shot learning, where other entries may provide guidance to solve other problems.  Something to note, is that when running in corpus mode, the program will split entries in half in order to fill the ```--block_size``` window.  If that is acceptable, consider this mode.

Individual entries mode means you include ```<|endoftext|>``` tokens to separate entries.  This means that entries contain one example with padding, which is useful for zero-shot problems or problems that can not be split in half in still make sense.

You can find more information on creating a dataset from this [video](https://www.youtube.com/watch?v=07ppAKvOhqk&ab_channel=Brillibits)

## Finetuning Arguments

Training and finetuning a model is equal parts art and science.  If you want the best model possible, you are going to need to do a hyperparameter sweep, meaning run with many different learning rates, weight decay, etc.

The values and flags in the ```example_run.txt``` are a good starting point.  Training arguments that are supported include those used in the ```transformers``` library.  See [here](https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/trainer#transformers.TrainingArguments) for all possible flags.

This repo uses GPUs, but it may be worthwhile to look at what the orginal TPU [project](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/howto_finetune.md) suggests for finetuning.

Some extra important settings are the ```learning_rate```, ```block_size```, ```gradient_accumulation_steps```, ```per_device_train_batch_size```, ```num_gpus```, ```extra_tokens_file``` and ```group_texts```.  

If your tasks never need more than say, 512 tokens, by decreasing the ```block_size``` arg, you can reduce the memory consumption and thus have a large batch size.  

The ```learning_rate``` at 5e-6 is good, but you could start higher and decay by 10x.  

I try to have ```gradient_accumulation_steps```, ```per_device_train_batch_size``` and ```num_gpus``` multipy to 32.  Larger product from these flags can allow a larger learning rate, but typically larger batch sizes generalize worse, see [here](https://arxiv.org/pdf/1609.04836.pdf).  This will need to be experimentally determined.  Going smaller than a 32 product may be a good idea.

If you want to add tokens to the tokenizer, use the ```--extra_tokens_file``` flag.  Have the flag point to a text file that has the extra tokens you want to add.  Each token should be on its own line.  Currently, this is only support for GPTJ, as it has extra room in its output dimensions to make this work.

If you want to run in an individual entries model, you do not need to run finetuning with any special flags and the individual entries will be padded.  If you want to run in corpus mode, you need to run with the ```--group_texts``` flag, which will combine entries as needed.

Using ```weight_decay``` may be a good idea as well to help the model generalize.  A value of 0.1 was used during pretraining.

```--pad_token_id``` is used with ```trl_finetune.py``` and is needed due to the fact some models like llama2 do not have pad tokens and we need to set them to something else other than EOS.  Ideally, you would add a new token, but to prevent that, we can use a token that is never used in the dataset.  ```Make sure that the token is not used```.  Some good candidates are 18610 and 18636.

```--split_model``` is used with Lora and Qlora for ```trl_finetune.py``` and is used to split a model onto multiple GPUs during training.  Use it if you have multiple GPUs.  You can then increase your batch size at the very least, or train larger models.

## Finetuning

There are now two programs to finetune.  That being ```run_clm.py``` and ```trl_finetune.py```.  In MOST cases, I now recommend ```trl_finetune```.  The arguments are mostly the same.  The trl option uses the ```trl``` repo and is more integrated with huggingface and by extension works well with accelerate  and allows one to easily train with many different options.

If you want to use accelerate, make sure you use ```accelerate config``` before launching ```trl_finetune.py```.  You then launch the program with ```accelerate launch trl_finetune.py``` with args.

### trl Full Finetuning

```accelerate config```

Configure Accelerate to use 1 or more GPUs, and select your data type.  If the model is really large, consider FSDP if it won't load into memory.  You may want to consider using DeepSpeed as well, especially if reduces hardware requirements.  Stage 3 will offload the parameters and optimizer.  I recommend using the ds_config files here.

For smaller models, use nothing but perhaps multiple GPUs speed things up.  Then for larger models, FSDP will be faster, but cost more, and DeepSpeed will be cheaper but likely run slower, especially stage 3.

Then, let's assume that with a batch size of 1 and a gradient accumulation of 16, there are 1000 steps in an epoch and we want to save and test every 0.1 epochs.  Lets also assume that the largest item in our dataset is 1024 tokens:

```
accelerate launch trl_finetune.py --block_size 1024 --eval_steps 100 --save_steps 100 -tf train.csv -vf validation.csv -m meta-llama/Llama-2-7b-hf -b 1 --log_steps 100 -lr 5e-6 -e 1 --gradient_accumulation_steps 16 --pad_token_id=18636 --disable_lora
```

### trl Lora Finetuning

For Lora finetuning, it uses less resouces than full finetuning while being near the same performance. 

Options for running are:
1. Lora
2. Lora with Deepspeed
3. Lora with a split model


For the first two options:

```accelerate config```

Configure Accelerate to use 1 or more GPUs, and select your data type.   You may want to consider using DeepSpeed as well, especially if reduces hardware requirements.  Stage 3 will offload the parameters and optimizer.  I recommend using the ds_config files here.

For smaller models, use nothing but perhaps multiple GPUs speed things up.  For larger models, DeepSpeed can allow further offloading parameters.  If you have multiple GPUs, you will likely want ot go with option 3 and split the model

Then, let's assume that with a batch size of 1 and a gradient accumulation of 16, there are 1000 steps in an epoch and we want to save and test every 0.1 epochs.  Lets also assume that the largest item in our dataset is 1024 tokens:

```
accelerate launch trl_finetune.py --block_size 1024 --eval_steps 100 --save_steps 100 -tf train.csv -vf validation.csv -m meta-llama/Llama-2-7b-hf -b 1 --log_steps 100 -lr 5e-6 -e 1 --gradient_accumulation_steps 16 --pad_token_id=18636
```

For option 3:

We can not use accelerate here, or at least there is no reason to do so and it may cause issues.  This method loads the based model by splitting the model onto all the GPUs and then trains the Lora adapter.

Then, let's assume that with a batch size of 1 and a gradient accumulation of 16, there are 1000 steps in an epoch and we want to save and test every 0.1 epochs.  Lets also assume that the largest item in our dataset is 1024 tokens:

```
python trl_finetune.py --block_size 1024 --eval_steps 100 --save_steps 100 -tf train.csv -vf validation.csv -m meta-llama/Llama-2-7b-hf -b 1 --log_steps 100 -lr 5e-6 -e 1 --gradient_accumulation_steps 16 --pad_token_id=18636 --split_model
```

### trl QLora Finetuning

For Qlora finetuning, it uses even less resources than full Lora but is a bit slower.  Since, this shrinks the base model, it makes it possible to train larger models. 

Options for running are:
1. QLora int8
2. Qlora int4
3. Qlora int8 split
4. Qlora int4 split

int 8 may or may not be faster.  I have not tested.  If its not, use int4


We can not use accelerate here, or at least there is no reason to do so and it may cause issues.  

Option 1:

Then, let's assume that with a batch size of 1 and a gradient accumulation of 16, there are 1000 steps in an epoch and we want to save and test every 0.1 epochs.  Lets also assume that the largest item in our dataset is 1024 tokens:

```
python trl_finetune.py --block_size 1024 --eval_steps 100 --save_steps 100 -tf train.csv -vf validation.csv -m meta-llama/Llama-2-7b-hf -b 1 --log_steps 100 -lr 5e-6 -e 1 --gradient_accumulation_steps 16 --pad_token_id=18636 --use_int8
```

Option 2:

Then, let's assume that with a batch size of 1 and a gradient accumulation of 16, there are 1000 steps in an epoch and we want to save and test every 0.1 epochs.  Lets also assume that the largest item in our dataset is 1024 tokens:

```
python trl_finetune.py --block_size 1024 --eval_steps 100 --save_steps 100 -tf train.csv -vf validation.csv -m meta-llama/Llama-2-7b-hf -b 1 --log_steps 100 -lr 5e-6 -e 1 --gradient_accumulation_steps 16 --pad_token_id=18636 --use_int4
```

Option 3:

Then, let's assume that with a batch size of 1 and a gradient accumulation of 16, there are 1000 steps in an epoch and we want to save and test every 0.1 epochs.  Lets also assume that the largest item in our dataset is 1024 tokens:

```
python trl_finetune.py --block_size 1024 --eval_steps 100 --save_steps 100 -tf train.csv -vf validation.csv -m meta-llama/Llama-2-7b-hf -b 1 --log_steps 100 -lr 5e-6 -e 1 --gradient_accumulation_steps 16 --pad_token_id=18636 --use_int8 --split_model
```

Option 4:

Then, let's assume that with a batch size of 1 and a gradient accumulation of 16, there are 1000 steps in an epoch and we want to save and test every 0.1 epochs.  Lets also assume that the largest item in our dataset is 1024 tokens:

```
python trl_finetune.py --block_size 1024 --eval_steps 100 --save_steps 100 -tf train.csv -vf validation.csv -m meta-llama/Llama-2-7b-hf -b 1 --log_steps 100 -lr 5e-6 -e 1 --gradient_accumulation_steps 16 --pad_token_id=18636 --use_int4 --split_model
```


### run_clm.py Full Finetuning Example(NOT RECCOMENDED)

```
deepspeed --num_gpus=1 run_clm.py --deepspeed ds_config_stage3.json --model_name_or_path EleutherAI/gpt-j-6B --train_file train.csv --validation_file validation.csv --do_train --do_eval --fp16 --overwrite_cache --evaluation_strategy=steps --output_dir finetuned --num_train_epochs 12  --eval_steps 20 --gradient_accumulation_steps 32 --per_device_train_batch_size 1 --use_fast_tokenizer False --learning_rate 5e-06 --warmup_steps 10 --save_total_limit 1 --save_steps 20 --save_strategy steps --tokenizer_name gpt2 --load_best_model_at_end=True --block_size=2048 --report_to=wandb
```

To change what model you run, simply change ```--model_name_or_path```.  Most causal models on HuggingFace are supported.  GPT2, GPT Neo, GPTJ, OPT, MPT, LLama, Falcon, etc

### run_clm.py QLora Example(NOT RECCOMENDED)

The arguments for QLora are the same except for a few things.  We want to raise the learning rate to something like 1e-4, not use deepspeed, and use two new flags.

Those flags are ```--use_lora``` and ```--lora_bits```

```
python run_clm.py --use_lora True --lora_bits 4 --model_name_or_path EleutherAI/gpt-j-6B --train_file train.csv --validation_file validation.csv --do_train --do_eval --fp16 --overwrite_cache --evaluation_strategy=steps --output_dir finetuned --num_train_epochs 12  --eval_steps 20 --gradient_accumulation_steps 32 --per_device_train_batch_size 1 --use_fast_tokenizer False --learning_rate 1e-04 --warmup_steps 10 --save_total_limit 1 --save_steps 20 --save_strategy steps --tokenizer_name gpt2 --load_best_model_at_end=True --block_size=2048 --report_to=wandb
```

## Lora and Qlora merging

To use your Lora models, you should merge the models.  To do so, use the ```lora_merge.py``` file.

See the flags using ```-h```

The flags are ```-bm```, ```-lm``` and ```-o```.  Those stand for base model, lora model, and output.

Give the name or path of the base model, give the path or the lora checkpoint, and give the name of the folder to output too:

Example:
```python lora_merge.py -bm meta-llama/Llama-2-7b-hf -lm checkpoints/checkpoint-2420 -o llama2_lora_model```
