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

# Original Repo Guide: Finetune GPT2-XL (1.5 Billion Parameters) and GPT-NEO (2.7 Billion Parameters) on a single GPU with Huggingface Transformers using DeepSpeed

This was the README from which I modified the code. Some useful insight may be here.

- Finetuning large language models like GPT2-xl is often difficult, as these models are too big to fit on a single GPU.
- This guide explains how to finetune GPT2-xl and GPT-NEO (2.7B Parameters) with just one command of the Huggingface Transformers library on a single GPU.
- This is made possible by using the DeepSpeed library and gradient checkpointing to lower the required GPU memory usage of the model.
- I also explain how to set up a server on Google Cloud with a V100 GPU (16GB VRAM), that you can use if you don't have a GPU with enough VRAM (12+ GB) or you don't have enough enough normal RAM (60 GB+).



## 1. (Optional) Setup VM with V100 in Google Compute Engine

Note: The GPT2-xl model does run on any server with a GPU with at least 16 GB VRAM and 60 GB RAM. The GPT-NEO model needs at least 70 GB RAM. If you use your own server and not the setup described here, you will need to install CUDA and Pytorch on it.

### Requirements

1. Install the Google Cloud SDK: [Click Here](https://cloud.google.com/sdk/docs/install)
2. Register a Google Cloud Account, create a project and set up billing (only once you set up billing, you can use the $300 dollar sign up credit for GPUs).
3. Request a quota limit increase for "GPU All Regions" to 1. [Here](https://nirbenz.github.io/gce-quota/) is a step by step guide. The UI changed a bit and looks now like [this](https://stackoverflow.com/a/62883281/15447124).
4. Log in and initialize the cloud sdk with `gcloud auth login` and `gcloud init` and follow the steps until you are set up.

### Create VM

- Replace YOURPROJECTID in the command below with the project id from your GCE project.
- You can add the `--preemptible` flag to the command below, this reduces your cost to about 1/3, but Google is then able to shut down your instance at any point. At the time of writing, this configuration only costs about $1.28 / hour in GCE, when using preemptible.
- You can change the zone, if there are no ressources available. [Here](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones) is a list of all zones and whether they have V100 GPUs. Depending on the time of the day you might need to try out a few.
- We need a GPU server with at least 60 GB RAM, otherwise the run will crash, whenever the script wants to save/pickle a model. This setup below gives us as much RAM as possible with 12 CPU cores in GCE (without paying for extended memory). You also can't use more than 12 CPU cores with a single V100 GPU in GCE.

Run this to create the instance:

```markdown
gcloud compute instances create gpuserver \
   --project YOURPROJECTID \
   --zone us-west1-b \
   --custom-cpu 12 \
   --custom-memory 78 \
   --maintenance-policy TERMINATE \
   --image-family pytorch-1-7-cu110 \
   --image-project deeplearning-platform-release \
   --boot-disk-size 200GB \
   --metadata "install-nvidia-driver=True" \
   --accelerator="type=nvidia-tesla-v100,count=1" \
```

After 5 minutes or so (the server needs to install nvidia drivers first), you can connect to your instance with the command below. If you changed the zone, you also will need to change it here.
- replace YOURSDKACCOUNT with your sdk account name 

```markdown
gcloud compute ssh YOURSDKACCOUNT@gpuserver --zone=us-west1-b
```

Don't forget to shut down the server once your done, otherwise you will keep getting billed for it. This can be done [here](https://console.cloud.google.com/compute/instance).

The next time you can restart the server from the same web ui [here](https://console.cloud.google.com/compute/instance).

## 2. Download script and install libraries

Run this to download the script and to install all libraries:

```markdown
git clone https://github.com/Xirider/finetune-gpt2xl.git
chmod -R 777 finetune-gpt2xl/
cd finetune-gpt2xl
pip install -r requirements.txt 

```

- This installs transformers from source, as the current release doesn't work well with deepspeed.

(Optional) If you want to use [Wandb.ai](http://wandb.ai) for experiment tracking, you have to login:

```markdown
wandb login
```

## 3. Finetune GPT2-xl (1.5 Billion Parameters)

Then add your training data:
- replace the example train.txt and validation.txt files in the folder with your own training data with the same names and then run `python text2csv.py`. This converts your .txt files into one column csv files with a "text" header and puts all the text into a single line. We need to use .csv files instead of .txt files, because Huggingface's dataloader removes line breaks when loading text from a .txt file, which does not happen with the .csv files.
- If you want to feed the model separate examples instead of one continuous block of text, you need to pack each of your examples into an separate line in the csv train and validation files.
- Be careful with the encoding of your text. If you don't clean your text files or if just copy text from the web into a text editor, the dataloader from the datasets library might not load them.

Run this:

```markdown
deepspeed --num_gpus=1 run_clm.py \
--deepspeed ds_config.json \
--model_name_or_path gpt2-xl \
--train_file train.csv \
--validation_file validation.csv \
--do_train \
--do_eval \
--fp16 \
--overwrite_cache \
--evaluation_strategy="steps" \
--output_dir finetuned \
--eval_steps 200 \
--num_train_epochs 1 \
--gradient_accumulation_steps 2 \
--per_device_train_batch_size 8

```

- This command runs the the standard run_clm.py file from Huggingface's examples with deepspeed, just with 2 lines added to enable gradient checkpointing to use less memory.
- Training on the Shakespeare example should take about 17 minutes. With gradient accumulation 2 and batch size 8, one gradient step takes about 9 seconds. This means the model training speed should be almost 2 examples / second. You can go up to batch size of 12 before running out of memory, but that doesn't provide any speedups.
- Note that the default huggingface optimizer hyperparameters and the hyperparameters given as flag overwrite the hyperparameters in the ds_config.json file. Therefore if you want to adjust learning rates, warmup and more, you need to set these as flags to the training command. For an example you can find further below the training command of GPT-NEO which changes the learning rate.
- You might want to try different hyperparameters like `--learning_rate` and `--warmup_steps` to improve the finetuning.



## 4. Generate text with your finetuned model

You can test your finetuned GPT2-xl model with this script from Huggingface Transfomers (is included in the folder):

```markdown
python run_generation.py --model_type=gpt2 --model_name_or_path=finetuned --length 200
```

Or you can use it now in your own code like this to generate text in batches:

```python
# credit to Niels Rogge - https://github.com/huggingface/transformers/issues/10704

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained('finetuned')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('finetuned').to(device)
print("model loaded")

# this is a single input batch with size 3
texts = ["From off a hill whose concave womb", "Another try", "A third test"]

encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)
with torch.no_grad():
    generated_ids = model.generate(**encoding, max_length=100)
generated_texts = tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True)

print(generated_texts)
```

- model inference runs on even small gpus or on cpus without any more additional changes

## Finetune GPT-NEO (2.7 Billion Parameters)

This works now. I tested it with a server with one V100 GPU (16 GB VRAM) and 78 GB normal RAM, but it might not actually need that much RAM.

Add your training data like you would for GPT2-xl:
- replace the example train.txt and validation.txt files in the folder with your own training data with the same names and then run `python text2csv.py`. This converts your .txt files into one column csv files with a "text" header and puts all the text into a single line. We need to use .csv files instead of .txt files, because Huggingface's dataloader removes line breaks when loading text from a .txt file, which does not happen with the .csv files.
- If you want to feed the model separate examples instead of one continuous block of text, you need to pack each of your examples into an separate line in the csv train and validation files.
- Be careful with the encoding of your text. If you don't clean your text files or if just copy text from the web into a text editor, the dataloader from the datasets library might not load them.

- Be sure to either login into wandb.ai with `wandb login` or uninstall it completely. Otherwise it might cause a memory error during the run.

Then start the training run this command:

```markdown
deepspeed --num_gpus=1 run_clm.py \
--deepspeed ds_config_gptneo.json \
--model_name_or_path EleutherAI/gpt-neo-2.7B \
--train_file train.csv \
--validation_file validation.csv \
--do_train \
--do_eval \
--fp16 \
--overwrite_cache \
--evaluation_strategy="steps" \
--output_dir finetuned \
--num_train_epochs 1 \
--eval_steps 15 \
--gradient_accumulation_steps 2 \
--per_device_train_batch_size 4 \
--use_fast_tokenizer False \
--learning_rate 5e-06 \
--warmup_steps 10
```

- This uses a smaller "allgather_bucket_size" setting in the ds_config_gptneo.json file and a smaller batch size to further reduce gpu memory.
- You might want to change and try hyperparameters to be closer to the orignal EleutherAi training config. You can find these [here](https://github.com/EleutherAI/gpt-neo/blob/master/configs/gpt3_2-7B_256.json).

## Generate text with a GPT-NEO 2.7 Billion Parameters model

I provided a script, that allows you to interactively prompt your GPT-NEO model. If you just want to sample from the pretrained model without finetuning it yourself, replace "finetuned" with "EleutherAI/gpt-neo-2.7B". Start it with this:

```markdown
python run_generate_neo.py finetuned
```

Or use this snippet to generate text from your finetuned model within your code:

```python
# credit to Suraj Patil - https://github.com/huggingface/transformers/pull/10848 - modified

from transformers import GPTNeoForCausalLM, AutoTokenizer

model = GPTNeoForCausalLM.from_pretrained("finetuned").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("finetuned")

text = "From off a hill whose concave"
ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

max_length = 400 + ids.shape[1] # add the length of the prompt tokens to match with the mesh-tf generation

gen_tokens = model.generate(
  ids,
  do_sample=True,
  min_length=max_length,
  max_length=max_length,
  temperature=0.9,
  use_cache=True
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)

```

## (Optional) Configuration

You can change the learning rate, weight decay and warmup by setting them as flags to the training command. Warm up and learning rates in the config are ignored, as the script always uses the Huggingface optimizer/trainer default values. If you want to overwrite them you need to use flags. You can check all the explanations here:

[https://huggingface.co/transformers/master/main_classes/trainer.html#deepspeed](https://huggingface.co/transformers/master/main_classes/trainer.html#deepspeed)

The rest of the training arguments can be provided as a flags and are all listed here:

[https://huggingface.co/transformers/master/main_classes/trainer.html#trainingarguments](https://huggingface.co/transformers/master/main_classes/trainer.html#trainingarguments)
