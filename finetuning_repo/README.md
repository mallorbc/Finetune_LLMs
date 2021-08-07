# Guide: Finetune GPT2-XL (1.5 Billion Parameters) and GPT-NEO (2.7 Billion Parameters) on a single GPU with Huggingface Transformers using DeepSpeed



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
